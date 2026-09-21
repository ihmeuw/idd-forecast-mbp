#!/usr/bin/env Rscript
# ============================================================================
# select_malaria_models_rocket.r
#
# Worker for the malaria model fits. One task fits a BUNDLE of specs (its cells come from
# the idd-tools manifest by --task-id) and writes ONE parquet, one row per spec. When asked
# it also writes each fitted object with a JSON sidecar (--save-fits) and each cell's
# predictions (--save-predictions). Which fits run is flag-driven:
#
#   --fit-is-fe    in-sample WITH country fixed effects        (is_*  columns)
#   --fit-oos      out-of-sample, FE present: one temporal window (--cv-strategy temporal,
#                  --train-lo/--train-hi/--test-lo/--test-hi) or k-fold CV (random /
#                  within_country)                             (oos_* / cv_* columns)
#
# Data: prepare_malaria_fit_frame() from --prep-script (default: lib/malaria_fit_frame.R
# next to this file's parent directory), read from --past-inputs, filtered by
# --inc-count-min / --pfpr-min (defaults = the selection run's 1 and 0.0001). The suitability
# variant is per spec: spec_table's optional `suit_variant` column, else mordecai_0_0.
# Formulas are NOT built here: spec_table.parquet$formula_text is fitted verbatim.
#
# Metrics are the validated 20260519 formulas, unchanged. In-sample metrics are computed on
# the rows the fit used (the formula's na.omit), read back from the fitted object.
#
# Structure mirrors forecast_malaria_admin_2s_rocket.r: argument-based helpers, no
# module-level run state; main() runs only under the --file guard at the bottom.
# ============================================================================

suppressPackageStartupMessages({
  library(glue); library(data.table); library(mgcv); library(scam); library(arrow)
  library(optparse); library(jsonlite)
})

# FE-only run: no lag covariates. MUST stay consistent with the orchestrator's --max-lag
# (asserted in main(): length(lags)==0 iff max_lag==0).
lags <- c()

# Cap arrow's COMPUTE pool to 1 (parquet I/O here is tiny). The I/O thread pool, though,
# MUST stay >= 2: io=1 is the confirmed root cause of the post-write teardown segfault
# ("caught segfault ... memory not mapped" after 'fin', wf 598192 and 599278). The output
# is always written before the crash, but the non-zero exit burns a jobmon retry.
arrow::set_cpu_count(1L)
arrow::set_io_thread_count(2L)

# constants.MAL_PAST_INPUTS_READ_PATH / MAL_PAST_INPUTS_FILENAME (the orchestrator passes
# --past-inputs explicitly; this default keeps the bare CLI behaving as before).
DEFAULT_PAST_INPUTS <- file.path(
  "/mnt/team/idd/pub/forecast-mbp", "03-modeling_data", "malaria", "past_inputs_nc", "lsae_1285",
  "current", "malaria_past_inputs.parquet")
DEFAULT_SUIT_VARIANT <- "mordecai_0_0"

this_file_path <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", ca[grepl("^--file=", ca)])
  if (length(fa)) normalizePath(fa[1]) else NA_character_
}
default_lib_dir <- function() {
  f <- this_file_path()
  if (is.na(f)) NA_character_ else normalizePath(file.path(dirname(f), "..", "lib"), mustWork = FALSE)
}
# the cell a task belongs to: task_id minus the trailing _n<ns>_s<nsc>_bin<k>
cell_of_task <- function(task_id) sub("_n[0-9]+_s[0-9]+_bin[0-9]+$", "", task_id)

env_flag <- function(name, default = FALSE) {
  v <- toupper(Sys.getenv(name, unset = if (default) "TRUE" else "FALSE"))
  v %in% c("TRUE", "T", "1", "YES")
}

safe <- function(expr) tryCatch(expr, error = function(e) NA_real_)

# ---------------------------------------------------------------------------
# Packing helpers for the per-term columns.
# ---------------------------------------------------------------------------
pack_kv <- function(nms, vals, digits = 6L) {
  if (!length(nms)) return(NA_character_)
  vstr <- ifelse(is.na(vals), "NA", formatC(vals, format = "g", digits = digits))
  paste(sprintf("%s=%s", nms, vstr), collapse = ";")
}
# Non-country parametric coefficients (intercept + linear covariate slopes).
# Excludes smooth-basis coefs (via fit$nsdf, mgcv puts parametric first) and the
# A0_af country dummies (by name).
nonfe_coefs_packed <- function(fit) {
  cf <- coef(fit)
  npar <- suppressWarnings(as.integer(fit$nsdf))
  # lm has no $nsdf -> as.integer(NULL) is integer(0) (length 0, not NA), which breaks a
  # bare `if (!is.na(npar) && ...)`. Require length-1 before testing.
  use_npar <- length(npar) == 1L && !is.na(npar) && npar >= 1L
  para_idx <- if (use_npar) {
    seq_len(npar)
  } else {
    smooth_idx <- unlist(lapply(fit$smooth, function(s) s$first.para:s$last.para))
    setdiff(seq_along(cf), smooth_idx)
  }
  p <- cf[para_idx]
  p <- p[!grepl("^A0_af", names(p))]
  pack_kv(names(p), unname(p))
}
smooth_edf_packed <- function(fit) {
  if (!length(fit$smooth)) return(NA_character_)
  lab <- vapply(fit$smooth, function(s) s$label, character(1))
  edf <- vapply(fit$smooth, function(s) sum(fit$edf[s$first.para:s$last.para]), numeric(1))
  pack_kv(lab, edf, digits = 4L)
}
smooth_sp_packed <- function(fit) {
  sp <- fit$sp
  if (!length(sp)) return(NA_character_)
  pack_kv(names(sp), unname(sp))
}

# ---------------------------------------------------------------------------
# One in-sample metric block (works for both the FE and the no-FE fit). Returns
# a generic-named list; the caller prefixes with "is" or "is_no_fe". Computes a
# fully-typed NA row when fit is NULL so chunk rows rbind cleanly.
# `pfpr_actual` must be the observed PfPR of exactly the rows the fit used.
# Metric formulas are identical to fit_malaria_models_rocket_bfgs.r.
# ---------------------------------------------------------------------------
is_block_metrics <- function(fit, pfpr_actual) {
  if (is.null(fit)) {
    return(list(
      aic = NA_real_, bic = NA_real_, loglik = NA_real_, deviance = NA_real_,
      null_deviance = NA_real_, dev_expl = NA_real_, r_sq = NA_real_, ssr = NA_real_,
      rmse = NA_real_, mae = NA_real_, pfpr_ssr = NA_real_, pfpr_rmse = NA_real_, pfpr_mae = NA_real_,
      pfpr_r = NA_real_, n_obs = NA_integer_, n_coef = NA_integer_,
      edf_sum = NA_real_, edf_smooth = NA_real_, edf_parametric = NA_real_,
      n_smooths = NA_integer_, max_sp = NA_real_, min_sp = NA_real_,
      smooth_labels = NA_character_, coef = NA_character_,
      edf_per_smooth = NA_character_, sp_per_smooth = NA_character_))
  }
  n_smooths_v <- length(fit$smooth)
  edf_sum     <- safe(sum(fit$edf))
  edf_smooth  <- if (n_smooths_v > 0) {
    safe(sum(sapply(fit$smooth, function(s) sum(fit$edf[s$first.para:s$last.para]))))
  } else 0
  smooth_labels <- if (n_smooths_v > 0) {
    paste(vapply(fit$smooth, function(s) s$label, character(1)), collapse = ";")
  } else NA_character_
  # Engine-agnostic response + fitted: lm has no $y/$deviance/$null.deviance; gam/scam do.
  # Gaussian identity link here, so response residuals reconstruct y, gaussian deviance
  # == RSS == fit$deviance, and gaussian null deviance == sum((y-mean(y))^2) -> gam/scam
  # numbers are unchanged; lm now computes cleanly instead of NULL/length-0.
  fitv       <- as.numeric(fitted(fit))
  y          <- fitv + as.numeric(residuals(fit))
  stopifnot("pfpr_actual must align with the rows the fit used" = length(pfpr_actual) == length(fitv))
  null_dev   <- safe(sum((y - mean(y))^2))
  pfpr_preds <- plogis(fitv)
  has_sp     <- length(fit$sp) > 0
  list(
    aic            = safe(AIC(fit)),
    bic            = safe(BIC(fit)),
    loglik         = safe(as.numeric(logLik(fit))),
    deviance       = safe(as.numeric(deviance(fit))),
    null_deviance  = null_dev,
    dev_expl       = safe(1 - as.numeric(deviance(fit)) / null_dev),
    r_sq           = safe(1 - sum((y - fitv)^2) / sum((y - mean(y))^2)),
    ssr            = safe(sum((y - fitv)^2)),   # in-sample residual sum of squares
    rmse           = safe(sqrt(mean((y - fitv)^2))),
    mae            = safe(mean(abs(y - fitv))),
    pfpr_ssr       = safe(sum((pfpr_actual - pfpr_preds)^2)),   # in-sample SSR, PfPR space
    pfpr_rmse      = safe(sqrt(mean((pfpr_actual - pfpr_preds)^2))),
    pfpr_mae       = safe(mean(abs(pfpr_actual - pfpr_preds))),
    pfpr_r         = safe(cor(pfpr_actual, pfpr_preds)),
    n_obs          = safe(as.integer(nobs(fit))),
    n_coef         = safe(as.integer(length(coef(fit)))),
    edf_sum        = edf_sum,
    edf_smooth     = edf_smooth,
    edf_parametric = safe(edf_sum - edf_smooth),
    n_smooths      = as.integer(n_smooths_v),
    max_sp         = if (has_sp) safe(max(fit$sp)) else NA_real_,
    min_sp         = if (has_sp) safe(min(fit$sp)) else NA_real_,
    smooth_labels  = smooth_labels,
    coef           = nonfe_coefs_packed(fit),
    edf_per_smooth = smooth_edf_packed(fit),
    sp_per_smooth  = smooth_sp_packed(fit))
}
prefix_list <- function(lst, prefix) setNames(lst, paste0(prefix, "_", names(lst)))

# ---------------------------------------------------------------------------
# OOS evaluation, FE-present throughout. Two strategies:
#   within_country / random : n_folds-fold CV (each row assigned a fold).
#   temporal                : ONE split by year_id -- fit on [train_lo, train_hi],
#                             predict [test_lo, test_hi]. Test rows whose A0 is
#                             absent from the training years are dropped (no FE to
#                             estimate); count reported. cv_fold_* then describe
#                             the single split.
# `on_fold(k, fold_res, train_idx, test_idx, preds_k)` is called after every successful
# fold fit (the caller saves the object / collects predictions there).
# When write_summary, each fold's training-fit stripped_summary is collated into
# oos_summary_<task_id>.txt (one section per fold).
# ---------------------------------------------------------------------------
run_oos <- function(past_data, cv_strategy, n_folds = 5L, cv_seed = 42L,
                    optimizer = "bfgs", maxit = 300L,
                    n_scams = NA_integer_, n_smooths = NA_integer_,
                    train_lo = NA, train_hi = NA, test_lo = NA, test_hi = NA,
                    out_dir = NULL, task_id = NULL, spec_index = NA, write_summary = FALSE, fml = NULL,
                    on_fold = NULL) {
  stopifnot("run_oos requires a model formula (fml)" = inherits(fml, "formula"))
  cv_t0 <- Sys.time()
  n <- nrow(past_data)

  # Build a per-row fold vector (NA = excluded). temporal_train_idx set only for temporal.
  temporal_train_idx <- NULL
  if (cv_strategy == "within_country") {
    set.seed(cv_seed)
    folds <- ave(seq_len(n), past_data$A0_location_id,
                 FUN = function(i) sample(rep_len(seq_len(n_folds), length(i))))
    n_eff <- n_folds
    message(glue("  Fold sizes: {paste(as.integer(table(folds)), collapse=', ')}"))
  } else if (cv_strategy == "random") {
    set.seed(cv_seed)
    folds <- sample(rep(seq_len(n_folds), length.out = n))
    n_eff <- n_folds
  } else if (cv_strategy == "temporal") {
    stopifnot("temporal needs train/test year bounds" =
                all(!is.na(c(train_lo, train_hi, test_lo, test_hi))))
    yr <- past_data$year_id
    train_mask <- yr >= train_lo & yr <= train_hi
    test_mask  <- yr >= test_lo  & yr <= test_hi
    train_a0   <- unique(past_data$A0_location_id[train_mask])
    unseen     <- test_mask & !(past_data$A0_location_id %in% train_a0)
    if (any(unseen))
      message(glue("  temporal: dropping {sum(unseen)} test rows w/ A0 absent from train years"))
    test_mask  <- test_mask & !unseen
    folds <- rep(NA_integer_, n); folds[test_mask] <- 1L
    temporal_train_idx <- which(train_mask)
    n_eff <- 1L
    message(glue("  temporal: train n={length(temporal_train_idx)} [{train_lo}-{train_hi}], ",
                 "test n={sum(test_mask)} [{test_lo}-{test_hi}]"))
  } else {
    stop(glue("unknown cv_strategy '{cv_strategy}' (use random / within_country / temporal)"))
  }

  cv_iter <- rep(NA_integer_, n_eff); cv_converged <- rep(NA, n_eff)
  cv_fold_rmse <- rep(NA_real_, n_eff); cv_fold_mae <- rep(NA_real_, n_eff)
  cv_fold_pfpr_r <- rep(NA_real_, n_eff)
  cv_actuals <- numeric(n); cv_preds <- rep(NA_real_, n); cv_fold_ok <- logical(n_eff)
  fold_summaries <- character(0)

  for (k in seq_len(n_eff)) {
    test_idx  <- which(folds == k)
    train_idx <- if (cv_strategy == "temporal") temporal_train_idx else which(folds != k)
    train_data <- past_data[train_idx, ]
    if (cv_strategy == "temporal") train_data$A0_af <- droplevels(train_data$A0_af)

    fold_res        <- fit_one_mod(fml, train_data, n_scams = n_scams, n_smooths = n_smooths,
                                   optimizer = optimizer, maxit = maxit, label = glue("CV_fold_{k}"))
    cv_iter[k]      <- fold_res$iter
    cv_converged[k] <- fold_res$converged

    if (fold_res$error) {
      cv_actuals[test_idx] <- past_data$logit_malaria_pfpr[test_idx]
      cv_fold_ok[k]        <- FALSE
    } else {
      preds_k <- predict(fold_res$fit, newdata = past_data[test_idx, ])
      cv_preds[test_idx]   <- preds_k
      cv_actuals[test_idx] <- past_data$logit_malaria_pfpr[test_idx]
      cv_fold_ok[k]        <- TRUE
      resid_k           <- past_data$logit_malaria_pfpr[test_idx] - preds_k
      cv_fold_rmse[k]   <- sqrt(mean(resid_k^2, na.rm = TRUE))
      cv_fold_mae[k]    <- mean(abs(resid_k), na.rm = TRUE)
      cv_fold_pfpr_r[k] <- safe(cor(past_data$malaria_pfpr[test_idx], plogis(preds_k), use = "complete.obs"))
      if (!is.null(on_fold)) on_fold(k, fold_res, train_idx, test_idx, preds_k)
      if (write_summary && !is.null(fold_res$fit)) {
        fold_summaries <- c(fold_summaries,
          sprintf("\n===== fold %d  (train n=%d, test n=%d) =====",
                  k, length(train_idx), length(test_idx)),
          capture.output(stripped_summary(fold_res$fit)))
      }
    }
  }
  cv_elapsed <- as.numeric(difftime(Sys.time(), cv_t0, units = "secs"))

  # Collate the per-fold training-fit summaries for this OOS experiment.
  if (write_summary && length(fold_summaries) && !is.null(out_dir) && !is.null(task_id)) {
    spath <- file.path(out_dir, glue("oos_summary_{task_id}_s{spec_index}.txt"))
    tryCatch(writeLines(fold_summaries, spath),
             error = function(e) message(glue("  [oos_summary] write failed: {conditionMessage(e)}")))
    if (file.exists(spath)) Sys.chmod(spath, "0775")
  }

  valid    <- !is.na(cv_preds)
  cv_resid <- cv_actuals[valid] - cv_preds[valid]
  oos_pfpr_preds  <- plogis(cv_preds[valid])
  oos_pfpr_actual <- past_data$malaria_pfpr[valid]

  fold_join     <- function(x) paste(ifelse(is.na(x), "NA", as.character(x)), collapse = ";")
  fold_join_num <- function(x, digits = 4) paste(
    ifelse(is.na(x), "NA", sprintf(paste0("%.", digits, "f"), x)), collapse = ";")

  list(
    cv_strategy           = cv_strategy,
    cv_n_folds            = as.integer(n_eff),
    cv_seed               = as.integer(cv_seed),
    cv_train_lo           = train_lo, cv_train_hi = train_hi,
    cv_test_lo            = test_lo,  cv_test_hi  = test_hi,
    cv_elapsed_sec        = cv_elapsed,
    cv_iter_per_fold      = fold_join(cv_iter),
    cv_converged_per_fold = fold_join(cv_converged),
    cv_n_converged        = as.integer(sum(cv_converged, na.rm = TRUE)),
    oos_ssr         = safe(sum(cv_resid^2)),                            # OOS SSR, logit space
    oos_rmse        = safe(sqrt(mean(cv_resid^2))),
    oos_mae         = safe(mean(abs(cv_resid))),
    oos_r_sq        = safe(1 - sum(cv_resid^2) / sum((cv_actuals[valid] - mean(cv_actuals[valid]))^2)),
    oos_pfpr_ssr    = safe(sum((oos_pfpr_actual - oos_pfpr_preds)^2)),   # OOS SSR, PfPR space
    oos_pfpr_rmse   = safe(sqrt(mean((oos_pfpr_actual - oos_pfpr_preds)^2))),
    oos_pfpr_mae    = safe(mean(abs(oos_pfpr_actual - oos_pfpr_preds))),
    oos_pfpr_r      = safe(cor(oos_pfpr_actual, oos_pfpr_preds)),
    oos_n_obs       = as.integer(sum(valid)),
    oos_n_folds_ok  = as.integer(sum(cv_fold_ok)),
    cv_fold_rmse_per_fold   = fold_join_num(cv_fold_rmse),
    cv_fold_mae_per_fold    = fold_join_num(cv_fold_mae),
    cv_fold_pfpr_r_per_fold = fold_join_num(cv_fold_pfpr_r),
    cv_fold_pfpr_r_min      = safe(min(cv_fold_pfpr_r, na.rm = TRUE)),
    cv_fold_pfpr_r_max      = safe(max(cv_fold_pfpr_r, na.rm = TRUE)),
    cv_fold_pfpr_r_sd       = safe(sd(cv_fold_pfpr_r,  na.rm = TRUE)))
}

# ---------------------------------------------------------------------------
# Human-readable per-model summary (residuals, parametric coefs, smooth-term
# table, adj R^2, deviance explained, natural-space R^2). Ported from
# malaria_model_explore.r; ilogit -> plogis so it depends on nothing external.
# Captured to summary_<task_id>.txt when --write-summary is set.
# ---------------------------------------------------------------------------
stripped_summary <- function(mod) {
  s <- summary(mod)
  is_gam_like <- inherits(mod, "scam") || inherits(mod, "gam")
  cat("\nResiduals:\n"); print(summary(mod$residuals))
  if (is_gam_like) {
    cat("\nParametric coefficients:\n")
    p <- s$p.table[!grepl("A0_", rownames(s$p.table)), , drop = FALSE]
    printCoefmat(p, signif.stars = TRUE, signif.legend = FALSE)
    cat("\nApproximate significance of smooth terms:\n")
    printCoefmat(s$s.table, has.Pvalue = TRUE, signif.stars = TRUE)
    df_resid <- if (!is.null(s$residual.df)) s$residual.df else mod$df.residual
    cat("\nResidual standard error:", round(sqrt(s$scale), 4),
        "on", round(df_resid, 1), "effective degrees of freedom\n")
    cat("Adjusted R-squared:", round(s$r.sq, 4),
        ",  Deviance explained:", paste0(round(100 * s$dev.expl, 2), "%"), "\n")
  } else {
    cat("\nCoefficients:\n")
    cf <- s$coefficients[!grepl("A0_", rownames(s$coefficients)), , drop = FALSE]
    printCoefmat(cf, signif.stars = TRUE)
    cat("\nResidual standard error:", round(s$sigma, 4), "on", s$df[2], "degrees of freedom\n")
    cat("Multiple R-squared:", round(s$r.squared, 4),
        ",  Adjusted R-squared:", round(s$adj.r.squared, 4), "\n")
    cat("F-statistic:", sprintf("%.3e", s$fstatistic[1]), "on",
        s$fstatistic[2], "and", s$fstatistic[3], "DF,  p-value: < 2.2e-16\n")
  }
  fitted_natural   <- plogis(mod$fitted.values)
  observed_natural <- plogis(mod$fitted.values + mod$residuals)
  cat("\nR-squared in natural space:", round(cor(fitted_natural, observed_natural)^2, 4), "\n")
}

# ---------------------------------------------------------------------------
# Fit one spec under the active flags -> one-row data.table. `run_info` is the run-level
# record every sidecar carries (thresholds, inputs and their hashes, task identity).
# ---------------------------------------------------------------------------
fit_one_spec <- function(spec_index, data, flags, task_id, fml_text,
                         n_scams, n_smooths, suit_variant = DEFAULT_SUIT_VARIANT,
                         run_info = list()) {
  optimizer <- flags$optimizer; maxit <- flags$maxit
  fml <- as.formula(fml_text)   # prep is the single source of truth for the formula
  if (!identical(suit_variant, attr(data, "suit_variant"))) data <- add_suit_terms(data, suit_variant)
  engine <- fit_engine(n_scams, n_smooths)
  cell   <- cell_of_task(task_id)
  row <- data.table(
    spec_index    = as.integer(spec_index),
    engine        = engine,
    suit_variant  = suit_variant,
    optimizer     = optimizer,
    maxit_setting = as.integer(maxit),
    r_version     = paste(R.version$major, R.version$minor, sep = "."),
    mgcv_version  = as.character(packageVersion("mgcv")),
    scam_version  = as.character(packageVersion("scam")),
    inc_count_min = as.numeric(flags$inc_count_min),
    pfpr_min      = as.numeric(flags$pfpr_min),
    n_rows_frame  = as.integer(nrow(data)),
    past_inputs_sha256 = if (is.null(run_info$past_inputs_sha256)) NA_character_ else run_info$past_inputs_sha256,
    prep_script_sha256 = if (is.null(run_info$prep_script_sha256)) NA_character_ else run_info$prep_script_sha256,
    did_is_fe     = flags$is_fe,
    did_oos       = flags$oos)
  spec_rec <- list(spec_index = as.integer(spec_index), formula_text = fml_text,
                   suit_variant = suit_variant, engine = engine)

  # Save one fitted object + its sidecar (object first, so a sidecar implies its object).
  save_fit <- function(fit, fit_res, cell_kind, fold = NULL, cell_extra = list()) {
    paths <- fit_paths(flags$out_dir, cell, spec_index, fold)
    sidecar <- fit_sidecar(fit, fit_res, fml, spec = spec_rec,
                           cell = c(list(cell = cell, cell_kind = cell_kind, fold = fold), cell_extra),
                           run = run_info, stripped = flags$strip_fits)
    write_rds_atomic(if (flags$strip_fits) strip_scam_for_predict(fit) else fit, paths$rds)
    write_json_atomic(sidecar, paths$json)
    message(glue("  saved {paths$rds}"))
  }

  # --- IS with FE ---
  if (flags$is_fe) {
    res  <- fit_one_mod(fml, data, n_scams = n_scams, n_smooths = n_smooths,
                        optimizer = optimizer, maxit = maxit, label = glue("IS_FE_{spec_index}"))
    used <- if (res$error) integer(0) else fit_rows_used(res$fit, data)
    m    <- prefix_list(is_block_metrics(res$fit, data$malaria_pfpr[used]), "is")
    row[, `:=`(is_formula = fml_text, is_elapsed_sec = res$elapsed,
               is_error = res$error, is_error_msg = res$error_msg,
               is_iter = res$iter, is_converged = res$converged)]
    for (nm in names(m)) row[, (nm) := m[[nm]]]
    if (!res$error) {
      if (flags$save_fits) save_fit(res$fit, res, "is", cell_extra = list(n_rows_test = 0L))
      if (flags$save_predictions) {
        write_parquet_atomic(build_predictions(data, used, fml, fitted(res$fit)),
                             prediction_path(flags$out_dir, cell, spec_index))
      }
    }
    if (flags$write_summary && !res$error && !is.null(res$fit)) {
      spath <- file.path(flags$out_dir, glue("summary_{task_id}_s{spec_index}.txt"))
      tryCatch(writeLines(capture.output(stripped_summary(res$fit)), spath),
               error = function(e) message(glue("  [summary] write failed: {conditionMessage(e)}")))
      if (file.exists(spath)) Sys.chmod(spath, "0775")
    }
  }

  # --- OOS: one temporal window or k-fold CV ---
  if (flags$oos) {
    temporal  <- flags$cv_strategy == "temporal"
    cell_kind <- paste0("oos_", flags$cv_strategy)
    pred_frames <- list()
    on_fold <- function(k, fold_res, train_idx, test_idx, preds_k) {
      fold <- if (temporal) NULL else as.integer(k)
      if (flags$save_fits) {
        save_fit(fold_res$fit, fold_res, cell_kind, fold = fold,
                 cell_extra = list(train_lo = flags$train_lo, train_hi = flags$train_hi,
                                   test_lo = flags$test_lo, test_hi = flags$test_hi,
                                   cv_n_folds = if (temporal) 1L else as.integer(flags$n_folds),
                                   cv_seed = as.integer(flags$cv_seed),
                                   n_rows_test = length(test_idx)))
      }
      if (flags$save_predictions) {
        pred_frames[[length(pred_frames) + 1L]] <<- build_predictions(data, test_idx, fml, preds_k, fold = fold)
      }
    }
    oos <- run_oos(data, flags$cv_strategy, flags$n_folds, flags$cv_seed, optimizer, maxit,
                   n_scams = n_scams, n_smooths = n_smooths,
                   train_lo = flags$train_lo, train_hi = flags$train_hi,
                   test_lo  = flags$test_lo,  test_hi  = flags$test_hi,
                   out_dir  = flags$out_dir, task_id = task_id, spec_index = spec_index,
                   write_summary = flags$write_summary, fml = fml, on_fold = on_fold)
    for (nm in names(oos)) row[, (nm) := oos[[nm]]]
    if (flags$save_predictions && length(pred_frames)) {
      write_parquet_atomic(do.call(rbind, pred_frames), prediction_path(flags$out_dir, cell, spec_index))
    }
  }

  row[, total_elapsed_sec := sum(c(
    if (flags$is_fe) row$is_elapsed_sec else 0,
    if (flags$oos)   row$cv_elapsed_sec else 0), na.rm = TRUE)]
  row
}

# ============================================================================
# MAIN (one task = one chunk of specs)
# ============================================================================
# CLI options. jobmon passes --flags (it does NOT set SLURM_ARRAY_TASK_ID, so
# --task-id is required there); the legacy sbatch path sets env vars. Per value:
# CLI > env > default, so both paths work unchanged.
cli_options <- function() {
  OptionParser(option_list = list(
    make_option("--task-id",     type = "character", default = NA, help = "manifest task_id this job fits"),
    make_option("--output-dir",  type = "character", default = NA),
    make_option("--manifest",    type = "character", default = NA, help = "idd-tools manifest JSON; this task's cells (spec list) read by --task-id; default <output-dir>/manifest.json"),
    make_option("--spec-table",  type = "character", default = NA, help = "spec_table parquet (formula_text [, suit_variant]); default <output-dir>/spec_table.parquet"),
    make_option("--past-inputs", type = "character", default = NA, help = "malaria_past_inputs.parquet; default the current past-inputs snapshot"),
    make_option("--prep-script", type = "character", default = NA, help = "R file defining prepare_malaria_fit_frame(); default <lib-dir>/malaria_fit_frame.R"),
    make_option("--lib-dir",     type = "character", default = NA, help = "directory holding scam_fit_helpers.R (+ the default prep script); default ../lib relative to this file"),
    make_option("--inc-count-min", type = "double",  default = NA, help = "row filter: malaria_inc_count >= this (default 1)"),
    make_option("--pfpr-min",    type = "double",    default = NA, help = "row filter: malaria_pfpr >= this (default 0.0001)"),
    make_option("--save-fits",   type = "character", default = NA, help = "TRUE: write fits/<cell>/spec_<i>.rds + .json per fit (default FALSE)"),
    make_option("--save-predictions", type = "character", default = NA, help = "TRUE: write predictions/<cell>/spec_<i>.parquet per cell (default FALSE)"),
    make_option("--strip-fits",  type = "character", default = NA, help = "TRUE (default): saved objects are predict-stripped; FALSE keeps the full object"),
    make_option("--cv-strategy", type = "character", default = NA),
    make_option("--optimizer",   type = "character", default = NA),
    make_option("--maxit",       type = "integer",   default = NA),
    make_option("--cv-n-folds",  type = "integer",   default = NA),
    make_option("--cv-seed",     type = "integer",   default = NA),
    make_option("--train-lo",    type = "integer",   default = NA, help = "temporal: min train year_id"),
    make_option("--train-hi",    type = "integer",   default = NA, help = "temporal: max train year_id"),
    make_option("--test-lo",     type = "integer",   default = NA, help = "temporal: min test year_id"),
    make_option("--test-hi",     type = "integer",   default = NA, help = "temporal: max test year_id"),
    make_option("--fit-is-fe",   type = "character", default = NA),
    make_option("--fit-oos",     type = "character", default = NA),
    make_option("--write-summary", type = "character", default = NA),
    make_option("--max-lag",     type = "integer",   default = NA,
                help = "orchestrator's max_lag; asserted consistent with this worker's `lags`")))
}

main <- function() {
  opt <- parse_args(cli_options())
  names(opt) <- gsub("-", "_", names(opt))   # robust to optparse dash/underscore naming

  pick      <- function(cli, env, default) if (!is.na(cli)) cli else { v <- Sys.getenv(env, unset = ""); if (nzchar(v)) v else default }
  pick_int  <- function(cli, env, default) { v <- if (!is.na(cli)) cli else { e <- Sys.getenv(env, unset = ""); if (nzchar(e)) e else default }; as.integer(v) }
  pick_num  <- function(cli, env, default) { v <- if (!is.na(cli)) cli else { e <- Sys.getenv(env, unset = ""); if (nzchar(e)) e else default }; as.numeric(v) }
  pick_flag <- function(cli, env, default) if (!is.na(cli)) toupper(as.character(cli)) %in% c("TRUE", "T", "1", "YES") else env_flag(env, default)

  # --- CONSISTENCY GUARD: worker `lags` must match the orchestrator's --max-lag ---
  # A silent mismatch corrupts the "full data" claim -- e.g. max_lag=0 in the orchestrator
  # (train_lo=2000) but `lags` set here so a lag-completeness block thins the early years.
  .max_lag <- pick_int(opt$max_lag, "MAX_LAG", NA_integer_)
  if (!is.na(.max_lag)) {
    .expected <- if (length(lags) == 0L) 0L else as.integer(max(lags))
    if (.max_lag != .expected)
      stop(sprintf(paste0("CONSISTENCY GUARD FAILED: orchestrator --max-lag=%d but worker ",
                          "`lags`=[%s] implies max_lag=%d. The full-data edits (worker lags / ",
                          "orchestrator max_lag) must move together."),
                   .max_lag, paste(lags, collapse = ","), .expected))
  }

  out_dir <- pick(opt$output_dir, "FIT_OUTPUT_DIR", NA_character_)
  stopifnot("output dir must be set (--output-dir or FIT_OUTPUT_DIR)" = !is.na(out_dir) && nzchar(out_dir))
  this_task <- pick(opt$task_id, "SLURM_ARRAY_TASK_ID", "1")

  # --- shared code: helpers and the data preparation (the contract is the function name) ---
  lib_dir <- pick(opt$lib_dir, "MBP_LIB_DIR", default_lib_dir())
  stopifnot("cannot locate lib dir (pass --lib-dir)" = !is.na(lib_dir) && dir.exists(lib_dir))
  source(file.path(lib_dir, "scam_fit_helpers.R"))
  prep_script <- pick(opt$prep_script, "PREP_SCRIPT", file.path(lib_dir, "malaria_fit_frame.R"))
  stopifnot("prep script not found" = file.exists(prep_script))
  source(prep_script)
  stopifnot("prep script defines no prepare_malaria_fit_frame()" =
              exists("prepare_malaria_fit_frame", mode = "function"))
  if (!exists("add_suit_terms", mode = "function")) source(file.path(lib_dir, "malaria_fit_frame.R"))
  past_inputs <- pick(opt$past_inputs, "PAST_INPUTS", DEFAULT_PAST_INPUTS)
  stopifnot("past inputs not found" = file.exists(past_inputs))

  flags <- list(
    is_fe       = pick_flag(opt$fit_is_fe,   "FIT_IS_FE",   FALSE),
    oos         = pick_flag(opt$fit_oos,     "FIT_OOS",     FALSE),
    write_summary = pick_flag(opt$write_summary, "WRITE_SUMMARY", TRUE),
    save_fits   = pick_flag(opt$save_fits,   "SAVE_FITS",   FALSE),
    save_predictions = pick_flag(opt$save_predictions, "SAVE_PREDICTIONS", FALSE),
    strip_fits  = pick_flag(opt$strip_fits,  "STRIP_FITS",  TRUE),
    inc_count_min = pick_num(opt$inc_count_min, "INC_COUNT_MIN", 1),
    pfpr_min    = pick_num(opt$pfpr_min, "PFPR_MIN", 0.0001),
    cv_strategy = pick(opt$cv_strategy, "CV_STRATEGY", "within_country"),
    n_folds     = pick_int(opt$cv_n_folds, "CV_N_FOLDS", 5L),
    cv_seed     = pick_int(opt$cv_seed,    "CV_SEED",    42L),
    train_lo    = pick_int(opt$train_lo, "TRAIN_LO", NA_integer_),
    train_hi    = pick_int(opt$train_hi, "TRAIN_HI", NA_integer_),
    test_lo     = pick_int(opt$test_lo,  "TEST_LO",  NA_integer_),
    test_hi     = pick_int(opt$test_hi,  "TEST_HI",  NA_integer_),
    optimizer   = pick(opt$optimizer, "OPTIMIZER", "efs"),
    maxit       = pick_int(opt$maxit, "MAXIT", 300L),
    out_dir     = out_dir)
  stopifnot("at least one of --fit-is-fe / --fit-oos must be TRUE" =
              flags$is_fe || flags$oos)
  stopifnot("cv-strategy must be 'random', 'within_country', or 'temporal'" =
              (!flags$oos) || flags$cv_strategy %in% c("random", "within_country", "temporal"))

  # Spec list comes from the idd-tools manifest, keyed by task_id. Each cell carries
  # spec_index + n_smooths + n_scams; formula_text (and any suit_variant) come from spec_table.
  manifest_path   <- pick(opt$manifest,   "MANIFEST",   file.path(out_dir, "manifest.json"))
  spec_table_path <- pick(opt$spec_table, "SPEC_TABLE", file.path(out_dir, "spec_table.parquet"))
  mani    <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  my_task <- Filter(function(t) identical(as.character(t$task_id), this_task), mani$tasks)
  stopifnot("no manifest task matching this task_id" = length(my_task) == 1L)
  cells   <- my_task[[1]]$task_args$cells
  stopifnot("manifest task carries no cells" = length(cells) > 0)
  my_specs     <- vapply(cells, function(c) as.integer(c$spec_index), integer(1))
  nsm_lookup   <- setNames(vapply(cells, function(c) as.integer(c$n_smooths), integer(1)),
                           as.character(my_specs))
  nscam_lookup <- setNames(vapply(cells, function(c) as.integer(c$n_scams), integer(1)),
                           as.character(my_specs))
  spec_table <- as.data.table(arrow::read_parquet(spec_table_path))
  fml_lookup <- setNames(as.character(spec_table$formula_text), as.character(spec_table$spec_index))
  variant_lookup <- if ("suit_variant" %in% names(spec_table)) {
    v <- as.character(spec_table$suit_variant); v[is.na(v) | !nzchar(v)] <- DEFAULT_SUIT_VARIANT
    setNames(v, as.character(spec_table$spec_index))
  } else NULL
  message(glue("[select task {this_task}] is_fe={flags$is_fe} oos={flags$oos} (cv={flags$cv_strategy}) | ",
               "{length(my_specs)} spec(s) | save_fits={flags$save_fits} save_predictions={flags$save_predictions}"))

  t_load <- Sys.time()
  data <- prepare_malaria_fit_frame(past_inputs, flags$inc_count_min, flags$pfpr_min, DEFAULT_SUIT_VARIANT)
  message(glue("  frame: {nrow(data)} rows x {ncol(data)} cols at inc_count >= {flags$inc_count_min}, ",
               "pfpr >= {flags$pfpr_min} in {round(as.numeric(difftime(Sys.time(), t_load, units = 'secs')), 1)}s"))
  # every column a formula names must be in the frame (the prep-script contract)
  for (si in my_specs) {
    fml_text <- fml_lookup[[as.character(si)]]
    stopifnot("spec_index missing formula_text in spec_table" =
                !is.null(fml_text) && !is.na(fml_text) && nzchar(fml_text))
    absent <- setdiff(all.vars(as.formula(fml_text)), names(data))
    if (length(absent))
      stop(glue("spec {si} names columns the prepared frame lacks: {paste(absent, collapse = ', ')}"))
  }
  run_info <- list(
    inc_count_min = flags$inc_count_min, pfpr_min = flags$pfpr_min,
    past_inputs = normalizePath(past_inputs), past_inputs_sha256 = sha256_file(past_inputs),
    prep_script = normalizePath(prep_script), prep_script_sha256 = sha256_file(prep_script),
    n_rows_frame = nrow(data), optimizer = flags$optimizer, maxit = as.integer(flags$maxit),
    task_id = this_task, worker_file = this_file_path())

  rows <- lapply(my_specs, function(si) {
    key      <- as.character(si)
    variant  <- if (is.null(variant_lookup)) DEFAULT_SUIT_VARIANT else variant_lookup[[key]]
    r <- fit_one_spec(si, data, flags, this_task, fml_lookup[[key]],
                      n_scams = nscam_lookup[[key]], n_smooths = nsm_lookup[[key]],
                      suit_variant = variant, run_info = run_info)
    r[, task_id := this_task]
    r
  })

  out <- rbindlist(rows, use.names = TRUE, fill = TRUE)
  setcolorder(out, c("task_id", "spec_index"))

  # Lean parquet, one row per spec. Atomic write: temp BESIDE the target (NOT /tmp),
  # metadata-only validation (row count), then rename.
  final <- file.path(out_dir, glue("select_summary_{this_task}.parquet"))
  write_parquet_atomic(out, final)
  Sys.chmod(final, "0775")
  message(glue("Wrote {final} ({nrow(out)} rows)"))
  message("fin")
}

# --- entrypoint guard: run main() only when this file is the Rscript --file ---
.is_rscript_entrypoint <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", ca[grepl("^--file=", ca)])
  length(fa) >= 1L && grepl("select_malaria_models_rocket", fa[1], fixed = TRUE)
}
if (.is_rscript_entrypoint()) {
  main()
  # main() has written + validated + renamed the parquet by here. Exit explicitly
  # with success so an R teardown segfault (compiled-lib / thread-pool unload
  # under singularity) can't turn a finished task into a jobmon "failure".
  quit(save = "no", status = 0)
}
