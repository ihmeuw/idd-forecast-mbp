#!/usr/bin/env Rscript
# ============================================================================
# select_malaria_models_rocket.r
#
# Unified, flag-driven worker for the 2-stage malaria pfpr model-SELECTION
# exercise. One task fits a CHUNK of specs (see param_map$task_id) and writes
# ONE lean parquet, one row per spec. Which of the three fits run is controlled
# by independent env flags set by the launcher:
#
#   FIT_IS_FE     in-sample WITH country fixed effects   (is_*    columns)
#   FIT_IS_NOFE   in-sample WITHOUT country fixed effects (is_no_fe_* columns)
#   FIT_OOS       k-fold out-of-sample CV (no-FE by default; oos_*/cv_* columns)
#
# Two-stage usage:
#   STAGE 1 (wide screen):  FIT_IS_NOFE=TRUE, others FALSE, WRITE_*=FALSE.
#       Cheapest fit (1 scam, no FE, no CV); run a huge grid, then cull.
#   STAGE 2 (survivors):    FIT_OOS=TRUE (+ FIT_IS_NOFE / FIT_IS_FE as desired).
#       The expensive 5-fold CV runs only on the culled survivor set.
#
# Calibration (why screen on IS-no-FE): on 20260519 (n=4289, country_no_fe),
# is_no_fe_pfpr_r predicts oos_pfpr_r ordering at Spearman rho=0.92, and keeping
# the top 25% by is_no_fe_pfpr_r retains 100% of the top-50 OOS models.
#
# The IS-FE / IS-no-FE / OOS computations are ported from
# fit_malaria_models_rocket_bfgs.r UNCHANGED (the validated 20260519 path) and
# only reorganized into flag-gated helpers; new *_coef / *_edf_per_smooth /
# *_sp_per_smooth columns are additive. Verify an OOS spec reproduces its
# 20260519 summary before trusting this at scale.
#
# Structure mirrors forecast_malaria_admin_2s_rocket.r: argument-based helpers,
# no module-level run state; main() runs only under the --file guard at bottom.
# ============================================================================

suppressPackageStartupMessages({
  library(glue); library(data.table); library(mgcv); library(scam); library(arrow)
  library(optparse); library(jsonlite)
})

# FE-only run: no lag covariates, so no complete.cases lag-completeness restriction ->
# the worker trains FE specs on the FULL unrestricted data (2000+). MUST stay consistent
# with the orchestrator's --max-lag (asserted in main(): length(lags)==0 iff max_lag==0).
lags <- c()


# Cap arrow's COMPUTE pool to 1 (parquet I/O here is tiny). The I/O thread pool,
# though, MUST stay >= 2: arrow warns that set_io_thread_count() < 2 "may cause
# certain operations to hang or crash", and io=1 is the confirmed root cause of the
# post-write teardown segfault ("caught segfault ... memory not mapped" after 'fin')
# seen in BOTH pre- and post-migration runs (wf 598192 @ 2026-07-07 with the old
# param_map worker, and wf 599278 with the manifest worker). The output parquet is
# always written before the crash, so data is fine, but the non-zero exit makes
# jobmon burn a retry (Attempt Error -> Attempt Done). 2 is arrow's recommended floor.
arrow::set_cpu_count(1L)
arrow::set_io_thread_count(2L)

REPO_DIR       <- "/mnt/team/idd/pub/forecast-mbp"   # == constants.MODEL_ROOT (verified)
LSAE_HIERARCHY <- "lsae_1285"                        # == constants.LSAE_HIERARCHY (verified)
SUIT_VARIANT   <- "mordecai_0_0"

# Canonical past-inputs path (mirrors fit_malaria_models_rocket_bfgs.r so the
# selection worker trains on exactly the same data).
past_inputs_parquet <- function() file.path(
  REPO_DIR, "03-modeling_data", "malaria", "past_inputs_nc", LSAE_HIERARCHY, "current",
  "malaria_past_inputs.parquet")

env_flag <- function(name, default = FALSE) {
  v <- toupper(Sys.getenv(name, unset = if (default) "TRUE" else "FALSE"))
  v %in% c("TRUE", "T", "1", "YES")
}

safe <- function(expr) tryCatch(expr, error = function(e) NA_real_)

# ---------------------------------------------------------------------------
# Data load + clean. COPIED VERBATIM from fit_malaria_models_rocket_bfgs.r so
# metrics match bit-for-bit (the calibration depends on identical preprocessing).
# ---------------------------------------------------------------------------

add_a0_pfpr_lag <- function(df, lag,
                            a0_col   = "a0_malaria_pfpr",
                            group_cols = c("A0_location_id", "year_id"),
                            out_col  = NULL) {
  stopifnot(lag >= 1)
  if (is.null(out_col)) out_col <- paste0(a0_col, "_lag", lag)

  a0_lag <- unique(df[, c(group_cols, a0_col)])
  # shift the *time* key forward so a target year Y receives the value from Y - lag
  time_col <- group_cols[2]
  a0_lag[[time_col]] <- a0_lag[[time_col]] + lag
  names(a0_lag)[names(a0_lag) == a0_col] <- out_col

  merge(df, a0_lag, by = group_cols, all.x = TRUE)
}

load_past_data <- function(parquet_path, suit_variant_pick = SUIT_VARIANT) {
  past_data <- as.data.frame(arrow::read_parquet(parquet_path))
  stopifnot(nrow(unique(past_data[,c("A0_location_id","year_id")])) == nrow(unique(past_data[,c("A0_location_id","year_id","a0_malaria_pfpr")])))
  # Add lags and drop NA rows.
  if (length(lags)) {
    for (L in lags) {
      past_data <- add_a0_pfpr_lag(past_data, lag = L)
    }
    past_data <- past_data[complete.cases(past_data[, paste0("a0_malaria_pfpr_lag", lags)]), ]
  }
  
  # Incidence COUNT = rate x population (the parquet carries rate + population, NOT
  # the count). Required by the modelled-rows filter at the end — without it that
  # filter references a NULL column and silently returns ZERO rows.
  past_data$malaria_inc_count <- past_data$malaria_inc_rate * past_data$population

  nan_toss <- function(df, var) {
    to_toss <- which(is.na(df[var]))
    if (length(to_toss)) df[-to_toss, ] else df
  }
  past_data <- nan_toss(past_data, "malaria_pfpr")
  past_data <- nan_toss(past_data, "gdppc_mean")
  past_data <- nan_toss(past_data, "mal_DAH_total_per_capita")

  suit_col <- paste0("malaria_suitability_", suit_variant_pick)
  past_data$malaria_suit <- past_data[[suit_col]]
  past_data$malaria_suit_fraction     <- past_data$malaria_suit / 365
  past_data$malaria_suit_fraction     <- pmin(pmax(past_data$malaria_suit_fraction, 0.001), 0.999)
  past_data$logit_malaria_suitability <- log(past_data$malaria_suit_fraction / (1 - past_data$malaria_suit_fraction))
  past_data$do30_fraction <- past_data$days_over_30C / 365
  past_data$do30_fraction <- pmin(pmax(past_data$do30_fraction, 0.001), 0.999)
  past_data$logit_do30    <- log(past_data$do30_fraction / (1 - past_data$do30_fraction))
  past_data$rh_fraction <- past_data$relative_humidity / 100
  past_data$rh_fraction <- pmin(pmax(past_data$rh_fraction, 0.001), 0.999)
  past_data$logit_relative_humidity <- log(past_data$rh_fraction / (1 - past_data$rh_fraction))

  log_covs <- c("mal_DAH_total_per_capita", "gdppc_mean", "ldipc_mean", "med_consumppc")
  for (cov in log_covs) {
    past_data[[paste0("log_", cov)]] <- log(past_data[[cov]])
  }

  past_data <- past_data[which(past_data$malaria_inc_count >= 1 & past_data$malaria_pfpr >= 0.0001),]
  
  past_data$A0_af <- as.factor(past_data$A0_location_id)

  return(past_data)
}

# ===========================================================================
# Formulas are NOT built here. malaria_spec_design.py is the single source of
# truth: it bakes each spec's formula (per-term k / bs codes included) into
# spec_table.parquet$formula_text, and this worker fits that string verbatim.
# There is deliberately no build_term/build_formula/K_DEFAULT here to drift out of
# sync with prep. The formula is looked up by spec_index in main() and threaded
# into fit_one_spec() / run_oos().
# ===========================================================================

# ---------------------------------------------------------------------------
# Fit one mod, capturing iter / convergence / timing / error.
# ---------------------------------------------------------------------------
fit_one_mod <- function(fml, data, n_scams, n_smooths,
                        optimizer = "bfgs", maxit = 300L, label = "fit") {
  t0  <- Sys.time()
  fit <- tryCatch(
    if (n_scams > 0) {
      scam(fml, data = data, optimizer = optimizer, control = list(maxit = maxit))
    } else if (n_smooths > 0) {
      mgcv::gam(fml, data = data, method = "REML")     # unconstrained s() only
    } else {
      lm(fml, data = data)                             # all linear + A0_af
    },
    error = function(e) e)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (inherits(fit, "error")) {
    message(glue("  [{label}] ERROR after {sprintf('%.1f', elapsed)}s: {conditionMessage(fit)}"))
    return(list(fit = NULL, iter = NA_integer_, converged = NA, elapsed = elapsed,
                error = TRUE, error_msg = conditionMessage(fit)))
  }
  iter <- tryCatch({ v <- fit$iter; if (length(v)) as.integer(v)[1] else NA_integer_ },
                   error = function(e) NA_integer_)   # gam/lm may have no $iter -> NA
  conv <- isTRUE(fit$conv)                            # scam $conv is a list -> FALSE; gam/lm no $conv -> FALSE
  message(glue("  [{label}] iter={iter} converged={conv} ({sprintf('%.1f', elapsed)}s)"))
  list(fit = fit, iter = iter, converged = conv, elapsed = elapsed,
       error = FALSE, error_msg = NA_character_)
}


# ---------------------------------------------------------------------------
# Packing helpers for the new per-term columns.
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
# When write_summary, each fold's training-fit stripped_summary is collated into
# oos_summary_<task_id>.txt (one section per fold).
# ---------------------------------------------------------------------------
run_oos <- function(past_data, cv_strategy, n_folds = 5L, cv_seed = 42L,
                    optimizer = "bfgs", maxit = 300L,
                    n_scams = NA_integer_, n_smooths = NA_integer_,
                    train_lo = NA, train_hi = NA, test_lo = NA, test_hi = NA,
                    out_dir = NULL, task_id = NULL, spec_index = NA, write_summary = FALSE, fml = NULL) {
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
      cv_fold_rmse[k]   <- sqrt(mean(resid_k^2))
      cv_fold_mae[k]    <- mean(abs(resid_k))
      cv_fold_pfpr_r[k] <- safe(cor(past_data$malaria_pfpr[test_idx], plogis(preds_k)))
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
# (maybe_write_artifacts removed.) The diagnostic RDS/PDF writer was the only
# consumer of the spec object / neighborhood_specs.rds. The spec design now comes
# from malaria_spec_design.py via spec_table.parquet (formula_text is the single
# source of truth), so the .rds and the WRITE_RDS/WRITE_PLOT flags are gone.
# ---------------------------------------------------------------------------

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
# Fit one spec under the active flags -> one-row data.table.
# ---------------------------------------------------------------------------
fit_one_spec <- function(spec_index, data, flags, task_id, fml_text,
                         n_scams, n_smooths) {
  optimizer <- flags$optimizer; maxit <- flags$maxit
  fml <- as.formula(fml_text)   # prep is the single source of truth for the formula
  row <- data.table(
    spec_index    = as.integer(spec_index),
    optimizer     = optimizer,
    maxit_setting = as.integer(maxit),
    r_version     = paste(R.version$major, R.version$minor, sep = "."),
    mgcv_version  = as.character(packageVersion("mgcv")),
    scam_version  = as.character(packageVersion("scam")),
    did_is_fe     = flags$is_fe,
    did_oos       = flags$oos)

  # --- IS with FE ---
  if (flags$is_fe) {
    res    <- fit_one_mod(fml, data, n_scams = n_scams, n_smooths = n_smooths,
                          optimizer = optimizer, maxit = maxit, label = glue("IS_FE_{spec_index}"))
    m      <- prefix_list(is_block_metrics(res$fit, data$malaria_pfpr), "is")
    row[, `:=`(is_formula = fml_text, is_elapsed_sec = res$elapsed,
               is_error = res$error, is_error_msg = res$error_msg,
               is_iter = res$iter, is_converged = res$converged)]
    for (nm in names(m)) row[, (nm) := m[[nm]]]
    if (flags$write_summary && !res$error && !is.null(res$fit)) {
      spath <- file.path(flags$out_dir, glue("summary_{task_id}_s{spec_index}.txt"))
      tryCatch(writeLines(capture.output(stripped_summary(res$fit)), spath),
               error = function(e) message(glue("  [summary] write failed: {conditionMessage(e)}")))
      if (file.exists(spath)) Sys.chmod(spath, "0775")
    }
  }

  # --- OOS k-fold CV ---
  if (flags$oos) {
    oos <- run_oos(data, flags$cv_strategy, flags$n_folds, flags$cv_seed, optimizer, maxit,
                   n_scams = n_scams, n_smooths = n_smooths,
                   train_lo = flags$train_lo, train_hi = flags$train_hi,
                   test_lo  = flags$test_lo,  test_hi  = flags$test_hi,
                   out_dir  = flags$out_dir, task_id = task_id, spec_index = spec_index,
                   write_summary = flags$write_summary, fml = fml)
    for (nm in names(oos)) row[, (nm) := oos[[nm]]]
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
    make_option("--task-id",     type = "character", default = NA, help = "param_map task_id this job fits"),
    make_option("--output-dir",  type = "character", default = NA),
    make_option("--manifest",    type = "character", default = NA, help = "idd-tools manifest JSON; this task's cells (spec list) read by --task-id; default <output-dir>/manifest.json"),
    make_option("--spec-table",  type = "character", default = NA, help = "spec_table parquet (formula_text); default <output-dir>/spec_table.parquet"),
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
  pick_flag <- function(cli, env, default) if (!is.na(cli)) toupper(as.character(cli)) %in% c("TRUE", "T", "1", "YES") else env_flag(env, default)

  # --- CONSISTENCY GUARD: worker `lags` must match the orchestrator's --max-lag ---
  # The two "full data" settings (this worker's `lags` and the orchestrator's max_lag)
  # must move together. A silent mismatch corrupts the "full data" claim -- e.g. max_lag=0
  # in the orchestrator (train_lo=2000) but `lags` still set here so the complete.cases
  # lag-completeness block thins 2000-2002 and lag-NA country-years, or the reverse.
  # Fail fast and loud rather than silently fitting on the wrong row set.
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

  flags <- list(
    is_fe       = pick_flag(opt$fit_is_fe,   "FIT_IS_FE",   FALSE),
    oos         = pick_flag(opt$fit_oos,     "FIT_OOS",     FALSE),
    write_summary = pick_flag(opt$write_summary, "WRITE_SUMMARY", TRUE),
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

  # Spec list comes from the idd-tools manifest, keyed by task_id (replaces param_map).
  # Each cell carries spec_index + n_smooths + n_scams, so the engine-dispatch counts come
  # straight from the manifest; only formula_text still needs the spec_table.
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
  # formula_text per spec_index from prep (single source of truth; per-term k/bs baked in).
  spec_table <- as.data.table(arrow::read_parquet(spec_table_path))
  fml_lookup <- setNames(as.character(spec_table$formula_text),
                        as.character(spec_table$spec_index))
  message(glue("[select task {this_task}] is_fe={flags$is_fe} ",
               "oos={flags$oos} (cv={flags$cv_strategy}) | {length(my_specs)} spec(s)"))

  t_load <- Sys.time()
  data <- load_past_data(past_inputs_parquet())
  message(glue("  loaded {nrow(data)} rows x {ncol(data)} cols in ",
               "{round(as.numeric(difftime(Sys.time(), t_load, units = 'secs')), 1)}s"))

  rows <- lapply(my_specs, function(si) {
    key      <- as.character(si)
    fml_text <- fml_lookup[[key]]
    n_sm     <- nsm_lookup[[key]]
    n_scam   <- nscam_lookup[[key]]
    stopifnot("spec_index missing formula_text in spec_table" =
                !is.null(fml_text) && !is.na(fml_text) && nzchar(fml_text))
    r <- fit_one_spec(si, data, flags, this_task, fml_text,
                      n_scams = n_scam, n_smooths = n_sm)
    r[, task_id := this_task]
    r
  })

  out <- rbindlist(rows, use.names = TRUE, fill = TRUE)
  setcolorder(out, c("task_id", "spec_index"))

  # Lean parquet, one row per spec. Atomic write: temp BESIDE the target (NOT
  # /tmp), metadata-only validation (row count), then rename.
  final <- file.path(out_dir, glue("select_summary_{this_task}.parquet"))
  tmp   <- file.path(out_dir, glue(".select_summary_{this_task}.tmp.parquet"))
  arrow::write_parquet(out, tmp)
  stopifnot("row-count mismatch on written parquet" =
              arrow::open_dataset(tmp)$num_rows == nrow(out))
  if (file.exists(final)) file.remove(final)
  file.rename(tmp, final)
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
