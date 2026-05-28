### This file will be called: fit_malaria_models_rocket.r
### Fits one scam pfpr model for the spec at task_id; writes THREE files:
###   fit_<NNN>.rds      — full scam fit + spec/metadata (large; ~30 MB)
###   summary_<NNN>.csv  — 1-row data.table: in-sample + 5-fold OOS metrics
###   plot_<NNN>.pdf     — fitted smooth curves on one page
###
### 2026-05-14 changes:
###   * Optimizer switched from "efs" to "bfgs". EFS silently hits maxit on
###     ~10% of specs in this grid and produces unconverged fits; bfgs takes
###     fewer outer iterations (~4) and reliably converges, though each
###     iteration is heavier.
###   * Summary now records iter + converged for every fit (IS-with-FE,
###     IS-no-FE, and each CV fold).
###   * Per-fold OOS metrics are now retained so fold-to-fold stability is
###     visible in summary CSVs.
###   * Optimizer / maxit / R+scam versions recorded for reproducibility.

rm(list = ls())

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)

OPTIMIZER    <- "bfgs"
MAXIT        <- 300L
N_FOLDS      <- 5L
CV_SEED      <- 42L
TESTING      <- FALSE

# -------------------------- Task ID + output dir --------------------------

task_id <- ifelse(is.na(as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))), 115L,
                  as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID")))

if (TESTING) {
  output_dir  <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/scam_fits/lsae_1285/20260511"
  CV_STRATEGY <- "country"
} else {
  output_dir  <- Sys.getenv("FIT_OUTPUT_DIR")
  CV_STRATEGY <- Sys.getenv("CV_STRATEGY", unset = "none")
  stopifnot("FIT_OUTPUT_DIR env var must be set by the launcher" = nzchar(output_dir))
}
stopifnot("CV_STRATEGY must be 'none', 'random', 'country', or 'country_no_fe'" =
            CV_STRATEGY %in% c("none", "random", "country", "country_no_fe"))

message(glue("Task ID:    {task_id}"))
message(glue("Output dir: {output_dir}"))
message(glue("Optimizer:  {OPTIMIZER} (maxit={MAXIT})"))

specs       <- readRDS(glue("{output_dir}/neighborhood_specs.rds"))
param_map   <- fread(glue("{output_dir}/param_map.csv"))
spec_index  <- param_map[task_id, spec_index]
spec        <- specs[[spec_index]]

# -------------------------- Formula --------------------------

K_DEFAULT <- 6
build_term <- function(var, form) {
  if (form == "linear") return(var)
  sprintf("s(%s, k = %d, bs = '%s')", var, K_DEFAULT, form)
}
build_formula <- function(spec, response) {
  rhs <- mapply(build_term, names(spec), unlist(spec), USE.NAMES = FALSE)
  reformulate(rhs, response = response)
}

fml          <- build_formula(spec, "logit_malaria_pfpr")
formula_text <- deparse1(fml)
message(glue("Formula: {formula_text}"))

# -------------------------- Load + clean past data --------------------------

parquet_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/past_inputs_nc/lsae_1285/current/malaria_past_inputs.parquet"
suit_variant_pick <- "mordecai_0_0"

past_data <- as.data.frame(arrow::read_parquet(parquet_path))
past_data$A0_af <- as.factor(past_data$A0_location_id)

nan_toss <- function(df, var) {
  to_toss <- which(is.na(df[var]))
  if (length(to_toss)) df[-to_toss, ] else df
}
past_data <- nan_toss(past_data, "malaria_pfpr")
past_data <- nan_toss(past_data, "gdppc_mean")
past_data <- nan_toss(past_data, "mal_DAH_total_per_capita")

suit_col <- paste0("malaria_suitability_", suit_variant_pick)
past_data$malaria_suit_fraction     <- past_data[[suit_col]] / 365
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

safe <- function(expr) tryCatch(expr, error = function(e) NA_real_)

# -------------------------- Fit helper --------------------------
# Wraps scam() to consistently capture iter + convergence + timing + errors.

fit_scam_one <- function(fml, data, label = "fit") {
  t0 <- Sys.time()
  fit <- tryCatch(
    scam(fml, data = data, optimizer = OPTIMIZER, control = list(maxit = MAXIT)),
    error = function(e) e
  )
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (inherits(fit, "error")) {
    message(glue("  [{label}] ERROR after {sprintf('%.1f', elapsed)}s: {conditionMessage(fit)}"))
    return(list(fit = fit, iter = NA_integer_, converged = NA,
                elapsed = elapsed, error = TRUE,
                error_msg = conditionMessage(fit)))
  }
  iter <- tryCatch(as.integer(fit$iter), error = function(e) NA_integer_)
  conv <- isTRUE(fit$conv)
  message(glue("  [{label}] iter={iter} converged={conv} ({sprintf('%.1f', elapsed)}s)"))
  list(fit = fit, iter = iter, converged = conv,
       elapsed = elapsed, error = FALSE, error_msg = NA_character_)
}

# ========================== In-sample fit (with country FE) ==========================

message("Starting IS fit (with country FE) ...")
is_res        <- fit_scam_one(fml, past_data, label = "IS_FE")
fit           <- is_res$fit
is_elapsed    <- is_res$elapsed
is_err        <- is_res$error
is_iter       <- is_res$iter
is_converged  <- is_res$converged
is_error_msg  <- is_res$error_msg

if (!is_err) {
  is_aic        <- safe(AIC(fit))
  is_bic        <- safe(BIC(fit))
  is_loglik     <- safe(as.numeric(logLik(fit)))
  is_deviance   <- safe(as.numeric(fit$deviance))
  is_null_dev   <- safe(as.numeric(fit$null.deviance))
  is_dev_expl   <- safe(1 - fit$deviance / fit$null.deviance)
  is_r_sq       <- safe(1 - sum((fit$y - fit$fitted.values)^2) /
                             sum((fit$y - mean(fit$y))^2))
  is_rmse       <- safe(sqrt(mean((fit$y - fit$fitted.values)^2)))
  is_mae        <- safe(mean(abs(fit$y - fit$fitted.values)))
  is_n_obs      <- safe(as.integer(nobs(fit)))
  is_n_coef     <- safe(length(coef(fit)))
  is_edf_sum    <- safe(sum(fit$edf))
  n_smooths_v   <- length(fit$smooth)
  is_edf_smooth <- if (n_smooths_v > 0) {
    safe(sum(sapply(fit$smooth, function(s) sum(fit$edf[s$first.para:s$last.para]))))
  } else 0
  is_edf_parametric <- is_edf_sum - is_edf_smooth
  is_max_sp     <- safe(suppressWarnings(max(fit$sp, na.rm = TRUE)))
  is_min_sp     <- safe(suppressWarnings(min(fit$sp, na.rm = TRUE)))
  smooth_labels <- if (n_smooths_v > 0) {
    paste(vapply(fit$smooth, function(s) s$label, character(1)), collapse = ";")
  } else NA_character_
  is_pfpr_preds  <- plogis(fit$fitted.values)
  is_pfpr_actual <- past_data$malaria_pfpr
  is_pfpr_rmse   <- safe(sqrt(mean((is_pfpr_actual - is_pfpr_preds)^2)))
  is_pfpr_mae    <- safe(mean(abs(is_pfpr_actual - is_pfpr_preds)))
  is_pfpr_r      <- safe(cor(is_pfpr_actual, is_pfpr_preds))
} else {
  is_aic <- is_bic <- is_loglik <- is_deviance <- is_null_dev <- is_dev_expl <-
    is_r_sq <- is_rmse <- is_mae <- is_edf_sum <- is_edf_smooth <-
    is_edf_parametric <- is_max_sp <- is_min_sp <- NA_real_
  is_n_obs <- is_n_coef <- NA_integer_
  n_smooths_v   <- NA_integer_
  smooth_labels  <- NA_character_
  is_pfpr_rmse <- is_pfpr_mae <- is_pfpr_r <- NA_real_
}

# ========================== In-sample fit (without country FE) ==========================
# Only computed for country_no_fe strategy.

if (CV_STRATEGY == "country_no_fe") {
  spec_no_fe     <- spec[names(spec) != "A0_af"]
  fml_no_fe      <- build_formula(spec_no_fe, "logit_malaria_pfpr")
  fml_no_fe_text <- deparse1(fml_no_fe)
  message(glue("No-FE formula: {fml_no_fe_text}"))
  message("Starting IS fit (no FE) ...")
  is_nofe_res        <- fit_scam_one(fml_no_fe, past_data, label = "IS_noFE")
  fit_no_fe          <- is_nofe_res$fit
  is_no_fe_elapsed   <- is_nofe_res$elapsed
  is_no_fe_err       <- is_nofe_res$error
  is_no_fe_iter      <- is_nofe_res$iter
  is_no_fe_converged <- is_nofe_res$converged

  if (!is_no_fe_err) {
    is_no_fe_aic      <- safe(AIC(fit_no_fe))
    is_no_fe_bic      <- safe(BIC(fit_no_fe))
    is_no_fe_dev_expl <- safe(1 - fit_no_fe$deviance / fit_no_fe$null.deviance)
    is_no_fe_r_sq     <- safe(1 - sum((fit_no_fe$y - fit_no_fe$fitted.values)^2) /
                                  sum((fit_no_fe$y - mean(fit_no_fe$y))^2))
    is_no_fe_rmse     <- safe(sqrt(mean((fit_no_fe$y - fit_no_fe$fitted.values)^2)))
    is_no_fe_mae      <- safe(mean(abs(fit_no_fe$y - fit_no_fe$fitted.values)))
    is_no_fe_pfpr_preds  <- plogis(fit_no_fe$fitted.values)
    is_no_fe_pfpr_actual <- past_data$malaria_pfpr
    is_no_fe_pfpr_rmse   <- safe(sqrt(mean((is_no_fe_pfpr_actual - is_no_fe_pfpr_preds)^2)))
    is_no_fe_pfpr_mae    <- safe(mean(abs(is_no_fe_pfpr_actual - is_no_fe_pfpr_preds)))
    is_no_fe_pfpr_r      <- safe(cor(is_no_fe_pfpr_actual, is_no_fe_pfpr_preds))
  } else {
    is_no_fe_aic <- is_no_fe_bic <- is_no_fe_dev_expl <- is_no_fe_r_sq <-
      is_no_fe_rmse <- is_no_fe_mae <- NA_real_
    is_no_fe_pfpr_rmse <- is_no_fe_pfpr_mae <- is_no_fe_pfpr_r <- NA_real_
  }
} else {
  is_no_fe_elapsed   <- 0
  is_no_fe_err       <- NA
  is_no_fe_iter      <- NA_integer_
  is_no_fe_converged <- NA
  is_no_fe_aic <- is_no_fe_bic <- is_no_fe_dev_expl <- is_no_fe_r_sq <-
    is_no_fe_rmse <- is_no_fe_mae <- NA_real_
  is_no_fe_pfpr_rmse <- is_no_fe_pfpr_mae <- is_no_fe_pfpr_r <- NA_real_
  fml_no_fe_text     <- NA_character_
}

# ========================== Cross-validation ==========================

# Per-fold trackers (used regardless of CV_STRATEGY; left NA when CV is skipped).
cv_iter         <- rep(NA_integer_, N_FOLDS)
cv_converged    <- rep(NA, N_FOLDS)
cv_fold_rmse    <- rep(NA_real_, N_FOLDS)
cv_fold_mae     <- rep(NA_real_, N_FOLDS)
cv_fold_pfpr_r  <- rep(NA_real_, N_FOLDS)

if (CV_STRATEGY != "none") {

message(glue("Starting {N_FOLDS}-fold CV (strategy={CV_STRATEGY}) ..."))
cv_t0 <- Sys.time()

set.seed(CV_SEED)
n <- nrow(past_data)

cv_fml <- if (CV_STRATEGY == "country_no_fe") fml_no_fe else fml

if (CV_STRATEGY == "random") {
  folds <- sample(rep(seq_len(N_FOLDS), length.out = n))

} else if (CV_STRATEGY %in% c("country", "country_no_fe")) {
  # Greedy bin-packing of countries into folds (descending by size).
  country_ids  <- unique(past_data$A0_location_id)
  country_n    <- table(past_data$A0_location_id)[as.character(country_ids)]
  ord          <- order(country_n, decreasing = TRUE)
  country_ids  <- country_ids[ord]
  country_n    <- country_n[ord]

  fold_sizes   <- rep(0L, N_FOLDS)
  country_fold <- integer(length(country_ids))
  for (i in seq_along(country_ids)) {
    smallest       <- which.min(fold_sizes)
    country_fold[i] <- smallest
    fold_sizes[smallest] <- fold_sizes[smallest] + country_n[i]
  }
  names(country_fold) <- as.character(country_ids)

  folds <- country_fold[as.character(past_data$A0_location_id)]
  message(glue("  Fold sizes: {paste(fold_sizes, collapse=', ')}"))
  message(glue("  Countries per fold: {paste(table(country_fold), collapse=', ')}"))
}

cv_actuals <- numeric(n)
cv_preds   <- numeric(n)
cv_fold_ok <- logical(N_FOLDS)

# For country-holdout with FE: held-out countries don't have a fitted FE,
# so predict on test rows using the mean of training countries' FEs.
predict_with_mean_country_fe <- function(fold_fit, test_data) {
  cf        <- coef(fold_fit)
  af_idx    <- grep("^A0_af", names(cf))
  n_train_countries <- length(af_idx) + 1L
  mean_fe   <- sum(cf[af_idx]) / n_train_countries
  ref_level <- levels(fold_fit$model$A0_af)[1]
  test_copy <- test_data
  test_copy$A0_af <- factor(ref_level, levels = levels(fold_fit$model$A0_af))
  predict(fold_fit, newdata = test_copy) + mean_fe
}

for (k in seq_len(N_FOLDS)) {
  test_idx  <- which(folds == k)
  train_idx <- which(folds != k)

  train_data <- past_data[train_idx, ]
  if (CV_STRATEGY == "country") {
    train_data$A0_af <- droplevels(train_data$A0_af)
  }

  fold_res         <- fit_scam_one(cv_fml, train_data, label = glue("CV_fold_{k}"))
  fold_fit         <- fold_res$fit
  cv_iter[k]       <- fold_res$iter
  cv_converged[k]  <- fold_res$converged

  if (fold_res$error) {
    cv_preds[test_idx]   <- NA_real_
    cv_actuals[test_idx] <- past_data$logit_malaria_pfpr[test_idx]
    cv_fold_ok[k]        <- FALSE
  } else {
    preds_k <- if (CV_STRATEGY == "country") {
      predict_with_mean_country_fe(fold_fit, past_data[test_idx, ])
    } else {
      predict(fold_fit, newdata = past_data[test_idx, ])
    }
    cv_preds[test_idx]   <- preds_k
    cv_actuals[test_idx] <- past_data$logit_malaria_pfpr[test_idx]
    cv_fold_ok[k]        <- TRUE

    # Per-fold OOS metrics (logit-space + PfPR-space)
    actual_logit_k    <- past_data$logit_malaria_pfpr[test_idx]
    resid_k           <- actual_logit_k - preds_k
    cv_fold_rmse[k]   <- sqrt(mean(resid_k^2))
    cv_fold_mae[k]    <- mean(abs(resid_k))
    cv_fold_pfpr_r[k] <- safe(cor(past_data$malaria_pfpr[test_idx], plogis(preds_k)))
  }
}

cv_elapsed <- as.numeric(difftime(Sys.time(), cv_t0, units = "secs"))

# -- Aggregate OOS metrics (across all held-out predictions) --
valid     <- !is.na(cv_preds)
cv_resid  <- cv_actuals[valid] - cv_preds[valid]
oos_rmse  <- safe(sqrt(mean(cv_resid^2)))
oos_mae   <- safe(mean(abs(cv_resid)))
oos_r_sq  <- safe(1 - sum(cv_resid^2) / sum((cv_actuals[valid] - mean(cv_actuals[valid]))^2))
oos_n_obs <- sum(valid)
oos_n_folds_ok <- sum(cv_fold_ok)

oos_pfpr_preds  <- plogis(cv_preds[valid])
oos_pfpr_actual <- past_data$malaria_pfpr[valid]
oos_pfpr_rmse   <- safe(sqrt(mean((oos_pfpr_actual - oos_pfpr_preds)^2)))
oos_pfpr_mae    <- safe(mean(abs(oos_pfpr_actual - oos_pfpr_preds)))
oos_pfpr_r      <- safe(cor(oos_pfpr_actual, oos_pfpr_preds))

message(glue("CV done: OOS RMSE={round(oos_rmse, 4)}, OOS R²={round(oos_r_sq, 4)}, ",
             "folds OK={oos_n_folds_ok}/{N_FOLDS} converged={sum(cv_converged, na.rm = TRUE)}/{N_FOLDS} ",
             "({sprintf('%.1f', cv_elapsed)}s)"))

} else {
  cv_elapsed     <- 0
  oos_rmse       <- NA_real_
  oos_mae        <- NA_real_
  oos_r_sq       <- NA_real_
  oos_n_obs      <- NA_integer_
  oos_n_folds_ok <- NA_integer_
  oos_pfpr_rmse  <- NA_real_
  oos_pfpr_mae   <- NA_real_
  oos_pfpr_r     <- NA_real_
  message("CV skipped (CV_STRATEGY='none')")
}

# Helpers for packing per-fold vectors into single CSV cells.
fold_join     <- function(x) paste(ifelse(is.na(x), "NA", as.character(x)), collapse = ";")
fold_join_num <- function(x, digits = 4) paste(
  ifelse(is.na(x), "NA", sprintf(paste0("%.", digits, "f"), x)), collapse = ";"
)

# ========================== Summary table ==========================

total_elapsed <- is_elapsed + is_no_fe_elapsed + cv_elapsed

summary_dt <- data.table(
  task_id          = task_id,
  spec_index       = spec_index,
  formula_text     = formula_text,
  # Run-environment metadata
  optimizer        = OPTIMIZER,
  maxit_setting    = MAXIT,
  r_version        = paste(R.version$major, R.version$minor, sep = "."),
  scam_version     = as.character(packageVersion("scam")),
  # Timing
  elapsed_sec          = total_elapsed,
  is_elapsed_sec       = is_elapsed,
  is_no_fe_elapsed_sec = is_no_fe_elapsed,
  cv_elapsed_sec       = cv_elapsed,
  # IS-with-FE: error / convergence / metrics
  error            = is_err,
  error_msg        = is_error_msg,
  is_iter          = is_iter,
  is_converged     = is_converged,
  is_aic           = is_aic,
  is_bic           = is_bic,
  is_loglik        = is_loglik,
  is_deviance      = is_deviance,
  is_null_deviance = is_null_dev,
  is_dev_expl      = is_dev_expl,
  is_r_sq          = is_r_sq,
  is_rmse          = is_rmse,
  is_mae           = is_mae,
  is_pfpr_rmse     = is_pfpr_rmse,
  is_pfpr_mae      = is_pfpr_mae,
  is_pfpr_r        = is_pfpr_r,
  is_n_obs         = is_n_obs,
  is_n_coef        = is_n_coef,
  is_edf_sum       = is_edf_sum,
  is_edf_smooth    = is_edf_smooth,
  is_edf_parametric = is_edf_parametric,
  n_smooths        = n_smooths_v,
  is_max_sp        = is_max_sp,
  is_min_sp        = is_min_sp,
  smooth_labels    = smooth_labels,
  # IS-without-FE (country_no_fe strategy only)
  is_no_fe_formula   = fml_no_fe_text,
  is_no_fe_error     = is_no_fe_err,
  is_no_fe_iter      = is_no_fe_iter,
  is_no_fe_converged = is_no_fe_converged,
  is_no_fe_aic       = is_no_fe_aic,
  is_no_fe_bic       = is_no_fe_bic,
  is_no_fe_dev_expl  = is_no_fe_dev_expl,
  is_no_fe_r_sq      = is_no_fe_r_sq,
  is_no_fe_rmse      = is_no_fe_rmse,
  is_no_fe_mae       = is_no_fe_mae,
  is_no_fe_pfpr_rmse = is_no_fe_pfpr_rmse,
  is_no_fe_pfpr_mae  = is_no_fe_pfpr_mae,
  is_no_fe_pfpr_r    = is_no_fe_pfpr_r,
  # CV setup + per-fold convergence
  cv_strategy           = CV_STRATEGY,
  cv_n_folds            = N_FOLDS,
  cv_seed               = CV_SEED,
  cv_iter_per_fold      = fold_join(cv_iter),
  cv_converged_per_fold = fold_join(cv_converged),
  cv_n_converged        = sum(cv_converged, na.rm = TRUE),
  # Aggregate OOS
  oos_rmse        = oos_rmse,
  oos_mae         = oos_mae,
  oos_r_sq        = oos_r_sq,
  oos_pfpr_rmse   = oos_pfpr_rmse,
  oos_pfpr_mae    = oos_pfpr_mae,
  oos_pfpr_r      = oos_pfpr_r,
  oos_n_obs       = oos_n_obs,
  oos_n_folds_ok  = oos_n_folds_ok,
  # Per-fold OOS for stability inspection
  cv_fold_rmse_per_fold   = fold_join_num(cv_fold_rmse),
  cv_fold_mae_per_fold    = fold_join_num(cv_fold_mae),
  cv_fold_pfpr_r_per_fold = fold_join_num(cv_fold_pfpr_r),
  cv_fold_pfpr_r_min      = safe(min(cv_fold_pfpr_r, na.rm = TRUE)),
  cv_fold_pfpr_r_max      = safe(max(cv_fold_pfpr_r, na.rm = TRUE)),
  cv_fold_pfpr_r_sd       = safe(sd(cv_fold_pfpr_r,  na.rm = TRUE))
)

# ========================== Write three files ==========================

tid_padded <- sprintf("%03d", task_id)

# 1) Full fit object (~30 MB) — in-sample fit only (CV fits are discarded).
fit_out_path <- glue("{output_dir}/fit_{tid_padded}.rds")
saveRDS(list(
  task_id      = task_id,
  spec_index   = spec_index,
  spec         = spec,
  formula_text = formula_text,
  optimizer    = OPTIMIZER,
  maxit        = MAXIT,
  elapsed_sec  = total_elapsed,
  fit          = fit,
  iter         = is_iter,
  converged    = is_converged,
  error        = is_err,
  error_msg    = is_error_msg
), file = fit_out_path)
Sys.chmod(fit_out_path, mode = "0775")

# 2) Per-fit summary (in-sample + OOS metrics).
summary_path <- glue("{output_dir}/summary_{tid_padded}.csv")
fwrite(summary_dt, file = summary_path)
Sys.chmod(summary_path, mode = "0775")

# 3) Plot of fitted smooths.
plot_path <- glue("{output_dir}/plot_{tid_padded}.pdf")
if (!is_err && n_smooths_v > 0) {
  ncol_pdf <- min(n_smooths_v, 3L)
  nrow_pdf <- ceiling(n_smooths_v / ncol_pdf)
  pdf(plot_path, width = 4 * ncol_pdf, height = 4 * nrow_pdf + 1)
  tryCatch({
    par(oma = c(0, 0, 3, 0))
    plot(fit, pages = 1, scale = 0, shade = TRUE, residuals = FALSE)
    oos_label  <- if (CV_STRATEGY != "none") glue("  OOS.R²={round(oos_r_sq, 3)}") else ""
    conv_label <- glue("  iter={is_iter} conv={is_converged}")
    title(main = glue("task_id={task_id}  AIC={round(is_aic, 1)}  dev.expl={round(is_dev_expl, 3)}",
                      "{conv_label}{oos_label}\n",
                      "{substr(formula_text, 1, 130)}"),
          outer = TRUE, cex.main = 0.9)
  }, error = function(e) {
    par(mfrow = c(1, 1), oma = c(0, 0, 0, 0), mar = c(5, 4, 4, 2) + 0.1)
    plot.new()
    title(main = glue("task_id={task_id}: plot failed"))
    mtext(conditionMessage(e), side = 1, line = -2, cex = 0.7)
  })
  dev.off()
  Sys.chmod(plot_path, mode = "0775")
} else if (is_err) {
  pdf(plot_path, width = 8, height = 6)
  plot.new()
  title(main = glue("task_id={task_id}: FIT ERROR"))
  mtext(is_error_msg, side = 1, line = -3, cex = 0.7)
  dev.off()
  Sys.chmod(plot_path, mode = "0775")
}

cv_msg <- if (CV_STRATEGY != "none") glue(", CV: {sprintf('%.1f', cv_elapsed)}s") else ""
message(glue("Saved fit/summary/plot for task_id={task_id} ",
             "(IS: {sprintf('%.1f', is_elapsed)}s{cv_msg}, ",
             "is_iter={is_iter} is_conv={is_converged} error={is_err})"))
print("fin")
