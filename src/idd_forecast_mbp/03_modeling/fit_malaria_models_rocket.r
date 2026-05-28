### This file will be called: fit_malaria_models_rocket.r
### Fits one scam pfpr model for the spec at task_id; writes THREE files:
###   fit_<NNN>.rds      — full scam fit + spec/metadata (large; ~30 MB)
###   summary_<NNN>.csv  — 1-row data.table: in-sample + 5-fold OOS metrics
###   plot_<NNN>.pdf     — fitted smooth curves on one page

rm(list = ls())

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)

N_FOLDS      <- 5L
CV_SEED      <- 42L
TESTING      <- FALSE

# -------------------------- Task ID + output dir --------------------------

task_id <- ifelse(is.na(as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))), 115L,
                  as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID")))

if (TESTING) {
  output_dir  <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/scam_fits/lsae_1285/20260511"
  CV_STRATEGY <- "country"   # "none", "random", or "country"
} else {
  output_dir  <- Sys.getenv("FIT_OUTPUT_DIR")
  CV_STRATEGY <- Sys.getenv("CV_STRATEGY", unset = "none")
  stopifnot("FIT_OUTPUT_DIR env var must be set by the launcher" = nzchar(output_dir))
}
stopifnot("CV_STRATEGY must be 'none', 'random', 'country', or 'country_no_fe'" =
            CV_STRATEGY %in% c("none", "random", "country", "country_no_fe"))

message(glue("Task ID:    {task_id}"))
message(glue("Output dir: {output_dir}"))

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

# ========================== In-sample fit ==========================

t0  <- Sys.time()
fit <- tryCatch(
  scam(fml, data = past_data, optimizer = "efs", control = list(maxit = 300)),
  error = function(e) e
)
is_elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
is_err     <- inherits(fit, "error")

# -- In-sample metrics --
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
  # Natural-space (PfPR) metrics
  is_pfpr_preds  <- plogis(fit$fitted.values)
  is_pfpr_actual <- past_data$malaria_pfpr
  is_pfpr_rmse   <- safe(sqrt(mean((is_pfpr_actual - is_pfpr_preds)^2)))
  is_pfpr_mae    <- safe(mean(abs(is_pfpr_actual - is_pfpr_preds)))
  is_pfpr_r   <- safe(cor(is_pfpr_actual, is_pfpr_preds))
} else {
  is_aic <- is_bic <- is_loglik <- is_deviance <- is_null_dev <- is_dev_expl <-
    is_r_sq <- is_rmse <- is_mae <- is_edf_sum <- is_edf_smooth <-
    is_edf_parametric <- is_max_sp <- is_min_sp <- NA_real_
  is_n_obs <- is_n_coef <- NA_integer_
  n_smooths_v   <- NA_integer_
  smooth_labels  <- NA_character_
  is_pfpr_rmse <- is_pfpr_mae <- is_pfpr_r <- NA_real_
}

# ========================== In-sample fit WITHOUT country FE ==========================
# Only computed for country_no_fe strategy.

if (CV_STRATEGY == "country_no_fe") {
  # Build formula without A0_af
  spec_no_fe      <- spec[names(spec) != "A0_af"]
  fml_no_fe       <- build_formula(spec_no_fe, "logit_malaria_pfpr")
  fml_no_fe_text  <- deparse1(fml_no_fe)
  message(glue("No-FE formula: {fml_no_fe_text}"))

  t0_no_fe <- Sys.time()
  fit_no_fe <- tryCatch(
    scam(fml_no_fe, data = past_data, optimizer = "efs", control = list(maxit = 300)),
    error = function(e) e
  )
  is_no_fe_elapsed <- as.numeric(difftime(Sys.time(), t0_no_fe, units = "secs"))
  is_no_fe_err     <- inherits(fit_no_fe, "error")

  if (!is_no_fe_err) {
    is_no_fe_aic     <- safe(AIC(fit_no_fe))
    is_no_fe_bic     <- safe(BIC(fit_no_fe))
    is_no_fe_dev_expl <- safe(1 - fit_no_fe$deviance / fit_no_fe$null.deviance)
    is_no_fe_r_sq    <- safe(1 - sum((fit_no_fe$y - fit_no_fe$fitted.values)^2) /
                                 sum((fit_no_fe$y - mean(fit_no_fe$y))^2))
    is_no_fe_rmse    <- safe(sqrt(mean((fit_no_fe$y - fit_no_fe$fitted.values)^2)))
    is_no_fe_mae     <- safe(mean(abs(fit_no_fe$y - fit_no_fe$fitted.values)))
    # Natural-space (PfPR) metrics
    is_no_fe_pfpr_preds  <- plogis(fit_no_fe$fitted.values)
    is_no_fe_pfpr_actual <- past_data$malaria_pfpr
    is_no_fe_pfpr_rmse   <- safe(sqrt(mean((is_no_fe_pfpr_actual - is_no_fe_pfpr_preds)^2)))
    is_no_fe_pfpr_mae    <- safe(mean(abs(is_no_fe_pfpr_actual - is_no_fe_pfpr_preds)))
    is_no_fe_pfpr_r   <- safe(cor(is_no_fe_pfpr_actual, is_no_fe_pfpr_preds))
  } else {
    is_no_fe_aic <- is_no_fe_bic <- is_no_fe_dev_expl <- is_no_fe_r_sq <-
      is_no_fe_rmse <- is_no_fe_mae <- NA_real_
    is_no_fe_pfpr_rmse <- is_no_fe_pfpr_mae <- is_no_fe_pfpr_r <- NA_real_
  }
} else {
  is_no_fe_elapsed <- 0
  is_no_fe_err     <- NA
  is_no_fe_aic <- is_no_fe_bic <- is_no_fe_dev_expl <- is_no_fe_r_sq <-
    is_no_fe_rmse <- is_no_fe_mae <- NA_real_
  is_no_fe_pfpr_rmse <- is_no_fe_pfpr_mae <- is_no_fe_pfpr_r <- NA_real_
  fml_no_fe_text   <- NA_character_
}

# ========================== Cross-validation (optional) ==========================

if (CV_STRATEGY != "none") {

message(glue("Starting {N_FOLDS}-fold CV (strategy={CV_STRATEGY}) ..."))
cv_t0 <- Sys.time()

set.seed(CV_SEED)
n <- nrow(past_data)

# -- Determine CV formula: drop FE for country_no_fe, keep for others --
cv_fml <- if (CV_STRATEGY == "country_no_fe") fml_no_fe else fml

if (CV_STRATEGY == "random") {
  # -- Random row-level folds --
  folds <- sample(rep(seq_len(N_FOLDS), length.out = n))

} else if (CV_STRATEGY %in% c("country", "country_no_fe")) {
  # -- Country-level folds: hold out entire A0_location_id groups --
  # Greedy bin-packing: assign each country to the smallest fold so far.
  country_ids  <- unique(past_data$A0_location_id)
  country_n    <- table(past_data$A0_location_id)[as.character(country_ids)]
  # Sort descending by size for better packing
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

# Helper: for country-holdout with FE, predict test rows using the mean country FE.
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
  message(glue("  Fold {k}/{N_FOLDS}"))
  test_idx  <- which(folds == k)
  train_idx <- which(folds != k)

  train_data <- past_data[train_idx, ]
  # For country strategies with FE, re-level A0_af to only training countries
  if (CV_STRATEGY == "country") {
    train_data$A0_af <- droplevels(train_data$A0_af)
  }

  fold_fit <- tryCatch(
    scam(cv_fml, data = train_data, optimizer = "efs",
         control = list(maxit = 300)),
    error = function(e) e
  )

  if (inherits(fold_fit, "error")) {
    message(glue("    Fold {k} fit failed: {conditionMessage(fold_fit)}"))
    cv_preds[test_idx]   <- NA_real_
    cv_actuals[test_idx] <- past_data$logit_malaria_pfpr[test_idx]
    cv_fold_ok[k]        <- FALSE
  } else if (CV_STRATEGY == "country") {
    cv_preds[test_idx]   <- predict_with_mean_country_fe(fold_fit, past_data[test_idx, ])
    cv_actuals[test_idx] <- past_data$logit_malaria_pfpr[test_idx]
    cv_fold_ok[k]        <- TRUE
  } else {
    # "random" and "country_no_fe": predict directly (no missing levels)
    cv_preds[test_idx]   <- predict(fold_fit, newdata = past_data[test_idx, ])
    cv_actuals[test_idx] <- past_data$logit_malaria_pfpr[test_idx]
    cv_fold_ok[k]        <- TRUE
  }
}

cv_elapsed <- as.numeric(difftime(Sys.time(), cv_t0, units = "secs"))

# -- OOS metrics (computed over all held-out predictions) --
valid     <- !is.na(cv_preds)
cv_resid  <- cv_actuals[valid] - cv_preds[valid]
oos_rmse  <- safe(sqrt(mean(cv_resid^2)))
oos_mae   <- safe(mean(abs(cv_resid)))
oos_r_sq  <- safe(1 - sum(cv_resid^2) / sum((cv_actuals[valid] - mean(cv_actuals[valid]))^2))
oos_n_obs <- sum(valid)
oos_n_folds_ok <- sum(cv_fold_ok)

# Natural-space (PfPR) OOS metrics
oos_pfpr_preds  <- plogis(cv_preds[valid])
oos_pfpr_actual <- past_data$malaria_pfpr[valid]
oos_pfpr_rmse   <- safe(sqrt(mean((oos_pfpr_actual - oos_pfpr_preds)^2)))
oos_pfpr_mae    <- safe(mean(abs(oos_pfpr_actual - oos_pfpr_preds)))
oos_pfpr_r   <- safe(cor(oos_pfpr_actual, oos_pfpr_preds))

message(glue("CV done: OOS RMSE={round(oos_rmse, 4)}, OOS R²={round(oos_r_sq, 4)}, ",
             "folds OK={oos_n_folds_ok}/{N_FOLDS} ({sprintf('%.1f', cv_elapsed)}s)"))

} else {
  # CV_STRATEGY == "none": no cross-validation
  cv_elapsed     <- 0
  oos_rmse       <- NA_real_
  oos_mae        <- NA_real_
  oos_r_sq       <- NA_real_
  oos_n_obs      <- NA_integer_
  oos_n_folds_ok <- NA_integer_
  oos_pfpr_rmse  <- NA_real_
  oos_pfpr_mae   <- NA_real_
  oos_pfpr_r  <- NA_real_
  message("CV skipped (CV_STRATEGY='none')")
}

# ========================== Summary table ==========================

total_elapsed <- is_elapsed + is_no_fe_elapsed + cv_elapsed

summary_dt <- data.table(
  task_id          = task_id,
  spec_index       = spec_index,
  formula_text     = formula_text,
  elapsed_sec      = total_elapsed,
  is_elapsed_sec   = is_elapsed,
  is_no_fe_elapsed_sec = is_no_fe_elapsed,
  cv_elapsed_sec   = cv_elapsed,
  error            = is_err,
  error_msg        = if (is_err) conditionMessage(fit) else NA_character_,
  # In-sample metrics (with country FE)
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
  is_pfpr_r     = is_pfpr_r,
  is_n_obs         = is_n_obs,
  is_n_coef        = is_n_coef,
  is_edf_sum       = is_edf_sum,
  is_edf_smooth    = is_edf_smooth,
  is_edf_parametric = is_edf_parametric,
  n_smooths        = n_smooths_v,
  is_max_sp        = is_max_sp,
  is_min_sp        = is_min_sp,
  smooth_labels    = smooth_labels,
  # In-sample metrics WITHOUT country FE (country_no_fe only)
  is_no_fe_formula = fml_no_fe_text,
  is_no_fe_error   = is_no_fe_err,
  is_no_fe_aic     = is_no_fe_aic,
  is_no_fe_bic     = is_no_fe_bic,
  is_no_fe_dev_expl = is_no_fe_dev_expl,
  is_no_fe_r_sq    = is_no_fe_r_sq,
  is_no_fe_rmse    = is_no_fe_rmse,
  is_no_fe_mae     = is_no_fe_mae,
  is_no_fe_pfpr_rmse = is_no_fe_pfpr_rmse,
  is_no_fe_pfpr_mae  = is_no_fe_pfpr_mae,
  is_no_fe_pfpr_r = is_no_fe_pfpr_r,
  # Out-of-sample metrics (5-fold CV)
  cv_strategy      = CV_STRATEGY,
  cv_n_folds       = N_FOLDS,
  cv_seed          = CV_SEED,
  oos_rmse         = oos_rmse,
  oos_mae          = oos_mae,
  oos_r_sq         = oos_r_sq,
  oos_pfpr_rmse    = oos_pfpr_rmse,
  oos_pfpr_mae     = oos_pfpr_mae,
  oos_pfpr_r    = oos_pfpr_r,
  oos_n_obs        = oos_n_obs,
  oos_n_folds_ok   = oos_n_folds_ok
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
  elapsed_sec  = total_elapsed,
  fit          = fit,
  error        = is_err,
  error_msg    = if (is_err) conditionMessage(fit) else NA_character_
), file = fit_out_path)
Sys.chmod(fit_out_path, mode = "0775")

# 2) Per-fit summary (in-sample + OOS metrics).
summary_path <- glue("{output_dir}/summary_{tid_padded}.csv")
fwrite(summary_dt, file = summary_path)
Sys.chmod(summary_path, mode = "0775")

# 3) Plot of fitted smooths (from full-data fit).
plot_path <- glue("{output_dir}/plot_{tid_padded}.pdf")
if (!is_err && n_smooths_v > 0) {
  ncol_pdf <- min(n_smooths_v, 3L)
  nrow_pdf <- ceiling(n_smooths_v / ncol_pdf)
  pdf(plot_path, width = 4 * ncol_pdf, height = 4 * nrow_pdf + 1)
  tryCatch({
    par(oma = c(0, 0, 3, 0))
    plot(fit, pages = 1, scale = 0, shade = TRUE, residuals = FALSE)
    oos_label <- if (CV_STRATEGY != "none") glue("  OOS.R²={round(oos_r_sq, 3)}") else ""
    title(main = glue("task_id={task_id}  AIC={round(is_aic, 1)}  dev.expl={round(is_dev_expl, 3)}",
                      "{oos_label}\n",
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
  mtext(conditionMessage(fit), side = 1, line = -3, cex = 0.7)
  dev.off()
  Sys.chmod(plot_path, mode = "0775")
}

cv_msg <- if (CV_STRATEGY != "none") glue(", CV: {sprintf('%.1f', cv_elapsed)}s") else ""
message(glue("Saved fit/summary/plot for task_id={task_id} ",
             "(IS: {sprintf('%.1f', is_elapsed)}s{cv_msg}, error={is_err})"))
print("fin")
