rm(list = ls())
#
# Fit one OR MORE malaria model formulations on the same past-inputs dataset.
#
# Each entry in FORMULATIONS is fit (pfpr -> inc/mort scams), stripped to a
# predict-only core, saved as {run_date}_{id}_malaria_models.RData, and recorded
# in the malaria model registry under run_date = "{run_date}_{id}". The registry
# key is an arbitrary string, so multiple same-day formulations coexist and each
# is forecastable by setting MODEL_RUN_DATE="{run_date}_{id}" in the 04 launcher.
#
# All are registered best = FALSE by default (inspect them, then flag your pick
# via FLAG_BEST_ID or a later registry edit). Set FLAG_BEST_ID = "f1" etc. to
# mark exactly one best.

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)

OPTIMIZER <- "bfgs"
MAXIT     <- 300L

# Row-inclusion thresholds applied to the past inputs before fitting. These are
# part of a RUN's data cleaning (all formulations in one invocation share them),
# and are recorded on every registry record so each model states what it used.
# Historical default (used by the f1-f6 comparison): inc_count >= 1 & pfpr >= 1e-4.
# Set both to 0 to fit on the full endemic past data (no threshold) -- the
# 2026_06_03 "hybrid" formulation was fit this way.
INC_COUNT_THRESHOLD <- 0
PFPR_THRESHOLD      <- 0

REPO_DIR  <- "/mnt/team/idd/pub/forecast-mbp"
run_date  <- format(Sys.Date(), "%Y_%m_%d")
data_path <- glue("{REPO_DIR}/03-modeling_data")

# Shared registry helpers (read/append, single-best enforcement).
SRC_REPO_DIR <- glue("/ihme/homes/{Sys.getenv('USER')}/repos/idd-forecast-mbp")
source(glue("{SRC_REPO_DIR}/src/idd_forecast_mbp/lib/model_registry.R"))

# ============================ FORMULATIONS ============================
# One list entry per formulation to try. Fields:
#   id           REQUIRED short tag (used in the .RData name + registry key).
#   desc         REQUIRED one-line description (stored in the registry).
#   pfpr         REQUIRED RHS of the PfPR model (response = logit_malaria_pfpr).
#   inc, mort    OPTIONAL RHS overrides (response = log_malaria_inc_rate /
#                log_malaria_mort_rate). Default to STD_INC_MORT below.
#   suit_variant OPTIONAL malaria_suitability variant used for
#                logit_malaria_suitability. Default "mordecai_0_0".
#
# Predictors available after the transforms further down:
#   logit_malaria_suitability, logit_do30, logit_relative_humidity,
#   log_gdppc_mean, log_ldipc_mean, log_med_consumppc,
#   log_mal_DAH_total_per_capita, gdppc_mean, mal_DAH_total_per_capita,
#   people_flood_days_per_capita, A0_af, and any raw column in
#   malaria_past_inputs.parquet (climate vars, the 14 suitability variants, ...).
STD_INC_MORT <- 's(logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af'

# Refit of the 2026_06_03 "hybrid" formulation on the current past inputs (corrected
# FGH 2026 July DAH). inc/mort use STD_INC_MORT and suit_variant defaults to
# mordecai_0_0 -- identical to the registered 2026_06_03 model. Run with
# INC_COUNT_THRESHOLD = PFPR_THRESHOLD = 0 (no thresholds; full endemic data).
# NOTE: the DAH error was future-only, so past inputs are unchanged -> this refit is
# numerically identical to 2026_07_14_hybrid; the new id just gives a clean key for
# the corrected FORECAST (the piece that actually changes).
FORMULATIONS <- list(
  list(id = "full_model_selection_results",
       desc = "Result of full model selection on the endemic data with corrected DAH",
       pfpr = 'logit_malaria_suitability + s(gdppc_mean, k = 4, bs = "mpd") + s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + A0_af')
)

# Set to a formulation id (e.g. "f1") to flag that run best=TRUE after fitting;
# "" leaves every new run best=FALSE (any pre-existing best is preserved).
FLAG_BEST_ID <- "full_model_selection_results"

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
                elapsed = elapsed, error = TRUE, error_msg = conditionMessage(fit)))
  }
  iter <- tryCatch(as.integer(fit$iter), error = function(e) NA_integer_)
  conv <- isTRUE(fit$conv)
  message(glue("  [{label}] iter={iter} converged={conv} ({sprintf('%.1f', elapsed)}s)"))
  list(fit = fit, iter = iter, converged = conv,
       elapsed = elapsed, error = FALSE, error_msg = NA_character_)
}

# -------------------------- Strip for prediction --------------------------
# Drop training-frame + n-length slots that predict() on new data never reads,
# shrinking the saved .RData and the forecast rocket's load(). verify_strip()
# proves it is lossless per model before saving.
strip_scam_for_predict <- function(mod) {
  for (slot in c("model", "y", "residuals", "fitted.values",
                 "linear.predictors", "weights", "prior.weights")) {
    mod[[slot]] <- NULL
  }
  mod
}

verify_strip <- function(full, stripped, newdata, label) {
  p_full     <- predict(full,     newdata = newdata)
  p_stripped <- predict(stripped, newdata = newdata)
  max_abs <- max(abs(p_full - p_stripped))
  if (!is.finite(max_abs) || max_abs > 1e-8) {
    stop(glue("[{label}] strip changed predictions (max |diff| = {max_abs}); ",
              "a dropped slot was needed for prediction — not saving."))
  }
  message(glue("  [{label}] strip verified (max |diff| = {signif(max_abs, 3)})"))
}

# -------------------------- Load + clean past data --------------------------
parquet_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/past_inputs_nc/lsae_1285/current/malaria_past_inputs.parquet"
DEFAULT_SUIT_VARIANT <- "mordecai_0_0"

past_data <- as.data.frame(arrow::read_parquet(parquet_path))
past_data$malaria_inc_count <- past_data$malaria_inc_rate * past_data$population

nan_toss <- function(df, var) {
  to_toss <- which(is.na(df[var]))
  if (length(to_toss)) df[-to_toss, ] else df
}

past_data <- nan_toss(past_data, "malaria_pfpr")
past_data <- nan_toss(past_data, "gdppc_mean")
past_data <- nan_toss(past_data, "mal_DAH_total_per_capita")
past_data <- nan_toss(past_data, "malaria_inc_rate")
past_data <- nan_toss(past_data, "malaria_mort_rate")

# Variant-independent transforms (computed once).
past_data$do30_fraction <- past_data$days_over_30C / 365
past_data$do30_fraction <- pmin(pmax(past_data$do30_fraction, 0.001), 0.999)
past_data$logit_do30    <- log(past_data$do30_fraction / (1 - past_data$do30_fraction))
past_data$rh_fraction <- past_data$relative_humidity / 100
past_data$rh_fraction <- pmin(pmax(past_data$rh_fraction, 0.001), 0.999)
past_data$logit_relative_humidity <- log(past_data$rh_fraction / (1 - past_data$rh_fraction))

log_covs <- c("mal_DAH_total_per_capita", "gdppc_mean", "ldipc_mean",
              "med_consumppc", "malaria_inc_rate", "malaria_mort_rate")
for (cov in log_covs) {
  past_data[[paste0("log_", cov)]] <- log(past_data[[cov]])
}
past_data <- past_data[which(past_data$malaria_inc_count >= INC_COUNT_THRESHOLD &
                             past_data$malaria_pfpr >= PFPR_THRESHOLD),]
past_data$A0_af <- as.factor(past_data$A0_location_id)

# Suitability terms for a chosen variant (variant is per-formulation):
#   malaria_suit              = raw variant value            -> for s(malaria_suit, ...)
#   logit_malaria_suitability = logit(clip(value / 365))     -> for the linear term
# NOTE: s(malaria_suit, bs="mpi") is invariant to a linear rescale of the raw
# value, BUT the forecast rocket must define malaria_suit the same way (the raw
# suitability) when predicting any formulation that uses it — else predict()
# errors on the missing column / a mismatched transform.
add_suit_terms <- function(df, suit_variant) {
  col <- paste0("malaria_suitability_", suit_variant)
  if (!col %in% names(df)) stop(glue("suit_variant '{suit_variant}' -> column '{col}' not in past inputs."))
  df$malaria_suit <- df[[col]]
  frac <- pmin(pmax(df[[col]] / 365, 0.001), 0.999)
  df$logit_malaria_suitability <- log(frac / (1 - frac))
  df
}

# -------------------------- Fit one formulation --------------------------
fit_formulation <- function(spec) {
  id  <- spec$id
  key <- glue("{run_date}_{id}")
  suit_variant <- if (!is.null(spec$suit_variant)) spec$suit_variant else DEFAULT_SUIT_VARIANT

  dat <- add_suit_terms(past_data, suit_variant)
  set.seed(1)
  verify_sample <- dat[sample(nrow(dat), min(5000L, nrow(dat))), ]

  pfpr_fml <- reformulate(spec$pfpr, response = "logit_malaria_pfpr")
  inc_fml  <- reformulate(if (!is.null(spec$inc))  spec$inc  else STD_INC_MORT,
                          response = "log_malaria_inc_rate")
  mort_fml <- reformulate(if (!is.null(spec$mort)) spec$mort else STD_INC_MORT,
                          response = "log_malaria_mort_rate")

  message(glue("\n=== [{id}] {spec$desc} (suit={suit_variant}) ==="))
  pfpr_res <- fit_scam_one(pfpr_fml, dat, label = glue("{id}:pfpr"))
  mort_res <- fit_scam_one(mort_fml, dat, label = glue("{id}:mort"))
  inc_res  <- fit_scam_one(inc_fml,  dat, label = glue("{id}:inc"))

  if (isTRUE(pfpr_res$error) || isTRUE(mort_res$error) || isTRUE(inc_res$error)) {
    message(glue("  [{id}] a fit errored -> NOT saved / registered."))
    return(list(id = id, key = key, ok = FALSE))
  }

  pfpr_mod <- pfpr_res$fit; mort_mod <- mort_res$fit; inc_mod <- inc_res$fit
  model_names <- c("pfpr_mod", "mort_mod", "inc_mod")
  for (nm in model_names) {
    full_mod <- get(nm)
    stripped <- strip_scam_for_predict(full_mod)
    verify_strip(full_mod, stripped, verify_sample, glue("{id}:{nm}"))
    assign(nm, stripped)
  }

  save(list = model_names, file = glue("{data_path}/{key}_malaria_models.RData"))
  upsert_malaria_model_run(
    path        = malaria_model_registry_path(data_path),
    run_date    = key,
    description = spec$desc,
    best        = identical(id, FLAG_BEST_ID),
    extra = list(
      rdata_file     = glue("{key}_malaria_models.RData"),
      formulation_id = id,
      models         = model_names,
      inc_count_threshold = INC_COUNT_THRESHOLD,
      pfpr_threshold      = PFPR_THRESHOLD,
      pfpr_formula   = deparse1(pfpr_fml),
      mort_formula   = deparse1(mort_fml),
      inc_formula    = deparse1(inc_fml),
      parquet_path   = normalizePath(parquet_path, mustWork = TRUE),
      suit_variant   = suit_variant,
      pfpr_converged = pfpr_res$converged, pfpr_iter = pfpr_res$iter,
      inc_converged  = inc_res$converged,  inc_iter  = inc_res$iter,
      mort_converged = mort_res$converged, mort_iter = mort_res$iter
    )
  )
  list(id = id, key = key, ok = TRUE,
       pfpr_conv = pfpr_res$converged, inc_conv = inc_res$converged, mort_conv = mort_res$converged)
}

# -------------------------- Run all formulations --------------------------
ids <- vapply(FORMULATIONS, function(s) s$id, character(1))
if (anyDuplicated(ids)) stop(glue("Duplicate formulation ids: {paste(ids[duplicated(ids)], collapse=', ')}"))
if (nzchar(FLAG_BEST_ID) && !(FLAG_BEST_ID %in% ids)) {
  stop(glue("FLAG_BEST_ID='{FLAG_BEST_ID}' is not one of the formulation ids: {paste(ids, collapse=', ')}"))
}

results <- lapply(FORMULATIONS, fit_formulation)

message("\n===== SUMMARY =====")
for (r in results) {
  if (isTRUE(r$ok)) {
    message(glue("[{r$id}] key={r$key} SAVED  (converged pfpr={r$pfpr_conv} inc={r$inc_conv} mort={r$mort_conv})"))
  } else {
    message(glue("[{r$id}] FAILED — not saved"))
  }
}
