#!/usr/bin/env Rscript
# ============================================================================
# fit_selected_malaria_model.r
#
# Fit the SELECTED malaria PfPR spec (read from a selection_result.json) plus the
# standard incidence / mortality scams on the past inputs, strip them to a
# predict-only core, and write malaria_models.RData + run.json into --out-dir.
#
# This worker writes into the directory it is given and nothing else. It keeps no
# registry: the launcher (fit_selected_malaria_model.py) owns the models node under
# idd_tools.versions, and "flag best" is `idd-versions <node> promote`. Every fit
# setting is a required flag; there are no defaults here (they live in the committed
# selection config's `final_fit:` section and arrive through the launcher).
# ============================================================================
suppressPackageStartupMessages({
  library(glue); library(mgcv); library(scam); library(arrow)
  library(data.table); library(jsonlite); library(optparse)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--result",        type = "character", default = NA, help = "selection_result.json written by rank_selection_run.py"),
  make_option("--out-dir",       type = "character", default = NA, help = "directory to write malaria_models.RData + run.json into (the node's working slot)"),
  make_option("--past-inputs",   type = "character", default = NA, help = "malaria_past_inputs.parquet"),
  make_option("--inc-count-min", type = "double",    default = NA, help = "row filter: malaria_inc_count >= this"),
  make_option("--pfpr-min",      type = "double",    default = NA, help = "row filter: malaria_pfpr >= this"),
  make_option("--optimizer",     type = "character", default = NA, help = "scam optimizer (bfgs / efs)"),
  make_option("--maxit",         type = "integer",   default = NA, help = "scam control maxit"),
  make_option("--suit-variant",  type = "character", default = NA, help = "malaria_suitability_<variant> column for logit_malaria_suitability"),
  make_option("--inc-mort-rhs",  type = "character", default = NA, help = "RHS shared by the incidence and mortality scams")
)))
names(opt) <- gsub("-", "_", names(opt))
required <- c("result", "out_dir", "past_inputs", "inc_count_min", "pfpr_min",
              "optimizer", "maxit", "suit_variant", "inc_mort_rhs")
missing <- required[vapply(required, function(k) is.null(opt[[k]]) || is.na(opt[[k]]), logical(1))]
if (length(missing)) stop(glue("missing required flags: --{paste(gsub('_', '-', missing), collapse=' --')}"))
if (!dir.exists(opt$out_dir)) stop(glue("--out-dir does not exist: {opt$out_dir} (the launcher creates it)"))

# -------------------------- The selected spec --------------------------
res  <- jsonlite::fromJSON(opt$result, simplifyVector = TRUE)
pick <- res$pick
pfpr_fml <- as.formula(pick$formula_text)
if (!identical(as.character(pfpr_fml[[2]]), "logit_malaria_pfpr")) {
  stop(glue("selected formula response is {deparse1(pfpr_fml[[2]])}, expected logit_malaria_pfpr"))
}
inc_fml  <- reformulate(opt$inc_mort_rhs, response = "log_malaria_inc_rate")
mort_fml <- reformulate(opt$inc_mort_rhs, response = "log_malaria_mort_rate")
message(glue("selected spec {pick$spec_index} from {res$run_dir} (result status: {res$status})"))
message(glue("  pfpr: {deparse1(pfpr_fml)}"))

# -------------------------- Fit helper --------------------------
fit_scam_one <- function(fml, data, label) {
  t0 <- Sys.time()
  fit <- tryCatch(
    scam(fml, data = data, optimizer = opt$optimizer, control = list(maxit = as.integer(opt$maxit))),
    error = function(e) e
  )
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (inherits(fit, "error")) stop(glue("[{label}] scam error after {sprintf('%.1f', elapsed)}s: {conditionMessage(fit)}"))
  iter <- tryCatch(as.integer(fit$iter), error = function(e) NA_integer_)
  conv <- isTRUE(fit$conv)
  message(glue("  [{label}] iter={iter} converged={conv} ({sprintf('%.1f', elapsed)}s)"))
  list(fit = fit, iter = iter, converged = conv, elapsed = elapsed)
}

# Drop training-frame slots predict() never reads; verify_strip proves it lossless.
strip_scam_for_predict <- function(mod) {
  for (slot in c("model", "y", "residuals", "fitted.values",
                 "linear.predictors", "weights", "prior.weights")) mod[[slot]] <- NULL
  mod
}
verify_strip <- function(full, stripped, newdata, label) {
  max_abs <- max(abs(predict(full, newdata = newdata) - predict(stripped, newdata = newdata)))
  if (!is.finite(max_abs) || max_abs > 1e-8) {
    stop(glue("[{label}] strip changed predictions (max |diff| = {max_abs}); not saving."))
  }
  message(glue("  [{label}] strip verified (max |diff| = {signif(max_abs, 3)})"))
}

# -------------------------- Load + clean past data --------------------------
past_data <- as.data.frame(arrow::read_parquet(opt$past_inputs))
past_data$malaria_inc_count <- past_data$malaria_inc_rate * past_data$population
for (var in c("malaria_pfpr", "gdppc_mean", "mal_DAH_total_per_capita", "malaria_inc_rate", "malaria_mort_rate")) {
  past_data <- past_data[!is.na(past_data[[var]]), ]
}
past_data$do30_fraction <- pmin(pmax(past_data$days_over_30C / 365, 0.001), 0.999)
past_data$logit_do30    <- log(past_data$do30_fraction / (1 - past_data$do30_fraction))
past_data$rh_fraction   <- pmin(pmax(past_data$relative_humidity / 100, 0.001), 0.999)
past_data$logit_relative_humidity <- log(past_data$rh_fraction / (1 - past_data$rh_fraction))
for (cov in c("mal_DAH_total_per_capita", "gdppc_mean", "ldipc_mean", "med_consumppc",
              "malaria_inc_rate", "malaria_mort_rate")) {
  past_data[[paste0("log_", cov)]] <- log(past_data[[cov]])
}
n_before <- nrow(past_data)
past_data <- past_data[past_data$malaria_inc_count >= opt$inc_count_min & past_data$malaria_pfpr >= opt$pfpr_min, ]
past_data$A0_af <- as.factor(past_data$A0_location_id)
suit_col <- paste0("malaria_suitability_", opt$suit_variant)
if (!suit_col %in% names(past_data)) stop(glue("suit_variant '{opt$suit_variant}' -> column '{suit_col}' not in past inputs"))
past_data$malaria_suit <- past_data[[suit_col]]
frac <- pmin(pmax(past_data[[suit_col]] / 365, 0.001), 0.999)
past_data$logit_malaria_suitability <- log(frac / (1 - frac))
message(glue("past inputs: {n_before} rows -> {nrow(past_data)} after filter (inc_count >= {opt$inc_count_min}, pfpr >= {opt$pfpr_min})"))

set.seed(1)
verify_sample <- past_data[sample(nrow(past_data), min(5000L, nrow(past_data))), ]

# -------------------------- Fit --------------------------
pfpr_res <- fit_scam_one(pfpr_fml, past_data, "pfpr")
mort_res <- fit_scam_one(mort_fml, past_data, "mort")
inc_res  <- fit_scam_one(inc_fml,  past_data, "inc")

# The fitted pfpr model must carry exactly the selected spec's terms.
fitted_terms <- attr(terms(pfpr_res$fit$formula), "term.labels")
spec_terms   <- attr(terms(pfpr_fml), "term.labels")
if (!setequal(fitted_terms, spec_terms)) {
  stop(glue("fitted pfpr terms differ from the selected spec:\n  fit : {paste(fitted_terms, collapse=' + ')}\n  spec: {paste(spec_terms, collapse=' + ')}"))
}

pfpr_mod <- strip_scam_for_predict(pfpr_res$fit); verify_strip(pfpr_res$fit, pfpr_mod, verify_sample, "pfpr")
mort_mod <- strip_scam_for_predict(mort_res$fit); verify_strip(mort_res$fit, mort_mod, verify_sample, "mort")
inc_mod  <- strip_scam_for_predict(inc_res$fit);  verify_strip(inc_res$fit,  inc_mod,  verify_sample, "inc")

# -------------------------- Write (tmp + rename) --------------------------
rdata <- file.path(opt$out_dir, "malaria_models.RData")
tmp   <- paste0(rdata, ".tmp")
save(list = c("pfpr_mod", "mort_mod", "inc_mod"), file = tmp)
file.rename(tmp, rdata); Sys.chmod(rdata, "0664")

run <- list(
  fitted_at    = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  models       = c("pfpr_mod", "mort_mod", "inc_mod"),
  rdata_file   = basename(rdata),
  pfpr_formula = deparse1(pfpr_fml), inc_formula = deparse1(inc_fml), mort_formula = deparse1(mort_fml),
  inc_count_threshold = opt$inc_count_min, pfpr_threshold = opt$pfpr_min,
  optimizer = opt$optimizer, maxit = as.integer(opt$maxit), suit_variant = opt$suit_variant,
  past_inputs = normalizePath(opt$past_inputs), n_rows_fit = nrow(past_data),
  pfpr_converged = pfpr_res$converged, pfpr_iter = pfpr_res$iter, pfpr_elapsed_sec = pfpr_res$elapsed,
  inc_converged  = inc_res$converged,  inc_iter  = inc_res$iter,  inc_elapsed_sec  = inc_res$elapsed,
  mort_converged = mort_res$converged, mort_iter = mort_res$iter, mort_elapsed_sec = mort_res$elapsed,
  r_version = R.version.string, scam_version = as.character(packageVersion("scam")),
  mgcv_version = as.character(packageVersion("mgcv")),
  selection = list(
    result_file = normalizePath(opt$result), run_dir = res$run_dir, result_status = res$status,
    spec_index = pick$spec_index, formula_text = pick$formula_text,
    spec_table_fingerprint = res$spec_table_fingerprint, rank = res$rank, code = res$code,
    config_source = res$config_source
  )
)
run_json <- file.path(opt$out_dir, "run.json")
tmp <- paste0(run_json, ".tmp")
jsonlite::write_json(run, tmp, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA)
file.rename(tmp, run_json); Sys.chmod(run_json, "0664")
message(glue("Wrote {rdata}\n      {run_json}"))
message("fin")
