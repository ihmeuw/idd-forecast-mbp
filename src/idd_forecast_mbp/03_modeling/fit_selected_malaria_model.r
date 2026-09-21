#!/usr/bin/env Rscript
# ============================================================================
# fit_selected_malaria_model.r
#
# Fit the SELECTED malaria PfPR spec (read from a selection_result.json) plus the
# standard incidence / mortality scams on the past inputs, strip them to a
# predict-only core, and write malaria_models.RData + run.json into --out-dir.
#
# The frame comes from prepare_malaria_fit_frame() in --prep-script (default:
# lib/malaria_fit_frame.R next to this file's parent directory), the same preparation
# the selection worker uses; fitting and stripping come from lib/scam_fit_helpers.R.
# Acceptance 2026-09-18: at thresholds (0, 0) on the 20260527 past inputs this
# preparation reproduces the registered 2026_07_31 fit to 3e-10 (coef, sp).
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

this_file_path <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", ca[grepl("^--file=", ca)])
  if (length(fa)) normalizePath(fa[1]) else NA_character_
}
default_lib_dir <- function() {
  f <- this_file_path()
  if (is.na(f)) NA_character_ else normalizePath(file.path(dirname(f), "..", "lib"), mustWork = FALSE)
}

opt <- parse_args(OptionParser(option_list = list(
  make_option("--result",        type = "character", default = NA, help = "selection_result.json written by rank_selection_run.py"),
  make_option("--out-dir",       type = "character", default = NA, help = "directory to write malaria_models.RData + run.json into (the node's working slot)"),
  make_option("--past-inputs",   type = "character", default = NA, help = "malaria_past_inputs.parquet"),
  make_option("--inc-count-min", type = "double",    default = NA, help = "row filter: malaria_inc_count >= this"),
  make_option("--pfpr-min",      type = "double",    default = NA, help = "row filter: malaria_pfpr >= this"),
  make_option("--optimizer",     type = "character", default = NA, help = "scam optimizer (bfgs / efs)"),
  make_option("--maxit",         type = "integer",   default = NA, help = "scam control maxit"),
  make_option("--suit-variant",  type = "character", default = NA, help = "malaria_suitability_<variant> column for logit_malaria_suitability"),
  make_option("--inc-mort-rhs",  type = "character", default = NA, help = "RHS shared by the incidence and mortality scams"),
  make_option("--prep-script",   type = "character", default = NA, help = "R file defining prepare_malaria_fit_frame(); default <lib-dir>/malaria_fit_frame.R"),
  make_option("--lib-dir",       type = "character", default = NA, help = "directory holding scam_fit_helpers.R; default ../lib relative to this file")
)))
names(opt) <- gsub("-", "_", names(opt))
required <- c("result", "out_dir", "past_inputs", "inc_count_min", "pfpr_min",
              "optimizer", "maxit", "suit_variant", "inc_mort_rhs")
missing <- required[vapply(required, function(k) is.null(opt[[k]]) || is.na(opt[[k]]), logical(1))]
if (length(missing)) stop(glue("missing required flags: --{paste(gsub('_', '-', missing), collapse=' --')}"))
if (!dir.exists(opt$out_dir)) stop(glue("--out-dir does not exist: {opt$out_dir} (the launcher creates it)"))

# -------------------------- Shared code --------------------------
lib_dir <- if (is.na(opt$lib_dir)) default_lib_dir() else opt$lib_dir
if (is.na(lib_dir) || !dir.exists(lib_dir)) stop("cannot locate lib dir (pass --lib-dir)")
source(file.path(lib_dir, "scam_fit_helpers.R"))
prep_script <- if (is.na(opt$prep_script)) file.path(lib_dir, "malaria_fit_frame.R") else opt$prep_script
if (!file.exists(prep_script)) stop(glue("prep script not found: {prep_script}"))
source(prep_script)
if (!exists("prepare_malaria_fit_frame", mode = "function")) stop("prep script defines no prepare_malaria_fit_frame()")

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
# Every model here is a scam; an error is fatal (nothing is written).
fit_scam_one <- function(fml, data, label) {
  r <- fit_one_mod(fml, data, n_scams = 1L, n_smooths = 1L,
                   optimizer = opt$optimizer, maxit = as.integer(opt$maxit), label = label)
  if (r$error) stop(glue("[{label}] scam error after {sprintf('%.1f', r$elapsed)}s: {r$error_msg}"))
  r
}

# -------------------------- Load + prepare past data --------------------------
past_data <- prepare_malaria_fit_frame(opt$past_inputs, opt$inc_count_min, opt$pfpr_min, opt$suit_variant)
for (fml in list(pfpr_fml, inc_fml, mort_fml)) {
  absent <- setdiff(all.vars(fml), names(past_data))
  if (length(absent)) stop(glue("formula names columns the prepared frame lacks: {paste(absent, collapse = ', ')}"))
}
message(glue("past inputs: {attr(past_data, 'n_read')} rows -> {nrow(past_data)} after filter (inc_count >= {opt$inc_count_min}, pfpr >= {opt$pfpr_min})"))

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
  past_inputs = normalizePath(opt$past_inputs), past_inputs_sha256 = sha256_file(opt$past_inputs),
  prep_script = normalizePath(prep_script), prep_script_sha256 = sha256_file(prep_script),
  n_rows_read = attr(past_data, "n_read"), n_rows_fit = nrow(past_data),
  n_non_finite_rows = attr(past_data, "n_non_finite_rows"), n_a0_levels = nlevels(past_data$A0_af),
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
write_json_atomic(run, file.path(opt$out_dir, "run.json"))
message(glue("Wrote {rdata}\n      {file.path(opt$out_dir, 'run.json')}"))
message("fin")
