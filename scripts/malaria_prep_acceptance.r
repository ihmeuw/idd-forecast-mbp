#!/usr/bin/env Rscript
# Acceptance test for the shared malaria data preparation (handoff 2026-09-18, D1).
#
# Refits the registered model's three formulas (read from the registered RData itself) on the
# frame the shared preparation builds, and compares coefficients, smoothing parameters,
# df.null and the country levels with the registered objects. Writes acceptance.json and the
# stripped refits into --out-dir (a scratch slot; the directory must already exist) and exits
# non-zero when any comparison is outside --tolerance. Never writes anywhere else.
suppressPackageStartupMessages({
  library(optparse); library(scam); library(jsonlite); library(glue)
})

this_file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
default_prep <- normalizePath(file.path(dirname(this_file[1]), "..", "src", "idd_forecast_mbp", "lib",
                                        "malaria_fit_frame.R"), mustWork = FALSE)

opt <- parse_args(OptionParser(option_list = list(
  make_option("--parquet",       type = "character", help = "past inputs the registered model was fit on"),
  make_option("--rdata",         type = "character", help = "registered malaria_models.RData (pfpr_mod, mort_mod, inc_mod)"),
  make_option("--out-dir",       type = "character", help = "existing scratch directory for acceptance.json + refits"),
  make_option("--prep-script",   type = "character", default = default_prep),
  make_option("--optimizer",     type = "character", default = "bfgs"),
  make_option("--maxit",         type = "integer",   default = 300L),
  make_option("--inc-count-min", type = "double",    default = 0),
  make_option("--pfpr-min",      type = "double",    default = 0),
  make_option("--suit-variant",  type = "character", default = "mordecai_0_0"),
  make_option("--tolerance",     type = "double",    default = 1e-6, help = "max relative difference allowed on coef and sp")
)))
names(opt) <- gsub("-", "_", names(opt))
for (k in c("parquet", "rdata", "out_dir")) if (is.null(opt[[k]])) stop("missing --", gsub("_", "-", k))
if (!dir.exists(opt$out_dir)) stop("--out-dir must exist: ", opt$out_dir)

source(opt$prep_script)
if (!exists("prepare_malaria_fit_frame", mode = "function")) stop("prep script defines no prepare_malaria_fit_frame()")

t0 <- Sys.time()
frame <- prepare_malaria_fit_frame(opt$parquet, opt$inc_count_min, opt$pfpr_min, opt$suit_variant)
message(glue("frame: {nrow(frame)} rows, {nlevels(frame$A0_af)} A0 levels ",
             "({round(as.numeric(difftime(Sys.time(), t0, units = 'secs')), 1)}s)"))

registered <- new.env()
load(opt$rdata, envir = registered)
models <- c("pfpr_mod", "mort_mod", "inc_mod")
stopifnot(all(models %in% ls(registered)))

rel_diff <- function(a, b) max(abs(a - b) / (1 + abs(b)))
strip_for_predict <- function(mod) {
  for (slot in c("model", "y", "residuals", "fitted.values", "linear.predictors", "weights", "prior.weights")) mod[[slot]] <- NULL
  mod
}

report <- list(
  ran_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  parquet = normalizePath(opt$parquet), rdata = normalizePath(opt$rdata),
  prep_script = normalizePath(opt$prep_script),
  optimizer = opt$optimizer, maxit = opt$maxit,
  inc_count_min = opt$inc_count_min, pfpr_min = opt$pfpr_min, suit_variant = opt$suit_variant,
  tolerance = opt$tolerance,
  frame = list(n_read = attr(frame, "n_read"), n_rows = nrow(frame), n_a0_levels = nlevels(frame$A0_af),
               n_non_finite_rows = attr(frame, "n_non_finite_rows")),
  r_version = R.version.string, scam_version = as.character(packageVersion("scam")),
  mgcv_version = as.character(packageVersion("mgcv")),
  models = list()
)

all_pass <- TRUE
for (nm in models) {
  reg <- registered[[nm]]
  fml <- reg$formula
  message(glue("\n== {nm}: {deparse1(fml)}"))
  t1 <- Sys.time()
  refit <- scam(fml, data = frame, optimizer = opt$optimizer, control = list(maxit = opt$maxit))
  elapsed <- as.numeric(difftime(Sys.time(), t1, units = "secs"))

  same_names <- identical(names(coef(refit)), names(coef(reg)))
  coef_rd <- if (same_names) rel_diff(coef(refit), coef(reg)) else NA_real_
  sp_rd   <- if (length(refit$sp) == length(reg$sp)) rel_diff(refit$sp, reg$sp) else NA_real_
  same_levels <- identical(refit$xlevels[["A0_af"]], reg$xlevels[["A0_af"]])
  same_dfnull <- isTRUE(all.equal(refit$df.null, reg$df.null))
  pass <- same_names && same_levels && same_dfnull &&
    is.finite(coef_rd) && coef_rd <= opt$tolerance && is.finite(sp_rd) && sp_rd <= opt$tolerance
  all_pass <- all_pass && pass

  message(glue("   coef names identical: {same_names} | max rel diff coef: {signif(coef_rd, 4)} | sp: {signif(sp_rd, 4)}"))
  message(glue("   df.null refit {refit$df.null} vs registered {reg$df.null} | A0 levels identical: {same_levels}"))
  message(glue("   iter {refit$iter} (registered {reg$iter}) | converged {isTRUE(refit$conv)} | {round(elapsed, 1)}s | PASS: {pass}"))

  report$models[[nm]] <- list(
    formula = deparse1(fml), pass = pass, elapsed_sec = elapsed,
    n_coef = length(coef(refit)), coef_names_identical = same_names,
    max_rel_diff_coef = coef_rd, max_rel_diff_sp = sp_rd,
    sp_refit = unname(refit$sp), sp_registered = unname(reg$sp),
    df_null_refit = refit$df.null, df_null_registered = reg$df.null, df_null_identical = same_dfnull,
    a0_levels_identical = same_levels, n_a0_levels = length(refit$xlevels[["A0_af"]]),
    deviance_refit = refit$deviance, deviance_registered = reg$deviance,
    sig2_refit = refit$sig2, sig2_registered = reg$sig2,
    iter_refit = refit$iter, iter_registered = reg$iter, converged_refit = isTRUE(refit$conv)
  )
  assign(paste0(nm, "_refit"), strip_for_predict(refit))
}
report$pass <- all_pass

rd <- file.path(opt$out_dir, "malaria_models_refit.RData")
save(list = paste0(models, "_refit"), file = paste0(rd, ".tmp")); file.rename(paste0(rd, ".tmp"), rd)
js <- file.path(opt$out_dir, "acceptance.json")
write_json(report, paste0(js, ".tmp"), auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")
file.rename(paste0(js, ".tmp"), js)
message(glue("\nwrote {js}\n      {rd}\nOVERALL PASS: {all_pass}"))
if (!all_pass) quit(save = "no", status = 1)
