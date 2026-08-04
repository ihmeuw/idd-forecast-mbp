#!/usr/bin/env Rscript
# ============================================================================
# finalize_malaria_forecast.r
#
# Repoint the malaria forecast output node's `current` symlink at this run's
# dated directory, AFTER the whole rocket array has succeeded. Submitted by
# 01_forecast_malaria_admin_2s_launcher.r as an afterok dependency job (or run
# by hand once all array tasks are confirmed successful).
#
# The run date is passed via the IDD_FORECAST_RUN_DATE env var (launcher
# --export), so this stays a tiny stateless finalizer.
# ============================================================================
suppressPackageStartupMessages({ library(glue); library(optparse) })

SRC_REPO <- glue("/ihme/homes/{Sys.getenv('USER')}/repos/idd-forecast-mbp")
source(glue("{SRC_REPO}/src/idd_forecast_mbp/lib/versioning.R"))   # finalize_artifact()

REPO_DIR       <- "/mnt/team/idd/pub/forecast-mbp"                 # == constants.MODEL_ROOT
LSAE_HIERARCHY <- "lsae_1285"
out_node <- file.path(REPO_DIR, "04-forecasting_data", "malaria", "forecast_outputs", LSAE_HIERARCHY)

# Run date passed as a flag by the jobmon orchestrator (--run-date {model_run_date}).
# Any non-empty name is accepted (e.g. 2026_07_14_hybrid); finalize_artifact() below
# is the real check -- it errors if that dated run dir does not exist.
opt <- parse_args(OptionParser(option_list = list(
  make_option("--run-date", type = "character", default = NA,
              help = "dated run dir under the forecast output node to point `current` at")
)))
names(opt) <- gsub("-", "_", names(opt))
run_date <- opt$run_date
if (is.na(run_date) || !nzchar(run_date)) {
  stop("--run-date not set. Pass the run dir name (e.g. 2026_07_14_hybrid).")
}

message(glue("Finalizing malaria forecast output node -> {run_date}"))
finalize_artifact(out_node, run_date)   # errors if the dated run dir is absent; replaces a symlink only
message("Finalize complete.")
