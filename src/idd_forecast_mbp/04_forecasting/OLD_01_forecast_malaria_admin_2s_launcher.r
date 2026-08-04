#!/usr/bin/env Rscript
# ============================================================================
# 01_forecast_malaria_admin_2s_launcher.r
#
# Builds the malaria forecast param map (one row per ssp x dah), submits a 1-N
# array of forecast_malaria_admin_2s_rocket.r (each task loops the 100 draws
# internally via mclapply), then repoints the output node's `current` symlink
# ONCE after the whole array succeeds (finalize job, afterok dependency).
#
# Design:        .claude/FORECAST_04_REWRITE_GUIDE.md
# Map + tests:   .claude/FORECAST_04_DRAFT_NOTES.md
# ============================================================================

suppressPackageStartupMessages({ library(glue); library(data.table) })

REPO_DIR    <- "/mnt/team/idd/pub/forecast-mbp"      # == constants.MODEL_ROOT (verified)
USER        <- Sys.getenv("USER")
SRC_REPO    <- glue("/ihme/homes/{USER}/repos/idd-forecast-mbp")
ROCKET      <- glue("{SRC_REPO}/src/idd_forecast_mbp/04_forecasting/forecast_malaria_admin_2s_rocket.r")
FINALIZE    <- glue("{SRC_REPO}/src/idd_forecast_mbp/04_forecasting/finalize_malaria_forecast.r")
message_dir <- "/mnt/team/idd/pub"

# ---------------------------- run-level choices ----------------------------
# Forecast one or more fitted FORMULATIONS. Each was saved by
# 02_fit_final_malaria_models.r as {MODEL_FIT_DATE}_{id}_malaria_models.RData and
# registered under run_date = "{MODEL_FIT_DATE}_{id}". Each formulation forecasts
# into its OWN output dir (forecast_outputs/lsae_1285/{MODEL_FIT_DATE}_{id}/) so
# they coexist for side-by-side comparison / ensembling.
MODEL_FIT_DATE     <- "2026_07_14"                          # date the formulations were fit
MODEL_IDS          <- c("hybrid")                           # <<< which formulations to forecast
# All six are forecastable once stage-08 has been re-run with mean_low_temperature
# (climate_mean, single realization) in its covariate set — the rocket now reads
# mean_low_temperature and malaria_suit. If you point at a stage-08 nc that
# predates that rebuild, drop f1-f4 (they use mean_low_temperature).
SSP_SCENARIOS      <- c("ssp126", "ssp245", "ssp585")
DAH_SCENARIOS      <- c("Baseline")                 # length-1 now; add "Constant" later, no code change
FORECAST_START     <- 2023L
FORECAST_END       <- 2100L
RAKE_YEAR          <- "2023"                        # 4-digit scalar OR a path to a per-loc rake-year parquet
ZERO_BURDEN_POLICY <- "drop"                        # "drop" (A) | "impute" (B, future)

# ---------------------------- param map ----------------------------
# One task per (formulation x ssp x dah). model_run_date = registry key that
# resolves the .RData; run_date = per-formulation output dir.
param_map <- CJ(model_id = MODEL_IDS, ssp_scenario = SSP_SCENARIOS, dah_scenario = DAH_SCENARIOS)
# NB: use paste0 (not glue) here — this expression is evaluated in the data.table
# column scope, where model_id is a COLUMN; glue would look it up as a variable
# in the calling frame and fail ("object 'model_id' not found").
param_map[, `:=`(
  model_run_date      = paste0(MODEL_FIT_DATE, "_", model_id),
  forecast_start_year = FORECAST_START,
  forecast_end_year   = FORECAST_END,
  rake_year           = RAKE_YEAR,
  zero_burden_policy  = ZERO_BURDEN_POLICY,
  run_date            = paste0(MODEL_FIT_DATE, "_", model_id)
)]
param_map_filepath <- glue("{REPO_DIR}/04-forecasting_data/malaria_forecast_param_map.csv")
fwrite(param_map, param_map_filepath)
message(glue("Wrote param map ({nrow(param_map)} tasks): {param_map_filepath}"))

# ---------------------------- submit the array ----------------------------
n_tasks   <- nrow(param_map)
# Pinned to the current stable image (latest.img -> ihme_rstudio_4523 as of 2026-06-02) for
# reproducibility — NOT latest.img, which silently repoints. See .claude/DECISIONS.md.
IMG       <- "/mnt/share/singularity-images/rstudio/ihme_rstudio_4523.img"
SHELL     <- "/ihme/singularity-images/rstudio/shells/execRscript.sh"
# cores -> in-task mclapply over draws; mem holds suit_dt (~21GB) + 10 forks + (kept x year x draw) arrays.
# Sized from the test-8 staged probe (5/10/20 draws): full 100-draw task ~25 min, ~42GB peak RSS.
# predict is memory-bandwidth-bound (5->10 forks: 50->135s), so >10 cores won't scale linearly.
cores_flag  <- "-c 10"
# Sized from the 2026-06-03 real run (array 47891353, 100 draws). ACTUAL per-scenario peak RSS:
#   ssp126 43.4G | ssp245 47.5G | ssp585 41.9G ; wall 29-31 min.
# PROJECT RULE: requested time must never exceed 2x the estimate.
mem_flag    <- "--mem=60G"    # ~12G headroom over the 47.5G peak; kept at 60G deliberately.
                              # REVISIT (bump) if the rocket does MORE per task: more draws,
                              # Constant dah alongside Baseline, or extra outcome variables.
time_flag   <- "-t 45"        # ~1.45x the ~31min actual (<= 2x rule)
# After the array completes, record actual usage to tune future runs (sacct is login-node only):
#   src/idd_forecast_mbp/04_forecasting/record_resources.sh forecast_malaria_real <array_jobid>
# BLAS taming so forked workers don't oversubscribe (DECISIONS 2026-05-14).
export_flag <- "--export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1"
err_flag    <- glue("-e {message_dir}/stderr/%x.e%j")
out_flag    <- glue("-o {message_dir}/stdout/%x.o%j")
proj_flag   <- "-A proj_rapidresponse"
queue_flag  <- "-p all.q"

# Full run = one task per param-map row. For the test-point-8 probe, set:
#   N_JOBS <- "1-1"   # just the first row, to measure mem/time on sacct first
N_JOBS    <- glue("1-{n_tasks}")
array_flag <- glue("-a '{N_JOBS}'")

qsub <- glue(
  "sbatch --parsable -J forecast_malaria {mem_flag} {cores_flag} {time_flag} ",
  "{proj_flag} {queue_flag} {export_flag} {array_flag} {err_flag} {out_flag} ",
  "{SHELL} -i {IMG} -s {ROCKET}"
)
array_jobid <- system(qsub, intern = TRUE)   # --parsable => the bare array job id
message(glue("Submitted array job {array_jobid} ({n_tasks} tasks)"))

# ---------------------------- finalize ----------------------------
# Multi-formulation exploratory run: each formulation writes its OWN dated dir
# ({MODEL_FIT_DATE}_{id}) and is inspected there by key, so we do NOT repoint the
# output node's `current` (it can point at only one; leave it on the production run).
# After you pick a winner, finalize that one by hand (afterok confirmed working here):
#   IDD_FORECAST_RUN_DATE=<MODEL_FIT_DATE>_<id>  execRscript.sh -i <IMG> -s finalize_malaria_forecast.r
message(glue("No finalize: {length(MODEL_IDS)} formulation(s) -> per-id dirs ",
             "({paste(paste0(MODEL_FIT_DATE, '_', MODEL_IDS), collapse = ', ')}); `current` left unchanged."))
