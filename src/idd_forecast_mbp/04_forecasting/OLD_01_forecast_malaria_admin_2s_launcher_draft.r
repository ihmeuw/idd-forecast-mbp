#!/usr/bin/env Rscript
# ============================================================================
# 01_forecast_malaria_admin_2s_launcher_draft.r
#
# DRAFT / BEST-GUESS SKELETON — *** NOT TESTED ***. Written in a Python session
# where R cannot run; treat every line as a starting point to verify, not as
# working code. Pairs with forecast_malaria_admin_2s_rocket_draft.r.
#
# Design:        .claude/FORECAST_04_REWRITE_GUIDE.md
# Map + tests:   .claude/FORECAST_04_DRAFT_NOTES.md
#
# Builds the param map (one row per ssp x dah), submits a 1-N array of the
# rocket, then repoints the output node's `current` symlink ONCE after the whole
# array succeeds.
# ============================================================================

suppressPackageStartupMessages({ library(glue); library(data.table) })

# Absolute REPO_DIR mirrors the other 04/03 R scripts; there is no R-side path
# config in this repo (the Python side uses constants.py). <<DEBT: see guide>>
REPO_DIR    <- "/mnt/team/idd/pub/forecast-mbp"
USER        <- Sys.getenv("USER")
SRC_REPO    <- glue("/ihme/homes/{USER}/repos/idd-forecast-mbp")
ROCKET      <- glue("{SRC_REPO}/src/idd_forecast_mbp/04_forecasting/forecast_malaria_admin_2s_rocket_draft.r")
FINALIZE    <- glue("{SRC_REPO}/src/idd_forecast_mbp/04_forecasting/finalize_malaria_forecast_draft.r")
message_dir <- "/mnt/team/idd/pub"

# ---------------------------- run-level choices ----------------------------
RUN_DATE           <- format(Sys.Date(), "%Y%m%d")  # output node version; ONE per launch
MODEL_RUN_DATE     <- "2026_06_02"                  # registry date; "" => rocket uses best
SSP_SCENARIOS      <- c("ssp126", "ssp245", "ssp585")
DAH_SCENARIOS      <- c("Baseline")                 # length-1 now; add "Constant" later, no code change
FORECAST_START     <- 2023L
FORECAST_END       <- 2100L
RAKE_YEAR          <- "2023"                        # scalar year OR a path to a per-loc rake-year parquet
ZERO_BURDEN_POLICY <- "drop"                        # "drop" (A) | "impute" (B, future)

# ---------------------------- param map ----------------------------
param_map <- CJ(ssp_scenario = SSP_SCENARIOS, dah_scenario = DAH_SCENARIOS)
param_map[, `:=`(
  model_run_date      = MODEL_RUN_DATE,
  forecast_start_year = FORECAST_START,
  forecast_end_year   = FORECAST_END,
  rake_year           = RAKE_YEAR,
  zero_burden_policy  = ZERO_BURDEN_POLICY,
  run_date            = RUN_DATE
)]
param_map_filepath <- glue("{REPO_DIR}/04-forecasting_data/malaria_forecast_param_map.csv")
fwrite(param_map, param_map_filepath)
message(glue("Wrote param map ({nrow(param_map)} tasks): {param_map_filepath}"))

# ---------------------------- submit the array ----------------------------
n_tasks   <- nrow(param_map)
IMG       <- "/ihme/singularity-images/rstudio/ihme_rstudio_4222.img"
SHELL     <- "/ihme/singularity-images/rstudio/shells/execRscript.sh"
# cores -> in-task mclapply over draws; mem holds (kept_loc x year x draw) x2 float arrays + forks.
# <<PROBE: run one task (n_jobs "1-1") and size mem/time from sacct before the full array.>>
cores_flag <- "-c 10"
mem_flag   <- "--mem=100G"
time_flag  <- "-t 240"
# BLAS taming so forked workers don't oversubscribe (DECISIONS 2026-05-14).
export_flag <- "--export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1"
err_flag <- glue("-e {message_dir}/stderr/%x.e%j")
out_flag <- glue("-o {message_dir}/stdout/%x.o%j")
proj_flag <- "-A proj_rapidresponse"
queue_flag <- "-p all.q"
array_flag <- glue("-a 1-{n_tasks}")

qsub <- glue(
  "sbatch --parsable -J forecast_malaria {mem_flag} {cores_flag} {time_flag} ",
  "{proj_flag} {queue_flag} {export_flag} {array_flag} {err_flag} {out_flag} ",
  "{SHELL} -i {IMG} -s {ROCKET}"
)
array_jobid <- system(qsub, intern = TRUE)   # --parsable => the bare array job id
message(glue("Submitted array job {array_jobid} ({n_tasks} tasks)"))

# ---------------------------- finalize after the whole array ----------------------------
# Repoint the output node's `current` ONLY after every task exits 0 (afterok).
# <<OPEN: confirm this cluster ALLOWS SLURM --dependency=afterok. If it does not,
#   delete this block and run finalize_malaria_forecast_draft.r by hand AFTER you
#   have verified all {n_tasks} tasks succeeded (never finalize on a partial array).>>
fin_qsub <- glue(
  "sbatch -J forecast_malaria_finalize --dependency=afterok:{array_jobid} ",
  "--mem=4G -c 1 -t 10 {proj_flag} {queue_flag} ",
  "--export=ALL,IDD_FORECAST_RUN_DATE={RUN_DATE} {err_flag} {out_flag} ",
  "{SHELL} -i {IMG} -s {FINALIZE}"
)
system(fin_qsub)
message("Submitted finalize job (afterok on the array).")
