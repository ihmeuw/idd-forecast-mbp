### 01a_fit_prelim_malaria_models_launcher.r
### STAGE 1 of the 2-stage malaria pfpr model selection: the cheap, wide SCREEN.
### Submits an sbatch array of select_malaria_models_rocket.r that fits ONLY the
### in-sample scam WITH country fixed effects per spec (no OOS CV, no rds, no
### plots) -> one lean parquet row per spec. The expensive OOS pipeline (stage 2)
### is run later on only the survivors of this screen.
###
### Screened WITH the FEs on purpose: the country FEs absorb most of the
### variation, so a no-FE screen would rank covariates on variance the FEs
### reclaim once reinstated -> the wrong objective. The screen fits FE-present.
###
### (a) FE in-sample only         -> FIT_IS_FE=TRUE, FIT_IS_NOFE/FIT_OOS=FALSE
### (b) balanced chunks           -> specs sorted by n_smooths, dealt round-robin
###                                  so big models aren't all in one chunk
### (c) probe N random chunks     -> PROBE_N_CHUNKS>0 submits that many random
###                                  chunks to measure runtime + MaxRSS on sacct

require(glue)
require(data.table)

USER <- Sys.getenv("USER")
repo_dir    <- glue("/ihme/homes/{USER}/repos/idd-forecast-mbp")
message_dir <- "/mnt/team/idd/pub"

script <- glue("{repo_dir}/src/idd_forecast_mbp/03_modeling/select_malaria_models_rocket.r")

# -------------------------- Build the specs (UNCHANGED) --------------------------

# Per-variable allowed forms. "linear" + bs codes from scam.
var_forms <- list(
  # Group 1 (decreasing, always in)
  "mal_DAH_total_per_capita"     = c("linear", "mpd"),
  "log_mal_DAH_total_per_capita" = c("linear"),

  # Group 2 (decreasing, always in)
  "gdppc_mean"        = c("linear", "mpd"),
  "log_gdppc_mean"    = c("linear"),
  "ldipc_mean"        = c("linear", "mpd"),
  "log_ldipc_mean"    = c("linear"),
  "med_consumppc"     = c("linear", "mpd"),
  "log_med_consumppc" = c("linear"),

  # Group 3 (decreasing)
  "weighted_1km_urban_threshold_300.0_simple_mean"   = c("linear", "mpd"),
  "weighted_1km_urban_threshold_1500.0_simple_mean"  = c("linear", "mpd"),
  "weighted_100m_urban_threshold_1500.0_simple_mean" = c("linear", "mpd"),

  # Group 4 (mpi or cv)
  "people_flood_days_per_capita" = c("linear", "mpi", "cv"),

  # Group 5 (mpi or cv)
  "total_precipitation"          = c("linear", "mpi", "cv"),

  # Group 6 (increasing)
  "precipitation_days"           = c("linear", "mpi", "cv"),
  "relative_humidity" = c("linear", "mpi"),
  "logit_relative_humidity" = c("linear", "mpi"),

  # Group 7 (mixed)
  "mean_temperature"                 = c("linear", "cv"),
  "mean_low_temperature"             = c("linear", "cv"),
  "mean_high_temperature"            = c("linear", "cv"),
  "days_over_30C"                    = c("linear", "cv"),
  "logit_do30"                       = c("linear"),
  "malaria_suitability_mordecai_0_0" = c("linear", "mpi"),
  "logit_malaria_suitability"        = c("linear"),

  # Group 8 (linear only, always in)
  "A0_af" = c("linear")
)

groups <- list(
  g1 = list(always_in = TRUE,  vars = c("mal_DAH_total_per_capita", "log_mal_DAH_total_per_capita")),
  g2 = list(always_in = TRUE,  vars = c("gdppc_mean", "log_gdppc_mean", "ldipc_mean", "log_ldipc_mean", "med_consumppc", "log_med_consumppc")),
  g3 = list(always_in = FALSE, vars = c("weighted_1km_urban_threshold_300.0_simple_mean", "weighted_1km_urban_threshold_1500.0_simple_mean", "weighted_100m_urban_threshold_1500.0_simple_mean")),
  g4 = list(always_in = FALSE, vars = c("people_flood_days_per_capita")),
  g5 = list(always_in = FALSE, vars = c("total_precipitation", "precipitation_days")),
  g6 = list(always_in = FALSE, vars = c("relative_humidity", "logit_relative_humidity")),
  g7 = list(always_in = FALSE, vars = c("mean_temperature", "mean_low_temperature", "mean_high_temperature", "days_over_30C", "logit_do30", "malaria_suitability_mordecai_0_0", "logit_malaria_suitability")),
  g8 = list(always_in = TRUE,  vars = c("A0_af"))
)
groups <- list(
  g1 = list(always_in = TRUE,  vars = c("mal_DAH_total_per_capita", "log_mal_DAH_total_per_capita")),
  g2 = list(always_in = TRUE,  vars = c("gdppc_mean", "log_gdppc_mean", "ldipc_mean", "log_ldipc_mean", "med_consumppc", "log_med_consumppc")),
  g3 = list(always_in = FALSE, vars = c("weighted_1km_urban_threshold_300.0_simple_mean", "weighted_1km_urban_threshold_1500.0_simple_mean", "weighted_100m_urban_threshold_1500.0_simple_mean")),
  g4 = list(always_in = FALSE, vars = c("people_flood_days_per_capita")),
  g5 = list(always_in = FALSE, vars = c("total_precipitation", "precipitation_days")),
  g6 = list(always_in = FALSE, vars = c("relative_humidity", "logit_relative_humidity")),
  g7 = list(always_in = FALSE, vars = c("mean_temperature", "mean_low_temperature", "mean_high_temperature", "days_over_30C", "logit_do30", "malaria_suitability_mordecai_0_0", "logit_malaria_suitability")),
  g8 = list(always_in = TRUE,  vars = c("A0_af"))
)

# Build all (var, form) options for a group; NA represents "skip" if optional
group_options <- function(group, var_forms) {
  opts <- list()
  if (!group$always_in) opts <- c(opts, list(NA))
  for (v in group$vars) {
    for (f in var_forms[[v]]) {
      opts <- c(opts, list(list(var = v, form = f)))
    }
  }
  opts
}

# Cartesian product across groups -> list of model specs
expand_models <- function(groups, var_forms) {
  per_group <- lapply(groups, group_options, var_forms = var_forms)
  ng  <- sapply(per_group, length)
  idx <- do.call(expand.grid, lapply(ng, seq_len))

  lapply(seq_len(nrow(idx)), function(i) {
    spec <- list()
    for (g in seq_along(groups)) {
      opt <- per_group[[g]][[idx[i, g]]]
      if (!identical(opt, NA)) spec[[opt$var]] <- opt$form
    }
    spec
  })
}

# Formula construction (build_term/K_DEFAULT kept in sync with the rocket).
K_DEFAULT <- 6
build_term <- function(var, form) {
  if (form == "linear") return(var)
  sprintf("s(%s, k = %d, bs = '%s')", var, K_DEFAULT, form)
}
build_formula <- function(spec, response) {
  rhs <- mapply(build_term, names(spec), unlist(spec), USE.NAMES = FALSE)
  reformulate(rhs, response = response)
}

# Filter: neighborhood around the original submission model.
# Holds these 2 vars at their submission-model forms; lets all others vary.
matches <- function(spec) {
  # identical(spec[["logit_malaria_suitability"]],  "linear") &&
  identical(spec[["gdppc_mean"]],                 "mpd")    &&
  identical(spec[["mal_DAH_total_per_capita"]],   "mpd")
}

n_smooths <- function(spec) {
  sum(unlist(spec) != "linear")
}

MAX_SMOOTHS <- 6
specs <- expand_models(groups, var_forms)
length(specs)
# stopifnot(length(specs) == 294840) # 73710 if we group g4 and g5
neighborhood_specs <- Filter(function(s) matches(s) && n_smooths(s) <= MAX_SMOOTHS, specs)
length(neighborhood_specs)
message(glue("Built {length(neighborhood_specs)} neighborhood specs."))

# -------------------------- Chunking (built ONCE; shared by both optimizers) --------------------------
# Deterministic: sort specs by n_smooths, deal round-robin -> equal-cost chunks, and a
# given task_id maps to the SAME specs in every dir below (no RNG in the assignment).
CHUNK_SIZE  <- 20L
ns_per_spec <- vapply(neighborhood_specs, n_smooths, integer(1))
n_specs     <- length(neighborhood_specs)
n_chunks    <- as.integer(ceiling(n_specs / CHUNK_SIZE))
ord         <- order(ns_per_spec)
task_id     <- integer(n_specs)
task_id[ord] <- ((seq_len(n_specs) - 1L) %% n_chunks) + 1L

formula_texts <- vapply(neighborhood_specs, function(s) {
  deparse1(build_formula(s, "logit_malaria_pfpr"))
}, FUN.VALUE = character(1))
param_map <- data.table(
  task_id      = task_id,
  spec_index   = seq_along(neighborhood_specs),
  n_smooths    = ns_per_spec,
  formula_text = formula_texts
)
setorder(param_map, task_id, spec_index)
message(glue("{n_specs} specs -> {n_chunks} chunk(s) of ~{CHUNK_SIZE}, balanced by n_smooths."))

# -------------------------- Run scope --------------------------
OPTIMIZERS <- c("efs")                          # k=6 EFS full screen
# Full grid = every chunk. For a probe instead, set RUN_CHUNKS <- c(<ids>).
RUN_CHUNKS <- seq_len(n_chunks)
stopifnot("RUN_CHUNKS out of range" = all(RUN_CHUNKS >= 1L & RUN_CHUNKS <= n_chunks))
array_spec <- if (length(RUN_CHUNKS) == n_chunks) glue("1-{n_chunks}") else paste(sort(RUN_CHUNKS), collapse = ",")

run_date    <- format(Sys.Date(), "%Y%m%d")
output_root <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/scam_prelim/lsae_1285"

# Shared resourcing. --mem sized for the heavier optimizer (BFGS peaked ~8 GB; EFS ~2.5 GB).
thread_flag         <- "-c 8"
mem_flag            <- "--mem=5G"     # ~1.5x the measured 3.4 GB peak (k=6 EFS)
time_flag           <- "-t 35"        # ~1.5x the measured 24-min/20-spec k=6 EFS chunk
queue_flag          <- "-p long.q"
project_flag        <- "-A proj_rapidresponse"
error_filepath      <- glue("-e {message_dir}/stderr/%x.e%j")
output_filepath     <- glue("-o {message_dir}/stdout/%x.o%j")
IMG   <- "/ihme/singularity-images/rstudio/ihme_rstudio_4524.img"   # == this interactive session's image
SHELL <- "/ihme/singularity-images/rstudio/shells/execRscript.sh"

for (opt in OPTIMIZERS) {
  output_dir <- glue("{output_root}/{run_date}_{opt}")
  if (dir.exists(output_dir)) {
    v <- 2L; while (dir.exists(glue("{output_dir}_v{v}"))) v <- v + 1L
    output_dir <- glue("{output_dir}_v{v}")
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  saveRDS(neighborhood_specs, file = glue("{output_dir}/neighborhood_specs.rds"))
  fwrite(param_map, glue("{output_dir}/param_map.csv"))
  writeLines(c(
    glue("Run timestamp: {format(Sys.time(), '%Y-%m-%d %H:%M:%S')}"),
    glue("Optimizer:     {opt}  (k=6 full screen)"),
    glue("Screen:        in-sample WITH country FE (FIT_IS_FE=TRUE)"),
    glue("Chunks:        {array_spec}   (CHUNK_SIZE={CHUNK_SIZE}, K_DEFAULT={K_DEFAULT}, MAXIT=50)"),
    glue("N specs:       {n_specs}  ->  {n_chunks} chunks"),
    glue("Output dir:    {output_dir}")
  ), con = glue("{output_dir}/run_info.txt"))

  rocket_env <- glue(
    "--export=ALL,FIT_OUTPUT_DIR='{output_dir}',",
    "FIT_IS_FE=TRUE,FIT_IS_NOFE=FALSE,FIT_OOS=FALSE,WRITE_RDS=FALSE,WRITE_PLOT=FALSE,",
    "OPTIMIZER={opt},MAXIT=50,K_DEFAULT={K_DEFAULT},OPENBLAS_NUM_THREADS=8,OMP_NUM_THREADS=8")
  qsub_command <- glue(
    "sbatch -J prelim_{opt} {mem_flag} {thread_flag} {project_flag} {time_flag} {queue_flag} ",
    "-a '{array_spec}' {rocket_env} {error_filepath} {output_filepath} ",
    "{SHELL} -i {IMG} -s {script}")
  message(glue("[{opt}] -> {output_dir}"))
  message(qsub_command)
  system(qsub_command)
}
message(glue("Submitted [{paste(OPTIMIZERS, collapse = ', ')}] -> array {array_spec} ",
             "({n_specs} specs, {n_chunks} chunks)."))

# After the probe finishes, measure actual usage to size the full run:
#   sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem,AllocCPUS,NCPUS
