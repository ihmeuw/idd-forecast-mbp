### This file will be called: 01_fit_malaria_models_launcher.r
### Launches an sbatch array job that fits 210 scam pfpr models, one per
### spec in the "neighborhood" around the original submission model.

require(glue)
require(data.table)

USER <- Sys.getenv("USER")
repo_dir    <- glue("/ihme/homes/{USER}/repos/idd-forecast-mbp")
message_dir <- "/mnt/team/idd/pub"

script <- glue("{repo_dir}/src/idd_forecast_mbp/03_modeling/fit_malaria_models_rocket_bfgs.r")

# -------------------------- Build the 210 specs --------------------------

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

# Formula construction (duplicated in rocket — keep in sync)
K_DEFAULT <- 6
build_term <- function(var, form) {
  if (form == "linear") return(var)
  sprintf("s(%s, k = %d, bs = '%s')", var, K_DEFAULT, form)
}
build_formula <- function(spec, response) {
  rhs <- mapply(build_term, names(spec), unlist(spec), USE.NAMES = FALSE)
  reformulate(rhs, response = response)
}

# Filter: 210-spec neighborhood around the original submission model.
# Holds these 3 vars at their submission-model forms; lets all others vary.
matches <- function(spec) {
  # identical(spec[["logit_malaria_suitability"]],  "linear") &&
  identical(spec[["gdppc_mean"]],                 "mpd")    &&
  identical(spec[["mal_DAH_total_per_capita"]],   "mpd")
}

n_smooths <- function(spec) {
  sum(unlist(spec) != "linear")
}

MAX_SMOOTHS <- 5
specs <- expand_models(groups, var_forms)
length(specs)
# stopifnot(length(specs) == 294840) # 73710 if we group g4 and g5
neighborhood_specs <- Filter(function(s) matches(s) && n_smooths(s) <= MAX_SMOOTHS, specs)
length(neighborhood_specs)

# stopifnot(length(neighborhood_specs) == 840) # 210 if we group g4 and g5
message(glue("Built {length(neighborhood_specs)} neighborhood specs."))

# -------------------------- Output dir + param map --------------------------

run_date    <- format(Sys.Date(), "%Y%m%d")
output_root <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/scam_fits/lsae_1285"
output_dir  <- glue("{output_root}/{run_date}")

# Bump to _v2, _v3 if a run already exists today (per STANDARDS versioning)
if (dir.exists(output_dir)) {
  v <- 2L
  while (dir.exists(glue("{output_dir}_v{v}"))) v <- v + 1L
  output_dir <- glue("{output_dir}_v{v}")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
message(glue("Output dir: {output_dir}"))

# Save the specs list (the rocket reads this to recover its spec from task_id).
saveRDS(neighborhood_specs, file = glue("{output_dir}/neighborhood_specs.rds"))

# Param map: one row per task_id. Rocket reads spec_index; formula_text is
# included so a human can `head` the CSV and see what each task fits.
formula_texts <- vapply(neighborhood_specs, function(s) {
  deparse1(build_formula(s, "logit_malaria_pfpr"))
}, FUN.VALUE = character(1))

param_map <- data.table(
  run_num      = seq_along(neighborhood_specs),
  spec_index   = seq_along(neighborhood_specs),
  formula_text = formula_texts
)
param_map_filepath <- glue("{output_dir}/param_map.csv")
fwrite(param_map, param_map_filepath)

# Stamp the run config for traceability.
writeLines(c(
  glue("Run timestamp: {format(Sys.time(), '%Y-%m-%d %H:%M:%S')}"),
  glue("User:          {USER}"),
  glue("Output dir:    {output_dir}"),
  glue("N specs:       {length(neighborhood_specs)}"),
  glue("Response:      logit_malaria_pfpr"),
  glue("Past data:     /mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/past_inputs_nc/lsae_1285/current/malaria_past_inputs.parquet"),
  glue("Source:        Extracted from src/idd_forecast_mbp/03_modeling/fit_malaria_models.qmd")
), con = glue("{output_dir}/run_info.txt"))

# -------------------------- Submit array job --------------------------

# CV_STRATEGY: "none" (in-sample only), "random", "country", or "country_no_fe"
CV_STRATEGY  <- "country_no_fe"

job_name     <- "fit_malaria_pfpr"
thread_flag  <- "-c 16"
mem_flag     <- "--mem=32G"
runtime_flag <- if (CV_STRATEGY == "none") "-t 30" else "-t 240"
queue_flag   <- "-p long.q"

n_jobs          <- paste0("1-", nrow(param_map))
error_filepath  <- glue("-e {message_dir}/stderr/%x.e%j")
output_filepath <- glue("-o {message_dir}/stdout/%x.o%j")
project_flag    <- "-A proj_rapidresponse"

# Pass output_dir to the rocket via env var so different launcher runs
# don't collide on a fixed param_map path.
rocket_env <- glue("--export=ALL,FIT_OUTPUT_DIR='{output_dir}',CV_STRATEGY='{CV_STRATEGY}',OPENBLAS_NUM_THREADS=8,OMP_NUM_THREADS=8")

qsub_command <- glue(
  "sbatch -J {job_name} {mem_flag} {thread_flag} {project_flag} {runtime_flag} {queue_flag} ",
  "-a '{n_jobs}' {rocket_env} {error_filepath} {output_filepath} ",
  "/ihme/singularity-images/rstudio/shells/execRscript.sh ",
  "-i /ihme/singularity-images/rstudio/ihme_rstudio_4222.img -s {script}"
)
message(qsub_command)
system(qsub_command)
