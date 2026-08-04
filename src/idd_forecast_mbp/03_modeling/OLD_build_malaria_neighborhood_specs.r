### build_malaria_neighborhood_specs.r
### Builds the malaria spec list and writes the two artifacts the orchestrator
### + worker consume:
###   neighborhood_specs.rds   the spec objects (worker reads by spec_index)
###   spec_table.parquet       spec_index:int32, n_smooths:int32, formula_text:str
### Does NOT submit anything. Prints the output dir to pass to the orchestrator.

suppressPackageStartupMessages({
  library(glue); library(data.table); library(arrow)
})

# >>> EDIT: where the spec list lands (also the orchestrator/worker run root) <<<
OUTPUT_ROOT <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/scam_prelim/lsae_1285"

# -------------------------- Build the specs --------------------------

# var_forms: for each covariate, which functional forms to consider. Names are the
# form codes; values are the basis dimension K for that (covariate, form) term:
#   linear               -> bare term            (K ignored; write NA)
#   smooth               -> s(var, k = K)         unconstrained thin-plate
#   mpi / mpd / cv / ...  -> s(var, k = K, bs='<code>')  scam shape-constrained
# K is per-(covariate, form): a var offered as both mpi and smooth can carry a
# different K for each. A non-linear form left off / set NA falls back to K_DEFAULT.

# FE-only neighborhood: lags dropped (see the lag experiment conclusion — lag beats FE
# only under observed-covariate OOS leakage; degrades with depth under honest compounded
# forecasting). Empty `lags` => g8 collapses to {A0_af} and no lag var_forms are added,
# so every spec carries the additive country FE and nothing else as its country term.
# MUST stay consistent with the worker's `lags` and the orchestrator's `max_lag` (guard
# in the worker asserts this).
lags <- c()


var_forms <- list(
  # DAH
  "mal_DAH_total_per_capita"     = c(mpd = 4L),
  "log_mal_DAH_total_per_capita" = c(linear = NA),
  # GDP
  "gdppc_mean"        = c(mpd = 4L),
  "log_gdppc_mean"    = c(linear = NA),
  # Temperature Suitability
  "malaria_suit" = c(mpi = 6L),
  "logit_malaria_suitability"        = c(linear = NA),
  # Temperature
  "mean_temperature"                 = c(linear = NA, smooth = 4L),
  "mean_low_temperature"             = c(linear = NA, smooth = 4L),
  # Urbanization
  "weighted_1km_urban_threshold_300.0_simple_mean"   = c(linear = NA, mpd = 4L),
  # Rainfall
  "total_precipitation"          = c(linear = NA, mpi = 4L),
  # Humidity
  "relative_humidity" = c(mpi = 6L),
  "logit_relative_humidity" = c(linear = NA),
  # Country fixed effect
  "A0_af" = c(linear = NA)
)

# Lagged country-level PfPR
for (L in lags) {
  col <- paste0("a0_malaria_pfpr_lag", L)
  var_forms[[col]] <- c(mpi = 6L)          # or whatever form(s) you want for the lag term
}

g8_vars <- c("A0_af", paste0("a0_malaria_pfpr_lag", lags))

groups <- list(
  g1 = list(always_in = FALSE,  vars = c("mal_DAH_total_per_capita")),
  g2 = list(always_in = FALSE,  vars = c("gdppc_mean")),
  g3 = list(always_in = FALSE, vars = c("malaria_suit", "logit_malaria_suitability")),
  g4 = list(always_in = FALSE, vars = c("mean_temperature", "mean_low_temperature")),
  g5 = list(always_in = FALSE, vars = c("weighted_1km_urban_threshold_300.0_simple_mean")),
  g6 = list(always_in = FALSE, vars = c("total_precipitation")),
  g7 = list(always_in = FALSE, vars = c("relative_humidity", "logit_relative_humidity")),
  g8 = list(always_in = TRUE,  vars = g8_vars)
)

group_options <- function(group, var_forms) {
  opts <- list()
  if (!group$always_in) opts <- c(opts, list(NA))
  for (v in group$vars) {
    for (f in names(var_forms[[v]])) {   # form codes are the names now; K is the value
      opts <- c(opts, list(list(var = v, form = f)))
    }
  }
  opts
}

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

# Fallback basis dimension for any non-linear form that doesn't carry an explicit
# K in var_forms above. Per-term K lives in var_forms; this is only the backstop.
K_DEFAULT <- 6L
resolve_k <- function(var, form) {
  k <- var_forms[[var]][form]          # K for this (var, form); NA / absent -> fallback
  if (length(k) == 0L || is.na(k)) K_DEFAULT else as.integer(k)
}

build_term <- function(var, form) {
  if (form == "linear") return(var)
  k <- resolve_k(var, form)
  if (form == "smooth") return(sprintf("s(%s, k = %d)", var, k))   # unconstrained tp spline
  sprintf("s(%s, k = %d, bs = '%s')", var, k, form)                # scam monotone/convex
}
build_formula <- function(spec, response) {
  rhs <- mapply(build_term, names(spec), unlist(spec), USE.NAMES = FALSE)
  reformulate(rhs, response = response)
}

# Force 
matches <- function(spec) {
  identical(spec[["gdppc_mean"]],               "mpd") &&
  identical(spec[["mal_DAH_total_per_capita"]], "mpd") &&
  # Drop specs where suit (either form) is present AND a temperature term enters
  # as mpi: suit absorbs the temperature signal, collapsing the mpi smooth to
  # edf=0 (20260701_efs_v2 diagnosis). Temp-as-linear alongside suit is fine.
  !(( !is.null(spec[["malaria_suit"]]) || !is.null(spec[["logit_malaria_suitability"]]) ) &&
    ( identical(spec[["mean_temperature"]],     "mpi") ||
      identical(spec[["mean_low_temperature"]], "mpi") ))
}
n_smooths <- function(spec) sum(unlist(spec) != "linear")
n_scams <- function(spec) sum(unlist(spec) != "linear" & unlist(spec) != "smooth")

MAX_SMOOTHS <- 7   # FE-only: DAH+gdppc + at most one each of suit/temp/urban/precip/humidity
specs <- expand_models(groups, var_forms)
# neighborhood_specs <- Filter(function(s) matches(s) && n_smooths(s) <= MAX_SMOOTHS, specs)
neighborhood_specs <- specs
message(glue("Built {length(neighborhood_specs)} specs."))

# -------------------------- Write artifacts --------------------------

output_dir <- glue("{OUTPUT_ROOT}/{format(Sys.Date(), '%Y%m%d')}_efs")
if (dir.exists(output_dir)) {
  v <- 2L; while (dir.exists(glue("{output_dir}_v{v}"))) v <- v + 1L
  output_dir <- glue("{output_dir}_v{v}")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

saveRDS(neighborhood_specs, file.path(output_dir, "neighborhood_specs.rds"))

spec_table <- data.table(
  spec_index   = as.integer(seq_along(neighborhood_specs)),
  n_smooths    = as.integer(vapply(neighborhood_specs, n_smooths, integer(1))),
  n_scams     = as.integer(vapply(neighborhood_specs, n_scams, integer(1))),
  formula_text = vapply(neighborhood_specs,
                        function(s) deparse1(build_formula(s, "logit_malaria_pfpr")),
                        character(1))
)
arrow::write_parquet(spec_table, file.path(output_dir, "spec_table.parquet"))
Sys.chmod(output_dir, "0775")

message(glue("n_smooths distribution: {paste(names(table(spec_table$n_smooths)), table(spec_table$n_smooths), sep='x', collapse=', ')}"))
message(glue("Wrote neighborhood_specs.rds + spec_table.parquet to:\n{output_dir}"))
cat(output_dir, "\n")
