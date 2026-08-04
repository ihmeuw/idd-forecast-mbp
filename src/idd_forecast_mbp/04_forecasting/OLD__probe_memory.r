#!/usr/bin/env Rscript
# ============================================================================
# _probe_memory.r  —  resource probe for the malaria forecast rocket.
#
# Runs ONE rocket task (ssp126 / Baseline) over the FULL location universe and
# FULL year window, but only the first PROBE_N_DRAWS draws, so a staged probe
# (e.g. 5 / 10 / 20 draws) can extrapolate runtime + memory to the real 100.
#
# Calls the rocket's *real* helpers (read_forecast_inputs -> make_predict_one_draw
# -> mclapply -> assemble_arrays -> write_forecast_netcdf), so it exercises the
# exact memory-heavy paths. Output goes to node-local /tmp (ephemeral; no shared
# storage written, nothing to clean up). Peak RSS comes from sacct/seff afterward;
# this script also prints phase timing + the R-heap peak as a cross-check.
# ============================================================================
suppressPackageStartupMessages({
  library(glue); library(data.table); library(tidync); library(ncdf4)
  library(mgcv); library(scam); library(arrow); library(parallel)
})
ROCKET <- "/ihme/homes/bcreiner/repos/idd-forecast-mbp/src/idd_forecast_mbp/04_forecasting/forecast_malaria_admin_2s_rocket.r"
source(ROCKET)   # defines helpers + paths; main() stays dormant (--file guard)

N_DRAWS        <- as.integer(Sys.getenv("PROBE_N_DRAWS", "5"))
ssp            <- "ssp126"; dah <- "Baseline"
forecast_years <- 2023:2100
rake_year_spec <- "2023"
mrd            <- "2026_06_02"

secs  <- function() as.numeric(Sys.time())
phase <- function(label, t0) message(glue("[probe] {label}: {sprintf('%.1f', secs() - t0)}s"))
t0 <- secs()
message(glue("[probe] === PROBE_N_DRAWS={N_DRAWS}  cores={Sys.getenv('SLURM_CPUS_PER_TASK','?')} ==="))

# --- load model ---
ts <- secs()
data_path <- file.path(REPO_DIR, "03-modeling_data")
load(file.path(data_path, glue("{mrd}_malaria_models.RData")))   # pfpr_mod, inc_mod, mort_mod
models <- list(pfpr = pfpr_mod, inc = inc_mod, mort = mort_mod)
fb     <- list(pfpr = fallback_level(pfpr_mod), inc = fallback_level(inc_mod), mort = fallback_level(mort_mod))
phase("load model", ts)

# --- read inputs (ALL draws — the dominant, draw-count-independent read) ---
ts <- secs()
src_locs       <- as.integer(tidync(forecast_inputs_nc(ssp))$transforms$location_id$location_id)
rake_years_vec <- resolve_rake_years(rake_year_spec, src_locs)
read_years     <- sort(unique(c(forecast_years, as.integer(rake_years_vec))))
inputs <- read_forecast_inputs(forecast_inputs_nc(ssp), read_years, dah)
phase("read inputs (all draws)", ts)
message(glue("[probe] suit_dt rows = {format(nrow(inputs$suit_dt), big.mark=',')}; n_draws available = {length(inputs$draws)}"))

# --- observed + classify ---
ts <- secs()
obs    <- read_rake_year_observed(raked_aa_parquet, inputs$all_locs, rake_years_vec)
status <- classify_zero_burden(obs, "drop")
kept_ids <- sort(status[status != "dropped", location_id])
inc_ids  <- status[status %in% c("kept_both", "inc_only"),  location_id]
mort_ids <- status[status %in% c("kept_both", "mort_only"), location_id]
phase("observed + classify", ts)
message(glue("[probe] kept={length(kept_ids)} dropped={status[status=='dropped',.N]} ",
             "inc_only={status[status=='inc_only',.N]} mort_only={status[status=='mort_only',.N]}"))

# --- predict N draws ---
ts <- secs()
predict_one_draw <- make_predict_one_draw(inputs, obs, models, fb,
                                          kept_ids, inc_ids, mort_ids,
                                          rake_years_vec, forecast_years)
draws_use <- head(inputs$draws, N_DRAWS)
n_cores   <- max(1L, as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")))
per_draw  <- mclapply(draws_use, predict_one_draw, mc.cores = n_cores)
failed    <- vapply(per_draw, function(x) inherits(x, "try-error"), logical(1))
if (any(failed)) stop(glue("{sum(failed)} fork(s) failed: ",
                           conditionMessage(attr(per_draw[[which(failed)[1]]], "condition"))))
phase(glue("predict {N_DRAWS} draws (mc.cores={n_cores})"), ts)

# --- assemble + write ---
# Throwaway probe output goes inside the project output dir (NEVER /tmp or any node-local
# temp — governance rule, ~/.claude/CLAUDE.md). Remove this subdir manually when done.
ts <- secs()
arrs    <- assemble_arrays(per_draw, kept_ids, forecast_years, draws_use)
out_dir <- file.path(forecast_output_node, "_probe_scratch"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_nc  <- file.path(out_dir, glue("probe_{ssp}_{dah}_n{N_DRAWS}.nc"))
write_forecast_netcdf(arrs$inc, arrs$mort, kept_ids, forecast_years, draws_use, out_nc)
phase("assemble + write", ts)
message(glue("[probe] wrote {out_nc} ({sprintf('%.1f', file.info(out_nc)$size/1e6)} MB) — project output dir, remove manually"))

phase("TOTAL", t0)
gct <- gc()
message(glue("[probe] R-heap max used: {round(sum(gct[, ncol(gct)]), 0)} MB ",
             "(R heap only; sacct/seff MaxRSS is the true peak incl. forks)"))
message("[probe] DONE")
