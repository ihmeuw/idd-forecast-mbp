#!/usr/bin/env Rscript
# ============================================================================
# forecast_malaria_admin_2s_rocket_draft.r
#
# DRAFT / BEST-GUESS SKELETON — *** NOT TESTED ***. Written in a Python session
# where R cannot run. Verify every step; the reader mirrors the verified demos
# (read_forecast_inputs_demo.r / lazy_subset_forecast_inputs_demo.r) but the
# predict / rake / write logic has never executed.
#
# Design:      .claude/FORECAST_04_REWRITE_GUIDE.md
# Map + tests: .claude/FORECAST_04_DRAFT_NOTES.md
#
# One task = one param_map row = one (ssp_scenario, dah_scenario). Loops the 100
# draws internally (mclapply). Reads 08 inputs lazily (tidync) + rake-year
# observed (raked AA parquet), predicts pfpr -> inc/mort with a uniform (or
# per-location) shift to the rake year, applies the zero-burden policy, and
# writes ONE compressed netCDF + a small location-status sidecar parquet.
# ============================================================================

suppressPackageStartupMessages({
  library(glue); library(data.table); library(tidync); library(ncdf4)
  library(mgcv); library(scam); library(arrow); library(parallel)
})

REPO_DIR <- "/mnt/team/idd/pub/forecast-mbp"                 # <<DEBT: mirrors other R scripts
SRC_REPO <- glue("/ihme/homes/{Sys.getenv('USER')}/repos/idd-forecast-mbp")
source(glue("{SRC_REPO}/src/idd_forecast_mbp/lib/model_registry.R"))   # get_malaria_model_run_date()
source(glue("{SRC_REPO}/src/idd_forecast_mbp/lib/netcdf_helpers.R"))   # grid_of()

Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1")  # tame BLAS under mclapply
LSAE_HIERARCHY <- "lsae_1285"                                  # <<CHECK: keep in sync with constants.py
data_path <- glue("{REPO_DIR}/03-modeling_data")

# ======================== 1. task params (nothing hardcoded) ========================
task_id <- suppressWarnings(as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID")))
if (is.na(task_id)) task_id <- 1L
pm  <- fread(glue("{REPO_DIR}/04-forecasting_data/malaria_forecast_param_map.csv"))
row <- pm[task_id]
ssp_scenario       <- row$ssp_scenario
dah_scenario       <- row$dah_scenario
model_run_date     <- as.character(row$model_run_date)
forecast_years     <- as.integer(row$forecast_start_year):as.integer(row$forecast_end_year)
rake_year_spec     <- as.character(row$rake_year)
zero_burden_policy <- as.character(row$zero_burden_policy)
out_run_date       <- as.character(row$run_date)
message(glue("[task {task_id}] ssp={ssp_scenario} dah={dah_scenario} ",
             "years={min(forecast_years)}-{max(forecast_years)} policy={zero_burden_policy}"))

# ======================== 2. resolve + load model ========================
mrd <- get_malaria_model_run_date(
  malaria_model_registry_path(data_path),
  best     = !nzchar(model_run_date),
  run_date = if (nzchar(model_run_date)) model_run_date else NULL
)
load(glue("{data_path}/{mrd}_malaria_models.RData"))           # -> pfpr_mod, inc_mod, mort_mod
stopifnot(all(c("pfpr_mod", "inc_mod", "mort_mod") %in% ls()))

# ======================== 3. lazy-read 08 inputs (tidync; from demos) ========================
inputs_dir <- glue("{REPO_DIR}/04-forecasting_data/malaria/forecast_inputs/{LSAE_HIERARCHY}/current")
nc_file    <- file.path(inputs_dir, glue("malaria_forecast_inputs_{ssp_scenario}.nc"))
src        <- tidync(nc_file)
all_locs   <- as.integer(src$transforms$location_id$location_id)   # lazy: coords only

# rake year per location (scalar -> constant vector; or a file with location_id, rake_year)
resolve_rake_years <- function(spec, location_ids) {
  if (grepl("^[0-9]{4}$", spec)) {
    setNames(rep(as.integer(spec), length(location_ids)), as.character(location_ids))
  } else {
    rk <- as.data.table(arrow::read_parquet(spec))               # <<CHECK schema: location_id, rake_year
    setNames(as.integer(rk$rake_year[match(location_ids, rk$location_id)]),
             as.character(location_ids))
  }
}
rake_years_vec <- resolve_rake_years(rake_year_spec, all_locs)
read_years     <- sort(unique(c(forecast_years, rake_years_vec)))  # must cover the rake year(s)

# pull(): lazily filter a grid to locs (+ optional years), realize as data.table.
pull <- function(exemplar_var, locs, years = NULL) {
  af <- activate(src, grid_of(src, exemplar_var))
  af <- hyper_filter(af, location_id = location_id %in% locs)
  if (!is.null(years) && "year_id" %in% names(src$transforms)) {
    af <- hyper_filter(af, year_id = year_id %in% years)
  }
  dt <- as.data.table(hyper_tibble(af))
  for (cc in intersect(c("location_id", "year_id", "draw"), names(dt))) dt[[cc]] <- as.integer(dt[[cc]])
  dt[]
}

a0_dt   <- pull("A0_location_id", all_locs)                                   # loc-only
gdp_dt  <- pull("gdppc_mean", all_locs, read_years)                           # loc x year
fld_dt  <- pull("people_flood_days_per_capita", all_locs, read_years)        # loc x year
dah_dt  <- pull("mal_DAH_total_per_capita", all_locs, read_years)            # loc x year x dah
dah_dt  <- dah_dt[dah_scenario == ..dah_scenario]                             # select scenario post-realize
suit_dt <- pull("malaria_suitability", all_locs, read_years)                  # loc x year x draw (KEEP draws)
draws   <- sort(unique(suit_dt$draw))

# ======================== 4. rake-year observed (raked AA parquet) ========================
# 08's locations ARE 07b's most-detailed prediction set, so filtering aa by them is
# equivalent to the level-5 filter (no hierarchy read needed).
aa_file <- glue("{REPO_DIR}/02-processed_data/malaria/raked_aa/{LSAE_HIERARCHY}/current/aa_full_malaria_df.parquet")
aa <- as.data.table(arrow::read_parquet(aa_file))
aa <- aa[location_id %in% all_locs & year_id %in% read_years,
         .(location_id, year_id, malaria_pfpr, malaria_inc_rate, malaria_mort_rate)]
# one row per location, at THAT location's rake year
obs <- aa[data.table(location_id = all_locs,
                     year_id     = unname(rake_years_vec[as.character(all_locs)])),
          on = .(location_id, year_id)]
obs[, `:=`(
  # match the fit's pfpr transform exactly (05 used 0.999 * pfpr)
  obs_logit_pfpr = log(0.999 * malaria_pfpr / (1 - 0.999 * malaria_pfpr)),
  obs_log_inc    = fifelse(malaria_inc_rate  > 0, log(malaria_inc_rate),  NA_real_),
  obs_log_mort   = fifelse(malaria_mort_rate > 0, log(malaria_mort_rate), NA_real_)
)]

# ======================== 5. zero-burden policy -> status + kept + mask ========================
classify_zero_burden <- function(obs, policy) {
  s <- obs[, .(location_id, pfpr = malaria_pfpr, inc = malaria_inc_rate, mort = malaria_mort_rate)]
  s[, `:=`(status = "kept_both", drop_reason = NA_character_)]
  s[is.na(pfpr) | pfpr == 0,              `:=`(status = "dropped",   drop_reason = "pfpr_zero")]
  s[status != "dropped" & inc == 0 & mort == 0, `:=`(status = "dropped", drop_reason = "inc_mort_both_zero")]
  s[status != "dropped" & inc == 0 & mort >  0, status := "mort_only"]   # predict mort, inc = NaN
  s[status != "dropped" & mort == 0 & inc >  0, status := "inc_only"]    # predict inc,  mort = NaN
  if (policy != "drop") {
    stop("zero_burden_policy='impute' (option B) not implemented in this draft.")  # <<TODO option B
  }
  s[]
}
status   <- classify_zero_burden(obs, zero_burden_policy)
kept_ids <- sort(status[status != "dropped", location_id])
inc_ids  <- status[status %in% c("kept_both", "inc_only"),  location_id]   # locations that get an inc value
mort_ids <- status[status %in% c("kept_both", "mort_only"), location_id]   # locations that get a mort value
message(glue("  kept={length(kept_ids)} (dropped={status[status=='dropped', .N]}, ",
             "inc_only={status[status=='inc_only', .N]}, mort_only={status[status=='mort_only', .N]})"))

# ======================== 6. orphan-A0 lowest-coef fallback (per model) ========================
fallback_level <- function(mod) {                       # level with the lowest total A0 effect (ref = 0)
  b   <- coef(mod); a0 <- b[grepl("^A0_af", names(b))]
  lev <- mod$xlevels$A0_af
  eff <- setNames(rep(0, length(lev)), lev)
  eff[sub("^A0_af", "", names(a0))] <- a0
  names(eff)[which.min(eff)]
}
fb <- list(pfpr = fallback_level(pfpr_mod),
           inc  = fallback_level(inc_mod),
           mort = fallback_level(mort_mod))

# build the A0_af factor for a model: real level if in the fit, else the fallback (orphan)
make_A0_af <- function(location_ids, mod, fb_level) {
  a0_id <- a0_dt$A0_location_id[match(location_ids, a0_dt$location_id)]
  af    <- factor(as.character(a0_id), levels = mod$xlevels$A0_af)
  af[is.na(af)] <- fb_level
  af
}

# ======================== 7. predict per draw (mclapply) ========================
# uniform-or-per-location shift: for each location, shift = obs - raw_pred at that
# location's rake year, applied to all years.
apply_shift <- function(dt, raw_col, obs_dt, obs_col) {
  rk <- data.table(location_id = as.integer(names(rake_years_vec)),
                   rake_year   = as.integer(rake_years_vec))
  at_rake <- dt[rk, on = .(location_id, year_id == rake_year),
                .(location_id, raw_at_rake = get(raw_col)), nomatch = NULL]
  sh <- merge(at_rake, obs_dt, by = "location_id")
  sh[, shift := get(obs_col) - raw_at_rake]
  dt <- merge(dt, sh[, .(location_id, shift)], by = "location_id", all.x = TRUE)
  dt[, shifted := get(raw_col) + shift]
  dt$shifted
}

predict_one_draw <- function(d) {
  df <- Reduce(function(a, b) merge(a, b, by = c("location_id", "year_id")),
               list(suit_dt[draw == d, .(location_id, year_id, malaria_suitability)],
                    gdp_dt[, .(location_id, year_id, gdppc_mean)],
                    fld_dt[, .(location_id, year_id, people_flood_days_per_capita)],
                    dah_dt[, .(location_id, year_id, mal_DAH_total_per_capita)]))
  df <- df[location_id %in% kept_ids & year_id %in% forecast_years]
  df[, `:=`(
    logit_malaria_suitability = log(pmin(pmax(malaria_suitability / 365, 1e-3), 1 - 1e-3) /
                                    (1 - pmin(pmax(malaria_suitability / 365, 1e-3), 1 - 1e-3))),
    log_gdppc_mean            = log(gdppc_mean)
  )]

  # --- pfpr: predict (raw), shift to observed logit pfpr, use shifted as predictor ---
  df[, A0_af := make_A0_af(location_id, pfpr_mod, fb$pfpr)]
  df[, pfpr_raw := as.numeric(predict(pfpr_mod, newdata = df))]
  df[, logit_malaria_pfpr := apply_shift(df, "pfpr_raw", obs[, .(location_id, obs_logit_pfpr)], "obs_logit_pfpr")]

  # --- incidence ---
  df[, A0_af := make_A0_af(location_id, inc_mod, fb$inc)]
  df[, inc_raw := as.numeric(predict(inc_mod, newdata = df))]
  df[, log_malaria_inc_rate_pred := apply_shift(df, "inc_raw", obs[, .(location_id, obs_log_inc)], "obs_log_inc")]

  # --- mortality ---
  df[, A0_af := make_A0_af(location_id, mort_mod, fb$mort)]
  df[, mort_raw := as.numeric(predict(mort_mod, newdata = df))]
  df[, log_malaria_mort_rate_pred := apply_shift(df, "mort_raw", obs[, .(location_id, obs_log_mort)], "obs_log_mort")]

  # --- outcome mask: NaN the outcome a location is not supposed to get ---
  df[!(location_id %in% inc_ids),  log_malaria_inc_rate_pred  := NA_real_]
  df[!(location_id %in% mort_ids), log_malaria_mort_rate_pred := NA_real_]

  df[order(location_id, year_id),
     .(location_id, year_id, log_malaria_inc_rate_pred, log_malaria_mort_rate_pred)]
}

n_cores <- max(1L, as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")))
per_draw <- mclapply(draws, predict_one_draw, mc.cores = n_cores)
# <<TODO: detect failed forks — any(vapply(per_draw, inherits, logical(1), "try-error")) -> stop loudly>>

# ======================== 8. assemble (kept_loc x year x draw) ========================
n_loc <- length(kept_ids); n_yr <- length(forecast_years); n_dr <- length(draws)
inc_arr  <- array(NA_real_, dim = c(n_loc, n_yr, n_dr))
mort_arr <- array(NA_real_, dim = c(n_loc, n_yr, n_dr))
loc_idx  <- setNames(seq_along(kept_ids), as.character(kept_ids))
yr_idx   <- setNames(seq_along(forecast_years), as.character(forecast_years))
for (di in seq_along(per_draw)) {
  d <- per_draw[[di]]
  li <- loc_idx[as.character(d$location_id)]; yi <- yr_idx[as.character(d$year_id)]
  inc_arr[cbind(li, yi, di)]  <- d$log_malaria_inc_rate_pred
  mort_arr[cbind(li, yi, di)] <- d$log_malaria_mort_rate_pred
}

# ======================== 9. write outputs ========================
out_node <- glue("{REPO_DIR}/04-forecasting_data/malaria/forecast_outputs/{LSAE_HIERARCHY}")  # _A04_MAL_FORECAST_OUTPUTS
out_dir  <- file.path(out_node, out_run_date)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_nc      <- file.path(out_dir, glue("malaria_forecast_{ssp_scenario}_{dah_scenario}.nc"))
out_sidecar <- file.path(out_dir, glue("malaria_forecast_{ssp_scenario}_{dah_scenario}_location_status.parquet"))

write_forecast_netcdf <- function(inc_arr, mort_arr, locs, years, draws, path) {
  d_loc  <- ncdim_def("location_id", "id",   vals = as.integer(locs))
  d_year <- ncdim_def("year_id",     "year", vals = as.integer(years))
  d_draw <- ncdim_def("draw",        "draw", vals = as.integer(draws))
  cz <- list(d_loc, d_year, d_draw)                      # <<CHECK dim order vs xarray read — see guide
  v_inc  <- ncvar_def("log_malaria_inc_rate_pred",  "unitless", cz, missval = NA,
                      prec = "float", compression = 4, shuffle = TRUE)
  v_mort <- ncvar_def("log_malaria_mort_rate_pred", "unitless", cz, missval = NA,
                      prec = "float", compression = 4, shuffle = TRUE)
  tmp <- paste0(path, ".tmp")
  nc  <- nc_create(tmp, list(v_inc, v_mort), force_v4 = TRUE)
  ncvar_put(nc, v_inc,  inc_arr)                          # array dim order must match cz
  ncvar_put(nc, v_mort, mort_arr)
  nc_close(nc)
  # metadata-only validation (NO ncvar_get of the data — would double memory)
  chk <- nc_open(tmp)
  stopifnot(identical(chk$var[["log_malaria_inc_rate_pred"]]$varsize, as.integer(dim(inc_arr))))
  nc_close(chk)
  file.rename(tmp, path)
  Sys.chmod(path, "0775")
}

write_forecast_netcdf(inc_arr, mort_arr, kept_ids, forecast_years, draws, out_nc)
arrow::write_parquet(status, out_sidecar)
Sys.chmod(out_sidecar, "0775")
message(glue("Wrote {out_nc}\n      {out_sidecar}"))
# NO finalize here — the launcher's afterok job repoints `current` once after the array.
message("fin")
