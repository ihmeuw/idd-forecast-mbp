#!/usr/bin/env Rscript
# ============================================================================
# forecast_malaria_admin_2s_rocket.r
#
# One task = one param_map row = one (ssp_scenario, dah_scenario). Loops the
# draws internally (mclapply). Reads 08 inputs lazily (tidync) + rake-year
# observed (raked AA parquet), predicts pfpr -> inc/mort with a per-location
# shift to the rake year, applies the zero-burden policy, and writes ONE
# compressed netCDF + a small location-status sidecar parquet.
#
# Design:      .claude/FORECAST_04_REWRITE_GUIDE.md
# Map + tests: .claude/FORECAST_04_DRAFT_NOTES.md
#
# Structure: all logic lives in small, argument-based helpers (no module-level
# globals) so a scratch harness can `source()` this file and exercise each
# function at tiny scale. main() is only run when the file is the Rscript
# entrypoint (the --file guard at the bottom == R's `if __name__ == "__main__"`).
# ============================================================================

suppressPackageStartupMessages({
  library(glue); library(data.table); library(tidync); library(ncdf4)
  library(mgcv); library(scam); library(arrow); library(parallel)
  library(optparse)
})

REPO_DIR <- "/mnt/team/idd/pub/forecast-mbp"          # == constants.MODEL_ROOT (verified)
SRC_REPO <- glue("/ihme/homes/{Sys.getenv('USER')}/repos/idd-forecast-mbp")
source(glue("{SRC_REPO}/src/idd_forecast_mbp/lib/model_registry.R"))   # get_malaria_model_run_date()
source(glue("{SRC_REPO}/src/idd_forecast_mbp/lib/netcdf_helpers.R"))   # grid_of()

Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1")  # tame BLAS under mclapply
LSAE_HIERARCHY <- "lsae_1285"                                  # == constants.LSAE_HIERARCHY (verified)

# ---------------------------------------------------------------------------
# Canonical paths (the R side has no path config; these mirror constants.py and
# are reconciled against it — see .claude/FORECAST_04_DRAFT_NOTES.md §3).
# ---------------------------------------------------------------------------
forecast_inputs_nc  <- function(ssp) file.path(
  REPO_DIR, "04-forecasting_data", "malaria", "forecast_inputs", LSAE_HIERARCHY, "current",
  glue("malaria_forecast_inputs_{ssp}.nc"))
raked_aa_parquet    <- file.path(
  REPO_DIR, "02-processed_data", "malaria", "raked_aa", LSAE_HIERARCHY, "current",
  "aa_full_malaria_df.parquet")
forecast_output_node <- file.path(
  REPO_DIR, "04-forecasting_data", "malaria", "forecast_outputs", LSAE_HIERARCHY)   # _A04_MAL_FORECAST_OUTPUTS
modeling_data_path  <- file.path(REPO_DIR, "03-modeling_data")

# ============================================================================
# READER (verified tidync pattern from the demos)
# ============================================================================

#' Lazily filter a tidync grid to `locs` (+ optional `years`) and realize it.
#' Mirrors lazy_subset_forecast_inputs_demo.r: index by variable->grid (grid_of),
#' hyper_filter the coords, realize with hyper_tibble. String axes (dah_scenario)
#' are NOT filtered here — select them after realizing (untested in tidync).
pull_grid <- function(src, exemplar_var, locs, years = NULL) {
  af <- activate(src, grid_of(src, exemplar_var))
  af <- hyper_filter(af, location_id = location_id %in% locs)
  if (!is.null(years) && "year_id" %in% names(src$transforms)) {
    af <- hyper_filter(af, year_id = year_id %in% years)
  }
  dt <- as.data.table(hyper_tibble(af))
  for (cc in intersect(c("location_id", "year_id", "draw"), names(dt))) {
    dt[[cc]] <- as.integer(dt[[cc]])
  }
  dt[]
}

#' Hold `value_cols` at their `hold_year` value for every later year.
#'
#' The counterfactual all the "hold X constant at 2023" sensitivities are built from.
#' Grouping matters: a loc x year covariate is keyed on location_id, but a draw-varying
#' one (suitability) must be keyed on (location_id, draw) so each draw keeps its OWN
#' hold-year value and the draw spread survives the freeze.
freeze_at_year <- function(dt, value_cols, hold_year, by_cols = "location_id") {
  value_cols <- intersect(value_cols, names(dt))
  if (!length(value_cols)) return(dt[])
  by_cols <- intersect(by_cols, names(dt))
  ref <- unique(dt[year_id == hold_year, c(by_cols, value_cols), with = FALSE], by = by_cols)
  if (!nrow(ref)) {
    stop(glue("freeze_at_year: no rows at hold year {hold_year} for ",
              "{paste(value_cols, collapse=', ')}"))
  }
  tmp <- paste0(".ref_", value_cols)
  setnames(ref, value_cols, tmp)
  out <- merge(dt, ref, by = by_cols, all.x = TRUE)
  for (i in seq_along(value_cols)) {
    out[year_id > hold_year, (value_cols[i]) := get(tmp[i])]
    out[, (tmp[i]) := NULL]
  }
  # A frozen column must be year-invariant after the hold year, by construction.
  chk <- out[year_id >= hold_year,
             lapply(.SD, function(x) length(unique(x))), .SDcols = value_cols,
             by = by_cols]
  stopifnot("freeze_at_year did not produce a year-invariant column" =
              all(unlist(chk[, value_cols, with = FALSE]) == 1L))
  out[]
}

# Which input table each hold token freezes. Adding a hold is a row here, not a
# new branch in the compute body.
HOLD_TARGETS <- list(
  gdppc       = list(table = "gdp_dt",  cols = "gdppc_mean",                    by = "location_id"),
  suitability = list(table = "suit_dt", cols = "malaria_suitability",           by = c("location_id", "draw")),
  temp        = list(table = "temp_dt", cols = "mean_low_temperature",          by = "location_id"),
  flood       = list(table = "fld_dt",  cols = "people_flood_days_per_capita",  by = "location_id"),
  dah         = list(table = "dah_dt",  cols = "mal_DAH_total_per_capita",      by = "location_id")
)

#' Apply every requested hold to the input list, in place of the raw trajectories.
apply_covariate_holds <- function(inputs, holds, hold_year) {
  holds <- holds[nzchar(holds)]
  if (!length(holds)) return(inputs)
  unknown <- setdiff(holds, names(HOLD_TARGETS))
  if (length(unknown)) {
    stop(glue("unknown --hold-covariate value(s): {paste(unknown, collapse=', ')}; ",
              "valid: {paste(names(HOLD_TARGETS), collapse=', ')}"))
  }
  for (h in holds) {
    spec <- HOLD_TARGETS[[h]]
    message(glue("  holding {h} ({spec$cols}) at {hold_year}"))
    inputs[[spec$table]] <- freeze_at_year(
      inputs[[spec$table]], spec$cols, hold_year, by_cols = spec$by)
  }
  inputs
}

#' Read every 08 input variable for the requested locs/years/dah scenario.
#' Returns a named list of data.tables + the draw vector + the location universe.
read_forecast_inputs <- function(nc_file, read_years, dah_scenario_sel) {
  src      <- tidync(nc_file)
  all_locs <- as.integer(src$transforms$location_id$location_id)  # coords only, no read
  a0_dt    <- pull_grid(src, "A0_location_id", all_locs)                          # loc-only
  gdp_dt   <- pull_grid(src, "gdppc_mean", all_locs, read_years)                  # loc x year
  fld_dt   <- pull_grid(src, "people_flood_days_per_capita", all_locs, read_years) # loc x year
  temp_dt  <- pull_grid(src, "mean_low_temperature", all_locs, read_years)         # loc x year (climate_mean; nc must contain it)
  dah_dt   <- pull_grid(src, "mal_DAH_total_per_capita", all_locs, read_years)    # loc x year x dah
  # select scenario post-realize; param is named *_sel so it doesn't collide with the column
  dah_dt   <- dah_dt[dah_scenario == dah_scenario_sel]
  suit_dt  <- pull_grid(src, "malaria_suitability", all_locs, read_years)         # loc x year x draw
  list(all_locs = all_locs, a0_dt = a0_dt, gdp_dt = gdp_dt, fld_dt = fld_dt,
       temp_dt = temp_dt, dah_dt = dah_dt, suit_dt = suit_dt, draws = sort(unique(suit_dt$draw)))
}

# ============================================================================
# RAKE-YEAR OBSERVED + ANCHORS
# ============================================================================

#' Rake year per location: a 4-digit scalar -> constant vector, or a parquet
#' with columns (location_id, rake_year) -> per-location vector.
resolve_rake_years <- function(spec, location_ids) {
  if (grepl("^[0-9]{4}$", spec)) {
    setNames(rep(as.integer(spec), length(location_ids)), as.character(location_ids))
  } else {
    rk <- as.data.table(arrow::read_parquet(spec))   # schema: location_id, rake_year
    setNames(as.integer(rk$rake_year[match(location_ids, rk$location_id)]),
             as.character(location_ids))
  }
}

#' Read the raked all-age observed values and build the rake-year anchors.
#' 08's locations ARE 07b's most-detailed prediction set, so filtering by them
#' is equivalent to the level-5 filter (no hierarchy read needed). One row per
#' location, taken at THAT location's own rake year.
read_rake_year_observed <- function(aa_file, all_locs, rake_years_vec) {
  read_years <- sort(unique(as.integer(rake_years_vec)))
  aa <- as.data.table(arrow::read_parquet(aa_file))
  aa <- aa[location_id %in% all_locs & year_id %in% read_years,
           .(location_id, year_id, malaria_pfpr, malaria_inc_rate, malaria_mort_rate)]
  obs <- aa[data.table(location_id = all_locs,
                       year_id     = unname(rake_years_vec[as.character(all_locs)])),
            on = .(location_id, year_id)]
  obs[, `:=`(
    # match the fit's pfpr transform exactly (05 used 0.999 * pfpr)
    obs_logit_pfpr = log(0.999 * malaria_pfpr / (1 - 0.999 * malaria_pfpr)),
    obs_log_inc    = fifelse(malaria_inc_rate  > 0, log(malaria_inc_rate),  NA_real_),
    obs_log_mort   = fifelse(malaria_mort_rate > 0, log(malaria_mort_rate), NA_real_)
  )]
  obs[]
}

# ============================================================================
# ZERO-BURDEN POLICY  (this run: "drop" / option A)
# ============================================================================
classify_zero_burden <- function(obs, policy) {
  if (policy != "drop") {
    stop("zero_burden_policy='impute' (option B) not implemented.")  # <<TODO option B
  }
  s <- obs[, .(location_id, pfpr = malaria_pfpr, inc = malaria_inc_rate, mort = malaria_mort_rate)]
  s[, `:=`(status = "kept_both", drop_reason = NA_character_)]
  s[is.na(pfpr) | pfpr == 0,                     `:=`(status = "dropped", drop_reason = "pfpr_zero")]
  s[status != "dropped" & inc == 0 & mort == 0,  `:=`(status = "dropped", drop_reason = "inc_mort_both_zero")]
  s[status != "dropped" & inc == 0 & mort >  0,  status := "mort_only"]   # predict mort, inc = NaN
  s[status != "dropped" & mort == 0 & inc >  0,  status := "inc_only"]    # predict inc,  mort = NaN
  s[]
}

# ============================================================================
# ORPHAN-A0 LOWEST-COEF FALLBACK (per model)
# ============================================================================

#' The A0_af level with the lowest total fitted effect (reference level = 0).
#' Used as the fallback for locations whose country was not in the fit.
fallback_level <- function(mod) {
  b   <- coef(mod); a0 <- b[grepl("^A0_af", names(b))]
  lev <- mod$xlevels$A0_af
  eff <- setNames(rep(0, length(lev)), lev)
  eff[sub("^A0_af", "", names(a0))] <- a0
  names(eff)[which.min(eff)]
}

#' Build the A0_af factor for a model: the real level if the country was in the
#' fit, else `fb_level` (orphan). Levels are pinned to model$xlevels$A0_af so
#' predict() sees only levels it knows.
make_A0_af <- function(location_ids, mod, fb_level, a0_dt) {
  a0_id <- a0_dt$A0_location_id[match(location_ids, a0_dt$location_id)]
  af    <- factor(as.character(a0_id), levels = mod$xlevels$A0_af)
  af[is.na(af)] <- fb_level    # fb_level is an existing level, so this lands correctly
  af
}

# ============================================================================
# RAKE SHIFT (order-safe; no non-equi join)
# ============================================================================

#' Shift a raw prediction column so that, at each location's rake year, the
#' shifted value equals the observed anchor. The shift is constant across years
#' within a location. Returns a vector aligned to `dt`'s current row order
#' (aligned by location_id via match() — never by position).
apply_shift <- function(dt, raw_col, obs_dt, obs_col, rake_years_vec) {
  ry      <- unname(rake_years_vec[as.character(dt$location_id)])  # each row's loc rake year
  at_rake <- dt[dt$year_id == ry, .(location_id, raw_at_rake = get(raw_col))]  # one row/loc
  sh      <- merge(at_rake, obs_dt, by = "location_id", all.x = TRUE)
  sh[, shift := get(obs_col) - raw_at_rake]
  dt[[raw_col]] + sh$shift[match(dt$location_id, sh$location_id)]
}

# ============================================================================
# PER-DRAW PREDICTION  (factory -> function(d) for mclapply)
# ============================================================================

#' Build the per-draw predictor closure. Closes over the (read-only) inputs so
#' mclapply can call it as function(d). pfpr is predicted, shifted to the rake
#' year, and the shifted logit pfpr feeds inc/mort. Prediction runs over
#' read_years (so the rake year is always present for the shift), then the
#' output is trimmed to forecast_years. Masked outcomes are set to NA.
make_predict_one_draw <- function(inputs, obs, models, fb,
                                   kept_ids, inc_ids, mort_ids,
                                   rake_years_vec, forecast_years,
                                   run_inc = TRUE, run_mort = TRUE) {
  a0_dt   <- inputs$a0_dt
  base_xy <- Reduce(function(a, b) merge(a, b, by = c("location_id", "year_id")),
                    list(inputs$gdp_dt[, .(location_id, year_id, gdppc_mean)],
                         inputs$fld_dt[, .(location_id, year_id, people_flood_days_per_capita)],
                         inputs$temp_dt[, .(location_id, year_id, mean_low_temperature)],
                         inputs$dah_dt[, .(location_id, year_id, mal_DAH_total_per_capita)]))
  base_xy <- base_xy[location_id %in% kept_ids]
  suit_dt <- inputs$suit_dt
  # key on draw ONCE here in the parent so each fork's suit_dt[.(d)] is a binary-search
  # subset (inherited sorted via copy-on-write) instead of a ~210M-row scan per draw.
  setkey(suit_dt, draw)

  function(d) {
    # keyed (binary-search) subset on draw — suit_dt is setkey'd on draw in the parent
    df <- merge(suit_dt[.(d), .(location_id, year_id, malaria_suitability), nomatch = NULL],
                base_xy, by = c("location_id", "year_id"))
    df <- df[location_id %in% kept_ids]    # all read_years; trimmed to forecast_years after shifting
    df[, `:=`(
      malaria_suit              = malaria_suitability,  # raw suitability; for formulations using s(malaria_suit,...). MUST match the fit's malaria_suit (raw variant value).
      logit_malaria_suitability = qlogis(pmin(pmax(malaria_suitability / 365, 1e-3), 1 - 1e-3)),
      log_gdppc_mean            = log(gdppc_mean)     # inc/mort term; pfpr uses raw gdppc_mean
    )]

    # pfpr: predict (raw), shift to observed logit pfpr, use shifted as the inc/mort predictor
    df[, A0_af := make_A0_af(location_id, models$pfpr, fb$pfpr, a0_dt)]
    df[, pfpr_raw := as.numeric(predict(models$pfpr, newdata = df))]
    df[, logit_malaria_pfpr := apply_shift(df, "pfpr_raw",
          obs[, .(location_id, obs_logit_pfpr)], "obs_logit_pfpr", rake_years_vec)]

    # incidence (only when requested; pfpr above always runs as the shared predictor).
    # The unrequested outcome's column is created as NA so the return schema and
    # assemble_arrays stay identical regardless of --outcomes.
    if (run_inc) {
      df[, A0_af := make_A0_af(location_id, models$inc, fb$inc, a0_dt)]
      df[, inc_raw := as.numeric(predict(models$inc, newdata = df))]
      df[, log_malaria_inc_rate_pred := apply_shift(df, "inc_raw",
            obs[, .(location_id, obs_log_inc)], "obs_log_inc", rake_years_vec)]
      # outcome mask: NaN the outcome a location is not supposed to receive
      df[!(location_id %in% inc_ids), log_malaria_inc_rate_pred := NA_real_]
    } else {
      df[, log_malaria_inc_rate_pred := NA_real_]
    }

    # mortality (only when requested)
    if (run_mort) {
      df[, A0_af := make_A0_af(location_id, models$mort, fb$mort, a0_dt)]
      df[, mort_raw := as.numeric(predict(models$mort, newdata = df))]
      df[, log_malaria_mort_rate_pred := apply_shift(df, "mort_raw",
            obs[, .(location_id, obs_log_mort)], "obs_log_mort", rake_years_vec)]
      df[!(location_id %in% mort_ids), log_malaria_mort_rate_pred := NA_real_]
    } else {
      df[, log_malaria_mort_rate_pred := NA_real_]
    }

    df[year_id %in% forecast_years][order(location_id, year_id),
       .(location_id, year_id, log_malaria_inc_rate_pred, log_malaria_mort_rate_pred)]
  }
}

# ============================================================================
# ASSEMBLE (kept_loc x year x draw)
# ============================================================================
assemble_arrays <- function(per_draw, kept_ids, forecast_years, draws) {
  n_loc <- length(kept_ids); n_yr <- length(forecast_years); n_dr <- length(draws)
  inc_arr  <- array(NA_real_, dim = c(n_loc, n_yr, n_dr))
  mort_arr <- array(NA_real_, dim = c(n_loc, n_yr, n_dr))
  loc_idx  <- setNames(seq_along(kept_ids), as.character(kept_ids))
  yr_idx   <- setNames(seq_along(forecast_years), as.character(forecast_years))
  for (di in seq_along(per_draw)) {
    d  <- per_draw[[di]]
    li <- loc_idx[as.character(d$location_id)]; yi <- yr_idx[as.character(d$year_id)]
    inc_arr[cbind(li, yi, di)]  <- d$log_malaria_inc_rate_pred
    mort_arr[cbind(li, yi, di)] <- d$log_malaria_mort_rate_pred
  }
  list(inc = inc_arr, mort = mort_arr)
}

# ============================================================================
# WRITE netCDF  (dim order chosen so an xarray reader sees (location_id, year_id, draw))
# ============================================================================

#' ncdf4 lists the FASTEST-varying dim first and maps it to netCDF's LAST (C-order)
#' dim; xarray reads C-order. So to make xarray report (location_id, year_id, draw)
#' we define ncdf4 dims REVERSED (draw, year_id, location_id) and pass the array as
#' [draw, year, loc]. (Verified by test point 6 against xarray — the demos showed
#' ncdf4 *reads* these files transposed, which is the same convention.)
write_forecast_netcdf <- function(inc_arr, mort_arr, locs, years, draws, path,
                                  run_inc = TRUE, run_mort = TRUE) {
  d_loc  <- ncdim_def("location_id", "id",   vals = as.integer(locs))
  d_year <- ncdim_def("year_id",     "year", vals = as.integer(years))
  d_draw <- ncdim_def("draw",        "draw", vals = as.integer(draws))
  cz <- list(d_draw, d_year, d_loc)                      # reversed -> xarray sees (loc, year, draw)

  # Only the requested outcomes become netCDF variables. assembled arrays are
  # [loc, year, draw]; reorder to match the reversed dims. shuffle is a no-op for float (ncdf4).
  vars <- list(); arrs <- list()
  if (run_inc) {
    vars[["log_malaria_inc_rate_pred"]] <- ncvar_def("log_malaria_inc_rate_pred", "unitless",
      cz, missval = NA, prec = "float", compression = 4, shuffle = FALSE)
    arrs[["log_malaria_inc_rate_pred"]] <- aperm(inc_arr, c(3, 2, 1))
  }
  if (run_mort) {
    vars[["log_malaria_mort_rate_pred"]] <- ncvar_def("log_malaria_mort_rate_pred", "unitless",
      cz, missval = NA, prec = "float", compression = 4, shuffle = FALSE)
    arrs[["log_malaria_mort_rate_pred"]] <- aperm(mort_arr, c(3, 2, 1))
  }
  stopifnot("write_forecast_netcdf: at least one outcome must be requested" = length(vars) > 0L)

  tmp <- paste0(path, ".tmp")
  nc  <- nc_create(tmp, unname(vars), force_v4 = TRUE)
  for (nm in names(vars)) ncvar_put(nc, vars[[nm]], arrs[[nm]])
  nc_close(nc)

  # metadata-only validation on the first written var (NO ncvar_get -> would double memory)
  chk <- nc_open(tmp)
  on.exit(nc_close(chk), add = TRUE)
  vname <- names(vars)[1]
  got <- vapply(chk$var[[vname]]$dim, function(d) d$len, integer(1))
  names(got) <- vapply(chk$var[[vname]]$dim, function(d) d$name, character(1))
  want <- c(location_id = length(locs), year_id = length(years), draw = length(draws))
  stopifnot(identical(got[names(want)], want))
  nc_close(chk); on.exit()
  file.rename(tmp, path)
  Sys.chmod(path, "0775")
  invisible(path)
}

# ============================================================================
# MAIN (one task)
# ============================================================================
main <- function() {
  # One task = one (ssp, dah) cell of the jobmon workflow. Args come as CLI flags
  # from 01_forecast_malaria_admin_2s_orchestrator.py (NOT the old array/CSV path).
  opt <- parse_args(OptionParser(option_list = list(
    make_option("--model-run-date",     type = "character", default = NA,
                help = "formulation run date = model-registry key AND output dir name"),
    make_option("--ssp-scenario",       type = "character", default = NA),
    make_option("--dah-scenario",       type = "character", default = NA),
    make_option("--forecast-start",     type = "integer",   default = 2023L),
    make_option("--forecast-end",       type = "integer",   default = 2100L),
    make_option("--rake-year",          type = "character", default = "2023",
                help = "4-digit scalar OR path to a per-loc rake-year parquet"),
    make_option("--zero-burden-policy", type = "character", default = "drop"),
    make_option("--output-key",         type = "character", default = NA,
                help = paste("output dir name; defaults to --model-run-date. Lets ONE",
                             "fitted model write several runs (the sensitivities),",
                             "each in its own dir with unchanged file names.")),
    make_option("--hold-covariate",     type = "character", default = "",
                help = paste("comma-separated covariates to hold constant:",
                             "gdppc, suitability, temp, flood, dah. Empty = none.")),
    make_option("--hold-year",          type = "integer",   default = 2023L,
                help = "year at which --hold-covariate values are frozen"),
    make_option("--outcomes",           type = "character", default = "both",
                help = "which outcomes to predict+write: inc | mort | both (pfpr always runs)")
  )))
  names(opt) <- gsub("-", "_", names(opt))   # robust to optparse dash/underscore naming

  model_run_date     <- as.character(opt$model_run_date)
  ssp_scenario       <- as.character(opt$ssp_scenario)
  dah_scenario       <- as.character(opt$dah_scenario)
  stopifnot(
    "--model-run-date must be set"     = !is.na(model_run_date) && nzchar(model_run_date),
    "--ssp-scenario must be set"       = !is.na(ssp_scenario)   && nzchar(ssp_scenario),
    "--dah-scenario must be set"       = !is.na(dah_scenario)   && nzchar(dah_scenario))
  forecast_years     <- as.integer(opt$forecast_start):as.integer(opt$forecast_end)
  rake_year_spec     <- as.character(opt$rake_year)
  zero_burden_policy <- as.character(opt$zero_burden_policy)
  outcomes           <- tolower(as.character(opt$outcomes))
  stopifnot("--outcomes must be one of inc|mort|both" = outcomes %in% c("inc", "mort", "both"))
  # pfpr ALWAYS runs (it is the shifted predictor feeding inc/mort); it is never written.
  run_inc  <- outcomes %in% c("inc",  "both")
  run_mort <- outcomes %in% c("mort", "both")
  # model_run_date is the registry key. The output dir name defaults to it -- the
  # single identity the 2026-07-14 rewrite established -- but --output-key can name
  # the dir separately, which is what the covariate-hold sensitivities need: ONE
  # fitted model writing several runs. The alternative (a hold token welded into the
  # file names) is what the 2025 chain did, and it is why adding an axis there meant
  # editing a nested filename expression in every downstream script.
  hold_covariates    <- trimws(strsplit(as.character(opt$hold_covariate), ",")[[1]])
  hold_covariates    <- hold_covariates[nzchar(hold_covariates)]
  hold_year          <- as.integer(opt$hold_year)
  out_run_date       <- if (!is.na(opt$output_key) && nzchar(as.character(opt$output_key))) {
    as.character(opt$output_key)
  } else {
    model_run_date
  }
  stopifnot("--output-key must differ from --model-run-date when holds are set" =
              !length(hold_covariates) || out_run_date != model_run_date)
  message(glue("[{model_run_date}] ssp={ssp_scenario} dah={dah_scenario} ",
               "years={min(forecast_years)}-{max(forecast_years)} ",
               "policy={zero_burden_policy} outcomes={outcomes}",
               if (length(hold_covariates)) {
                 glue(" hold={paste(hold_covariates, collapse='+')}@{hold_year}")
               } else "",
               if (out_run_date != model_run_date) glue(" -> {out_run_date}") else ""))

  # resolve + load model
  mrd <- get_malaria_model_run_date(
    malaria_model_registry_path(modeling_data_path),
    best     = !nzchar(model_run_date),
    run_date = if (nzchar(model_run_date)) model_run_date else NULL)
  load(file.path(modeling_data_path, glue("{mrd}_malaria_models.RData")))  # -> pfpr_mod (+ inc_mod/mort_mod)
  # pfpr is always needed; inc_mod/mort_mod only for the requested outcomes -- so a
  # formulation missing an outcome model can still run the other via --outcomes.
  stopifnot("pfpr_mod missing from the model .RData" = exists("pfpr_mod"))
  if (run_inc)  stopifnot("inc_mod missing but --outcomes needs incidence"  = exists("inc_mod"))
  if (run_mort) stopifnot("mort_mod missing but --outcomes needs mortality" = exists("mort_mod"))
  models <- list(pfpr = pfpr_mod)
  if (run_inc)  models$inc  <- inc_mod
  if (run_mort) models$mort <- mort_mod

  # rake years + the year window we must read (cover the rake year)
  src_locs       <- as.integer(tidync(forecast_inputs_nc(ssp_scenario))$transforms$location_id$location_id)
  rake_years_vec <- resolve_rake_years(rake_year_spec, src_locs)
  read_years     <- sort(unique(c(forecast_years, as.integer(rake_years_vec))))

  # inputs + observed
  inputs <- read_forecast_inputs(forecast_inputs_nc(ssp_scenario), read_years, dah_scenario)
  # Holds are applied to the covariate trajectories BEFORE any transform or predict,
  # so a frozen gdppc_mean also freezes the log_gdppc_mean the inc/mort models use.
  inputs <- apply_covariate_holds(inputs, hold_covariates, hold_year)
  obs    <- read_rake_year_observed(raked_aa_parquet, inputs$all_locs, rake_years_vec)

  # zero-burden classification
  status   <- classify_zero_burden(obs, zero_burden_policy)
  kept_ids <- sort(status[status != "dropped", location_id])
  inc_ids  <- status[status %in% c("kept_both", "inc_only"),  location_id]
  mort_ids <- status[status %in% c("kept_both", "mort_only"), location_id]
  message(glue("  kept={length(kept_ids)} (dropped={status[status=='dropped', .N]}, ",
               "inc_only={status[status=='inc_only', .N]}, mort_only={status[status=='mort_only', .N]})"))

  # orphan-A0 fallback (per model; only for the models we loaded)
  fb <- list(pfpr = fallback_level(models$pfpr))
  if (run_inc)  fb$inc  <- fallback_level(models$inc)
  if (run_mort) fb$mort <- fallback_level(models$mort)

  # predict per draw
  predict_one_draw <- make_predict_one_draw(inputs, obs, models, fb,
                                            kept_ids, inc_ids, mort_ids,
                                            rake_years_vec, forecast_years,
                                            run_inc = run_inc, run_mort = run_mort)
  n_cores  <- max(1L, as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")))
  per_draw <- mclapply(inputs$draws, predict_one_draw, mc.cores = n_cores)
  failed   <- vapply(per_draw, function(x) inherits(x, "try-error"), logical(1))
  if (any(failed)) {
    stop(glue("{sum(failed)} draw(s) failed in mclapply: ",
              "{conditionMessage(attr(per_draw[[which(failed)[1]]], 'condition'))}"))
  }

  # assemble (kept_loc x year x draw)
  arrs <- assemble_arrays(per_draw, kept_ids, forecast_years, inputs$draws)

  # write
  out_dir <- file.path(forecast_output_node, out_run_date)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_nc      <- file.path(out_dir, glue("malaria_forecast_{ssp_scenario}_{dah_scenario}.nc"))
  out_sidecar <- file.path(out_dir, glue("malaria_forecast_{ssp_scenario}_{dah_scenario}_location_status.parquet"))
  write_forecast_netcdf(arrs$inc, arrs$mort, kept_ids, forecast_years, inputs$draws, out_nc,
                        run_inc = run_inc, run_mort = run_mort)
  # atomic sidecar write (tmp + rename) so a crash mid-write can't leave a partial
  # sidecar that the orchestrator's done() would count as present.
  sidecar_tmp <- paste0(out_sidecar, ".tmp")
  arrow::write_parquet(status, sidecar_tmp)
  file.rename(sidecar_tmp, out_sidecar)
  Sys.chmod(out_sidecar, "0775")
  message(glue("Wrote {out_nc}\n      {out_sidecar}"))
  # NO finalize here -- the launcher's afterok job repoints `current` once after the array.
  message("fin")
}

# --- entrypoint guard: run main() only when this file is the Rscript --file ---
# (R's analog of `if __name__ == "__main__"`; lets a test harness source() the
#  helpers above without executing the task.)
.is_rscript_entrypoint <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", ca[grepl("^--file=", ca)])
  length(fa) >= 1L && grepl("forecast_malaria_admin_2s_rocket", fa[1], fixed = TRUE)
}
if (.is_rscript_entrypoint()) main()
