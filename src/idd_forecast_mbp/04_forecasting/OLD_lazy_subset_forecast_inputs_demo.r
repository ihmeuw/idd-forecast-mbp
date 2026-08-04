#!/usr/bin/env Rscript
# Lazy-load + subset demo:
#   - SSP  = which file we open
#   - dah_scenario = subset one scenario
#   - then manipulate (join mixed-dim vars, derive transforms)
#
# tidync is lazy: tidync() only indexes; hyper_filter() defines the slab;
# nothing is read until hyper_tibble(). We filter on location_id lazily, then
# pick the dah_scenario after realizing (string-axis filtering in tidync is the
# one behavior I haven't confirmed, so we do it in the data.table — guaranteed).

suppressPackageStartupMessages({ library(tidync); library(data.table) })
source("/ihme/homes/bcreiner/repos/idd-forecast-mbp/src/idd_forecast_mbp/lib/netcdf_helpers.R")

# ---- choices ----
SSP        <- "ssp126"
DAH_SCEN   <- "Baseline"
N_LOC      <- 5L          # how many locations to pull (demo)

base <- "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/malaria/forecast_inputs/lsae_1285/20260527"
f    <- file.path(base, sprintf("malaria_forecast_inputs_%s.nc", SSP))

# ---- 1. LAZY open (no data read yet) ----
src <- tidync(f)
locs <- as.integer(src$transforms$location_id$location_id)   # coord values, still no var read
target_locs <- head(locs, N_LOC)
cat(sprintf("Opened %s lazily. Subsetting to %d locs, dah_scenario=%s\n\n",
            basename(f), N_LOC, DAH_SCEN))

# small helper: lazily filter a grid to target_locs, then realize as data.table
pull <- function(exemplar_var) {
  g  <- grid_of(src, exemplar_var)
  dt <- as.data.table(hyper_tibble(
    hyper_filter(activate(src, g), location_id = location_id %in% target_locs)
  ))
  for (cc in intersect(c("location_id","year_id","draw"), names(dt)))
    dt[[cc]] <- as.integer(dt[[cc]])
  dt[]
}

# ---- 2. realize the slabs (this is where reads happen) ----
dah  <- pull("mal_DAH_total_per_capita")[dah_scenario == DAH_SCEN]   # subset scenario
gdp  <- pull("gdppc_mean")
suit <- pull("malaria_suitability")[, .(suit_draw_mean = mean(malaria_suitability)),
                                    by = .(location_id, year_id)]     # collapse draws

# ---- 3. manipulate: join mixed-dim vars into one (loc x year) table, derive ----
out <- Reduce(function(a, b) merge(a, b, by = c("location_id","year_id")),
              list(dah[, .(location_id, year_id, mal_DAH_total_per_capita)], gdp, suit))
out[, `:=`(log_gdppc          = log(gdppc_mean),
           logit_suit_frac    = qlogis(pmin(pmax(suit_draw_mean/365, 1e-3), 1-1e-3)))]

cat(sprintf("Result: %d rows (%d locs x %d years), dah_scenario=%s\n",
            nrow(out), uniqueN(out$location_id), uniqueN(out$year_id), DAH_SCEN))
print(out[location_id == target_locs[1] & year_id %in% c(2000, 2050, 2100)])
cat("\nDONE.\n")
