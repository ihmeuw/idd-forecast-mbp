#!/usr/bin/env Rscript
# Demo / smoke test: read a malaria forecast-input netCDF in R, handling the
# mixed dimensionalities and the string `dah_scenario` coordinate.
#
#   malaria_suitability          : (location_id, year_id, draw)   float
#   gdppc_mean / people_flood_*  : (location_id, year_id)         float
#   mal_DAH_total_per_capita     : (location_id, year_id, dah_scenario) float
#   A0_location_id               : (location_id,)                 int
#   dah_scenario                 : NC_STRING coord ('Baseline','Constant')
#
# Tests two read paths:
#   PART 1 — ncdf4: per-variable ncvar_get + string-coord indexing
#   PART 2 — tidync: grid-based hyperslabbing
#
# Run anywhere with ncdf4 + tidync (e.g. the rstudio singularity image, or a
# plain Rscript that has the packages).

suppressPackageStartupMessages({
  library(ncdf4)
  library(tidync)
  library(data.table)
})

f <- "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/malaria/forecast_inputs/lsae_1285/20260527/malaria_forecast_inputs_ssp126.nc"

# ============================================================================
# PART 1 — ncdf4
# ============================================================================
cat("######## PART 1: ncdf4 ########\n\n")
nc <- nc_open(f)

# ncdf4 reports each var's dims in nc$var[[v]]$dim; ncvar_get() returns the
# array in that same order. We never hardcode the order — we read it and match
# dimnames by NAME, so the code is correct whichever order ncdf4 uses.
dim_names <- function(v) vapply(nc$var[[v]]$dim, function(d) d$name, character(1))

cat("-- variable dim order as ncdf4 reports it --\n")
for (v in names(nc$var)) {
  cat(sprintf("  %-32s [%s]\n", v, paste(dim_names(v), collapse = ", ")))
}

# Coordinate vectors (all small). dah_scenario is the NC_STRING coord.
loc      <- as.integer(ncvar_get(nc, "location_id"))
yr       <- as.integer(ncvar_get(nc, "year_id"))
drw      <- as.integer(ncvar_get(nc, "draw"))
dah_scen <- tryCatch(ncvar_get(nc, "dah_scenario"),
                     error = function(e) paste("ERROR reading NC_STRING coord:",
                                               conditionMessage(e)))
cat("\n-- dah_scenario via ncvar_get --\n")
cat("   class =", class(dah_scen), " | values =",
    paste(sQuote(dah_scen), collapse = ", "), "\n")

coord_vals <- list(location_id = loc, year_id = yr, draw = drw,
                   dah_scenario = dah_scen)

# Read the first n_loc locations of variable v as a tidy data.table.
# Hyperslab via start/count (contiguous), then label dims BY NAME and melt.
read_var_slice <- function(v, n_loc = 3L) {
  dn    <- dim_names(v)
  start <- rep(1L, length(dn))
  count <- rep(-1L, length(dn))                 # -1 = full extent
  count[match("location_id", dn)] <- n_loc
  arr <- ncvar_get(nc, v, start = start, count = count, collapse_degen = FALSE)

  labs <- coord_vals[dn]
  labs$location_id <- head(coord_vals$location_id, n_loc)
  stopifnot(identical(as.integer(dim(arr)), as.integer(lengths(labs))))  # order sanity
  dimnames(arr) <- labs

  dt <- as.data.table(as.data.frame.table(arr, responseName = v,
                                          stringsAsFactors = FALSE))
  # coordinate columns come back as character from dimnames; restore numerics
  for (cc in intersect(c("location_id", "year_id", "draw"), names(dt))) {
    dt[[cc]] <- as.integer(dt[[cc]])
  }
  dt[]
}

cat("\n-- mal_DAH_total_per_capita (loc x year x dah_scenario), 3 locs, year 2050 --\n")
dah <- read_var_slice("mal_DAH_total_per_capita", 3L)
print(dah[year_id == 2050])
cat("   dah_scenario column class:", class(dah$dah_scenario),
    "| distinct:", paste(sQuote(unique(dah$dah_scenario)), collapse = ", "), "\n")

cat("\n-- malaria_suitability (loc x year x draw): draw-mean for loc 1, a few years --\n")
suit <- read_var_slice("malaria_suitability", 2L)
print(suit[location_id == loc[1] & year_id %in% c(2000, 2050, 2100),
           .(suit_draw_mean = mean(malaria_suitability),
             suit_draw_sd   = sd(malaria_suitability)), by = year_id])

cat("\n-- gdppc_mean (loc x year): loc 1, a few years --\n")
gdp <- read_var_slice("gdppc_mean", 2L)
print(gdp[location_id == loc[1] & year_id %in% c(2000, 2050, 2100)])

cat("\n-- A0_location_id (loc-only): first 5 locations --\n")
a0 <- as.integer(ncvar_get(nc, "A0_location_id"))
print(data.table(location_id = head(loc, 5), A0_location_id = head(a0, 5)))

nc_close(nc)

# ============================================================================
# PART 2 — tidync
# ============================================================================
cat("\n\n######## PART 2: tidync ########\n\n")
source("/ihme/homes/bcreiner/repos/idd-forecast-mbp/src/idd_forecast_mbp/lib/netcdf_helpers.R")

src <- tidync(f)
cat("-- tidync grids --\n")
print(src)

g_dah <- grid_of(src, "mal_DAH_total_per_capita")
cat("\n-- DAH grid id from grid_of():", g_dah, "--\n")

da <- activate(src, g_dah)
da <- hyper_filter(da, location_id = location_id %in% loc[1:2])
dah_tib <- as.data.table(hyper_tibble(da))
cat("-- tidync hyper_tibble of DAH grid (2 locs) — does it carry the string coord? --\n")
print(head(dah_tib))
cat("   dah_scenario column class:", class(dah_tib$dah_scenario),
    "| distinct:", paste(sQuote(unique(dah_tib$dah_scenario)), collapse = ", "), "\n")

cat("\nDONE.\n")
