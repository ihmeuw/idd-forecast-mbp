# Shared helpers for reading pipeline netCDF files in R via tidync.
#
# Usage:
#   source("src/idd_forecast_mbp/lib/netcdf_helpers.R")
#   src   <- tidync(nc_path)
#   ly_df <- as.data.frame(hyper_tibble(activate(src, grid_of(src, "malaria_pfpr"))))

#' Resolve the tidync grid identifier containing a given variable.
#'
#' tidync uses positional grid IDs (D0,D1 / D1,D0 / D2,D1,D0 ...) that depend
#' on dimension write order and break if the file changes. This function looks up
#' the ID by variable name, which is stable.
#'
#' src$grid$variables is a list-column of tibbles (each with a `variable` column),
#' not a plain character column — sapply over the list is required.
#'
#' @param src   tidync object returned by tidync()
#' @param varname  name of any variable on the target grid
#' @return grid identifier string (e.g. "D1,D0") suitable for activate()
grid_of <- function(src, varname) {
  hits <- sapply(src$grid$variables, function(t) varname %in% t$variable)
  src$grid$grid[hits][1]
}

#' Read the grid_map attribute from a netCDF file.
#'
#' Pipeline netCDFs embed a grid_map JSON attribute naming one canonical exemplar
#' variable per logical grid (e.g. list(loc_year = "malaria_pfpr")). Use this to
#' get the right exemplar to pass to grid_of() without hardcoding variable names.
#'
#' @param nc_path  path to the netCDF file
#' @return named list mapping logical grid name -> exemplar variable name,
#'         or NULL if the attribute is absent
read_grid_map <- function(nc_path) {
  nc  <- ncdf4::nc_open(nc_path)
  on.exit(ncdf4::nc_close(nc))
  raw <- ncdf4::ncatt_get(nc, 0, "grid_map")
  if (!raw$hasatt) return(NULL)
  jsonlite::fromJSON(raw$value)
}
