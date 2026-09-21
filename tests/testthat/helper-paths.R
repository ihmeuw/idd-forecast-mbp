# Shared fixtures for the R tests.
repo_root <- normalizePath(testthat::test_path("..", ".."))
lib_file <- function(name) file.path(repo_root, "src", "idd_forecast_mbp", "lib", name)

# Temp files stay inside the repo (never /tmp): .r_test_tmp/ is gitignored.
r_test_tmp <- function() {
  d <- file.path(repo_root, ".r_test_tmp")
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  d
}
tmp_parquet <- function(frame, name) {
  path <- file.path(r_test_tmp(), name)
  arrow::write_parquet(frame, path)
  path
}

# A tiny past-inputs frame with every column the preparation touches.
synthetic_past_inputs <- function() {
  data.frame(
    location_id = 1:6, year_id = c(2000L, 2000L, 2001L, 2001L, 2002L, 2002L),
    population = c(1000, 2000, 500, 4000, 100, 3000),
    malaria_pfpr = c(0.2, 0.00005, 0.1, 0.3, 0.05, 0.4),
    malaria_inc_rate = c(0.01, 0.01, 0.0005, 0.02, 0.005, 0.03),
    malaria_mort_rate = c(1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4),
    logit_malaria_pfpr = c(-1.386, -9.9, -2.197, -0.847, -2.944, -0.405),
    A0_location_id = c(10L, 10L, 20L, 20L, 30L, 30L),
    mal_DAH_total_per_capita = c(1.5, 0, 2.5, -1, 3.5, 4.5),
    gdppc_mean = c(1000, 2000, 3000, 4000, 5000, 6000),
    ldipc_mean = c(800, NA, 2400, 3200, 4000, 4800),
    med_consumppc = c(50, 60, NA, 80, 90, 100),
    days_over_30C = c(0, 100, 200, 365, 400, 50),
    relative_humidity = c(0, 50, 75, 100, 120, 30),
    malaria_suitability_mordecai_0_0 = c(0, 100, 200, 365, 400, 50),
    malaria_suitability_villena_0_0 = c(10, 20, 30, 40, 50, 60)
  )
}

# A larger synthetic past-inputs frame: 3 countries x n_years, enough rows for a temporal
# split and a k-fold CV with the lm and gam engines.
synthetic_past_inputs_large <- function(n_years = 24L, locs_per_country = 4L) {
  set.seed(3)
  years <- 2000L + seq_len(n_years) - 1L
  a0 <- rep(c(10L, 20L, 30L), each = locs_per_country)
  grid <- expand.grid(location_id = seq_along(a0), year_id = years)
  grid$A0_location_id <- a0[grid$location_id]
  n <- nrow(grid)
  gdppc <- exp(7 + 0.03 * (grid$year_id - 2000L) + 0.2 * grid$location_id)
  suit <- 100 + 50 * sin(grid$location_id) + 2 * (grid$year_id - 2000L)
  logit_pfpr <- -3 + 0.4 * log(suit / 365 / (1 - suit / 365)) - 0.3 * (log(gdppc) - 8) +
    c(0, 0.5, -0.5)[match(grid$A0_location_id, c(10L, 20L, 30L))] + rnorm(n, sd = 0.15)
  pfpr <- plogis(logit_pfpr)
  data.frame(
    location_id = as.integer(grid$location_id), year_id = as.integer(grid$year_id),
    population = 1e5 + 1e3 * grid$location_id, malaria_pfpr = pfpr,
    malaria_inc_rate = pfpr * 0.3, malaria_mort_rate = pfpr * 1e-3, logit_malaria_pfpr = logit_pfpr,
    A0_location_id = grid$A0_location_id, mal_DAH_total_per_capita = runif(n, 0.5, 5),
    gdppc_mean = gdppc, ldipc_mean = gdppc * 0.8, med_consumppc = gdppc * 0.05,
    days_over_30C = runif(n, 0, 200), relative_humidity = runif(n, 30, 90),
    malaria_suitability_mordecai_0_0 = suit, malaria_suitability_villena_0_0 = suit * 0.9
  )
}
