# The forecast rocket's draw-varying covariates beyond suitability (wind_speed) and the holds that
# freeze them. The rocket defines functions only when sourced: main() runs under the --file guard.
rocket_file <- file.path(repo_root, "src", "idd_forecast_mbp", "04_forecasting", "forecast_malaria_admin_2s_rocket.r")
source(rocket_file)

locs  <- 1:2
years <- 2022:2025
grid  <- data.table::CJ(location_id = locs, year_id = years)

# wind differs by draw after 2023 so a draw mix-up shows in the forecast
wind_by_draw <- function(w0, w1) {
  rbind(
    data.table::data.table(grid, draw = 0L, wind_speed = ifelse(grid$year_id <= 2023, 2, w0)),
    data.table::data.table(grid, draw = 1L, wind_speed = ifelse(grid$year_id <= 2023, 2, w1))
  )
}

synthetic_inputs <- function(wind_dt) {
  list(
    all_locs = locs,
    a0_dt    = data.table::data.table(location_id = locs, A0_location_id = c(10L, 20L)),
    gdp_dt   = data.table::data.table(grid, gdppc_mean = 2000),
    fld_dt   = data.table::data.table(grid, people_flood_days_per_capita = 0),
    temp_dt  = data.table::data.table(grid, mean_low_temperature = 20),
    dah_dt   = data.table::data.table(grid, mal_DAH_total_per_capita = 1),
    suit_dt  = rbind(data.table::data.table(grid, draw = 0L, malaria_suitability = 150),
                     data.table::data.table(grid, draw = 1L, malaria_suitability = 150)),
    wind_dt  = wind_dt,
    draws    = 0:1
  )
}

synthetic_models <- function() {
  set.seed(1)
  n  <- 60
  tr <- data.frame(
    logit_malaria_suitability = rnorm(n), wind_speed = runif(n, 1, 4),
    A0_af = factor(rep(c("10", "20"), length.out = n)), log_gdppc_mean = rnorm(n, 8)
  )
  tr$logit_malaria_pfpr    <- -1 + 0.5 * tr$logit_malaria_suitability - 0.4 * tr$wind_speed + rnorm(n, sd = 0.1)
  tr$log_malaria_inc_rate  <- -3 + 0.8 * tr$logit_malaria_pfpr - 0.1 * tr$log_gdppc_mean + rnorm(n, sd = 0.1)
  tr$log_malaria_mort_rate <- -7 + 0.9 * tr$logit_malaria_pfpr - 0.1 * tr$log_gdppc_mean + rnorm(n, sd = 0.1)
  list(
    pfpr = lm(logit_malaria_pfpr ~ logit_malaria_suitability + wind_speed + A0_af, data = tr),
    inc  = lm(log_malaria_inc_rate ~ logit_malaria_pfpr + log_gdppc_mean + A0_af, data = tr),
    mort = lm(log_malaria_mort_rate ~ logit_malaria_pfpr + log_gdppc_mean + A0_af, data = tr)
  )
}

run_draws <- function(wind_dt) {
  models <- synthetic_models()
  fb  <- lapply(models, fallback_level)
  obs <- data.table::data.table(location_id = locs, obs_logit_pfpr = c(-1, -2),
                                obs_log_inc = c(-3, -4), obs_log_mort = c(-7, -8))
  rake <- setNames(c(2023L, 2023L), as.character(locs))
  f <- make_predict_one_draw(synthetic_inputs(wind_dt), obs, models, fb, locs, locs, locs,
                             rake, 2023:2025)
  list(d0 = f(0L), d1 = f(1L))
}

test_that("model_draw_vars finds wind_speed only in a model that uses it", {
  m <- synthetic_models()
  expect_equal(model_draw_vars(m$pfpr), "wind_speed")
  expect_length(model_draw_vars(m$inc), 0L)
})

test_that("wind enters by draw number: swapping the wind draws swaps the forecasts", {
  a <- run_draws(wind_by_draw(w0 = 1, w1 = 4))
  b <- run_draws(wind_by_draw(w0 = 4, w1 = 1))
  expect_false(isTRUE(all.equal(a$d0, a$d1)))     # the two draws differ after 2023
  expect_equal(a$d0, b$d1)                         # draw 0 with wind 1 == draw 1 with wind 1
  expect_equal(a$d1, b$d0)
  at_rake <- a$d0[year_id == 2023]
  expect_equal(at_rake$log_malaria_inc_rate_pred, c(-3, -4))   # the 2023 anchor still holds
  expect_equal(at_rake$log_malaria_mort_rate_pred, c(-7, -8))
})

test_that("the wind hold freezes each (location, draw) at its own hold-year value", {
  w <- wind_by_draw(w0 = 1, w1 = 4)
  w[year_id == 2023, wind_speed := ifelse(draw == 0L, 2.5, 3.5)]
  held <- apply_covariate_holds(synthetic_inputs(w), "wind", 2023L)$wind_dt
  expect_true(all(held[year_id > 2023 & draw == 0L, wind_speed] == 2.5))
  expect_true(all(held[year_id > 2023 & draw == 1L, wind_speed] == 3.5))
  expect_equal(held[year_id <= 2023][order(location_id, year_id, draw), wind_speed],
               w[year_id <= 2023][order(location_id, year_id, draw), wind_speed])   # the past is untouched
})

test_that("a hold on a covariate the model's inputs lack is refused", {
  expect_error(apply_covariate_holds(synthetic_inputs(NULL), "wind", 2023L),
               "needs wind_speed, which this model's inputs do not carry")
})
