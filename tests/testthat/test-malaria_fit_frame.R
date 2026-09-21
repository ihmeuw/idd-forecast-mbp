source(lib_file("malaria_fit_frame.R"))

frame_at <- function(inc_count_min, pfpr_min, ...) {
  path <- tmp_parquet(synthetic_past_inputs(), "synthetic_past_inputs.parquet")
  withr::defer(unlink(path), envir = parent.frame())
  suppressMessages(prepare_malaria_fit_frame(path, inc_count_min, pfpr_min, ...))
}

test_that("derived columns are computed unconditionally and match the old scripts' arithmetic", {
  f <- frame_at(0, 0)
  raw <- synthetic_past_inputs()
  expect_equal(f$malaria_inc_count, raw$malaria_inc_rate * raw$population)
  expect_equal(f$log_gdppc_mean, log(raw$gdppc_mean))
  expect_equal(f$log_malaria_inc_rate, log(raw$malaria_inc_rate))
  expect_equal(f$log_malaria_mort_rate, log(raw$malaria_mort_rate))
  do30 <- pmin(pmax(raw$days_over_30C / 365, 0.001), 0.999)
  expect_equal(f$logit_do30, log(do30 / (1 - do30)))
  rh <- pmin(pmax(raw$relative_humidity / 100, 0.001), 0.999)
  expect_equal(f$logit_relative_humidity, log(rh / (1 - rh)))
  suit <- pmin(pmax(raw$malaria_suitability_mordecai_0_0 / 365, 0.001), 0.999)
  expect_equal(f$malaria_suit, raw$malaria_suitability_mordecai_0_0)
  expect_equal(f$logit_malaria_suitability, log(suit / (1 - suit)))
})

test_that("the stored response is kept as stored, never recomputed", {
  f <- frame_at(0, 0)
  expect_equal(f$logit_malaria_pfpr, synthetic_past_inputs()$logit_malaria_pfpr)
})

test_that("non-finite transforms become NA and are counted once per row", {
  path <- tmp_parquet(synthetic_past_inputs(), "synthetic_nonfinite.parquet")
  withr::defer(unlink(path))
  expect_message(f <- prepare_malaria_fit_frame(path, 0, 0), "2 rows carry a non-finite transform")
  expect_true(is.na(f$log_mal_DAH_total_per_capita[2]))   # log(0)
  expect_true(is.na(f$log_mal_DAH_total_per_capita[4]))   # log(-1)
  expect_false(anyNA(f$log_mal_DAH_total_per_capita[c(1, 3, 5, 6)]))
  expect_true(all(is.finite(f$log_gdppc_mean)))
  expect_true(is.na(f$log_ldipc_mean[2]))                 # an input NA stays NA
})

test_that("thresholds drop rows on inc_count and pfpr, and only there", {
  expect_equal(nrow(frame_at(0, 0)), 6L)
  f <- frame_at(1, 0.0001)
  # row 2 fails pfpr (5e-5); row 3 fails inc_count (0.25); row 5 fails inc_count (0.5)
  expect_equal(f$location_id, c(1L, 4L, 6L))
  expect_equal(nrow(frame_at(20, 0)), sum(synthetic_past_inputs()$malaria_inc_rate *
                                          synthetic_past_inputs()$population >= 20))
})

test_that("A0_af is built after the filter, from the surviving countries", {
  expect_equal(levels(frame_at(0, 0)$A0_af), c("10", "20", "30"))
  f <- frame_at(60, 0)   # keeps location 4 (80) and 6 (90): countries 20 and 30 only
  expect_equal(levels(f$A0_af), c("20", "30"))
})

test_that("the suitability variant is an argument and a missing variant column is an error", {
  f <- frame_at(0, 0, suit_variant = "villena_0_0")
  expect_equal(f$malaria_suit, synthetic_past_inputs()$malaria_suitability_villena_0_0)
  expect_error(frame_at(0, 0, suit_variant = "nope"), "column 'malaria_suitability_nope' not in past inputs")
})

test_that("attributes record what was read and kept", {
  f <- frame_at(1, 0.0001)
  expect_equal(attr(f, "n_read"), 6L)
  expect_equal(attr(f, "n_non_finite_rows"), 2L)
  expect_equal(unname(attr(f, "thresholds")), c(1, 0.0001))
  expect_equal(attr(f, "suit_variant"), "mordecai_0_0")
})

test_that("a missing parquet is an error", {
  expect_error(prepare_malaria_fit_frame(file.path(r_test_tmp(), "absent.parquet"), 0, 0), "past inputs not found")
})

# --- slow: the real past inputs (set MBP_PAST_INPUTS to the parquet the fits read) -----
real_parquet <- Sys.getenv("MBP_PAST_INPUTS", unset = "")

test_that("selection thresholds (1, 1e-4) reproduce the selection frame on the real past inputs", {
  skip_if(!nzchar(real_parquet) || !file.exists(real_parquet), "MBP_PAST_INPUTS not set")
  f <- suppressMessages(prepare_malaria_fit_frame(real_parquet, 1, 0.0001))
  expect_equal(nrow(f), 167649L)
  expect_equal(nlevels(f$A0_af), 70L)
})

test_that("final-fit thresholds (0, 0) reproduce the registered model's frame on the real past inputs", {
  skip_if(!nzchar(real_parquet) || !file.exists(real_parquet), "MBP_PAST_INPUTS not set")
  f <- suppressMessages(prepare_malaria_fit_frame(real_parquet, 0, 0))
  expect_equal(nrow(f), 319073L)
  expect_equal(nlevels(f$A0_af), 70L)
  expect_true(all(is.finite(f$logit_malaria_pfpr)))
})
