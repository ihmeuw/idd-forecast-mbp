# The worker's fit_one_spec with the switches on: saved objects, sidecars, predictions,
# in-sample metrics on the rows the fit used, per-spec suitability variant.
source(lib_file("scam_fit_helpers.R"))
source(lib_file("malaria_fit_frame.R"))
worker_file <- file.path(repo_root, "src", "idd_forecast_mbp", "03_modeling", "select_malaria_models_rocket.r")
source(worker_file)   # defines functions only: main() runs under the --file guard

worker_frame <- function() {
  path <- tmp_parquet(synthetic_past_inputs_large(), "synthetic_large.parquet")
  withr::defer(unlink(path), envir = parent.frame())
  suppressMessages(prepare_malaria_fit_frame(path, 0, 0))
}
run_info <- list(inc_count_min = 0, pfpr_min = 0, past_inputs = "/p", past_inputs_sha256 = "p0",
                 prep_script = "/s", prep_script_sha256 = "s0", n_rows_frame = 288L, optimizer = "efs",
                 maxit = 30L, task_id = "IS_n0_s0_bin0", worker_file = "/w")
base_flags <- function(out_dir, ...) {
  modifyList(list(is_fe = TRUE, oos = FALSE, write_summary = FALSE, save_fits = FALSE,
                  save_predictions = FALSE, strip_fits = TRUE, inc_count_min = 0, pfpr_min = 0,
                  cv_strategy = "temporal", n_folds = 5L, cv_seed = 42L,
                  train_lo = 2000L, train_hi = 2015L, test_lo = 2016L, test_hi = 2023L,
                  optimizer = "efs", maxit = 30L, out_dir = out_dir), list(...))
}
fml_lm  <- "logit_malaria_pfpr ~ logit_malaria_suitability + log_gdppc_mean + A0_af"
fml_gam <- "logit_malaria_pfpr ~ s(logit_malaria_suitability, k = 4) + log_gdppc_mean + A0_af"

test_that("cell_of_task strips the bundle suffix", {
  expect_equal(cell_of_task("IS_n0_s0_bin0"), "IS")
  expect_equal(cell_of_task("narrow_full_n2_s1_bin13"), "narrow_full")
  expect_equal(cell_of_task("random_n1_s0_bin0"), "random")
})

test_that("with the switches off an IS fit writes nothing and the row carries the new provenance", {
  out <- file.path(r_test_tmp(), "worker_off"); unlink(out, recursive = TRUE); dir.create(out)
  withr::defer(unlink(out, recursive = TRUE))
  data <- worker_frame()
  row <- suppressMessages(fit_one_spec(1L, data, base_flags(out), "IS_n0_s0_bin0", fml_lm, 0L, 0L,
                                       run_info = run_info))
  expect_equal(nrow(row), 1L)
  expect_equal(row$engine, "lm"); expect_equal(row$suit_variant, "mordecai_0_0")
  expect_equal(c(row$inc_count_min, row$pfpr_min, row$n_rows_frame), c(0, 0, nrow(data)))
  expect_equal(row$past_inputs_sha256, "p0"); expect_equal(row$prep_script_sha256, "s0")
  expect_false(row$is_error); expect_equal(row$is_n_obs, nrow(data))
  expect_true(is.finite(row$is_pfpr_r) && row$is_pfpr_r > 0.9)
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0)
})

test_that("in-sample metrics use the rows the fit kept when a formula drops NA rows", {
  out <- file.path(r_test_tmp(), "worker_na"); unlink(out, recursive = TRUE); dir.create(out)
  withr::defer(unlink(out, recursive = TRUE))
  data <- worker_frame()
  data$log_gdppc_mean[c(5, 50, 100)] <- NA
  row <- suppressMessages(fit_one_spec(1L, data, base_flags(out), "IS_n0_s0_bin0", fml_lm, 0L, 0L,
                                       run_info = run_info))
  expect_equal(row$is_n_obs, nrow(data) - 3L)
  fit <- lm(as.formula(fml_lm), data = data)
  expect_equal(row$is_pfpr_rmse, sqrt(mean((data$malaria_pfpr[-c(5, 50, 100)] - plogis(unname(fitted(fit))))^2)))
})

test_that("--save-fits and --save-predictions write the object, its sidecar and the predictions for IS", {
  out <- file.path(r_test_tmp(), "worker_is"); unlink(out, recursive = TRUE); dir.create(out)
  withr::defer(unlink(out, recursive = TRUE))
  data <- worker_frame()
  flags <- base_flags(out, save_fits = TRUE, save_predictions = TRUE)
  row <- suppressMessages(fit_one_spec(7L, data, flags, "IS_n1_s0_bin0", fml_gam, 0L, 1L, run_info = run_info))
  rds <- file.path(out, "fits", "IS", "spec_7.rds"); js <- file.path(out, "fits", "IS", "spec_7.json")
  pq  <- file.path(out, "predictions", "IS", "spec_7.parquet")
  expect_true(file.exists(rds)); expect_true(file.exists(js)); expect_true(file.exists(pq))
  fit <- readRDS(rds)
  expect_s3_class(fit, "gam"); expect_null(fit$model)                     # stripped by default
  expect_length(predict(fit, newdata = data[1:5, ]), 5)
  sc <- jsonlite::fromJSON(js)
  expect_equal(sc$spec_index, 7L); expect_equal(sc$formula_text, fml_gam); expect_equal(sc$engine, "gam")
  expect_equal(sc$cell, "IS"); expect_equal(sc$cell_kind, "is"); expect_equal(sc$n_rows_train, nrow(data))
  expect_equal(sort(sc$a0_levels_train), c(10L, 20L, 30L)); expect_equal(sc$task_id, "IS_n0_s0_bin0")
  expect_equal(sc$response, "logit_malaria_pfpr"); expect_equal(sc$response_natural_column, "malaria_pfpr")
  expect_true(sc$stripped); expect_equal(sc$inc_count_min, 0)
  p <- arrow::read_parquet(pq)
  expect_equal(names(p), c("location_id", "year_id", "observed", "predicted_lp", "predicted_response", "observed_response"))
  expect_equal(nrow(p), nrow(data)); expect_equal(p$observed_response, data$malaria_pfpr)
  expect_equal(p$predicted_response, plogis(p$predicted_lp))
  expect_length(list.files(out, pattern = "tmp", recursive = TRUE, all.files = TRUE), 0)
})

test_that("a temporal OOS cell saves the training fit with its window and predicts only the test rows", {
  out <- file.path(r_test_tmp(), "worker_oos"); unlink(out, recursive = TRUE); dir.create(out)
  withr::defer(unlink(out, recursive = TRUE))
  data <- worker_frame()
  flags <- base_flags(out, is_fe = FALSE, oos = TRUE, save_fits = TRUE, save_predictions = TRUE)
  row <- suppressMessages(fit_one_spec(3L, data, flags, "wide_full_n0_s0_bin2", fml_lm, 0L, 0L,
                                       run_info = modifyList(run_info, list(task_id = "wide_full_n0_s0_bin2"))))
  expect_equal(row$cv_strategy, "temporal"); expect_equal(row$oos_n_obs, sum(data$year_id >= 2016))
  sc <- jsonlite::fromJSON(file.path(out, "fits", "wide_full", "spec_3.json"))
  expect_equal(sc$cell, "wide_full"); expect_equal(sc$cell_kind, "oos_temporal")
  expect_equal(c(sc$train_lo, sc$train_hi, sc$test_lo, sc$test_hi), c(2000L, 2015L, 2016L, 2023L))
  expect_equal(sc$n_rows_train, sum(data$year_id <= 2015)); expect_equal(sc$n_rows_test, sum(data$year_id >= 2016))
  expect_null(sc$fold)
  p <- arrow::read_parquet(file.path(out, "predictions", "wide_full", "spec_3.parquet"))
  expect_equal(nrow(p), sum(data$year_id >= 2016)); expect_true(all(p$year_id >= 2016))
  expect_false("fold" %in% names(p))
})

test_that("a random k-fold cell saves one object per fold and predictions carry the fold", {
  out <- file.path(r_test_tmp(), "worker_kfold"); unlink(out, recursive = TRUE); dir.create(out)
  withr::defer(unlink(out, recursive = TRUE))
  data <- worker_frame()
  flags <- base_flags(out, is_fe = FALSE, oos = TRUE, cv_strategy = "random", n_folds = 3L,
                      save_fits = TRUE, save_predictions = TRUE)
  row <- suppressMessages(fit_one_spec(2L, data, flags, "random_n0_s0_bin0", fml_lm, 0L, 0L, run_info = run_info))
  expect_equal(row$cv_n_folds, 3L)
  expect_setequal(list.files(file.path(out, "fits", "random")),
                  c(sprintf("spec_2_fold%d.rds", 1:3), sprintf("spec_2_fold%d.json", 1:3)))
  sc <- jsonlite::fromJSON(file.path(out, "fits", "random", "spec_2_fold2.json"))
  expect_equal(sc$fold, 2L); expect_equal(sc$cv_n_folds, 3L); expect_equal(sc$cv_seed, 42L)
  p <- arrow::read_parquet(file.path(out, "predictions", "random", "spec_2.parquet"))
  expect_equal(nrow(p), nrow(data)); expect_setequal(unique(p$fold), 1:3)
})

test_that("the spec's suitability variant is applied per spec and recorded", {
  out <- file.path(r_test_tmp(), "worker_variant"); unlink(out, recursive = TRUE); dir.create(out)
  withr::defer(unlink(out, recursive = TRUE))
  data <- worker_frame()
  flags <- base_flags(out, save_fits = TRUE)
  a <- suppressMessages(fit_one_spec(1L, data, flags, "IS_n0_s0_bin0", fml_lm, 0L, 0L, run_info = run_info))
  b <- suppressMessages(fit_one_spec(2L, data, flags, "IS_n0_s0_bin0", fml_lm, 0L, 0L,
                                     suit_variant = "villena_0_0", run_info = run_info))
  expect_equal(a$suit_variant, "mordecai_0_0"); expect_equal(b$suit_variant, "villena_0_0")
  expect_false(isTRUE(all.equal(a$is_coef, b$is_coef)))    # a different covariate column was fitted
  expect_equal(jsonlite::fromJSON(file.path(out, "fits", "IS", "spec_2.json"))$suit_variant, "villena_0_0")
  expect_equal(attr(data, "suit_variant"), "mordecai_0_0")   # the caller's frame is untouched
})

test_that("--strip-fits FALSE keeps the full object", {
  out <- file.path(r_test_tmp(), "worker_full"); unlink(out, recursive = TRUE); dir.create(out)
  withr::defer(unlink(out, recursive = TRUE))
  data <- worker_frame()
  flags <- base_flags(out, save_fits = TRUE, strip_fits = FALSE)
  suppressMessages(fit_one_spec(1L, data, flags, "IS_n0_s0_bin0", fml_lm, 0L, 0L, run_info = run_info))
  fit <- readRDS(file.path(out, "fits", "IS", "spec_1.rds"))
  expect_false(is.null(fit$model))
  expect_false(jsonlite::fromJSON(file.path(out, "fits", "IS", "spec_1.json"))$stripped)
})
