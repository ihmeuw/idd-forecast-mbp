source(lib_file("scam_fit_helpers.R"))

# 60 rows, 3 countries, one covariate with a single NA so na.omit has work to do.
helper_frame <- function() {
  set.seed(11)
  n <- 60
  x <- seq(0, 3, length.out = n)
  a0 <- rep(c(10L, 20L, 30L), each = n / 3)
  y <- -2 + 0.8 * x + c(0, 0.5, -0.5)[match(a0, c(10L, 20L, 30L))] + rnorm(n, sd = 0.1)
  f <- data.frame(location_id = 1:n, year_id = 2000L + (1:n) %% 24L, x = x, A0_location_id = a0,
                  logit_malaria_pfpr = y, malaria_pfpr = plogis(y), A0_af = factor(a0))
  f$x[7] <- NA
  rownames(f) <- as.character(100 + seq_len(n))   # non-default row names, like a filtered frame
  f
}

test_that("fit_engine dispatches on the spec's counts", {
  expect_equal(fit_engine(1, 2), "scam")
  expect_equal(fit_engine(0, 1), "gam")
  expect_equal(fit_engine(0, 0), "lm")
})

test_that("fit_one_mod fits lm and gam, and reports an error instead of throwing", {
  f <- helper_frame()
  lm_res <- suppressMessages(fit_one_mod(logit_malaria_pfpr ~ x + A0_af, f, 0, 0, label = "lm"))
  expect_false(lm_res$error); expect_s3_class(lm_res$fit, "lm"); expect_true(lm_res$elapsed >= 0)
  gam_res <- suppressMessages(fit_one_mod(logit_malaria_pfpr ~ s(x, k = 4) + A0_af, f, 0, 1, label = "gam"))
  expect_false(gam_res$error); expect_s3_class(gam_res$fit, "gam")
  bad <- suppressMessages(fit_one_mod(logit_malaria_pfpr ~ nope + A0_af, f, 0, 0, label = "bad"))
  expect_true(bad$error); expect_null(bad$fit); expect_match(bad$error_msg, "nope")
})

test_that("a fit carries no copy of the training frame in its formula environment", {
  f <- helper_frame()
  wrapper <- function(frame) {
    big_local <- frame   # would be serialised with the fit if the formula kept this environment
    fml <- as.formula("logit_malaria_pfpr ~ s(x, k = 4) + A0_af")
    suppressMessages(fit_one_mod(fml, frame, 0, 1))$fit
  }
  fit <- wrapper(f)
  # gam resets the environment to globalenv (never serialised); scam keeps the clean one
  expect_false(exists("big_local", envir = environment(fit$formula), inherits = FALSE))
  expect_false(exists("frame", envir = environment(fit$formula), inherits = FALSE))
  expect_identical(environment(fit$terms), environment(fit$formula))
  path <- file.path(r_test_tmp(), "small_fit.rds"); withr::defer(unlink(path))
  saveRDS(strip_scam_for_predict(fit), path)
  expect_lt(file.size(path), 2e5)
  expect_equal(predict(readRDS(path), newdata = f[1:4, ]), predict(fit, newdata = f[1:4, ]))
})

test_that("fit_rows_used returns the rows na.omit kept, by row name", {
  f <- helper_frame()
  fit <- lm(logit_malaria_pfpr ~ x + A0_af, data = f)
  used <- fit_rows_used(fit, f)
  expect_equal(length(used), nrow(f) - 1L)
  expect_false(7L %in% used)
  expect_equal(f$logit_malaria_pfpr[used], unname(fitted(fit) + residuals(fit)))
  gam_fit <- mgcv::gam(logit_malaria_pfpr ~ s(x, k = 4) + A0_af, data = f, method = "REML")
  expect_equal(length(fit_rows_used(gam_fit, f)), nrow(f) - 1L)
  expect_error(fit_rows_used(fit, f[-1, ]), "row names changed")
})

test_that("response_info maps the response name to its inverse link and natural column", {
  f <- helper_frame()
  i <- response_info(logit_malaria_pfpr ~ x, f)
  expect_equal(i$response, "logit_malaria_pfpr"); expect_identical(i$inverse, stats::plogis)
  expect_equal(i$natural_column, "malaria_pfpr")
  j <- response_info(log_malaria_inc_rate ~ x, f)          # natural column absent from this frame
  expect_identical(j$inverse, exp); expect_true(is.na(j$natural_column))
  k <- response_info(log_malaria_inc_rate ~ x)               # no frame: name only
  expect_equal(k$natural_column, "malaria_inc_rate")
  expect_identical(response_info(y ~ x)$inverse, identity)
})

test_that("build_predictions carries identifiers, both scales and an optional fold", {
  f <- helper_frame()
  fit <- lm(logit_malaria_pfpr ~ x + A0_af, data = f)
  used <- fit_rows_used(fit, f)
  p <- build_predictions(f, used, logit_malaria_pfpr ~ x + A0_af, fitted(fit))
  expect_equal(names(p), c("location_id", "year_id", "observed", "predicted_lp", "predicted_response", "observed_response"))
  expect_type(p$location_id, "integer"); expect_type(p$year_id, "integer")
  expect_equal(nrow(p), length(used))
  expect_equal(p$observed, f$logit_malaria_pfpr[used])
  expect_equal(p$predicted_response, plogis(unname(fitted(fit))))
  expect_equal(p$observed_response, f$malaria_pfpr[used])
  q <- build_predictions(f, 1:3, logit_malaria_pfpr ~ x + A0_af, c(0, 1, 2), fold = 2L)
  expect_equal(q$fold, c(2L, 2L, 2L))
})

test_that("atomic writers leave no temp file and the rds predicts as the fit does", {
  f <- helper_frame()
  fit <- mgcv::gam(logit_malaria_pfpr ~ s(x, k = 4) + A0_af, data = f, method = "REML")
  out <- file.path(r_test_tmp(), "helpers_out"); unlink(out, recursive = TRUE)
  withr::defer(unlink(out, recursive = TRUE))
  paths <- fit_paths(out, "IS", 7L)
  expect_equal(paths$rds, file.path(out, "fits", "IS", "spec_7.rds"))
  expect_equal(fit_paths(out, "random", 7L, fold = 3L)$json, file.path(out, "fits", "random", "spec_7_fold3.json"))
  expect_equal(prediction_path(out, "IS", 7L), file.path(out, "predictions", "IS", "spec_7.parquet"))

  stripped <- strip_scam_for_predict(fit)
  expect_null(stripped$model); expect_silent(suppressMessages(verify_strip(fit, stripped, f[1:5, ], "gam")))
  write_rds_atomic(stripped, paths$rds)
  back <- readRDS(paths$rds)
  expect_equal(predict(back, newdata = f[10:15, ]), predict(fit, newdata = f[10:15, ]))
  expect_false(file.exists(paste0(paths$rds, ".tmp")))

  write_json_atomic(list(a = 1L, b = "x", v = c(1L, 2L)), paths$json)
  j <- jsonlite::fromJSON(paths$json)
  expect_equal(j$a, 1L); expect_equal(j$v, c(1L, 2L))
  expect_false(file.exists(paste0(paths$json, ".tmp")))

  used <- fit_rows_used(fit, f)
  p <- build_predictions(f, used, logit_malaria_pfpr ~ s(x, k = 4) + A0_af, fitted(fit))
  pp <- prediction_path(out, "IS", 7L)
  write_parquet_atomic(p, pp)
  expect_equal(arrow::open_dataset(pp)$num_rows, nrow(p))
  expect_equal(names(arrow::read_parquet(pp)), names(p))
  expect_length(list.files(dirname(pp), all.files = TRUE, pattern = "tmp"), 0)
  write_parquet_atomic(p[1:3, ], pp)                       # replaces, never appends
  expect_equal(arrow::open_dataset(pp)$num_rows, 3)
})

test_that("fit_sidecar records the spec, the cell, the run and the fit's own row set", {
  f <- helper_frame()
  res <- suppressMessages(fit_one_mod(logit_malaria_pfpr ~ x + A0_af, f, 0, 0))
  sc <- fit_sidecar(
    res$fit, res, logit_malaria_pfpr ~ x + A0_af,
    spec = list(spec_index = 7L, formula_text = "logit_malaria_pfpr ~ x + A0_af", suit_variant = "mordecai_0_0", engine = "lm"),
    cell = list(cell = "narrow_full", cell_kind = "oos_temporal", train_lo = 2000L, train_hi = 2012L,
                test_lo = 2013L, test_hi = 2023L, fold = NULL, cv_n_folds = 1L, cv_seed = 42L, n_rows_test = 9L),
    run = list(inc_count_min = 1, pfpr_min = 1e-4, past_inputs = "/p.parquet", past_inputs_sha256 = "0",
               prep_script = "/prep.R", prep_script_sha256 = "1", n_rows_frame = nrow(f), optimizer = "efs",
               maxit = 30L, task_id = "narrow_full_n0_s0_bin0", worker_file = "/w.r"))
  required <- c("spec_index", "formula_text", "response", "response_natural_column", "engine", "cell", "cell_kind",
                "train_lo", "train_hi", "test_lo", "test_hi", "cv_n_folds", "cv_seed", "inc_count_min", "pfpr_min",
                "suit_variant", "past_inputs", "past_inputs_sha256", "prep_script", "prep_script_sha256", "n_rows_frame",
                "n_rows_train", "n_rows_test", "a0_levels_train", "optimizer", "maxit", "iter", "converged",
                "elapsed_sec", "r_version", "scam_version", "mgcv_version", "stripped", "task_id", "worker_file", "written_at")
  expect_true(all(required %in% names(sc)), info = paste("missing:", paste(setdiff(required, names(sc)), collapse = ", ")))
  expect_equal(sc$n_rows_train, nrow(f) - 1L)
  expect_equal(sc$a0_levels_train, c(10L, 20L, 30L))
  expect_equal(sc$response_natural_column, "malaria_pfpr")
  expect_true(sc$stripped)
})

test_that("sha256_file is a 64-hex digest that changes with the content", {
  p <- file.path(r_test_tmp(), "hash_me.txt"); withr::defer(unlink(p))
  writeLines("a", p); h1 <- sha256_file(p)
  expect_match(h1, "^[0-9a-f]{64}$")
  writeLines("b", p); expect_false(identical(sha256_file(p), h1))
})
