# Fitting, stripping and saving helpers shared by the malaria fit worker and the final fitter.
#
# Engine dispatch (scam / gam / lm), the predict-only strip with its verification, the rows a
# fit actually used, prediction frames, the per-fit sidecar, and atomic writers (tmp beside the
# target, then rename). Packages are referenced with `::`; callers attach what they attach.

fit_engine <- function(n_scams, n_smooths) {
  if (n_scams > 0) "scam" else if (n_smooths > 0) "gam" else "lm"
}

# Fit one model, capturing iterations, convergence, timing and any error (never throws).
fit_one_mod <- function(fml, data, n_scams, n_smooths,
                        optimizer = "bfgs", maxit = 300L, label = "fit") {
  # A formula built inside a function carries that function's environment, and saveRDS
  # serialises it with the fit: the whole training frame, 150 MB, hidden in every object.
  # Every variable comes from `data`, so a clean environment is safe and predict() is unchanged.
  environment(fml) <- new.env(parent = globalenv())
  t0  <- Sys.time()
  fit <- tryCatch(
    switch(fit_engine(n_scams, n_smooths),
      scam = scam::scam(fml, data = data, optimizer = optimizer, control = list(maxit = maxit)),
      gam  = mgcv::gam(fml, data = data, method = "REML"),
      lm   = stats::lm(fml, data = data)),
    error = function(e) e)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (inherits(fit, "error")) {
    message(sprintf("  [%s] ERROR after %.1fs: %s", label, elapsed, conditionMessage(fit)))
    return(list(fit = NULL, iter = NA_integer_, converged = NA, elapsed = elapsed,
                error = TRUE, error_msg = conditionMessage(fit)))
  }
  iter <- tryCatch({ v <- fit$iter; if (length(v)) as.integer(v)[1] else NA_integer_ },
                   error = function(e) NA_integer_)
  conv <- isTRUE(fit$conv)
  message(sprintf("  [%s] iter=%s converged=%s (%.1fs)", label, iter, conv, elapsed))
  list(fit = fit, iter = iter, converged = conv, elapsed = elapsed,
       error = FALSE, error_msg = NA_character_)
}

# Drop training-frame slots predict() never reads; verify_strip proves it lossless.
strip_scam_for_predict <- function(mod) {
  for (slot in c("model", "y", "residuals", "fitted.values",
                 "linear.predictors", "weights", "prior.weights")) mod[[slot]] <- NULL
  mod
}
verify_strip <- function(full, stripped, newdata, label = "fit", tol = 1e-8) {
  max_abs <- max(abs(predict(full, newdata = newdata) - predict(stripped, newdata = newdata)))
  if (!is.finite(max_abs) || max_abs > tol) {
    stop(sprintf("[%s] strip changed predictions (max |diff| = %g); not saving.", label, max_abs))
  }
  message(sprintf("  [%s] strip verified (max |diff| = %.3g)", label, max_abs))
  invisible(max_abs)
}

# Row positions in `data` that the fit used (the formula's na.omit decides), via the model
# frame's row names. Works for lm, gam and scam alike.
fit_rows_used <- function(fit, data) {
  idx <- match(rownames(fit$model), rownames(data))
  if (anyNA(idx)) stop("fitted rows not found in the frame (row names changed between prep and fit)")
  idx
}

# The response column a formula names, its inverse link by naming convention, and the
# natural-scale column it was derived from when the frame has one.
response_info <- function(fml, data = NULL) {
  lhs <- as.character(fml[[2]])
  inverse <- if (startsWith(lhs, "logit_")) stats::plogis else if (startsWith(lhs, "log_")) exp else identity
  natural <- sub("^(logit|log)_", "", lhs)
  if (natural == lhs || (!is.null(data) && !natural %in% names(data))) natural <- NA_character_
  list(response = lhs, inverse = inverse, natural_column = natural)
}

# One row per predicted observation: identifiers, observed and predicted on the link scale,
# the prediction on the natural scale, and the observed natural-scale value when known.
build_predictions <- function(data, rows, fml, predicted_lp, fold = NULL) {
  info <- response_info(fml, data)
  out <- data.frame(
    location_id        = as.integer(data$location_id[rows]),
    year_id            = as.integer(data$year_id[rows]),
    observed           = as.numeric(data[[info$response]][rows]),
    predicted_lp       = as.numeric(predicted_lp),
    predicted_response = as.numeric(info$inverse(predicted_lp)),
    observed_response  = if (is.na(info$natural_column)) NA_real_ else as.numeric(data[[info$natural_column]][rows]))
  if (!is.null(fold)) out$fold <- as.integer(fold)
  out
}

sha256_file <- function(path) digest::digest(file = path, algo = "sha256")

# Where a cell's saved fit and predictions live under the run dir.
fit_paths <- function(out_dir, cell, spec_index, fold = NULL) {
  stem <- if (is.null(fold)) sprintf("spec_%d", spec_index) else sprintf("spec_%d_fold%d", spec_index, fold)
  dir  <- file.path(out_dir, "fits", cell)
  list(dir = dir, rds = file.path(dir, paste0(stem, ".rds")), json = file.path(dir, paste0(stem, ".json")))
}
prediction_path <- function(out_dir, cell, spec_index) {
  file.path(out_dir, "predictions", cell, sprintf("spec_%d.parquet", spec_index))
}

# The record beside a saved fit: enough to rebuild the rows it was fitted on.
# `spec`, `cell` and `run` are lists the caller assembles; fit-derived fields are added here.
fit_sidecar <- function(fit, fit_res, fml, spec, cell, run, stripped = TRUE) {
  info <- response_info(fml)
  c(spec,
    list(response = info$response, response_natural_column = info$natural_column),
    cell,
    list(n_rows_train = nrow(fit$model),
         a0_levels_train = as.integer(fit$xlevels[["A0_af"]]),
         iter = fit_res$iter, converged = fit_res$converged, elapsed_sec = fit_res$elapsed,
         r_version = R.version.string,
         scam_version = as.character(utils::packageVersion("scam")),
         mgcv_version = as.character(utils::packageVersion("mgcv")),
         stripped = stripped,
         written_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
    run)
}

# --- atomic writers: temp beside the target, validate, replace, rename --------------------
.replace_with <- function(tmp, path, mode = "0664") {
  if (file.exists(path)) file.remove(path)
  if (!file.rename(tmp, path)) stop("rename failed: ", tmp, " -> ", path)
  Sys.chmod(path, mode)
  invisible(path)
}
write_rds_atomic <- function(obj, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  tmp <- paste0(path, ".tmp")
  saveRDS(obj, tmp)
  .replace_with(tmp, path)
}
write_json_atomic <- function(x, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  tmp <- paste0(path, ".tmp")
  jsonlite::write_json(x, tmp, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")
  .replace_with(tmp, path)
}
write_parquet_atomic <- function(df, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  tmp <- file.path(dirname(path), paste0(".", sub("\\.parquet$", "", basename(path)), ".tmp.parquet"))
  arrow::write_parquet(df, tmp)
  if (arrow::open_dataset(tmp)$num_rows != nrow(df)) stop("row-count mismatch on written parquet: ", tmp)
  .replace_with(tmp, path)
}
