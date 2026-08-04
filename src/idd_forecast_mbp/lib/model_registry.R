# model_registry.R — shared malaria model run registry (JSON).
#
# Single source of truth for the fitted malaria-model runs and which one is
# "best". One JSON array of run records; each record carries a `best` flag and
# exactly one record is TRUE at a time. Read by both R (this repo's fit +
# forecast scripts) and Python (idd_forecast_mbp.constants.get_malaria_model_run_date).
#
# Record shape (minimum):
#   { "run_date": "2026_06_01", "best": true|false,
#     "description": "...", "recorded_at": "2026-06-01 14:32:10", ... }
#
# JSON (not YAML) is used because jsonlite is confirmed present in the cluster R
# image and json is Python stdlib — zero new dependency on either side.

require(jsonlite)
require(glue)

# Canonical registry location: alongside the saved {run_date}_malaria_models.RData
# files, i.e. inside the 03-modeling_data output dir.
malaria_model_registry_path <- function(data_path) {
  file.path(data_path, "malaria_model_registry.json")
}

# Return the registry as a list of records (list of named lists), or an empty
# list if the file does not exist yet (first run creates it).
read_malaria_model_registry <- function(path) {
  if (!file.exists(path)) return(list())
  recs <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (is.null(recs)) list() else recs
}

# Atomic write: serialize to a temp file, confirm it parses back, then rename
# over the live file. Never leaves a half-written registry on disk.
.write_malaria_model_registry <- function(recs, path) {
  tmp <- paste0(path, ".tmp")
  writeLines(
    jsonlite::toJSON(recs, auto_unbox = TRUE, pretty = TRUE, null = "null"),
    tmp
  )
  invisible(jsonlite::fromJSON(tmp, simplifyVector = FALSE))  # validate round-trip
  file.rename(tmp, path)
}

# Sort best-first, then most-recently-recorded first among the rest.
.sort_malaria_model_registry <- function(recs) {
  if (length(recs) <= 1) return(recs)
  is_best <- vapply(recs, function(r) isTRUE(r$best), logical(1))
  rec_at  <- vapply(recs, function(r) if (is.null(r$recorded_at)) "" else as.character(r$recorded_at),
                    character(1))
  best_part <- recs[is_best]
  rest      <- recs[!is_best]
  if (length(rest) > 1) rest <- rest[order(rec_at[!is_best], decreasing = TRUE)]
  c(best_part, rest)
}

# Append or update the record for `run_date`, then persist.
#  - description: required, non-empty (this is the per-run note).
#  - best: TRUE makes this run the single best (every other record demoted to
#    FALSE). FALSE preserves an existing record's prior best status (so re-running
#    the current best without re-flagging does not silently un-best it).
#  - extra: named list of additional provenance fields to store on the record.
# Re-running the same run_date updates that record in place (the .RData file for a
# given date is likewise overwritten), so the registry never accumulates dupes.
upsert_malaria_model_run <- function(path, run_date, description, best = FALSE, extra = list()) {
  stopifnot(is.character(run_date), length(run_date) == 1, nzchar(run_date))
  if (!nzchar(trimws(description))) {
    stop("upsert_malaria_model_run(): 'description' must be a non-empty string.")
  }

  recs <- read_malaria_model_registry(path)
  run_dates <- vapply(recs, function(r) as.character(r$run_date), character(1))
  idx <- which(run_dates == run_date)

  keep_best <- if (length(idx)) isTRUE(recs[[idx[1]]]$best) else FALSE
  this_best <- isTRUE(best) || keep_best

  rec <- c(
    list(
      run_date    = run_date,
      best        = this_best,
      description = trimws(description),
      recorded_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    ),
    extra
  )

  if (length(idx)) recs[[idx[1]]] <- rec else recs <- c(recs, list(rec))

  # Enforce exactly-one-best: if this run is best, demote every other record.
  if (isTRUE(this_best)) {
    for (i in seq_along(recs)) {
      if (!identical(as.character(recs[[i]]$run_date), run_date)) recs[[i]]$best <- FALSE
    }
  }

  recs <- .sort_malaria_model_registry(recs)
  .write_malaria_model_registry(recs, path)
  message(glue::glue(
    "[registry] {if (length(idx)) 'updated' else 'added'} run_date={run_date} ",
    "best={this_best} -> {path}"
  ))
  invisible(recs)
}

# Resolve a run_date from the registry.
#  - run_date set: verify it exists, return it.
#  - best = TRUE (default): return the run_date of the single best record.
# Errors clearly when the registry is empty/missing, the date is absent, or the
# best flag is ambiguous (0 or >1 records flagged).
get_malaria_model_run_date <- function(path, best = TRUE, run_date = NULL) {
  recs <- read_malaria_model_registry(path)
  if (!length(recs)) stop(glue::glue("Malaria model registry is empty or missing: {path}"))

  if (!is.null(run_date)) {
    hit <- Filter(function(r) as.character(r$run_date) == run_date, recs)
    if (!length(hit)) stop(glue::glue("No malaria model run with run_date='{run_date}' in {path}"))
    return(as.character(hit[[1]]$run_date))
  }

  if (isTRUE(best)) {
    best_hits <- Filter(function(r) isTRUE(r$best), recs)
    if (length(best_hits) == 0) stop(glue::glue("No malaria model flagged best=TRUE in {path}"))
    if (length(best_hits) > 1) {
      stop(glue::glue("Multiple malaria models flagged best=TRUE in {path}; exactly one expected."))
    }
    return(as.character(best_hits[[1]]$run_date))
  }

  stop("get_malaria_model_run_date(): specify run_date=, or leave best=TRUE.")
}

# Remove the records for `run_dates` from the registry and persist (atomic).
# Does NOT delete any .RData files — the caller handles file deletion so this
# stays a pure index mutator (single registry writer).
#  - allow_best: FALSE (default) stop()s if any target is best=TRUE, since
#    deleting the best model leaves the forecaster with no run to resolve.
# Returns invisibly: list(removed = <list of removed records>,
#                         missing = <run_dates not present in the registry>).
remove_malaria_model_runs <- function(path, run_dates, allow_best = FALSE) {
  stopifnot(is.character(run_dates), length(run_dates) >= 1L)
  recs <- read_malaria_model_registry(path)
  if (!length(recs)) stop(glue::glue("Malaria model registry is empty or missing: {path}"))

  rec_dates <- vapply(recs, function(r) as.character(r$run_date), character(1))
  is_target <- rec_dates %in% run_dates
  removing  <- recs[is_target]
  missing   <- setdiff(run_dates, rec_dates)

  if (!isTRUE(allow_best)) {
    best_hit <- Filter(function(r) isTRUE(r$best), removing)
    if (length(best_hit)) {
      bd <- vapply(best_hit, function(r) as.character(r$run_date), character(1))
      stop(glue::glue(
        "Refusing to remove best=TRUE run(s): {paste(bd, collapse = ', ')}. ",
        "Pass allow_best=TRUE only if you intend to leave no best model."
      ))
    }
  }

  .write_malaria_model_registry(recs[!is_target], path)
  invisible(list(removed = removing, missing = missing))
}
