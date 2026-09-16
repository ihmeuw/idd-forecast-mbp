# model_registry.R -- LEGACY reader of the pre-2026-09-16 malaria model registry (JSON).
#
# Until 2026-09-16 the fitted malaria models were flat {run_date}_malaria_models.RData
# files beside malaria_model_registry.json, each record carrying a `best` flag. Fitted
# models now live on the models node under idd_tools.versions
# (03-modeling_data/malaria/models/<hierarchy>/<snapshot>/malaria_models.RData + run.json;
# `current` = the model in use), written by 03_modeling/fit_selected_malaria_model.py and
# promoted with `idd-versions <node> promote`. No writer of this JSON remains: the
# upsert / remove functions that lived here were deleted so nothing can move `best`
# behind the node's back. The readers stay so the legacy records (formulas,
# thresholds, convergence) remain queryable from R.
#
# Record shape: { "run_date": "2026_06_01", "best": true|false,
#                 "description": "...", "recorded_at": "2026-06-01 14:32:10", ... }

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

