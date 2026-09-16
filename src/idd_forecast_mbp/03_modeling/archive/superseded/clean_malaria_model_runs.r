#!/usr/bin/env Rscript
# clean_malaria_model_runs.r — remove stale malaria model runs from the registry
# and (optionally) delete their saved .RData files.
#
# SAFETY:
#   - DRY-RUN by default. Nothing is removed unless you pass --apply.
#   - Refuses to delete a best=TRUE run unless --force-best (deleting the best
#     model leaves the forecaster with no run to resolve).
#   - Deletes only the .RData file named in each registry record (no guessing).
#   - This DELETES files on shared storage — run it yourself, after reviewing
#     the dry-run output.
#
# Usage:
#   Rscript clean_malaria_model_runs.r --doctor
#   Rscript clean_malaria_model_runs.r --run-dates 2026_06_01
#   Rscript clean_malaria_model_runs.r --run-dates 2026_06_01 2026_05_30 --apply
#   Rscript clean_malaria_model_runs.r --run-dates <best_date> --apply --force-best

require(glue)

# NOTE: absolute paths mirror the other 03_modeling R scripts (02_fit_*.r), which
# hardcode REPO_DIR the same way. There is no R-side path config in this repo yet.
REPO_DIR     <- "/mnt/team/idd/pub/forecast-mbp"
data_path    <- glue("{REPO_DIR}/03-modeling_data")
SRC_REPO_DIR <- glue("/ihme/homes/{Sys.getenv('USER')}/repos/idd-forecast-mbp")
source(glue("{SRC_REPO_DIR}/src/idd_forecast_mbp/lib/model_registry.R"))

# ---- arg parsing (base R, zero deps) ----
args          <- commandArgs(trailingOnly = TRUE)
apply_changes <- "--apply"      %in% args
force_best    <- "--force-best" %in% args
doctor        <- "--doctor"     %in% args

# --run-dates consumes the contiguous non-flag tokens that follow it.
parse_run_dates <- function(args) {
  i <- match("--run-dates", args)
  if (is.na(i)) return(character(0))
  out <- character(0); j <- i + 1L
  while (j <= length(args) && !startsWith(args[j], "--")) {
    out <- c(out, args[j]); j <- j + 1L
  }
  out
}
run_dates <- parse_run_dates(args)

registry_path <- malaria_model_registry_path(data_path)

# Full path to a record's saved models, from its rdata_file field (fallback to
# the conventional name if an older record predates that field).
rdata_path_for <- function(rec) {
  fname <- if (is.null(rec$rdata_file)) {
    paste0(rec$run_date, "_malaria_models.RData")
  } else {
    rec$rdata_file
  }
  file.path(data_path, fname)
}

# ---- --doctor: registry vs disk reconciliation (READ-ONLY) ----
if (doctor) {
  recs <- read_malaria_model_registry(registry_path)
  message(glue("Registry: {registry_path}"))
  message(glue("  {length(recs)} record(s)."))
  reg_files <- character(0)
  for (r in recs) {
    p <- rdata_path_for(r)
    reg_files <- c(reg_files, basename(p))
    flag   <- if (isTRUE(r$best)) " [BEST]" else ""
    status <- if (file.exists(p)) "ok" else "MISSING FILE"
    message(glue("  - {r$run_date}{flag}: {basename(p)} ({status})"))
  }
  on_disk <- list.files(data_path, pattern = "_malaria_models\\.RData$")
  orphans <- setdiff(on_disk, reg_files)
  if (length(orphans)) {
    message(glue("\nOrphan .RData files on disk with NO registry entry ({length(orphans)}):"))
    for (f in orphans) message(glue("  - {f}"))
  } else {
    message("\nNo orphan .RData files.")
  }
  quit(save = "no", status = 0)
}

if (!length(run_dates)) {
  stop("Specify --run-dates <date> [<date> ...], or --doctor. Nothing to do.")
}

# ---- resolve targets ----
recs      <- read_malaria_model_registry(registry_path)
rec_dates <- vapply(recs, function(r) as.character(r$run_date), character(1))
present   <- intersect(run_dates, rec_dates)
missing   <- setdiff(run_dates, rec_dates)

if (length(missing)) {
  message(glue("Not in registry (skipped): {paste(missing, collapse = ', ')}"))
}
if (!length(present)) {
  stop("None of the requested run-dates are in the registry. Nothing to remove.")
}

targets      <- Filter(function(r) as.character(r$run_date) %in% present, recs)
best_targets <- Filter(function(r) isTRUE(r$best), targets)

message(glue("\n{if (apply_changes) 'APPLY' else 'DRY RUN'} — {length(targets)} run(s) targeted:"))
for (r in targets) {
  p    <- rdata_path_for(r)
  flag <- if (isTRUE(r$best)) " [BEST]" else ""
  fst  <- if (file.exists(p)) "exists" else "file already gone"
  message(glue("  - {r$run_date}{flag}: drop registry entry + rm {basename(p)} ({fst})"))
}

if (length(best_targets) && !force_best) {
  bd <- vapply(best_targets, function(r) as.character(r$run_date), character(1))
  stop(glue("Target(s) flagged best=TRUE: {paste(bd, collapse = ', ')}. ",
            "Refusing without --force-best (this would leave no best model)."))
}

if (!apply_changes) {
  message("\nDRY RUN — nothing removed. Re-run with --apply to execute.")
  quit(save = "no", status = 0)
}

# ---- apply: registry first (atomic), then the files ----
res <- remove_malaria_model_runs(registry_path, present, allow_best = force_best)
message(glue("\nRemoved {length(res$removed)} registry entr(y/ies)."))
for (r in res$removed) {
  p <- rdata_path_for(r)
  if (file.exists(p)) {
    ok <- file.remove(p)
    message(glue("  {if (ok) 'rm' else 'FAILED to rm'}: {p}"))
  } else {
    message(glue("  (file already absent): {p}"))
  }
}
message("Done.")
