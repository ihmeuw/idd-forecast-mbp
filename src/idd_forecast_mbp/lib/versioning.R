# versioning.R — output-node versioning for R pipeline stages.
#
# R port of the symlink mechanics in lib/versioning.py. Each artifact directory
# has the structure:
#   {artifact_root}/{RUN_DATE}/         <- output files for this run
#   {artifact_root}/current -> RUN_DATE <- symlink, repointed after a successful run
#
# Path semantics are kept identical to the Python helpers so a node written by R
# is read the same way by either language. The forecast stage writes from R, so
# this is the R-side equivalent of finalize_artifact().

# Write path for an artifact: artifact_root/run_date. Caller creates it with
# dir.create(..., recursive = TRUE).
artifact_write_path <- function(artifact_root, run_date) {
  file.path(artifact_root, run_date)
}

# Read path for an artifact via its current/ symlink.
artifact_read_path <- function(artifact_root) {
  file.path(artifact_root, "current")
}

# Repoint {artifact_root}/current at run_date after a successful run. Mirrors
# lib/versioning.py::finalize_artifact:
#   - errors if the dated run directory does not exist,
#   - only replaces an existing *symlink* (a real directory at current/ is left
#     for file.symlink to fail on loudly, rather than being silently removed),
#   - writes a *relative* symlink target (run_date), matching Path.symlink_to().
finalize_artifact <- function(artifact_root, run_date) {
  run_dir <- file.path(artifact_root, run_date)
  if (!dir.exists(run_dir)) {
    stop(sprintf(
      "Run directory does not exist: %s\nDid the stage complete successfully?",
      run_dir
    ))
  }
  current <- file.path(artifact_root, "current")
  if (nzchar(Sys.readlink(current))) unlink(current)  # replace an existing symlink only
  ok <- file.symlink(run_date, current)               # relative target, matches Python
  if (!ok) stop(sprintf("Failed to create symlink: %s -> %s", current, run_date))
  message(sprintf("  %s -> %s", current, run_date))
  invisible(current)
}
