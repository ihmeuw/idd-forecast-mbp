# versioning.R -- R side of the output-node contract (idd_tools.versions).
#
# A stage writes into <node>/working/ (the launcher passes the path down; a worker
# never chooses one) and reads <node>/current/. Nothing here creates a dated
# directory or moves a symlink: freeze / promote happen from Python
# (idd_tools.versions, or a launcher started with --current), or from R through
# idd-tools' own helper:
#
#   source(<idd_tools.versions.r_helper_path()>)     # read_registry, current_version,
#   snap <- finish_run(node, "why", current = TRUE,   # resolve_version, finish_run
#                      cli = Sys.getenv("IDD_VERSIONS_CLI"))
#
# The two helpers below are the R spellings of constants._artifact_write / _artifact_read
# so a node written by R is read the same way by either language.

artifact_write_path <- function(artifact_root) file.path(artifact_root, "working")
artifact_read_path  <- function(artifact_root) file.path(artifact_root, "current")
