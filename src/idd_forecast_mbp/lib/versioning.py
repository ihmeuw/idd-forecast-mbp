"""Output versioning utilities.

Each artifact directory has the structure:
    {artifact_root}/{RUN_DATE}/          ← output files for this run
    {artifact_root}/current  -> RUN_DATE ← symlink updated after successful run
    {artifact_root}/first_submission -> ... ← named tag for a deliverable version

After a stage completes successfully, call finalize_artifact() to update the
current/ symlink so downstream scripts read the latest output.

Artifact roots are defined in constants.py as _A02_* and _A03_* variables,
with corresponding WRITE and READ path pairs.

Example — finalizing a full pipeline run:
    from idd_forecast_mbp.lib.versioning import finalize_all_artifacts
    finalize_all_artifacts()
"""

from pathlib import Path
from idd_forecast_mbp import constants as mbpc


def assert_artifact_ready(artifact_root: Path) -> Path:
    """Check that an artifact has readable output and return the resolved path.

    Prefers current/ symlink. Falls back to the most recent dated subdirectory
    if current/ is missing, with a warning. Raises FileNotFoundError if nothing
    exists at all.

    Call at the top of main() for each artifact the script reads from — not at
    module level. Fires only for the artifacts this script actually needs.

    Returns the resolved read path so callers can use it directly if needed.

    Example:
        assert_artifact_ready(mbpc._A02_POPULATION)
        assert_artifact_ready(mbpc._A02_MAL_RAKED_AA)
    """
    import warnings
    current = artifact_root / "current"
    if current.exists():
        return current
    if artifact_root.exists():
        dated = sorted(
            [p for p in artifact_root.iterdir()
             if p.is_dir() and not p.is_symlink() and p.name[:8].isdigit()],
            reverse=True,
        )
        if dated:
            warnings.warn(
                f"No current/ symlink for {artifact_root.name}. "
                f"Falling back to {dated[0].name}. "
                "Run finalize_artifact() after writing to suppress this.",
                stacklevel=2,
            )
            return dated[0]
    raise FileNotFoundError(
        f"\nArtifact not ready: {artifact_root}\n"
        "Run and finalize the upstream stage that produces this artifact first."
    )


def finalize_artifact(artifact_root: Path, run_date: str = mbpc.RUN_DATE) -> None:
    """Update the current/ symlink for an artifact after a successful run.

    Args:
        artifact_root: The artifact directory containing dated run subdirs,
                       e.g. constants._A02_HIERARCHY.
        run_date: The run date string to point current/ at.
    """
    run_dir = artifact_root / run_date
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}\n"
            "Did the stage complete successfully?"
        )
    current = artifact_root / "current"
    if current.is_symlink():
        current.unlink()
    current.symlink_to(run_date)
    print(f"  {current} -> {run_date}")


def finalize_all_artifacts(run_date: str = mbpc.RUN_DATE) -> None:
    """Update current/ symlinks for all standard pipeline artifacts."""
    artifact_roots = [
        mbpc._A02_HIERARCHY,
        mbpc._A02_POPULATION,
        mbpc._A02_DAH,
        mbpc._A02_MED_CONSUMPPC,
        mbpc._A02_MAL_RAKED_AA,
        mbpc._A02_MAL_RAKED_AS,
        mbpc._A02_DEN_RAKED_AA,
        mbpc._A02_DEN_RAKED_AS,
        mbpc._A03_MAL_MODELING,
        mbpc._A03_DEN_MODELING,
        mbpc._A03_MAL_PAST_INPUTS,
        mbpc._A03_DEN_PAST_INPUTS,
    ]
    print(f"Finalizing run {run_date}:")
    for root in artifact_roots:
        run_dir = root / run_date
        if run_dir.exists():
            finalize_artifact(root, run_date)
        else:
            print(f"  skipping {root.relative_to(mbpc.MODEL_ROOT)} (no output found)")


def tag_artifact(artifact_root: Path, tag: str, run_date: str = mbpc.RUN_DATE) -> None:
    """Create a named symlink alongside current/ pointing to a specific run.

    Use this to bookmark deliverable versions, e.g.:
        tag_artifact(mbpc._A02_HIERARCHY, "manuscript_v1")
        tag_artifact(mbpc._A03_MAL_MODELING, "pre_revision_1", run_date="20260315")

    Args:
        artifact_root: The artifact directory.
        tag: Name for the symlink (e.g. "manuscript_v1").
        run_date: The run date string to point the tag at.
    """
    run_dir = artifact_root / run_date
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    tag_path = artifact_root / tag
    if tag_path.exists() or tag_path.is_symlink():
        raise FileExistsError(
            f"Tag already exists: {tag_path}\n"
            "Remove it manually before re-tagging."
        )
    tag_path.symlink_to(run_date)
    print(f"  {tag_path} -> {run_date}")


def list_runs(artifact_root: Path) -> list[dict]:
    """List all dated run directories and symlink status for an artifact.

    Returns a list of dicts with keys: run_date, symlinks, is_current.
    """
    if not artifact_root.exists():
        return []

    symlink_targets: dict[str, list[str]] = {}
    for p in artifact_root.iterdir():
        if p.is_symlink():
            target = p.resolve().name
            symlink_targets.setdefault(target, []).append(p.name)

    runs = []
    for p in sorted(artifact_root.iterdir()):
        if p.is_dir() and not p.is_symlink():
            symlinks = symlink_targets.get(p.name, [])
            runs.append({
                "run_date": p.name,
                "symlinks": symlinks,
                "is_current": "current" in symlinks,
            })
    return runs


def list_unlinked_runs(artifact_root: Path) -> list[Path]:
    """Return dated run directories with no symlink pointing to them."""
    return [
        artifact_root / r["run_date"]
        for r in list_runs(artifact_root)
        if not r["symlinks"]
    ]
