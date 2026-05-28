"""Enforce that every pipeline stage script that writes to a versioned artifact
also calls finalize_artifact().

A script "writes to a versioned artifact" if it imports any *_WRITE_PATH or
FORECASTING_DATA_PATH constant from mbpc. Such a script MUST also import
finalize_artifact — either directly or via finalize_all_artifacts.

Parallel worker scripts (which process one chunk and are called many times) are
explicitly exempted; the orchestrator is responsible for calling finalize_artifact
after all workers complete.
"""
import ast
from pathlib import Path

import pytest

STAGE_DIRS = [
    "02_data_prep",
    "04_forecasting",
    "05_aggregation",
    "06_upload",
]

# Scripts that process a single chunk in a parallel job array.
# The orchestrator (not the worker) is responsible for finalize_artifact.
PARALLEL_WORKERS = {
    "forecasted_draw_specific_malaria_dataframes.py",
    "forecasted_draw_specific_dengue_dataframes.py",
    "as_malaria_fractions.py",
    "rake_dengue.py",
    "as_dengue_shifts.py",
    "cause_as_aggregation_by_draw.py",
    "cause_as_aggregation_by_draw_raked.py",
    "make_population_hold_variables_by_draw.py",
    "create_as_dalys_by_draw_raked_parallel.py",
}

# Scripts with known deferred finalize_artifact work — tracked in DECISIONS.md.
# Remove entries here as they are fixed.
DEFERRED = {
    "01_cause_as_aggregation_by_draw_parallel.py",  # stage 05 orchestrator
    "create_and_combine_as_and_aa_draws.py",        # stage 06
    "fhs_upload_as_draws.py",                       # stage 06
    "make_full_means_ds.py",                        # stage 06
}

WRITE_PATH_MARKERS = {
    "WRITE_PATH",
    "FORECASTING_DATA_PATH",
    "UPLOAD_DATA_PATH",
    "FIGURES_PATH",
    "VISUALIZATION_PATH",
    "MANUSCRIPT_PATH",
}

FINALIZE_NAMES = {"finalize_artifact", "finalize_all_artifacts"}

SRC_ROOT = Path(__file__).parent.parent / "src" / "idd_forecast_mbp"


def _imports_write_path(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in WRITE_PATH_MARKERS:
            return True
        if isinstance(node, ast.Name) and node.id in WRITE_PATH_MARKERS:
            return True
    return False


def _imports_finalize(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            if any(n in FINALIZE_NAMES for n in names):
                return True
    return False


def collect_offending_scripts():
    offenders = []
    for stage in STAGE_DIRS:
        stage_dir = SRC_ROOT / stage
        if not stage_dir.exists():
            continue
        for path in sorted(stage_dir.glob("*.py")):
            if path.name.startswith("test_") or path.name in PARALLEL_WORKERS or path.name in DEFERRED:
                continue
            source = path.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            if _imports_write_path(tree) and not _imports_finalize(tree):
                offenders.append(path.relative_to(SRC_ROOT))
    return offenders


OFFENDERS = collect_offending_scripts()


@pytest.mark.parametrize("script", OFFENDERS, ids=str)
def test_script_calls_finalize_artifact(script):
    pytest.fail(
        f"{script} writes to a versioned artifact but does not import "
        "finalize_artifact or finalize_all_artifacts.\n"
        "Add: from idd_forecast_mbp.lib.versioning import finalize_artifact\n"
        "and call finalize_artifact(mbpc._AXX_*) at the end of main()."
    )


def test_no_offending_scripts():
    """Single summary test — fails if any scripts are missing finalize_artifact."""
    if OFFENDERS:
        names = "\n  ".join(str(p) for p in OFFENDERS)
        pytest.fail(
            f"{len(OFFENDERS)} script(s) write to versioned artifacts without "
            f"calling finalize_artifact:\n  {names}"
        )
