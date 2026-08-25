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

# Retired scripts, kept for reference but not run by the pipeline. The
# finalize_artifact rule governs scripts that actually execute a stage, so
# retired copies are out of its scope. Naming conventions in this repo:
# OLD_/alt_ prefixes and an _old suffix.
RETIRED_PREFIXES = ("OLD_", "alt_")
RETIRED_SUFFIXES = ("_old.py",)


def _is_retired(name: str) -> bool:
    return name.startswith(RETIRED_PREFIXES) or name.endswith(RETIRED_SUFFIXES)


# Any constant whose name ends in _WRITE_PATH, plus these stage-level paths that
# do not follow that suffix. Matched by SUFFIX, not equality: the previous exact-
# match set contained the bare string "WRITE_PATH", which matches no real constant
# (they are all <NODE>_WRITE_PATH), so the rule silently never fired for them.
WRITE_PATH_SUFFIX = "_WRITE_PATH"
WRITE_PATH_MARKERS = {
    "FORECASTING_DATA_PATH",
    "UPLOAD_DATA_PATH",
    "FIGURES_PATH",
    "VISUALIZATION_PATH",
    "MANUSCRIPT_PATH",
}


def _is_write_path_name(name: str) -> bool:
    return name.endswith(WRITE_PATH_SUFFIX) or name in WRITE_PATH_MARKERS

FINALIZE_NAMES = {"finalize_artifact", "finalize_all_artifacts"}

SRC_ROOT = Path(__file__).parent.parent / "src" / "idd_forecast_mbp"


def _module_level_assignments(tree: ast.Module) -> set[str]:
    """Names the module defines for itself at module level."""
    assigned = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
    return assigned


def _imports_write_path(tree: ast.Module) -> bool:
    """True if the script uses one of the pipeline's versioned write-path constants.

    Attribute access (`mbpc.X_WRITE_PATH`) is unambiguous. A BARE name is only
    the constant if the module did not define that name itself: several stage
    scripts assign their own local `UPLOAD_DATA_PATH = rfc.MODEL_ROOT / ...`,
    which is a hand-built path into a non-versioned subtree, not the versioned
    constant. Counting those produced false positives that this rule then
    demanded finalize_artifact for -- against an artifact root that has no
    dated-run/current convention at all.
    """
    shadowed = _module_level_assignments(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_write_path_name(node.attr):
            return True
        if (isinstance(node, ast.Name) and _is_write_path_name(node.id)
                and node.id not in shadowed):
            return True
    return False


def _imports_finalize(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            if any(n in FINALIZE_NAMES for n in names):
                return True
    return False


def collect_offending_scripts(src_root: Path = SRC_ROOT):
    offenders = []
    for stage in STAGE_DIRS:
        stage_dir = src_root / stage
        if not stage_dir.exists():
            continue
        for path in sorted(stage_dir.glob("*.py")):
            if (path.name.startswith("test_") or _is_retired(path.name)
                    or path.name in PARALLEL_WORKERS or path.name in DEFERRED):
                continue
            source = path.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            if _imports_write_path(tree) and not _imports_finalize(tree):
                offenders.append(path.relative_to(src_root))
    return offenders


OFFENDERS = collect_offending_scripts()


@pytest.mark.parametrize("script", OFFENDERS, ids=str)
def test_script_calls_finalize_artifact(script):
    pytest.fail(  # pragma: no cover - only runs when an offender exists
        f"{script} writes to a versioned artifact but does not import "
        "finalize_artifact or finalize_all_artifacts.\n"
        "Add: from idd_forecast_mbp.lib.versioning import finalize_artifact\n"
        "and call finalize_artifact(mbpc._AXX_*) at the end of main()."
    )


def test_no_offending_scripts():
    """Single summary test — fails if any scripts are missing finalize_artifact."""
    if OFFENDERS:  # pragma: no cover - only runs when an offender exists
        names = "\n  ".join(str(p) for p in OFFENDERS)
        pytest.fail(
            f"{len(OFFENDERS)} script(s) write to versioned artifacts without "
            f"calling finalize_artifact:\n  {names}"
        )


# ---------------------------------------------------------------------------
# Unit tests for the detector. These exist because the rule above was silently
# broken twice: it scanned retired scripts, and its marker set held the bare
# string "WRITE_PATH" matched by equality, so no real <NODE>_WRITE_PATH constant
# ever matched. The rule passed while checking almost nothing.
# ---------------------------------------------------------------------------
def _tree(source: str) -> ast.Module:
    return ast.parse(source)


@pytest.mark.parametrize("name, expected", [
    ("OLD_forecast.py", True),
    ("alt_forecast.py", True),
    ("forecast_old.py", True),
    ("forecast.py", False),
    ("02b_full_population.py", False),
])
def test_is_retired(name, expected):
    assert _is_retired(name) is expected


@pytest.mark.parametrize("name, expected", [
    ("POPULATION_WRITE_PATH", True),
    ("MAL_VACCINE_COHORTS_WRITE_PATH", True),
    ("UPLOAD_DATA_PATH", True),
    ("FORECASTING_DATA_PATH", True),
    ("WRITE_PATH", False),          # the bare marker that used to be the whole rule
    ("POPULATION_READ_PATH", False),
    ("MODEL_ROOT", False),
])
def test_is_write_path_name(name, expected):
    assert _is_write_path_name(name) is expected


def test_module_level_assignments_finds_plain_and_annotated():
    tree = _tree("A = 1\nB: int = 2\n\ndef f():\n    C = 3\n")
    assert _module_level_assignments(tree) == {"A", "B"}


def test_detects_attribute_access_of_write_path():
    assert _imports_write_path(_tree("import x as mbpc\np = mbpc.POPULATION_WRITE_PATH\n"))


def test_detects_bare_imported_write_path():
    """The name is imported, not defined locally, so it IS the constant."""
    assert _imports_write_path(
        _tree("from idd_forecast_mbp.constants import HIERARCHY_WRITE_PATH\n"
              "p = HIERARCHY_WRITE_PATH / 'a.parquet'\n")
    )


def test_ignores_locally_assigned_lookalike():
    """A script building its own path into a non-versioned subtree is not using
    the versioned constant, even though the variable shares its name."""
    assert not _imports_write_path(
        _tree("import x as rfc\nUPLOAD_DATA_PATH = rfc.MODEL_ROOT / '05-upload_data'\n"
              "p = f'{UPLOAD_DATA_PATH}/folders'\n")
    )


def test_no_write_path_at_all():
    assert not _imports_write_path(_tree("import pandas as pd\nx = pd.DataFrame()\n"))


@pytest.mark.parametrize("source, expected", [
    ("from idd_forecast_mbp.lib.versioning import finalize_artifact\n", True),
    ("from idd_forecast_mbp.lib.versioning import finalize_all_artifacts\n", True),
    ("import finalize_artifact\n", True),
    ("from idd_forecast_mbp.lib.versioning import something_else\n", False),
    ("x = 1\n", False),
])
def test_imports_finalize(source, expected):
    assert _imports_finalize(_tree(source)) is expected


def _stage_file(root: Path, name: str, source: str) -> None:
    stage = root / STAGE_DIRS[0]
    stage.mkdir(parents=True, exist_ok=True)
    (stage / name).write_text(source)


WRITER = "import x as mbpc\np = mbpc.POPULATION_WRITE_PATH\n"
WRITER_WITH_FINALIZE = ("from idd_forecast_mbp.lib.versioning import finalize_artifact\n"
                        "import x as mbpc\np = mbpc.POPULATION_WRITE_PATH\n")


def test_collector_flags_a_writer_without_finalize(tmp_path):
    _stage_file(tmp_path, "writes_without_finalize.py", WRITER)
    assert [str(o) for o in collect_offending_scripts(tmp_path)] == [
        f"{STAGE_DIRS[0]}/writes_without_finalize.py"
    ]


def test_collector_accepts_a_writer_with_finalize(tmp_path):
    _stage_file(tmp_path, "compliant.py", WRITER_WITH_FINALIZE)
    assert collect_offending_scripts(tmp_path) == []


@pytest.mark.parametrize("name", ["OLD_writer.py", "test_writer.py"])
def test_collector_skips_exempt_names(tmp_path, name):
    _stage_file(tmp_path, name, WRITER)
    assert collect_offending_scripts(tmp_path) == []


def test_collector_skips_parallel_workers(tmp_path):
    _stage_file(tmp_path, sorted(PARALLEL_WORKERS)[0], WRITER)
    assert collect_offending_scripts(tmp_path) == []


def test_collector_tolerates_unparseable_file(tmp_path):
    """A stage script that does not parse is skipped, not fatal -- the rule is
    about versioning discipline, not syntax checking."""
    _stage_file(tmp_path, "broken.py", "def f(:\n")
    _stage_file(tmp_path, "writer.py", WRITER)
    assert [str(o) for o in collect_offending_scripts(tmp_path)] == [
        f"{STAGE_DIRS[0]}/writer.py"
    ]


def test_collector_handles_missing_stage_dir(tmp_path):
    """Only one stage dir exists in the fixture; the others must be skipped."""
    _stage_file(tmp_path, "writer.py", WRITER)
    assert not (tmp_path / STAGE_DIRS[1]).exists()
    assert len(collect_offending_scripts(tmp_path)) == 1
