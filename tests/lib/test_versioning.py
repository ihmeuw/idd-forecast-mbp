"""
Tests for lib/versioning.py

Uses tmp_path to simulate artifact directories — no real pipeline paths.
The artifact-level API takes a Path directly, so no mbpc patching is needed
for most functions.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from idd_forecast_mbp.lib.versioning import (
    finalize_artifact,
    finalize_all_artifacts,
    tag_artifact,
    list_runs,
    list_unlinked_runs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def artifact_root(tmp_path):
    """Simulate an artifact directory (e.g. _A02_HIERARCHY)."""
    root = tmp_path / "02-processed_data" / "hierarchy" / "lsae_1285"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def artifact_with_run(artifact_root):
    """Artifact root with a dated run directory."""
    run_dir = artifact_root / "20260405"
    run_dir.mkdir()
    return artifact_root


# ---------------------------------------------------------------------------
# finalize_artifact
# ---------------------------------------------------------------------------

def test_finalize_artifact_creates_symlink(artifact_with_run):
    finalize_artifact(artifact_with_run, run_date="20260405")

    symlink = artifact_with_run / "current"
    assert symlink.is_symlink()
    assert symlink.resolve().name == "20260405"


def test_finalize_artifact_updates_existing_symlink(artifact_with_run):
    run2 = artifact_with_run / "20260406"
    run2.mkdir()

    finalize_artifact(artifact_with_run, run_date="20260405")
    finalize_artifact(artifact_with_run, run_date="20260406")

    symlink = artifact_with_run / "current"
    assert symlink.resolve().name == "20260406"


def test_finalize_artifact_missing_run_raises(artifact_root):
    with pytest.raises(FileNotFoundError, match="Run directory does not exist"):
        finalize_artifact(artifact_root, run_date="99991231")


# ---------------------------------------------------------------------------
# tag_artifact
# ---------------------------------------------------------------------------

def test_tag_artifact_creates_named_symlink(artifact_with_run):
    tag_artifact(artifact_with_run, "pre_revision_1", run_date="20260405")

    tag = artifact_with_run / "pre_revision_1"
    assert tag.is_symlink()
    assert tag.resolve().name == "20260405"


def test_tag_artifact_missing_run_raises(artifact_root):
    with pytest.raises(FileNotFoundError):
        tag_artifact(artifact_root, "mytag", run_date="99991231")


def test_tag_artifact_existing_tag_raises(artifact_with_run):
    tag_artifact(artifact_with_run, "v1", run_date="20260405")
    with pytest.raises(FileExistsError, match="Tag already exists"):
        tag_artifact(artifact_with_run, "v1", run_date="20260405")


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

def test_list_runs_empty_artifact(artifact_root):
    result = list_runs(artifact_root)
    assert result == []


def test_list_runs_returns_run_dirs(artifact_with_run):
    result = list_runs(artifact_with_run)
    run_dates = [r['run_date'] for r in result]
    assert "20260405" in run_dates


def test_list_runs_shows_current_symlink(artifact_with_run):
    finalize_artifact(artifact_with_run, run_date="20260405")
    result = list_runs(artifact_with_run)

    current_runs = [r for r in result if r['is_current']]
    assert len(current_runs) == 1
    assert current_runs[0]['run_date'] == "20260405"


def test_list_runs_nonexistent_artifact(tmp_path):
    result = list_runs(tmp_path / "99-nonexistent")
    assert result == []


# ---------------------------------------------------------------------------
# list_unlinked_runs
# ---------------------------------------------------------------------------

def test_list_unlinked_runs_no_symlinks(artifact_with_run):
    """Run directory with no symlink pointing to it is unlinked."""
    result = list_unlinked_runs(artifact_with_run)
    assert len(result) == 1


def test_list_unlinked_runs_empty_after_finalize(artifact_with_run):
    """After finalize_artifact, the run has current/ pointing to it — no longer unlinked."""
    finalize_artifact(artifact_with_run, run_date="20260405")
    result = list_unlinked_runs(artifact_with_run)
    assert result == []


# ---------------------------------------------------------------------------
# finalize_all_artifacts
# ---------------------------------------------------------------------------

def test_finalize_all_artifacts_skips_missing(tmp_path):
    """finalize_all_artifacts skips artifacts with no run dir — no error."""
    fake_root = tmp_path / "02-processed_data" / "hierarchy" / "lsae_1285"
    (fake_root / "20260405").mkdir(parents=True)

    artifact_roots = [fake_root]
    with patch('idd_forecast_mbp.lib.versioning.mbpc') as mock_mbpc:
        mock_mbpc.MODEL_ROOT = tmp_path
        mock_mbpc.RUN_DATE = "20260405"
        mock_mbpc._A02_HIERARCHY = fake_root
        mock_mbpc._A02_POPULATION = tmp_path / "missing_artifact"
        mock_mbpc._A02_DAH = tmp_path / "missing_artifact2"
        mock_mbpc._A02_MAL_RAKED_AA = tmp_path / "missing_artifact3"
        mock_mbpc._A02_MAL_RAKED_AS = tmp_path / "missing_artifact4"
        mock_mbpc._A02_DEN_RAKED_AA = tmp_path / "missing_artifact5"
        mock_mbpc._A02_DEN_RAKED_AS = tmp_path / "missing_artifact6"
        mock_mbpc._A03_MAL_MODELING = tmp_path / "missing_artifact7"
        mock_mbpc._A03_DEN_MODELING = tmp_path / "missing_artifact8"
        mock_mbpc._A03_COV_MEANS = tmp_path / "missing_artifact9"
        finalize_all_artifacts(run_date="20260405")

    symlink = fake_root / "current"
    assert symlink.is_symlink()
    assert not (tmp_path / "missing_artifact" / "current").exists()
