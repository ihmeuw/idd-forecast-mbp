"""Tests for lib/versioning.py, the adapter over idd_tools.versions.

Nodes live in tmp_path. GIT_CEILING_DIRECTORIES keeps tmp nodes outside any repo;
finish_stage records provenance against the repo itself (REPO_DIR), read-only.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from idd_tools.versions import VersionsOptions

from idd_forecast_mbp.lib import versioning as v


def _restore_write(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.lstat().st_mode) | 0o200)


@pytest.fixture(autouse=True)
def _outside_any_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", os.path.realpath(tmp_path))
    for key in list(os.environ):
        if key.startswith(v.ENV_PREFIX):
            monkeypatch.delenv(key)


@pytest.fixture
def node(tmp_path):
    n = tmp_path / "node"
    n.mkdir()
    yield n
    _restore_write(n)


# ------------------------------------------------------------------ paths
def test_write_path_creates_working(node):
    assert v.write_path(node) == node / "working"
    assert (node / "working").is_dir()


def test_scratch_path_creates_label_dir(node):
    assert v.scratch_path(node, "probe") == node / "scratch" / "probe"
    assert (node / "scratch" / "probe").is_dir()


def test_read_path_requires_current(node):
    with pytest.raises(FileNotFoundError, match="No current snapshot"):
        v.read_path(node)
    (node / "20260101").mkdir()
    (node / "current").symlink_to("20260101")
    assert v.read_path(node) == node / "current"


def test_assert_artifact_ready_prefers_current_then_working(node):
    with pytest.raises(FileNotFoundError, match="Artifact not ready"):
        v.assert_artifact_ready(node)
    (node / "working").mkdir()
    with pytest.raises(FileNotFoundError):
        v.assert_artifact_ready(node)  # empty working/ does not count
    (node / "working" / "a.parquet").write_text("x")
    with pytest.warns(UserWarning, match="reading working/"):
        assert v.assert_artifact_ready(node) == node / "working"
    (node / "20260101").mkdir()
    (node / "current").symlink_to("20260101")
    assert v.assert_artifact_ready(node) == node / "current"


# ------------------------------------------------------------------ environment options
def test_versions_from_env_defaults_to_nothing():
    opts = v.versions_from_env({})
    assert opts == VersionsOptions()
    assert not opts.finishes


def test_versions_from_env_current_implies_freeze():
    opts = v.versions_from_env(
        {
            "IDD_VERSIONS_CURRENT": "1",
            "IDD_VERSIONS_DESCRIPTION": "stage 02",
            "IDD_VERSIONS_LABEL": "x",
        }
    )
    assert opts.current
    assert opts.freeze
    assert opts.description == "stage 02"
    assert opts.label == "x"


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"IDD_VERSIONS_CURRENT": "maybe"}, "must be a boolean"),
        ({"IDD_VERSIONS_CURRENT": "1"}, "need --description"),
        (
            {
                "IDD_VERSIONS_FREEZE": "yes",
                "IDD_VERSIONS_DESCRIPTION": "d",
                "IDD_VERSIONS_SCRATCH": "probe",
            },
            "never frozen",
        ),
        ({"IDD_VERSIONS_TAG": "1"}, "--tag needs"),
    ],
)
def test_versions_from_env_refuses_bad_launches(env, match):
    with pytest.raises(ValueError, match=match):
        v.versions_from_env(env)


def test_stage_target_honours_scratch(node, monkeypatch):
    monkeypatch.setenv("IDD_VERSIONS_SCRATCH", "probe")
    assert v.stage_target(node) == node / "scratch" / "probe"
    monkeypatch.delenv("IDD_VERSIONS_SCRATCH")
    assert v.stage_target(node) == node / "working"


# ------------------------------------------------------------------ finish
def test_finish_stage_does_nothing_unless_asked(node):
    (v.write_path(node) / "a.txt").write_text("x")
    assert v.finish_stage(node) is None
    assert not (node / "registry.json").exists()
    assert not (node / "current").exists()


def test_finish_stage_current_freezes_and_promotes(node, monkeypatch):
    (v.write_path(node) / "a.txt").write_text("x")
    monkeypatch.setenv("IDD_VERSIONS_CURRENT", "1")
    monkeypatch.setenv("IDD_VERSIONS_DESCRIPTION", "test stage")
    res = v.finish_stage(node)
    assert res is not None
    assert res.promoted
    assert not res.reused
    assert (node / res.snapshot / "a.txt").is_file()
    assert (node / "current").resolve() == (node / res.snapshot).resolve()
    assert (node / "working").is_dir()
    assert not any((node / "working").iterdir())
    assert res.record.git_commit  # provenance recorded against the repo


def test_finish_stage_identical_rerun_reuses_snapshot(node):
    (v.write_path(node) / "a.txt").write_text("same")
    opts = VersionsOptions(current=True, description="first")
    first = v.finish_stage(node, opts)
    (v.write_path(node) / "a.txt").write_text("same")
    second = v.finish_stage(
        node, VersionsOptions(current=True, description="again", label="keep")
    )
    assert first is not None
    assert second is not None
    assert second.reused
    assert second.snapshot == first.snapshot
    assert (node / "keep").resolve() == (node / first.snapshot).resolve()


def test_finish_stage_refuses_empty_working(node):
    v.write_path(node)
    with pytest.raises(Exception, match=r"empty|nothing"):
        v.finish_stage(node, VersionsOptions(freeze=True, description="d"))
