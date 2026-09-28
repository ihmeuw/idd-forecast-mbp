"""Tests for idd_forecast_mbp.select.gate (the two-button gate as functions)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.select import gate
from idd_forecast_mbp.select import rank as rk
from idd_forecast_mbp.select.malaria_spec_design import build_universe
from tests.select.synthetic import ANCHOR_SPEC, subset_with_downset, synthetic_summary

if TYPE_CHECKING:
    from idd_tools.model_selection.space import ModelUniverse

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "reports" / "model_selection" / "malaria_selection_config.yaml"


@pytest.fixture(scope="module")
def universe() -> ModelUniverse:
    return build_universe()


@pytest.fixture
def run_dir(tmp_path: Path, universe: ModelUniverse) -> Path:
    """A run dir holding a synthetic summary where the anchor spec wins every criterion."""
    d = tmp_path / "run"
    d.mkdir()
    specs = subset_with_downset(universe, ANCHOR_SPEC)
    synthetic_summary(universe, specs, best=ANCHOR_SPEC).to_parquet(
        d / "selection_summary.parquet", index=False
    )
    pd.DataFrame({"spec_index": specs}).to_parquet(
        d / "spec_table.parquet", index=False
    )
    return d


@pytest.fixture
def session(run_dir: Path) -> gate.GateSession:
    return gate.GateSession.open(CONFIG, run_dir)


# ------------------------------------------------------------------ session
def test_open_resolves_the_run_dir_and_loads_the_summary(
    session: gate.GateSession, run_dir: Path
) -> None:
    assert session.run_dir == run_dir
    assert session.result_path == run_dir / rk.RESULT_FILE
    assert len(session.summary) == len(pd.read_parquet(run_dir / "spec_table.parquet"))
    assert session.config.cause == "malaria"


def test_open_without_override_uses_the_config_run_dir_under_root(
    tmp_path: Path, run_dir: Path
) -> None:
    cfg = rk.load_config(CONFIG)
    root = tmp_path / "stage"
    target = root / cfg.run_dir
    target.mkdir(parents=True)
    (run_dir / "selection_summary.parquet").rename(target / "selection_summary.parquet")
    s = gate.GateSession.open(CONFIG, root=root)
    assert s.run_dir == target


def test_params_override_only_the_named_keys(session: gate.GateSession) -> None:
    base = session.config.rank
    p = session.params(tolerance_value=0.0, profile_top_n=3)
    assert p.tolerance_value == 0.0
    assert p.profile_top_n == 3
    assert p.tau_prune_threshold == base.tau_prune_threshold
    assert p.focus_metric == base.focus_metric
    assert session.params() == base


def test_params_refuse_a_key_the_gate_may_not_change(session: gate.GateSession) -> None:
    with pytest.raises(ValueError, match="may not change"):
        session.params(metric_space="logit")


def test_params_still_validate_like_rank_params(session: gate.GateSession) -> None:
    with pytest.raises(ValueError, match="focus_metric"):
        session.params(focus_metric="consensus_rank")


def test_rerank_zero_tolerance_returns_the_anchor(session: gate.GateSession) -> None:
    res = session.rerank(tolerance_value=0.0)
    assert int(res.pick["spec_index"]) == int(res.anchor["spec_index"]) == ANCHOR_SPEC
    assert res.params.tolerance_value == 0.0


def test_rerank_large_tolerance_picks_something_simpler(
    session: gate.GateSession,
) -> None:
    res = session.rerank(tolerance_value=50.0)
    assert int(res.pick["n_terms"]) <= int(res.anchor["n_terms"])
    assert int(res.pick["spec_index"]) in set(res.candidates["spec_index"].astype(int))


def test_record_pick_writes_recorded_status_with_the_overrides(
    session: gate.GateSession, run_dir: Path
) -> None:
    res = session.rerank(tolerance_value=0.0, profile_top_n=4)
    paths = session.record_pick(res, render=False)
    assert set(paths) == {"result", "ranking", "candidates"}
    rec = rk.read_result(run_dir)
    assert rec["status"] == gate.RECORDED_STATUS
    assert rec["rank"]["tolerance"]["value"] == 0.0
    assert rec["rank"]["profile_top_n"] == 4
    assert int(rec["pick"]["spec_index"]) == ANCHOR_SPEC
    assert rec["spec_table_fingerprint"].startswith("sha256:")


def test_record_pick_renders_when_asked(
    session: gate.GateSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, Path, Path | None]] = []

    def fake_render(
        qmd: str | Path,
        run_dir: str | Path,
        *,
        quarto: Path | None = None,
        output_name: str = "report.html",
    ) -> Path:
        calls.append((Path(qmd), Path(run_dir), quarto))
        return Path(run_dir) / output_name

    monkeypatch.setattr(gate, "render_report", fake_render)
    res = session.rerank()
    paths = session.record_pick(res, quarto=tmp_path / "quarto")
    assert paths["report"] == session.run_dir / "report.html"
    assert calls == [(gate.QMD, session.run_dir, tmp_path / "quarto")]


# ------------------------------------------------------------------ the final fit
def test_fit_command_names_config_run_dir_image_and_shell_and_never_freezes() -> None:
    argv = gate.fit_command(
        "cfg.yaml", "/runs/x", r_image="img.img", r_shell="exec.sh", python="py"
    )
    assert argv[:2] == ["py", str(gate.FIT_LAUNCHER)]
    pairs = dict(zip(argv[2::2], argv[3::2], strict=True))
    assert pairs == {
        "--config": "cfg.yaml",
        "--run-dir": "/runs/x",
        "--r-image": "img.img",
        "--r-shell": "exec.sh",
    }
    assert not {"--freeze", "--current", "--label"} & set(argv)


def test_fit_command_passes_optional_inputs_through() -> None:
    argv = gate.fit_command(
        "c",
        "r",
        r_image="i",
        r_shell="s",
        past_inputs="/p.parquet",
        prep_script="/prep.R",
    )
    assert argv[-4:] == ["--past-inputs", "/p.parquet", "--prep-script", "/prep.R"]


def _proc(returncode: int) -> subprocess.CompletedProcess[Any]:
    return subprocess.CompletedProcess(args=[], returncode=returncode)


def test_launch_fit_returns_the_process_on_success() -> None:
    seen: list[tuple[list[str], bool]] = []

    def runner(argv: list[str], *, check: bool) -> subprocess.CompletedProcess[Any]:
        seen.append((argv, check))
        return _proc(0)

    proc = gate.launch_fit(["a", "b"], runner=runner)
    assert proc.returncode == 0
    assert seen == [(["a", "b"], False)]


def test_launch_fit_refuses_a_failed_fit() -> None:
    def runner(argv: list[str], *, check: bool) -> subprocess.CompletedProcess[Any]:
        del argv, check
        return _proc(3)

    with pytest.raises(RuntimeError, match="exit code 3"):
        gate.launch_fit(["x"], runner=runner)


# ------------------------------------------------------------------ fit status / flag best
def _write_fit(fit_dir: Path, result_file: str | list[str] | None) -> None:
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / mbpc.MAL_MODELS_RDATA).write_bytes(b"rdata")
    record: dict[str, Any] = {"models": ["pfpr_mod"]}
    if result_file is not None:
        record["selection"] = {"result_file": result_file}
    (fit_dir / mbpc.MAL_MODELS_RUN_JSON).write_text(json.dumps(record))


def test_fit_status_none_when_the_node_is_empty_or_missing(tmp_path: Path) -> None:
    assert gate.fit_status(tmp_path / "absent", tmp_path / "r.json").location == "none"
    (tmp_path / "node").mkdir()
    assert gate.fit_status(tmp_path / "node", tmp_path / "r.json").location == "none"


def test_fit_status_working_match_and_mismatch(tmp_path: Path) -> None:
    node, result = tmp_path / "node", tmp_path / "run" / "selection_result.json"
    _write_fit(node / "working", str(result))
    st = gate.fit_status(node, result)
    assert (st.location, st.matches, st.snapshot) == ("working", True, None)
    assert st.path == node / "working"

    _write_fit(node / "working", str(tmp_path / "other.json"))
    st = gate.fit_status(node, result)
    assert (st.location, st.matches) == ("working", False)


def test_fit_status_ignores_working_without_both_outputs(tmp_path: Path) -> None:
    node, result = tmp_path / "node", tmp_path / "r.json"
    (node / "working").mkdir(parents=True)
    (node / "working" / mbpc.MAL_MODELS_RUN_JSON).write_text(
        json.dumps({"selection": {"result_file": str(result)}})
    )
    assert gate.fit_status(node, result).location == "none"


@dataclass
class _Rec:
    version: str


def test_fit_status_finds_the_newest_matching_snapshot_and_unboxes_jsonlite_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node, result = tmp_path / "node", tmp_path / "r.json"
    _write_fit(node / "20260101", [str(result)])
    _write_fit(node / "20260102", str(tmp_path / "other.json"))
    (node / "current").symlink_to("20260102")
    (node / "registry.json").write_text("[]")

    def fake_registry(node_path: str | Path) -> list[_Rec]:
        del node_path
        return [_Rec("20260101")]

    monkeypatch.setattr(gate, "read_registry", fake_registry)
    st = gate.fit_status(node, result)
    assert (st.location, st.snapshot, st.registered, st.matches) == (
        "snapshot",
        "20260101",
        True,
        True,
    )
    assert st.path == node / "20260101"


@dataclass
class _Frozen:
    snapshot: str


def _never(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    pytest.fail("this action must not run here")


def test_flag_best_freezes_a_working_fit_as_current(tmp_path: Path) -> None:
    node, result = tmp_path / "node", tmp_path / "r.json"
    _write_fit(node / "working", str(result))
    calls: list[tuple[Path, str, str | None, bool]] = []

    def freezer(
        node_path: str | Path, description: str, *, label: str | None, current: bool
    ) -> _Frozen:
        calls.append((Path(node_path), description, label, current))
        return _Frozen("20260928")

    name = gate.flag_best(
        node,
        result,
        description="why",
        label="best_2026",
        freezer=freezer,
        promoter=_never,
    )
    assert name == "20260928"
    assert calls == [(node, "why", "best_2026", True)]


def test_flag_best_promotes_and_labels_a_frozen_snapshot(tmp_path: Path) -> None:
    node, result = tmp_path / "node", tmp_path / "r.json"
    _write_fit(node / "20260901", str(result))
    promoted: list[tuple[Path, str]] = []
    labelled: list[tuple[Path, str, str]] = []

    def promoter(node_path: str | Path, snapshot: str) -> None:
        promoted.append((Path(node_path), snapshot))

    def labeller(node_path: str | Path, snapshot: str, name: str) -> None:
        labelled.append((Path(node_path), snapshot, name))

    name = gate.flag_best(
        node,
        result,
        description="why",
        label="best",
        freezer=_never,
        promoter=promoter,
        labeller=labeller,
    )
    assert name == "20260901"
    assert promoted == [(node, "20260901")]
    assert labelled == [(node, "20260901", "best")]


def test_flag_best_refuses_when_no_fit_points_at_this_result(tmp_path: Path) -> None:
    node, result = tmp_path / "node", tmp_path / "r.json"
    _write_fit(node / "working", str(tmp_path / "other.json"))
    with pytest.raises(RuntimeError, match="no fit for"):
        gate.flag_best(node, result, description="why", freezer=_never)
