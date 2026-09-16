"""Tests for idd_forecast_mbp.select.rank (the notebook's ranking as functions)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from idd_tools.model_selection import config_key, down_set

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.select import rank as rk
from idd_forecast_mbp.select.malaria_spec_design import build_universe

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "reports" / "model_selection" / "malaria_selection_config.yaml"
METRICS = {"oos_pfpr_r": "higher", "oos_pfpr_rmse": "lower", "oos_pfpr_mae": "lower"}
ANCHOR_SPEC = (
    1585  # the fullest config in the 20260727_efs top tier; its down-set is large
)


@pytest.fixture(scope="module")
def universe():
    return build_universe()


@pytest.fixture(scope="module")
def config():
    return rk.load_config(CONFIG)


def _synthetic_summary(
    universe, spec_indices: list[int], best: int, windows=("w1", "w2"), seed=0
) -> pd.DataFrame:
    """A summary frame with random windowed metrics where ``best`` is ideal on every criterion."""
    rng = np.random.default_rng(seed)
    n = len(spec_indices)
    frame = pd.DataFrame({"spec_index": spec_indices})
    for w in windows:
        for m, direction in METRICS.items():
            col = f"{w}__{m}"
            vals = (
                rng.uniform(0.5, 0.9, n)
                if direction == "higher"
                else rng.uniform(0.05, 0.2, n)
            )
            frame[col] = vals
            i = spec_indices.index(best)
            frame.loc[i, col] = (
                vals.max() + 0.05 if direction == "higher" else vals.min() - 0.01
            )
    return frame


def _subset_with_downset(universe, anchor_spec: int, n_extra: int = 15) -> list[int]:
    cfg_of = {i + 1: c for i, c in enumerate(universe.configs)}
    key_to_spec = {config_key(c): i + 1 for i, c in enumerate(universe.configs)}
    ds = [
        key_to_spec[config_key(c)]
        for c in down_set(cfg_of[anchor_spec], universe.space, order="complexity")
    ]
    others = [s for s in range(1, n_extra + 1) if s not in ds and s != anchor_spec]
    return sorted({anchor_spec, *ds[:20], *others})


# ------------------------------------------------------------------ config
def test_load_config_reads_committed_file(config):
    assert config.cause == "malaria"
    assert config.rank.focus_metric in rk.FOCUS_DIRECTION
    assert config.rank.tolerance_rule == "std_fraction"
    assert set(config.fit) >= {"cv_strategy", "gaps", "test_windows"}


def test_load_config_missing_file_refuses(tmp_path):
    with pytest.raises(FileNotFoundError):
        rk.load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "mutation", ["drop_rank_key", "add_rank_key", "add_top_key", "bad_focus"]
)
def test_load_config_refuses_bad_shapes(tmp_path, mutation):
    raw = yaml.safe_load(CONFIG.read_text())
    if mutation == "drop_rank_key":
        del raw["rank"]["profile_top_n"]
    elif mutation == "add_rank_key":
        raw["rank"]["something_else"] = 1
    elif mutation == "add_top_key":
        raw["extra"] = {}
    elif mutation == "bad_focus":
        raw["rank"]["focus_metric"] = "consensus_rank"
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, TypeError), match=r"rank|focus_metric|top-level"):
        rk.load_config(p)


def test_rank_params_have_no_defaults():
    fields = dataclasses.fields(rk.RankParams)
    assert all(
        f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        for f in fields
    )


# ------------------------------------------------------------------ inputs
def test_attach_universe_bridges_axes_and_counts(universe):
    summary = pd.DataFrame({"spec_index": [1, 2, 3, ANCHOR_SPEC]})
    df = rk.attach_universe(summary, universe)
    assert df[list(rk.AXES)].notna().all().all()
    assert {"n_scams", "n_smooths", "n_terms", "formula_text"} <= set(df.columns)
    assert (df["n_scams"] <= df["n_smooths"]).all()
    assert (df["n_smooths"] <= df["n_terms"]).all()


def test_attach_universe_refuses_unknown_spec(universe):
    with pytest.raises(ValueError, match="bridge is incomplete"):
        rk.attach_universe(pd.DataFrame({"spec_index": [1, 10_000]}), universe)


def test_detect_windows_requires_agreement():
    df = pd.DataFrame(
        {"a__oos_pfpr_r": [1], "b__oos_pfpr_r": [1], "a__oos_pfpr_rmse": [1]}
    )
    assert rk.detect_windows(df, ["oos_pfpr_r"]) == ["a", "b"]
    with pytest.raises(ValueError, match="disagree on windows"):
        rk.detect_windows(df, ["oos_pfpr_r", "oos_pfpr_rmse"])


# ------------------------------------------------------------------ ranking
def test_run_selection_zero_tolerance_returns_anchor(universe, config):
    specs = _subset_with_downset(universe, ANCHOR_SPEC)
    summary = _synthetic_summary(universe, specs, best=ANCHOR_SPEC)
    params = rk.rank_params_from_dict(
        {**config.rank.to_dict(), "tolerance": {"rule": "std_fraction", "value": 0.0}}
    )
    res = rk.run_selection(summary, universe, params)
    assert int(res.anchor["spec_index"]) == ANCHOR_SPEC
    assert int(res.pick["spec_index"]) == ANCHOR_SPEC
    assert res.tolerance == 0.0
    assert res.windows == ["w1", "w2"]
    assert len(res.criteria) == 6


def test_run_selection_pick_is_simpler_and_within_band(universe, config):
    specs = _subset_with_downset(universe, ANCHOR_SPEC)
    summary = _synthetic_summary(universe, specs, best=ANCHOR_SPEC)
    res = rk.run_selection(summary, universe, config.rank)
    a, k = res.anchor, res.pick
    order = list(config.rank.complexity_order)
    assert tuple(int(k[c]) for c in order) <= tuple(int(a[c]) for c in order)
    assert float(k["topsis_score"]) >= float(a["topsis_score"]) - res.tolerance - 1e-12
    assert res.tolerance == pytest.approx(res.scored["topsis_score"].std() * 0.25)
    cfg_of = {i + 1: c for i, c in enumerate(universe.configs)}
    ds_keys = {
        config_key(c)
        for c in down_set(cfg_of[ANCHOR_SPEC], universe.space, order="complexity")
    }
    pick_key = config_key({ax: k[ax] for ax in rk.AXES})
    assert pick_key in ds_keys or int(k["spec_index"]) == ANCHOR_SPEC
    assert int(k["spec_index"]) in set(res.candidates["spec_index"])


def test_run_selection_large_tolerance_picks_simplest_in_downset(universe, config):
    specs = _subset_with_downset(universe, ANCHOR_SPEC)
    summary = _synthetic_summary(universe, specs, best=ANCHOR_SPEC)
    params = rk.rank_params_from_dict(
        {**config.rank.to_dict(), "tolerance": {"rule": "std_fraction", "value": 1e6}}
    )
    res = rk.run_selection(summary, universe, params)
    order = list(params.complexity_order)
    simplest = res.candidates.sort_values(
        [*order, "topsis_score"], ascending=[True] * len(order) + [False], kind="stable"
    ).iloc[0]
    assert int(res.pick["spec_index"]) == int(simplest["spec_index"])
    assert int(res.pick["n_scams"]) <= int(res.anchor["n_scams"])


def test_borda_focus_direction_is_lower(universe, config):
    specs = _subset_with_downset(universe, ANCHOR_SPEC)
    summary = _synthetic_summary(universe, specs, best=ANCHOR_SPEC)
    params = rk.rank_params_from_dict(
        {
            **config.rank.to_dict(),
            "focus_metric": "borda_score",
            "tolerance": {"rule": "std_fraction", "value": 0.0},
        }
    )
    res = rk.run_selection(summary, universe, params)
    assert params.focus_direction == "lower"
    assert int(res.anchor["spec_index"]) == ANCHOR_SPEC


# ------------------------------------------------------------------ outputs
def test_write_and_read_result_roundtrip(tmp_path, universe, config):
    specs = _subset_with_downset(universe, ANCHOR_SPEC)
    summary = _synthetic_summary(universe, specs, best=ANCHOR_SPEC)
    res = rk.run_selection(summary, universe, config.rank)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = rk.write_result(res, config, run_dir)
    rec = rk.read_result(run_dir)
    assert rec["status"] == "tentative"
    assert rec["pick"]["spec_index"] == int(res.pick["spec_index"])
    assert rec["anchor"]["spec_index"] == ANCHOR_SPEC
    assert rec["rank"] == config.rank.to_dict()
    assert rec["fit"] == config.fit
    assert rec["n_candidates"] == len(res.candidates)
    ranking = pd.read_parquet(paths["ranking"])
    assert len(ranking) == len(specs)
    assert str(ranking["spec_index"].dtype) == "int32"
    cands = pd.read_parquet(paths["candidates"])
    assert set(cands["spec_index"]) == set(rec["candidates"])
    json.loads(paths["result"].read_text())  # valid JSON on disk


def test_format_pick_mentions_both_specs(universe, config):
    specs = _subset_with_downset(universe, ANCHOR_SPEC)
    res = rk.run_selection(
        _synthetic_summary(universe, specs, best=ANCHOR_SPEC), universe, config.rank
    )
    text = rk.format_pick(res)
    assert f"spec {int(res.anchor['spec_index'])}" in text
    assert f"spec {int(res.pick['spec_index'])}" in text


# ------------------------------------------------------------------ regression on the real run
REAL_RUN = (
    mbpc._MODELING_STAGE  # noqa: SLF001 - stage roots are underscore-named in constants
    / "malaria"
    / "scam_prelim"
    / "lsae_1285"
    / "20260727_efs"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not (REAL_RUN / "selection_summary.parquet").is_file(),
    reason="20260727_efs not mounted",
)
def test_regression_20260727_efs_picks_1486(universe, config):
    """The 2026-07 selection: anchor 1585, tolerance 0.0371 on TOPSIS, 6 candidates, pick 1486."""
    res = rk.run_selection(rk.load_summary(REAL_RUN), universe, config.rank)
    assert res.n_specs == 1620
    assert len(res.windows) == 10
    assert int(res.anchor["spec_index"]) == 1585
    assert int(res.pick["spec_index"]) == 1486
    assert res.tolerance == pytest.approx(0.0371, abs=5e-4)
    assert len(res.candidates) == 6
    assert res.pick["formula_text"].startswith("logit_malaria_pfpr ~")
