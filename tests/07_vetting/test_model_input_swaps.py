"""Unit tests for 07_vetting/model_input_swaps.py (the pure pieces and one toy year)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from idd_forecast_mbp.lib.processing.scenario_gap import Rebuild

SCRIPT_PATH = (
    Path(__file__).parents[2] / "src/idd_forecast_mbp/07_vetting/model_input_swaps.py"
)
_spec = importlib.util.spec_from_file_location("model_input_swaps", SCRIPT_PATH)
mis = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mis
_spec.loader.exec_module(mis)


def test_swap_variants_are_own_each_shared_covariate_and_all():
    v = dict(mis.swap_variants())
    assert v["own"] == ()
    assert v["swap:gdp"] == ("gdp",)
    assert v["swap:suitability"] == ("suit",)
    assert set(v["swap:all"]) == {"gdp", "dah", "suit"}
    assert len(v) == 5


def test_swap_inputs_replaces_only_the_named_keys_and_keeps_own_extras():
    own = {
        "ssp245": {
            "gdp": np.array([1.0]),
            "dah": np.array([2.0]),
            "flood": np.array([3.0]),
        }
    }
    other = {"ssp245": {"gdp": np.array([10.0]), "dah": np.array([20.0])}}
    out = mis.swap_inputs(own, other, ("gdp",))
    assert out["ssp245"]["gdp"][0] == 10.0
    assert out["ssp245"]["dah"][0] == 2.0
    assert out["ssp245"]["flood"][0] == 3.0


def _rebuild(gdp_t, flood_slope=0.0):
    gx = np.geomspace(1.0, 1e5, 500)
    px = np.linspace(-20.0, 20.0, 4001)
    dx = np.linspace(0.0, 10.0, 11)
    x = {
        s: {
            "gdp": np.array([[100.0, gdp_t * k]]),
            "dah": np.array([[1.0, 0.5]]),
            "suit": np.stack(
                [np.full((1, 2), 100.0), np.full((1, 2), 100.0 + 20 * k)], axis=1
            ),
            "flood": np.array([[0.0, 2.0 * k]]),
        }
        for k, s in enumerate(mis.SSPS, start=1)
    }
    slopes = {
        ("pfpr", "logit_malaria_suitability"): 0.5,
        ("inc", "log_gdppc_mean"): -0.05,
        ("mort", "log_gdppc_mean"): -0.05,
        ("pfpr", "people_flood_days_per_capita"): flood_slope,
    }
    return Rebuild(
        grids={
            ("pfpr", "gdppc_mean"): (gx, -0.3 * np.log(gx)),
            ("pfpr", "mal_DAH_total_per_capita"): (dx, -0.1 * dx),
            ("inc", "logit_malaria_pfpr"): (px, 0.8 * px),
            ("mort", "logit_malaria_pfpr"): (px, 0.8 * px),
        },
        slopes=slopes,
        inputs=x,
        l_obs=np.array([[-1.0]]),
        obs_log={"inc": np.array([[-3.0]]), "mort": np.array([[-7.0]])},
        ry=0,
    )


def test_flood_term_moves_the_rate_only_when_the_model_has_it():
    src = {"gdp": "ssp245", "suitability": "ssp245", "wind": "ssp245"}
    without, with_flood = _rebuild(150.0), _rebuild(150.0, flood_slope=0.1)
    diff = with_flood.log_rate(with_flood.year_parts(1), "inc", src) - without.log_rate(
        without.year_parts(1), "inc", src
    )
    # flood change 4 x slope 0.1 x g slope 0.8
    np.testing.assert_allclose(diff, 0.32)


def test_run_year_counts_every_model_variant_ssp_and_returns_own_rates():
    a, b = _rebuild(150.0), _rebuild(120.0, flood_slope=0.1)
    rebuilds = {"current": {"own": a, "swap:gdp": a}, "first_submission": {"own": b}}
    masks = {
        f"{s}:{m}": np.zeros((1, 2, 2), bool) for s in mis.SSPS for m in mis.MEASURES
    }
    rows, own = mis.run_year(
        rebuilds,
        1,
        year=2024,
        population=np.array([[1e5, 1e5]]),
        masks=masks,
        factor={m: np.ones((1, 2)) for m in mis.MEASURES},
        groups=np.array([166.0]),
    )
    assert {(r["source"], r["variant"]) for r in rows} == {
        ("current", "own"),
        ("current", "swap:gdp"),
        ("first_submission", "own"),
    }
    assert set(own) == {"current", "first_submission"}
    lr = a.log_rate(
        a.year_parts(1),
        "inc",
        {"gdp": "ssp245", "suitability": "ssp245", "wind": "ssp245"},
    )
    np.testing.assert_allclose(own["current"]["ssp245:inc"], lr)
