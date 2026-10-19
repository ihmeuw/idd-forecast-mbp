"""Unit tests for 07_vetting/scenario_variant_counts.py (the pure pieces and a toy two-unit run)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from idd_forecast_mbp.lib.processing.scenario_gap import INPUT_FACTORS, Rebuild

SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "src/idd_forecast_mbp/07_vetting/scenario_variant_counts.py"
)
_spec = importlib.util.spec_from_file_location("scenario_variant_counts", SCRIPT_PATH)
svc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = svc
_spec.loader.exec_module(svc)


def test_hold_variants_cover_every_subset_once():
    names = [n for n, _ in svc.hold_variants()]
    assert len(names) == 8
    assert names[0] == "hold:none"
    assert "hold:gdp+dah+suitability" in names
    assert len(set(names)) == 8


# toy: two admin-2 units, years (2023, 2024), two draws, linear terms
def _rebuild(gdp_t):
    gx = np.geomspace(1.0, 1e5, 500)
    px = np.linspace(-20.0, 20.0, 4001)
    dx = np.linspace(0.0, 10.0, 11)
    x = {
        s: {
            "gdp": np.array([[100.0, gdp_t * k], [200.0, gdp_t * k]]),
            "dah": np.array([[1.0, 0.5], [1.0, 0.5]]),
            "suit": np.stack(
                [np.full((2, 2), 100.0), np.full((2, 2), 100.0 + 20 * k)], axis=1
            ),
        }
        for k, s in enumerate(svc.SSPS, start=1)
    }
    return Rebuild(
        grids={
            ("pfpr", "gdppc_mean"): (gx, -0.3 * np.log(gx)),
            ("pfpr", "mal_DAH_total_per_capita"): (dx, -0.1 * dx),
            ("inc", "logit_malaria_pfpr"): (px, 0.8 * px),
            ("mort", "logit_malaria_pfpr"): (px, 0.8 * px),
        },
        slopes={
            ("pfpr", "logit_malaria_suitability"): 0.5,
            ("inc", "log_gdppc_mean"): -0.05,
            ("mort", "log_gdppc_mean"): -0.05,
        },
        inputs=x,
        l_obs=np.array([[-1.0], [0.0]]),
        obs_log={"inc": np.array([[-3.0], [-2.0]]), "mort": np.array([[-7.0], [-6.0]])},
        ry=0,
    )


def _run():
    cur, old = _rebuild(150.0), _rebuild(120.0)
    pop = np.array([[1e5, 1e5], [2e5, 2e5]])
    fc = {
        s: {
            m: np.stack(
                [
                    np.zeros((2, 2)),
                    cur.log_rate(cur.year_parts(1), m, dict.fromkeys(INPUT_FACTORS, s)),
                ],
                axis=1,
            )
            for m in svc.MEASURES
        }
        for s in svc.SSPS
    }
    return svc.VariantRun(
        cur=cur,
        old=old,
        forecasts=fc,
        held_forecasts=fc,
        years=np.array([2023, 2024]),
        population=pop,
        groups=np.array([166.0, 158.0]),
        factor={m: np.full((2, 2), 0.9) for m in svc.MEASURES},
    )


def test_counts_apply_mask_and_factor():
    rb = _rebuild(150.0)
    parts = rb.year_parts(1)
    src = dict.fromkeys(INPUT_FACTORS, "ssp245")
    out = svc.counts(
        rb,
        parts,
        measure="inc",
        src=src,
        population=np.array([10.0, 20.0]),
        mask=np.array([[False, False], [True, True]]),
        factor=np.array([0.5, 0.5]),
    )
    expected = np.exp(rb.log_rate(parts, "inc", src))[0].mean() * 10.0 * 0.5
    assert out[0] == pytest.approx(expected)
    assert np.isnan(out[1])


def test_year_rows_cover_every_variant_and_hold_none_matches_the_forecast():
    run = _run()
    rows = run.year_rows(1)
    variants = {r["variant"] for r in rows}
    assert variants == {n for n, _ in svc.hold_variants()} | {
        f"gdp_from:{s}" for s in svc.SSPS
    }
    assert {r["source"] for r in rows} == {"current", "2025"}
    assert run.worst["current_hold_none_vs_forecast"] == pytest.approx(0.0, abs=1e-12)
    # holding GDP and suitability leaves the three scenarios identical (DAH is the same in all of them)
    held = {
        r["ssp"]: r["count"]
        for r in rows
        if r["source"] == "current"
        and r["variant"] == "hold:gdp+suitability"
        and r["measure"] == "inc"
        and r["location_id"] == 1
    }
    assert held["ssp126"] == pytest.approx(held["ssp585"])


def test_population_rows_and_trajectories():
    run = _run()
    pops = {r["location_id"]: r["population"] for r in run.population_rows(1)}
    assert pops == {1: 3e5, 158: 2e5, 166: 1e5}
    traj = run.trajectories(np.array([1.0, 1.0]))
    g = [
        r["value"]
        for r in traj
        if r["source"] == "current"
        and r["covariate"] == "gdppc"
        and r["ssp"] == "ssp126"
        and r["location_id"] == 1
    ]
    assert g == [150.0, 150.0]
