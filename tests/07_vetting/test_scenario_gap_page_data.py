"""Unit tests for 07_vetting/scenario_gap_page_data.py (the pure pieces)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "src/idd_forecast_mbp/07_vetting/scenario_gap_page_data.py"
)
_spec = importlib.util.spec_from_file_location("scenario_gap_page_data", SCRIPT_PATH)
pdata = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pdata
_spec.loader.exec_module(pdata)


def _agg():
    rows = [
        {
            "pair": "ssp245:ssp126",
            "measure": "inc",
            "level": 3,
            "location_id": loc,
            "location_name": name,
            "year_id": year,
            "count_a": 10.0 + gap,
            "count_b": 10.0,
            "part_gdp": gap * 0.6,
            "part_suitability": gap * 0.4,
        }
        for loc, name, gap in (
            (11, "Alpha", 5.0),
            (12, "Beta", -9.0),
            (13, "Gamma", 1.0),
        )
        for year in (2099, 2100)
    ]
    return pd.DataFrame(rows)


def test_ranked_parts_orders_by_absolute_gap_and_sums_the_years():
    out = pdata.ranked_parts(
        _agg(),
        level=3,
        years=[2099, 2100],
        part_cols=["part_gdp", "part_suitability"],
        n=2,
    )
    assert out.location_name.tolist() == ["Beta", "Alpha"]
    np.testing.assert_allclose(out.gap, [-18.0, 10.0])
    np.testing.assert_allclose(out.part_gdp + out.part_suitability, out.gap)


def test_draw_mean_parts_broadcasts_draw_invariant_pieces():
    parts = {
        "dah": np.array([[1.0], [2.0]]),
        "gdp_pfpr": np.array([[0.5], [0.0]]),
        "log_gdp": np.array([[0.1], [0.2]]),
        "suitability": np.array([[1.0, 3.0], [0.0, 2.0]]),
        "wind": np.zeros((2, 2)),
    }
    out = pdata.draw_mean_parts(parts, 2)
    np.testing.assert_allclose(out["dah"], [1.0, 2.0])
    np.testing.assert_allclose(out["suitability"], [2.0, 1.0])
    assert set(out) == set(pdata.LINK_PARTS)


def test_scenario_block_lists_each_ssp_in_year_order():
    rows = [
        {
            "location_id": 1,
            "year_id": y,
            "ssp": ssp,
            "inc_count_mean": float(y),
            "inc_rate_mean": 0.1,
            "mort_count_mean": 1.0,
            "mort_rate_mean": np.nan,
        }
        for ssp in pdata.SSPS
        for y in (2025, 2024)
    ]
    out = pdata.scenario_block(pd.DataFrame(rows), [1])
    assert out[1]["ssp245"]["year"] == [2024, 2025]
    assert out[1]["ssp245"]["inc_count_mean"] == [2024.0, 2025.0]
    assert out[1]["ssp126"]["mort_rate_mean"] == [None, None]


def test_previous_block_puts_measures_side_by_side():
    rows = [
        {"location_id": 1, "year_id": 2050, "ssp": ssp, "measure": m, "count": v}
        for ssp in pdata.SSPS
        for m, v in (("inc", 5.0), ("mort", 0.5))
    ]
    out = pdata.previous_block(pd.DataFrame(rows), [1])
    assert out[1]["ssp585"] == {"year": [2050], "inc_count": [5.0], "mort_count": [0.5]}


def test_gap_and_ranked_blocks_key_by_pair_and_measure():
    agg = _agg()
    glob = agg.groupby(["pair", "measure", "year_id"], as_index=False)[
        ["count_a", "count_b", "part_gdp", "part_suitability"]
    ].sum()
    agg = pd.concat(
        [agg, glob.assign(level=0, location_id=1, location_name="Global")],
        ignore_index=True,
    )
    parts = ["part_gdp", "part_suitability"]
    gap = pdata.gap_block(agg, parts, [1])
    assert gap["ssp245:ssp126"]["inc"][1]["year"] == [2099, 2100]
    ranked = pdata.ranked_block(agg, parts)
    countries = ranked["ssp245:ssp126"]["inc"]["countries_y2100"]
    assert [r["location_name"] for r in countries] == ["Beta", "Alpha", "Gamma"]


def test_apply_factors_scales_counts_and_parts_by_measure():
    admin2 = pd.DataFrame(
        {
            "pair": "a:b",
            "measure": ["inc", "mort"],
            "location_id": 1,
            "year_id": 2050,
            "count_a": [10.0, 4.0],
            "count_b": [6.0, 2.0],
            "part_gdp": [4.0, 2.0],
        }
    )
    factors = pd.DataFrame(
        {
            "location_id": 1,
            "year_id": 2050,
            "measure": ["inc", "mort"],
            "factor": [0.5, 0.25],
        }
    )
    out = pdata.apply_factors(admin2, factors, ["count_a", "count_b", "part_gdp"])
    np.testing.assert_allclose(out.count_a, [5.0, 1.0])
    np.testing.assert_allclose(out.part_gdp, [2.0, 0.5])
    assert "factor" not in out


def test_apply_factors_rejects_a_forecast_row_without_a_factor():
    admin2 = pd.DataFrame(
        {
            "pair": "a:b",
            "measure": "inc",
            "location_id": [1],
            "year_id": 2050,
            "count_a": [1.0],
            "count_b": [1.0],
        }
    )
    factors = pd.DataFrame(
        {"location_id": [2], "year_id": 2050, "measure": "inc", "factor": 1.0}
    )
    with pytest.raises(ValueError, match="no vaccine factor"):
        pdata.apply_factors(admin2, factors, ["count_a"])


def test_total_check_reports_the_worst_relative_difference():
    agg = pd.DataFrame(
        {
            "pair": "ssp245:ssp126",
            "measure": "inc",
            "level": 0,
            "location_id": 1,
            "year_id": [2050, 2051],
            "count_a": [10.0, 20.0],
            "count_b": [5.0, 5.0],
        }
    )
    products = pd.DataFrame(
        {
            "location_id": 1,
            "ssp": ["ssp245", "ssp245", "ssp126", "ssp126"],
            "year_id": [2050, 2051, 2050, 2051],
            "inc_count_mean": [10.0, 19.0, 5.0, 5.0],
        }
    )
    assert pdata.total_check(agg, products) == pytest.approx(1 / 19)


def _held_agg():
    return pd.DataFrame(
        {
            "arm": ["ssp245:gdp_held", "ssp245:gdp_held", "ssp126:gdp_held"],
            "measure": "inc",
            "level": 0,
            "location_id": 1,
            "year_id": [2051, 2050, 2050],
            "count": [3.0, 2.0, 4.0],
        }
    )


def test_held_block_nests_by_hold_measure_location_and_ssp():
    out = pdata.held_block(_held_agg(), [1])
    assert out["gdp_held"]["inc"][1]["ssp245"] == {
        "year": [2050, 2051],
        "count": [2.0, 3.0],
    }
    assert out["gdp_held"]["inc"][1]["ssp126"]["count"] == [4.0]


def test_held_check_compares_each_ssp_with_the_hold_arm():
    products = pd.DataFrame(
        {
            "location_id": 1,
            "ssp": ["ssp245", "ssp245", "ssp126"],
            "year_id": [2050, 2051, 2050],
            "inc_count_mean": [2.0, 3.0, 5.0],
        }
    )
    assert pdata.held_check(_held_agg(), products, "gdp_held") == pytest.approx(0.2)


def test_factor_spread_is_the_largest_difference_from_the_first_ssp():
    base = pd.DataFrame(
        {"location_id": [1, 2], "year_id": 2050, "measure": "inc", "factor": [0.9, 1.0]}
    )
    frames = {
        "ssp126": base,
        "ssp245": base.assign(factor=[0.9, 0.97]),
        "ssp585": base.assign(factor=[0.91, 1.0]),
    }
    assert pdata.factor_spread(frames) == pytest.approx(0.03)


def test_render_page_injects_json_that_cannot_close_the_script_tag():
    out = pdata.render_page("<script>__PAGE_DATA__</script>", {"name": "a</script>b"})
    assert out == '<script>{"name":"a<\\/script>b"}</script>'
    assert json.loads(out[len("<script>") : -len("</script>")]) == {
        "name": "a</script>b"
    }


def test_render_page_needs_exactly_one_placeholder():
    with pytest.raises(ValueError, match="exactly once"):
        pdata.render_page("no placeholder", {})


def test_apply_factors_checks_the_first_value_column_of_any_frame():
    held = pd.DataFrame(
        {
            "arm": "ssp245:gdp_held",
            "measure": "inc",
            "location_id": [1, 2],
            "year_id": 2050,
            "count": [4.0, np.nan],
        }
    )
    factors = pd.DataFrame(
        {"location_id": [1], "year_id": 2050, "measure": "inc", "factor": 0.5}
    )
    out = pdata.apply_factors(held, factors, ["count"])
    assert out["count"].iloc[0] == 2.0
    assert np.isnan(out["count"].iloc[1])


def test_ranked_block_adds_population_summed_over_the_same_years():
    agg = _agg()
    pop = pd.DataFrame(
        {
            "location_id": [11, 11, 12, 12, 13, 13],
            "year_id": [2099, 2100] * 3,
            "population": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    out = pdata.ranked_block(agg, ["part_gdp", "part_suitability"], pop)
    recs = {
        r["location_name"]: r for r in out["ssp245:ssp126"]["inc"]["countries_y2100"]
    }
    assert recs["Beta"]["population"] == 4.0
    cum = {
        r["location_name"]: r
        for r in out["ssp245:ssp126"]["inc"]["countries_cumulative"]
    }
    assert cum["Beta"]["population"] == 7.0


def test_population_block_orders_years():
    pop = pd.DataFrame(
        {
            "location_id": [1, 1, 4],
            "year_id": [2001, 2000, 2000],
            "population": [2.0, 1.0, 9.0],
        }
    )
    assert pdata.population_block(pop, [1]) == {
        1: {"year": [2000, 2001], "population": [1.0, 2.0]}
    }


def test_variant_block_nests_source_variant_measure_location_ssp():
    counts = pd.DataFrame(
        {
            "source": "2025",
            "variant": "hold:dah",
            "measure": "inc",
            "location_id": 1,
            "ssp": "ssp245",
            "year_id": [2025, 2024],
            "count": [2.0, 1.0],
        }
    )
    pop = pd.DataFrame({"location_id": [1], "year_id": [2024], "population": [10.0]})
    out = pdata.variant_block(counts, pop, {"n": 1})
    assert out["counts"]["2025"]["hold:dah"]["inc"][1]["ssp245"] == {
        "year": [2024, 2025],
        "count": [1.0, 2.0],
    }
    assert out["population"][1]["population"] == [10.0]
    assert out["checks"] == {"n": 1}


def test_trajectory_block_nests_covariate_source_ssp_location():
    traj = pd.DataFrame(
        {
            "covariate": "gdppc",
            "source": "current",
            "ssp": "ssp126",
            "location_id": 166,
            "year_id": [2001, 2000],
            "value": [5.0, 4.0],
        }
    )
    assert pdata.trajectory_block(traj) == {
        "gdppc": {
            "current": {"ssp126": {166: {"year": [2000, 2001], "value": [4.0, 5.0]}}}
        }
    }
