"""Unit tests for the pure helpers in plot_dengue_run_comparison.py.

The loaders and figure functions need cluster artifacts and a display, so only the two
transforms that carry logic are covered here: the unit-scale chooser and the wide->long
reshape that every figure function consumes.

The module lives in a numbered stage directory, which is not an importable package name,
so it is loaded by path — the same pattern as tests/04_forecasting.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

#: The fixture's two rows, times {inc, mort}.
EXPECTED_LONG_ROWS = 4
#: Anchor year, and the first of the fixture's two years.
ANCHOR_YEAR = 2023

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "idd_forecast_mbp" / "05_aggregation" / "plot_dengue_run_comparison.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("plot_dengue_run_comparison", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wide(**overrides) -> pd.DataFrame:
    """A two-row wide summary in the schema the forecast driver writes."""
    base = {
        "formulation": ["GBD-esque_w_time"] * 2,
        "ssp_scenario": ["ssp245", "ssp245"],
        "decay": ["logistic_k8", "logistic_k8"],
        "location_id": [1, 1],
        "year_id": [2023, 2100],
    }
    for measure, scale in (("inc", 1.0), ("mort", 0.003)):
        for metric, mult in (("count", 2.0e7), ("rate", 2.5e-3)):
            for stat, bump in (("mean", 1.0), ("lower", 0.8), ("upper", 1.2)):
                base[f"dengue_{measure}_{metric}_{stat}"] = [
                    mult * scale * bump, mult * scale * bump * 10,
                ]
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# pick_scale
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("peak", "expected"),
    [
        (3.5e9, (1e9, "billions")),
        (2.0e7, (1e6, "millions")),
        (6.1e4, (1e3, "thousands")),
        (52.0, (1.0, "")),
    ],
)
def test_pick_scale_picks_unit_from_largest_magnitude(mod, peak, expected):
    assert mod.pick_scale(np.array([0.0, peak])) == expected


def test_pick_scale_ignores_non_finite(mod):
    """A NaN or inf in a rate column must not decide the unit — or crash the panel."""
    assert mod.pick_scale(np.array([np.nan, np.inf, 2.0e6])) == (1e6, "millions")


def test_pick_scale_handles_empty_and_all_nan(mod):
    assert mod.pick_scale(np.array([])) == (1.0, "")
    assert mod.pick_scale(np.array([np.nan, np.nan])) == (1.0, "")


def test_pick_scale_uses_magnitude_not_sign(mod):
    """Difference-style inputs are negative; the unit comes from |value|."""
    assert mod.pick_scale(np.array([-4.0e6, 1.0])) == (1e6, "millions")


# ---------------------------------------------------------------------------
# current_to_long
# ---------------------------------------------------------------------------

def test_current_to_long_emits_one_row_per_measure(mod):
    out = mod.current_to_long(_wide())
    assert len(out) == EXPECTED_LONG_ROWS
    assert set(out.measure) == {"inc", "mort"}
    assert set(out.run) == {"current"}


def test_current_to_long_preserves_the_decay_and_ssp_axes(mod):
    """Both are figure dimensions; losing either silently merges arms."""
    wide = _wide(
        decay=["logistic_k8", "no_decay"], ssp_scenario=["ssp126", "ssp585"],
    )
    out = mod.current_to_long(wide)
    assert set(out.decay) == {"logistic_k8", "no_decay"}
    assert set(out.ssp) == {"ssp126", "ssp585"}
    # decay and ssp must stay paired to their own row, not cross-joined.
    pairs = set(zip(out.decay, out.ssp, strict=True))
    assert pairs == {("logistic_k8", "ssp126"), ("no_decay", "ssp585")}


def test_current_to_long_renames_measure_prefixed_columns(mod):
    """The long schema drops the dengue_/measure prefix; values must survive intact."""
    out = mod.current_to_long(_wide())
    inc = out[(out.measure == "inc") & (out.year_id == ANCHOR_YEAR)].iloc[0]
    assert inc.count_mean == pytest.approx(2.0e7)
    assert inc.count_lower == pytest.approx(2.0e7 * 0.8)
    assert inc.count_upper == pytest.approx(2.0e7 * 1.2)
    assert inc.rate_mean == pytest.approx(2.5e-3)
    mort = out[(out.measure == "mort") & (out.year_id == ANCHOR_YEAR)].iloc[0]
    assert mort.count_mean == pytest.approx(2.0e7 * 0.003)


def test_current_to_long_output_columns_are_exactly_the_long_schema(mod):
    out = mod.current_to_long(_wide())
    assert set(out.columns) == {
        "location_id", "year_id", "ssp", "decay", "measure", "run",
        "count_mean", "count_lower", "count_upper",
        "rate_mean", "rate_lower", "rate_upper",
    }


# ---------------------------------------------------------------------------
# format_year_slopes
# ---------------------------------------------------------------------------

def _coefs() -> pd.DataFrame:
    """Coefficients in the schema the fit's extractor writes."""
    return pd.DataFrame({
        "outcome": ["mortality", "incidence", "incidence"],
        "term": ["year_sr_4", "year_sr_4", "log_dengue_mort_rate"],
        "coef": [-0.008365, -0.008067, 0.494216],
        "p_coef": [2.4e-3, 2.7e-2, 9.4e-278],
        "super_region_id": pd.array([4, 4, None], dtype="Int64"),
    })


def test_format_year_slopes_reports_both_outcomes_and_the_mediator(mod):
    line = mod.format_year_slopes(_coefs(), 4)
    assert "mortality -0.00836" in line
    assert "incidence -0.00807" in line
    # The mortality->incidence pathway is why a negative mortality slope drags
    # incidence down; omitting it makes the incidence row unreadable.
    assert "+0.494" in line
    assert "log mortality rate" in line


def test_format_year_slopes_keeps_the_sign_explicit(mod):
    """A leading '+' distinguishes a rising from a falling super-region at a glance."""
    coefs = _coefs()
    coefs.loc[coefs.outcome == "mortality", "coef"] = 0.079799
    assert "mortality +0.07980" in mod.format_year_slopes(coefs, 4)


def test_format_year_slopes_empty_for_global(mod):
    """Global spans all six slopes, so there is no single number to report."""
    assert mod.format_year_slopes(_coefs(), None) == ""


def test_format_year_slopes_empty_for_unknown_super_region(mod):
    assert mod.format_year_slopes(_coefs(), 31) == ""


def test_format_year_slopes_empty_without_coefficients(mod):
    """An absent coefficient file must degrade to no subtitle, not an error."""
    assert mod.format_year_slopes(pd.DataFrame(), 4) == ""
    assert mod.format_year_slopes(None, 4) == ""


def test_load_coefficients_returns_empty_frame_when_absent(mod, tmp_path):
    out = mod.load_coefficients(tmp_path / "nope.parquet")
    assert out.empty
    # Must still be shaped so format_year_slopes can filter it without a KeyError.
    assert mod.format_year_slopes(out, 4) == ""


def test_load_coefficients_returns_none_path_as_empty(mod):
    assert mod.load_coefficients(None).empty


def test_current_to_long_raises_on_missing_columns(mod):
    """A silently-absent stat would plot as an empty panel; fail loudly instead."""
    wide = _wide().drop(columns=["dengue_mort_rate_upper"])
    with pytest.raises(KeyError, match="dengue_mort_rate_upper"):
        mod.current_to_long(wide)
