"""
Tests for lib/processing/scenarios.py
"""

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.scenarios import generate_dah_scenarios


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def baseline_df():
    """Minimal baseline DataFrame with years 2020–2030."""
    rows = []
    for loc in [10, 20]:
        for year in range(2020, 2031):
            rows.append({
                'location_id': loc,
                'year_id': year,
                'A0_location_id': loc,
                'aa_population': 1_000_000.0,
                'mal_DAH_total_per_capita': 5.0,
                'mal_DAH_total': 5_000_000.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

def test_returns_four_scenarios(baseline_df):
    scenarios, names = generate_dah_scenarios(baseline_df, ssp_scenario='ssp245')
    assert len(scenarios) == 4
    assert len(names) == 4


def test_default_scenario_names(baseline_df):
    _, names = generate_dah_scenarios(baseline_df, ssp_scenario='ssp245')
    assert names == ['Baseline', 'Constant', 'Increasing', 'Decreasing']


def test_custom_scenario_names(baseline_df):
    custom = ['A', 'B', 'C', 'D']
    _, names = generate_dah_scenarios(baseline_df, ssp_scenario='ssp245', dah_scenario_names=custom)
    assert names == custom


def test_dah_scenario_column_set(baseline_df):
    scenarios, names = generate_dah_scenarios(baseline_df, ssp_scenario='ssp245')
    for df, name in zip(scenarios, names):
        assert (df['dah_scenario'] == name).all()


def test_ssp_scenario_column_set(baseline_df):
    scenarios, _ = generate_dah_scenarios(baseline_df, ssp_scenario='ssp585')
    for df in scenarios:
        assert (df['ssp_scenario'] == 'ssp585').all()


# ---------------------------------------------------------------------------
# year_start filter
# ---------------------------------------------------------------------------

def test_year_start_filters_rows(baseline_df):
    # Use reference_year within the filtered range to avoid empty constant-scenario merge
    scenarios, _ = generate_dah_scenarios(
        baseline_df, ssp_scenario='ssp245', year_start=2025, reference_year=2025
    )
    for df in scenarios:
        assert df['year_id'].min() >= 2025


# ---------------------------------------------------------------------------
# Constant scenario
# ---------------------------------------------------------------------------

def test_constant_scenario_holds_reference_year(baseline_df):
    scenarios, _ = generate_dah_scenarios(baseline_df, ssp_scenario='ssp245', reference_year=2023)
    constant_df = scenarios[1]

    ref_vals = constant_df[constant_df['year_id'] == 2023][
        ['location_id', 'mal_DAH_total_per_capita']
    ].set_index('location_id')
    future_vals = constant_df[constant_df['year_id'] == 2028][
        ['location_id', 'mal_DAH_total_per_capita']
    ].set_index('location_id')

    for loc in ref_vals.index:
        assert np.isclose(
            ref_vals.loc[loc, 'mal_DAH_total_per_capita'],
            future_vals.loc[loc, 'mal_DAH_total_per_capita'],
        )


# ---------------------------------------------------------------------------
# Increasing scenario
# ---------------------------------------------------------------------------

def test_increasing_scenario_multiplies_start_year(baseline_df):
    scenarios, _ = generate_dah_scenarios(
        baseline_df, ssp_scenario='ssp245', modification_start_year=2026
    )
    increasing_df = scenarios[2]

    base = baseline_df[baseline_df['year_id'] == 2026]['mal_DAH_total_per_capita'].iloc[0]
    inc = increasing_df[increasing_df['year_id'] == 2026]['mal_DAH_total_per_capita'].iloc[0]
    assert np.isclose(inc, base * 1.2)


def test_increasing_scenario_caps_at_2x_after_5_years(baseline_df):
    scenarios, _ = generate_dah_scenarios(
        baseline_df, ssp_scenario='ssp245', modification_start_year=2026
    )
    increasing_df = scenarios[2]

    base = baseline_df[baseline_df['year_id'] == 2030]['mal_DAH_total_per_capita'].iloc[0]
    val_2030 = increasing_df[increasing_df['year_id'] == 2030]['mal_DAH_total_per_capita'].iloc[0]
    assert np.isclose(val_2030, base * 2.0)


# ---------------------------------------------------------------------------
# Decreasing scenario
# ---------------------------------------------------------------------------

def test_decreasing_scenario_multiplies_start_year(baseline_df):
    scenarios, _ = generate_dah_scenarios(
        baseline_df, ssp_scenario='ssp245', modification_start_year=2026
    )
    decreasing_df = scenarios[3]

    base = baseline_df[baseline_df['year_id'] == 2026]['mal_DAH_total_per_capita'].iloc[0]
    dec = decreasing_df[decreasing_df['year_id'] == 2026]['mal_DAH_total_per_capita'].iloc[0]
    assert np.isclose(dec, base * 0.8)


def test_decreasing_scenario_zeroes_after_5_years(baseline_df):
    scenarios, _ = generate_dah_scenarios(
        baseline_df, ssp_scenario='ssp245', modification_start_year=2026
    )
    decreasing_df = scenarios[3]

    val_2030 = decreasing_df[decreasing_df['year_id'] == 2030]['mal_DAH_total_per_capita'].iloc[0]
    assert np.isclose(val_2030, 0.0)


# ---------------------------------------------------------------------------
# Input not modified
# ---------------------------------------------------------------------------

def test_does_not_modify_input(baseline_df):
    original_len = len(baseline_df)
    original_cols = set(baseline_df.columns)
    generate_dah_scenarios(baseline_df, ssp_scenario='ssp245')
    assert len(baseline_df) == original_len
    assert set(baseline_df.columns) == original_cols
