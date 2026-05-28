"""
Tests for lib/utils/diagnostics.py

Uses small synthetic DataFrames — no real pipeline data.
"""

import pytest
import pandas as pd
import numpy as np

from idd_forecast_mbp.lib.utils.diagnostics import (
    check_concordance,
    check_column_for_problematic_values,
)


# ---------------------------------------------------------------------------
# check_concordance
# ---------------------------------------------------------------------------

@pytest.fixture
def concordant_dfs():
    """Two DataFrames with identical 'rate' values — should pass."""
    aa = pd.DataFrame({
        'location_id': [1, 2, 3],
        'year_id': [2020, 2020, 2020],
        'rate': [0.1, 0.2, 0.3],
    })
    gbd = pd.DataFrame({
        'location_id': [1, 2, 3],
        'year_id': [2020, 2020, 2020],
        'rate': [0.1, 0.2, 0.3],
    })
    return aa, gbd


@pytest.fixture
def discordant_dfs():
    """Two DataFrames where 'rate' differs by 0.5 — should fail tolerance=0.01."""
    aa = pd.DataFrame({
        'location_id': [1, 2],
        'year_id': [2020, 2020],
        'rate': [0.1, 0.2],
    })
    gbd = pd.DataFrame({
        'location_id': [1, 2],
        'year_id': [2020, 2020],
        'rate': [0.6, 0.7],
    })
    return aa, gbd


def test_check_concordance_passes_returns_empty_dict(concordant_dfs):
    aa, gbd = concordant_dfs
    result = check_concordance('rate', aa, gbd, tolerance=0.01)
    assert result == {}


def test_check_concordance_fails_returns_stats_dict(discordant_dfs):
    aa, gbd = discordant_dfs
    result = check_concordance('rate', aa, gbd, tolerance=0.01)
    assert isinstance(result, dict)
    assert 'max_absolute_diff' in result
    assert result['max_absolute_diff'] > 0.01


def test_check_concordance_stats_keys(discordant_dfs):
    aa, gbd = discordant_dfs
    result = check_concordance('rate', aa, gbd, tolerance=0.01)
    expected_keys = {
        'mean_absolute_diff', 'median_absolute_diff', 'max_absolute_diff',
        'std_absolute_diff', 'mean_relative_diff_pct', 'pearson_correlation',
        'within_1_pct', 'within_5_pct', 'within_10_pct',
        'p95_absolute_diff', 'p99_absolute_diff',
    }
    assert expected_keys == set(result.keys())


def test_check_concordance_inner_join(concordant_dfs):
    """Locations not in both DFs are excluded — no error."""
    aa, gbd = concordant_dfs
    aa_extra = pd.concat([aa, pd.DataFrame({
        'location_id': [99], 'year_id': [2020], 'rate': [0.9]
    })], ignore_index=True)
    # Should still pass — location 99 isn't in gbd so it's excluded
    result = check_concordance('rate', aa_extra, gbd.copy(), tolerance=0.01)
    assert result == {}


# ---------------------------------------------------------------------------
# check_column_for_problematic_values
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_df():
    return pd.DataFrame({
        'location_id': [1, 2, 3],
        'value': [1.0, 2.0, 3.0],
    })


@pytest.fixture
def nan_df():
    return pd.DataFrame({'value': [1.0, float('nan'), 3.0]})


@pytest.fixture
def inf_df():
    return pd.DataFrame({'value': [1.0, float('inf'), 3.0]})


@pytest.fixture
def negative_df():
    return pd.DataFrame({'value': [1.0, -2.0, 3.0]})


@pytest.fixture
def string_df():
    return pd.DataFrame({'value': ['a', 'not_a_number', 'c']})


def test_clean_column_returns_none(clean_df):
    result = check_column_for_problematic_values('value', clean_df)
    assert result is None


def test_clean_column_with_report(clean_df):
    result = check_column_for_problematic_values('value', clean_df, return_report=True)
    assert isinstance(result, dict)
    assert result['total_problematic'] == 0


def test_nan_detected(nan_df):
    result = check_column_for_problematic_values('value', nan_df, return_report=True)
    assert result['nan_count'] == 1


def test_inf_detected(inf_df):
    result = check_column_for_problematic_values('value', inf_df, return_report=True)
    assert result['inf_count'] == 1


def test_negative_detected(negative_df):
    result = check_column_for_problematic_values('value', negative_df, return_report=True)
    assert result['negative_count'] == 1


def test_non_numeric_string_column(string_df):
    result = check_column_for_problematic_values('value', string_df, return_report=True)
    assert result['non_numeric_count'] > 0


def test_missing_column_raises(clean_df):
    with pytest.raises(ValueError, match="not found in dataframe"):
        check_column_for_problematic_values('nonexistent', clean_df)


def test_report_contains_summary_df(clean_df):
    result = check_column_for_problematic_values('value', clean_df, return_report=True)
    assert 'summary_df' in result
    assert isinstance(result['summary_df'], pd.DataFrame)


def test_verbose_mode_no_error(clean_df, capsys):
    check_column_for_problematic_values('value', clean_df, verbose=True)
    captured = capsys.readouterr()
    assert 'Summary' in captured.out


def test_problematic_rows_in_report(nan_df):
    result = check_column_for_problematic_values('value', nan_df, return_report=True)
    assert len(result['problematic_rows']) == 1
