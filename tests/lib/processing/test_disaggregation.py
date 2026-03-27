"""
Tests for lib/processing/disaggregation.py
"""

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.disaggregation import (
    disaggregate_age_sex_dengue,
    disaggregate_age_sex_malaria,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def malaria_df():
    """Two locations × 1 year × 2 age groups × 2 sexes = 8 rows."""
    rows = []
    for loc in [1, 2]:
        for age in [3, 10]:  # age_group_id 2 would be zeroed; use others
            for sex in [1, 2]:
                rows.append({
                    'location_id': loc,
                    'year_id': 2025,
                    'age_group_id': age,
                    'sex_id': sex,
                    'population': 1000.0,
                    'rr_inc_as': 1.0,
                    'rr_mort_as': 1.0,
                    'aa_malaria_inc_count': 100.0,
                    'aa_malaria_mort_count': 10.0,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def dengue_df():
    """Two locations × 2 age groups × 1 sex = 4 rows."""
    rows = []
    for loc in [1, 2]:
        for age in [10, 20]:
            rows.append({
                'location_id': loc,
                'year_id': 2025,
                'age_group_id': age,
                'sex_id': 1,
                'population': 1000.0,
                'rr_inc_as': 1.5,
                'base_log_dengue_inc_rate_pred': np.log(0.01),
                'dengue_cfr_pred': 0.02,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# disaggregate_age_sex_malaria
# ---------------------------------------------------------------------------

def test_malaria_adds_output_columns(malaria_df):
    result = disaggregate_age_sex_malaria(malaria_df)
    assert 'malaria_inc_count_pred' in result.columns
    assert 'malaria_mort_count_pred' in result.columns


def test_malaria_drops_intermediate_columns(malaria_df):
    result = disaggregate_age_sex_malaria(malaria_df)
    for col in ['rr_inc_as_pop', 'rr_mort_as_pop', 'sum_rr_inc_as_pop',
                'sum_rr_mort_as_pop', 'inc_fraction', 'mort_fraction']:
        assert col not in result.columns


def test_malaria_counts_sum_to_aa_count(malaria_df):
    result = disaggregate_age_sex_malaria(malaria_df)
    # With equal RRs and equal populations, fractions are equal (1/4 each).
    # Sum per location should equal aa count (4 strata × 1/4 × 100 = 100).
    totals = result.groupby(['location_id', 'year_id'])['malaria_inc_count_pred'].sum()
    # aa_malaria_inc_count = 100 for all rows
    assert np.allclose(totals.values, 100.0)


def test_malaria_zeros_age_group_2(malaria_df):
    # Add rows with age_group_id = 2
    extra = malaria_df.iloc[:4].copy()
    extra['age_group_id'] = 2
    combined = pd.concat([malaria_df, extra], ignore_index=True)

    result = disaggregate_age_sex_malaria(combined)
    zeroed = result[result['age_group_id'] == 2]
    assert (zeroed['malaria_inc_count_pred'] == 0).all()
    assert (zeroed['malaria_mort_count_pred'] == 0).all()


def test_malaria_nonzero_age_groups_not_zeroed(malaria_df):
    result = disaggregate_age_sex_malaria(malaria_df)
    nonzero = result[result['age_group_id'] != 2]
    assert (nonzero['malaria_inc_count_pred'] > 0).all()


def test_malaria_does_not_modify_input(malaria_df):
    original_cols = set(malaria_df.columns)
    _ = disaggregate_age_sex_malaria(malaria_df)
    assert set(malaria_df.columns) == original_cols


def test_malaria_unequal_rr(malaria_df):
    # Make rr_inc_as unequal: age_group_id=3 → rr=2, age_group_id=10 → rr=1
    df = malaria_df.copy()
    df.loc[df['age_group_id'] == 3, 'rr_inc_as'] = 2.0
    df.loc[df['age_group_id'] == 10, 'rr_inc_as'] = 1.0

    result = disaggregate_age_sex_malaria(df)

    # Within one location, age_group_id=3 should have larger pred than age_group_id=10
    for loc in [1, 2]:
        for sex in [1, 2]:
            r3 = result[(result['location_id'] == loc) & (result['age_group_id'] == 3) & (result['sex_id'] == sex)]['malaria_inc_count_pred'].iloc[0]
            r10 = result[(result['location_id'] == loc) & (result['age_group_id'] == 10) & (result['sex_id'] == sex)]['malaria_inc_count_pred'].iloc[0]
            assert r3 > r10


# ---------------------------------------------------------------------------
# disaggregate_age_sex_dengue
# ---------------------------------------------------------------------------

def test_dengue_adds_output_columns(dengue_df):
    result = disaggregate_age_sex_dengue(dengue_df)
    assert 'dengue_inc_count_pred' in result.columns
    assert 'dengue_mort_count_pred' in result.columns


def test_dengue_inc_count_formula(dengue_df):
    result = disaggregate_age_sex_dengue(dengue_df)
    # inc = population * exp(log_rate) * rr_inc = 1000 * 0.01 * 1.5 = 15.0
    expected_inc = 1000.0 * np.exp(np.log(0.01)) * 1.5
    assert np.allclose(result['dengue_inc_count_pred'].values, expected_inc)


def test_dengue_mort_count_formula(dengue_df):
    result = disaggregate_age_sex_dengue(dengue_df)
    expected_inc = 1000.0 * 0.01 * 1.5
    expected_mort = expected_inc * 0.02
    assert np.allclose(result['dengue_mort_count_pred'].values, expected_mort)


def test_dengue_does_not_modify_input(dengue_df):
    original_cols = set(dengue_df.columns)
    _ = disaggregate_age_sex_dengue(dengue_df)
    assert set(dengue_df.columns) == original_cols


def test_dengue_zero_cfr_gives_zero_mort(dengue_df):
    df = dengue_df.copy()
    df['dengue_cfr_pred'] = 0.0
    result = disaggregate_age_sex_dengue(df)
    assert (result['dengue_mort_count_pred'] == 0).all()
