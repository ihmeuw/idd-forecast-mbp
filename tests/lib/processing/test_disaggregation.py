"""
Tests for lib/processing/disaggregation.py
"""

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.disaggregation import (
    compute_as_rr,
    disaggregate_age_sex_dengue,
    disaggregate_age_sex_malaria,
    disaggregate_malaria_draws,
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


def test_malaria_preserves_total_with_nonzero_age2_rr(malaria_df):
    """Regression: age zero_age_group_id must be excluded from the denominator, not
    zeroed after normalizing. With a NONZERO age-2 rr, the retained-age counts must
    still sum to the all-age total. Under the old normalize-then-zero order this sum
    was (1 - fraction_age2) * aa_count — a silent undercount."""
    extra = pd.DataFrame([
        {'location_id': loc, 'year_id': 2025, 'age_group_id': 2, 'sex_id': sex,
         'population': 1000.0, 'rr_inc_as': 1.0, 'rr_mort_as': 1.0,
         'aa_malaria_inc_count': 100.0, 'aa_malaria_mort_count': 10.0}
        for loc in [1, 2] for sex in [1, 2]
    ])
    combined = pd.concat([malaria_df, extra], ignore_index=True)

    result = disaggregate_age_sex_malaria(combined)

    inc_tot = result.groupby(['location_id', 'year_id'])['malaria_inc_count_pred'].sum()
    mort_tot = result.groupby(['location_id', 'year_id'])['malaria_mort_count_pred'].sum()
    assert np.allclose(inc_tot.values, 100.0)   # aa_malaria_inc_count
    assert np.allclose(mort_tot.values, 10.0)    # aa_malaria_mort_count

    zeroed = result[result['age_group_id'] == 2]
    assert (zeroed['malaria_inc_count_pred'] == 0).all()
    assert (zeroed['malaria_mort_count_pred'] == 0).all()


def test_malaria_zero_burden_group_no_nan(malaria_df):
    """A (location, year) group with no retained-age burden gets fraction 0, not 0/0=NaN."""
    df = malaria_df.copy()
    df['rr_inc_as'] = 0.0
    df['aa_malaria_inc_count'] = 0.0
    result = disaggregate_age_sex_malaria(df)
    assert result['malaria_inc_count_pred'].notna().all()
    assert (result['malaria_inc_count_pred'] == 0).all()


# ---------------------------------------------------------------------------
# compute_as_rr
# ---------------------------------------------------------------------------

def test_compute_as_rr_anchor_year_and_ratio():
    df = pd.DataFrame([
        {'location_id': 1, 'year_id': 2023, 'age_group_id': 3, 'sex_id': 1,
         'malaria_inc_rate': 2.0, 'aa_malaria_inc_rate': 1.0,
         'malaria_mort_rate': 0.4, 'aa_malaria_mort_rate': 0.2},
        # a non-anchor year that must be ignored
        {'location_id': 1, 'year_id': 2020, 'age_group_id': 3, 'sex_id': 1,
         'malaria_inc_rate': 9.0, 'aa_malaria_inc_rate': 1.0,
         'malaria_mort_rate': 9.0, 'aa_malaria_mort_rate': 0.2},
    ])
    rr = compute_as_rr(df, anchor_year=2023)
    assert set(rr.columns) == {'location_id', 'age_group_id', 'sex_id', 'rr_inc_as', 'rr_mort_as'}
    assert 'year_id' not in rr.columns
    assert len(rr) == 1  # only the anchor-year row survives
    assert np.isclose(rr['rr_inc_as'].iloc[0], 2.0)   # 2.0 / 1.0
    assert np.isclose(rr['rr_mort_as'].iloc[0], 2.0)  # 0.4 / 0.2


def test_compute_as_rr_zero_aa_rate_guarded():
    df = pd.DataFrame([{'location_id': 1, 'year_id': 2023, 'age_group_id': 3, 'sex_id': 1,
                        'malaria_inc_rate': 2.0, 'aa_malaria_inc_rate': 0.0,
                        'malaria_mort_rate': 0.0, 'aa_malaria_mort_rate': 0.0}])
    rr = compute_as_rr(df, anchor_year=2023)
    assert (rr['rr_inc_as'] == 0).all()
    assert (rr['rr_mort_as'] == 0).all()


def test_compute_as_rr_missing_anchor_year_raises():
    df = pd.DataFrame([{'location_id': 1, 'year_id': 2020, 'age_group_id': 3, 'sex_id': 1,
                        'malaria_inc_rate': 2.0, 'aa_malaria_inc_rate': 1.0,
                        'malaria_mort_rate': 0.4, 'aa_malaria_mort_rate': 0.2}])
    with pytest.raises(ValueError):
        compute_as_rr(df, anchor_year=2023)


# ---------------------------------------------------------------------------
# disaggregate_malaria_draws
# ---------------------------------------------------------------------------

@pytest.fixture
def aa_draws_df():
    """All-age draws: 2 loc × 2 year × 2 draw. aa count varies by draw."""
    rows = []
    for loc in [1, 2]:
        for year in [2025, 2026]:
            for draw in [0, 1]:
                rows.append({
                    'location_id': loc, 'year_id': year, 'draw': draw,
                    'aa_malaria_inc_count': 100.0 + draw * 10,
                    'aa_malaria_mort_count': 10.0 + draw,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def rr_df():
    """Static age/sex rr, incl. age_group_id 2. 2 loc × 3 age × 2 sex."""
    rows = []
    for loc in [1, 2]:
        for age in [2, 3, 10]:
            for sex in [1, 2]:
                rows.append({'location_id': loc, 'age_group_id': age, 'sex_id': sex,
                             'rr_inc_as': 1.0, 'rr_mort_as': 1.0})
    return pd.DataFrame(rows)


@pytest.fixture
def as_pop_df():
    """Year-varying age/sex population. 2 loc × 2 year × 3 age × 2 sex; uniform 1000."""
    rows = []
    for loc in [1, 2]:
        for year in [2025, 2026]:
            for age in [2, 3, 10]:
                for sex in [1, 2]:
                    rows.append({'location_id': loc, 'year_id': year,
                                 'age_group_id': age, 'sex_id': sex, 'population': 1000.0})
    return pd.DataFrame(rows)


def test_draws_output_shape(aa_draws_df, rr_df, as_pop_df):
    out = disaggregate_malaria_draws(aa_draws_df, rr_df, as_pop_df)
    assert set(out.columns) == {
        'location_id', 'year_id', 'age_group_id', 'sex_id', 'draw',
        'malaria_inc_count_pred', 'malaria_mort_count_pred'}
    # loc(2) × year(2) × draw(2) × age(3) × sex(2) = 48
    assert len(out) == 48


def test_draws_totals_preserved_per_draw(aa_draws_df, rr_df, as_pop_df):
    out = disaggregate_malaria_draws(aa_draws_df, rr_df, as_pop_df)
    tot = (out.groupby(['location_id', 'year_id', 'draw'])
           [['malaria_inc_count_pred', 'malaria_mort_count_pred']].sum().reset_index())
    m = tot.merge(aa_draws_df, on=['location_id', 'year_id', 'draw'])
    assert np.allclose(m['malaria_inc_count_pred'], m['aa_malaria_inc_count'])
    assert np.allclose(m['malaria_mort_count_pred'], m['aa_malaria_mort_count'])


def test_draws_age2_zeroed(aa_draws_df, rr_df, as_pop_df):
    out = disaggregate_malaria_draws(aa_draws_df, rr_df, as_pop_df)
    z = out[out.age_group_id == 2]
    assert (z['malaria_inc_count_pred'] == 0).all()
    assert (z['malaria_mort_count_pred'] == 0).all()


def test_draws_year_varying_pop_shifts_fractions(aa_draws_df, rr_df, as_pop_df):
    """Choice (b): holding rr fixed but aging the population moves the age fractions."""
    p = as_pop_df.copy()
    p.loc[(p.year_id == 2026) & (p.age_group_id == 10), 'population'] = 3000.0
    out = disaggregate_malaria_draws(aa_draws_df, rr_df, p)

    def age10_frac(year):
        sub = out[(out.location_id == 1) & (out.year_id == year) & (out.draw == 0)]
        return (sub[sub.age_group_id == 10]['malaria_inc_count_pred'].sum()
                / sub['malaria_inc_count_pred'].sum())

    assert age10_frac(2026) > age10_frac(2025)


def test_draws_missing_rr_location_no_nan(aa_draws_df, rr_df, as_pop_df):
    """A location absent from rr contributes rr 0 → fractions 0 → counts 0, never NaN."""
    rr_loc1_only = rr_df[rr_df.location_id == 1]
    out = disaggregate_malaria_draws(aa_draws_df, rr_loc1_only, as_pop_df)
    l2 = out[out.location_id == 2]
    assert l2['malaria_inc_count_pred'].notna().all()
    assert (l2['malaria_inc_count_pred'] == 0).all()


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


def test_dengue_incidence_only_when_no_cfr(dengue_df):
    # An incidence-only frame (no CFR column) yields inc but not mort.
    df = dengue_df.drop(columns=['dengue_cfr_pred'])
    result = disaggregate_age_sex_dengue(df)
    assert 'dengue_inc_count_pred' in result.columns
    assert 'dengue_mort_count_pred' not in result.columns
    expected_inc = 1000.0 * np.exp(np.log(0.01)) * 1.5
    assert np.allclose(result['dengue_inc_count_pred'].values, expected_inc)
