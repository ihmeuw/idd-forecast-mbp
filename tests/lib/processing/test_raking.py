"""
Tests for lib/processing/raking.py
"""

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.raking import (
    logit_shift_rake,
    rake_aa_count_lsae_to_gbd,
    rake_level,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_hierarchy():
    """Consistent 6-level hierarchy: each level's parent is exactly one level up.

    L0: [1]
    L1: [2, 3]
    L2: [4, 5, 6, 7]
    L3: [8..15]   parent = (i - 8) // 2 + 4  → 4..7
    L4: [16..31]  parent = (i - 16) // 2 + 8  → 8..15
    L5: [32..63]  parent = (i - 32) // 2 + 16 → 16..31
    """
    rows = []
    rows.append({'location_id': 1, 'parent_id': 0, 'level': 0})
    for i in range(2, 4):
        rows.append({'location_id': i, 'parent_id': 1, 'level': 1})
    for i in range(4, 8):
        rows.append({'location_id': i, 'parent_id': (i - 4) // 2 + 2, 'level': 2})
    for i in range(8, 16):
        rows.append({'location_id': i, 'parent_id': (i - 8) // 2 + 4, 'level': 3})
    for i in range(16, 32):
        rows.append({'location_id': i, 'parent_id': (i - 16) // 2 + 8, 'level': 4})
    for i in range(32, 64):
        rows.append({'location_id': i, 'parent_id': (i - 32) // 2 + 16, 'level': 5})
    return pd.DataFrame(rows)


@pytest.fixture
def problematic_rules():
    return {
        'rate_max': {3: 1.0, 4: 1.0, 5: 1.0},
        'count_raking_factor_max': 100.0,
        'count_raking_factor_conditional': 10.0,
        'rate_max_conditional': 0.5,
    }


def make_level_df(hierarchy_df, level, count_val, pop_val=1000.0, include_set_by_gbd=True):
    """Build a simple level DataFrame from hierarchy."""
    locs = hierarchy_df[hierarchy_df['level'] == level]['location_id'].tolist()
    rows = []
    for loc in locs:
        for year in [2020, 2021]:
            row = {
                'location_id': loc,
                'year_id': year,
                'count': count_val,
                'population': pop_val,
            }
            if include_set_by_gbd:
                row['set_by_gbd'] = False
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# rake_level
# ---------------------------------------------------------------------------

def test_rake_level_returns_dataframe(simple_hierarchy, problematic_rules):
    level5_df = make_level_df(simple_hierarchy, 5, count_val=5.0)
    level4_df = make_level_df(simple_hierarchy, 4, count_val=10.0)

    result = rake_level('count', level5_df, level4_df, problematic_rules, simple_hierarchy, level=5)
    assert isinstance(result, pd.DataFrame)


def test_rake_level_output_has_count_column(simple_hierarchy, problematic_rules):
    level5_df = make_level_df(simple_hierarchy, 5, count_val=5.0)
    level4_df = make_level_df(simple_hierarchy, 4, count_val=10.0)

    result = rake_level('count', level5_df, level4_df, problematic_rules, simple_hierarchy, level=5)
    assert 'count' in result.columns


def test_rake_level_no_intermediate_columns(simple_hierarchy, problematic_rules):
    level5_df = make_level_df(simple_hierarchy, 5, count_val=5.0)
    level4_df = make_level_df(simple_hierarchy, 4, count_val=10.0)

    result = rake_level('count', level5_df, level4_df, problematic_rules, simple_hierarchy, level=5)
    for col in result.columns:
        assert 'raking' not in col
        assert 'based' not in col
    assert 'parent_id' not in result.columns


def test_rake_level_set_by_gbd_unchanged(simple_hierarchy, problematic_rules):
    """Rows with set_by_gbd=True must not have their count changed."""
    level5_df = make_level_df(simple_hierarchy, 5, count_val=5.0)
    level5_df.loc[level5_df['location_id'] == 17, 'set_by_gbd'] = True
    original_count = level5_df.loc[level5_df['location_id'] == 17, 'count'].values.copy()

    level4_df = make_level_df(simple_hierarchy, 4, count_val=20.0)
    result = rake_level('count', level5_df, level4_df, problematic_rules, simple_hierarchy, level=5)

    result_count = result.loc[result['location_id'] == 17, 'count'].values
    assert np.allclose(result_count, original_count)


def test_rake_level_sums_match_parent(simple_hierarchy, problematic_rules):
    """After raking, sum of level-5 children per parent should equal level-4 target."""
    level5_df = make_level_df(simple_hierarchy, 5, count_val=3.0)
    level4_df = make_level_df(simple_hierarchy, 4, count_val=10.0)
    # level4_df has 'set_by_gbd' column; rename to avoid conflicts
    level4_target = level4_df[['location_id', 'year_id', 'count']].copy()

    result = rake_level('count', level5_df, level4_df, problematic_rules, simple_hierarchy, level=5)
    result_with_parent = result.merge(
        simple_hierarchy[['location_id', 'parent_id']], on='location_id'
    )

    for (parent, year), grp in result_with_parent.groupby(['parent_id', 'year_id']):
        child_sum = grp['count'].sum()
        target_rows = level4_target[
            (level4_target['location_id'] == parent) &
            (level4_target['year_id'] == year)
        ]
        if len(target_rows) == 0:  # pragma: no cover
            continue  # parent not in level4 (shouldn't happen with consistent hierarchy)
        target = target_rows['count'].iloc[0]
        assert np.isclose(child_sum, target, rtol=1e-6)


# ---------------------------------------------------------------------------
# rake_aa_count_lsae_to_gbd
# ---------------------------------------------------------------------------

@pytest.fixture
def rake_aa_inputs(simple_hierarchy):
    """LSAE and GBD inputs for rake_aa_count_lsae_to_gbd tests.

    LSAE has levels 4 and 5 (no set_by_gbd — the function sets that up internally).
    GBD has levels 0–4, used both as replacement values and as level-3 raking targets.
    """
    lsae_all = pd.concat([
        make_level_df(simple_hierarchy, 4, count_val=6.0, include_set_by_gbd=False),
        make_level_df(simple_hierarchy, 5, count_val=3.0, include_set_by_gbd=False),
    ], ignore_index=True)

    gbd_df = pd.concat([
        make_level_df(simple_hierarchy, l, count_val=20.0, include_set_by_gbd=False)
        for l in [0, 1, 2, 3, 4]
    ], ignore_index=True)

    return lsae_all, gbd_df


def test_rake_aa_count_returns_none_by_default(simple_hierarchy, problematic_rules, rake_aa_inputs):
    lsae_all, gbd_df = rake_aa_inputs
    result = rake_aa_count_lsae_to_gbd(
        'count', simple_hierarchy, gbd_df, lsae_all, problematic_rules
    )
    assert result is None


def test_rake_aa_count_return_full_df(simple_hierarchy, problematic_rules, rake_aa_inputs):
    lsae_all, gbd_df = rake_aa_inputs
    result = rake_aa_count_lsae_to_gbd(
        'count', simple_hierarchy, gbd_df, lsae_all, problematic_rules,
        return_full_df=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert 'count' in result.columns


def test_rake_aa_count_gbd_locations_unchanged(simple_hierarchy, problematic_rules, rake_aa_inputs):
    """Locations present in GBD at level 3 should have their GBD values in the output."""
    lsae_all, gbd_df = rake_aa_inputs
    result = rake_aa_count_lsae_to_gbd(
        'count', simple_hierarchy, gbd_df, lsae_all, problematic_rules,
        return_full_df=True,
    )
    level3_ids = simple_hierarchy[simple_hierarchy['level'] == 3]['location_id'].tolist()
    level3_result = result[result['location_id'].isin(level3_ids)]
    assert (level3_result['count'] == 20.0).all()


# ---------------------------------------------------------------------------
# logit_shift_rake
# ---------------------------------------------------------------------------

@pytest.fixture
def logit_rake_dfs():
    """Forecast and observed DataFrames for logit shift test."""
    locs = [1, 2, 3]
    years = [2020, 2021, 2022, 2023]
    rows = []
    for loc in locs:
        for year in years:
            rows.append({
                'location_id': loc,
                'year_id': year,
                'cfr_raw': 0.3,
            })
    forecast_df = pd.DataFrame(rows)

    obs_rows = [
        {'location_id': 1, 'cfr': 0.5},
        {'location_id': 2, 'cfr': 0.2},
        {'location_id': 3, 'cfr': 0.4},
    ]
    observed_df = pd.DataFrame(obs_rows)
    return forecast_df, observed_df


def test_logit_shift_rake_adds_output_col(logit_rake_dfs):
    forecast_df, observed_df = logit_rake_dfs
    result = logit_shift_rake(forecast_df, observed_df, rate_column='cfr', pred_column='cfr', rake_year=2022)
    assert 'cfr' in result.columns


def test_logit_shift_rake_drops_raw_col(logit_rake_dfs):
    forecast_df, observed_df = logit_rake_dfs
    result = logit_shift_rake(forecast_df, observed_df, rate_column='cfr', pred_column='cfr', rake_year=2022)
    assert 'cfr_raw' not in result.columns


def test_logit_shift_rake_at_rake_year_matches_observed(logit_rake_dfs):
    """At rake_year, cfr should equal logit(obs) because shift = logit(obs) - pred_raw."""
    from idd_forecast_mbp.lib.utils.transforms import logit as logit_fn

    forecast_df, observed_df = logit_rake_dfs
    result = logit_shift_rake(
        forecast_df, observed_df, rate_column='cfr', pred_column='cfr', rake_year=2022
    )
    # pred_raw = 0.3 (in logit space)
    # shift = logit(obs) - 0.3
    # cfr at rake_year = pred_raw + shift = logit(obs)
    obs_dict = dict(zip(observed_df['location_id'], observed_df['cfr']))
    rake_rows = result[result['year_id'] == 2022]
    for _, row in rake_rows.iterrows():
        obs = obs_dict[int(row['location_id'])]
        expected = logit_fn(np.array([obs]))[0]
        assert np.isclose(row['cfr'], expected, atol=1e-6)


def test_logit_shift_rake_preserves_row_count(logit_rake_dfs):
    forecast_df, observed_df = logit_rake_dfs
    result = logit_shift_rake(forecast_df, observed_df, rate_column='cfr', pred_column='cfr', rake_year=2022)
    assert len(result) == len(forecast_df)


def test_rake_aa_count_writes_parquet(tmp_path, simple_hierarchy, problematic_rules, rake_aa_inputs):
    """aa_full_count_df_path is not None → write_parquet is called (covers line 281)."""
    lsae_all, gbd_df = rake_aa_inputs
    out = tmp_path / 'raked.parquet'
    rake_aa_count_lsae_to_gbd(
        'count', simple_hierarchy, gbd_df, lsae_all, problematic_rules,
        aa_full_count_df_path=str(out),
    )
    assert out.exists()
