"""
Tests for lib/processing/locations.py — dengue eligibility gate + all-grain
fit-location set (levels 3-5 tagged with the most_detailed_* flags).
"""

import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.locations import (
    dengue_eligible_a0_ids,
    dengue_fit_locations,
)


@pytest.fixture
def hierarchy_df():
    # Country A0=10 (eligible): admin-1 20, admin-2s 30/31.
    # Country A0=11 (not eligible): admin-2 41.
    return pd.DataFrame([
        {'location_id': 10, 'A0_location_id': 10, 'level': 3,
         'most_detailed_lsae': 0, 'most_detailed_fhs': 1, 'most_detailed_gbd': 1},
        {'location_id': 20, 'A0_location_id': 10, 'level': 4,
         'most_detailed_lsae': 0, 'most_detailed_fhs': 0, 'most_detailed_gbd': 0},
        {'location_id': 30, 'A0_location_id': 10, 'level': 5,
         'most_detailed_lsae': 1, 'most_detailed_fhs': 0, 'most_detailed_gbd': 0},
        {'location_id': 31, 'A0_location_id': 10, 'level': 5,
         'most_detailed_lsae': 1, 'most_detailed_fhs': 0, 'most_detailed_gbd': 0},
        {'location_id': 11, 'A0_location_id': 11, 'level': 3,
         'most_detailed_lsae': 0, 'most_detailed_fhs': 1, 'most_detailed_gbd': 1},
        {'location_id': 41, 'A0_location_id': 11, 'level': 5,
         'most_detailed_lsae': 1, 'most_detailed_fhs': 0, 'most_detailed_gbd': 0},
    ])


@pytest.fixture
def aa_df():
    # Gate reads A0 rows (location_id == A0_location_id) in the gate year.
    # A0=10 clears (mort>0 AND inc>0); A0=11 fails (mort==0).
    return pd.DataFrame([
        {'location_id': 10, 'year_id': 2023, 'dengue_mort_count': 5.0, 'dengue_inc_count': 100.0},
        {'location_id': 11, 'year_id': 2023, 'dengue_mort_count': 0.0, 'dengue_inc_count': 50.0},
        # non-gate-year / non-A0 rows must be ignored by the gate:
        {'location_id': 10, 'year_id': 2010, 'dengue_mort_count': 0.0, 'dengue_inc_count': 0.0},
        {'location_id': 30, 'year_id': 2023, 'dengue_mort_count': 3.0, 'dengue_inc_count': 40.0},
    ])


def test_eligible_a0_selects_only_gated_country(aa_df, hierarchy_df):
    elig = dengue_eligible_a0_ids(aa_df, hierarchy_df,
                                  mort_threshold=0.0, inc_threshold=0.0, gate_year=2023)
    assert set(elig) == {10}


def test_fit_locations_spans_levels_3_to_5_within_eligible_a0(aa_df, hierarchy_df):
    out = dengue_fit_locations(aa_df, hierarchy_df,
                               mort_threshold=0.0, inc_threshold=0.0, gate_year=2023)
    assert set(out['location_id']) == {10, 20, 30, 31}   # A0=11's 11/41 excluded
    assert set(out['level']) == {3, 4, 5}


def test_fit_locations_has_all_three_grain_flags(aa_df, hierarchy_df):
    out = dengue_fit_locations(aa_df, hierarchy_df,
                               mort_threshold=0.0, inc_threshold=0.0, gate_year=2023)
    for col in ['location_id', 'A0_location_id', 'level',
                'most_detailed_lsae', 'most_detailed_fhs', 'most_detailed_gbd']:
        assert col in out.columns


def test_fit_locations_flag_values_match_hierarchy(aa_df, hierarchy_df):
    out = dengue_fit_locations(aa_df, hierarchy_df,
                               mort_threshold=0.0, inc_threshold=0.0, gate_year=2023).set_index('location_id')
    # country 10 is FHS/GBD most-detailed but not LSAE most-detailed
    assert out.loc[10, 'most_detailed_fhs'] == 1
    assert out.loc[10, 'most_detailed_lsae'] == 0
    # admin-2 30 is LSAE most-detailed only
    assert out.loc[30, 'most_detailed_lsae'] == 1
    assert out.loc[30, 'most_detailed_fhs'] == 0


def test_fit_locations_custom_levels_subset(aa_df, hierarchy_df):
    out = dengue_fit_locations(aa_df, hierarchy_df, levels=(5,),
                               mort_threshold=0.0, inc_threshold=0.0, gate_year=2023)
    assert set(out['location_id']) == {30, 31}
    assert set(out['level']) == {5}
