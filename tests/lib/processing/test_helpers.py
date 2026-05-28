"""
Tests for lib/processing/helpers.py and lib/processing/_helpers.py
"""

import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.helpers import (
    level_filter,
    make_aa_df_square,
    prep_df,
)
from idd_forecast_mbp.lib.processing._helpers import (  # re-export module
    level_filter as level_filter_alias,
    make_aa_df_square as make_aa_df_square_alias,
    prep_df as prep_df_alias,
)


@pytest.fixture
def hierarchy_df():
    rows = []
    loc = 1
    for level in range(1, 6):
        for _ in range(level):
            rows.append({'location_id': loc, 'level': level, 'parent_id': loc - 1})
            loc += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _helpers re-export (covers _helpers.py import statements)
# ---------------------------------------------------------------------------

def test_helpers_reexport_is_same_function():
    assert level_filter_alias is level_filter
    assert make_aa_df_square_alias is make_aa_df_square
    assert prep_df_alias is prep_df


# ---------------------------------------------------------------------------
# level_filter
# ---------------------------------------------------------------------------

def test_level_filter_single_level(hierarchy_df):
    f = level_filter(hierarchy_df, 5)
    assert f[0] == 'location_id'
    assert f[1] == 'in'
    assert set(f[2]) == {11, 12, 13, 14, 15}


def test_level_filter_end_level_defaults_to_start(hierarchy_df):
    f1 = level_filter(hierarchy_df, 3)
    f2 = level_filter(hierarchy_df, 3, end_level=3)
    assert set(f1[2]) == set(f2[2])


def test_level_filter_range(hierarchy_df):
    f = level_filter(hierarchy_df, 3, end_level=5)
    ids = set(f[2])
    assert {4, 5, 6} <= ids        # level 3
    assert {7, 8, 9, 10} <= ids    # level 4
    assert {11, 12, 13, 14, 15} <= ids  # level 5
    assert 1 not in ids


def test_level_filter_return_ids(hierarchy_df):
    f, ids = level_filter(hierarchy_df, 5, return_ids=True)
    assert isinstance(ids, list)
    assert set(ids) == {11, 12, 13, 14, 15}
    assert f == ('location_id', 'in', ids)


def test_level_filter_return_ids_false(hierarchy_df):
    result = level_filter(hierarchy_df, 4, return_ids=False)
    assert isinstance(result, tuple)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# prep_df
# ---------------------------------------------------------------------------

def test_prep_df_adds_level(hierarchy_df):
    df = pd.DataFrame({'location_id': [1, 2, 3], 'count': [1.0, 2.0, 3.0]})
    result = prep_df(df, hierarchy_df)
    assert 'level' in result.columns


def test_prep_df_drops_parent_id(hierarchy_df):
    df = pd.DataFrame({
        'location_id': [1, 2],
        'parent_id': [0, 1],
        'count': [1.0, 2.0],
    })
    result = prep_df(df, hierarchy_df)
    assert 'parent_id' not in result.columns


def test_prep_df_level_already_present(hierarchy_df):
    df = pd.DataFrame({'location_id': [1], 'level': [99], 'count': [1.0]})
    result = prep_df(df, hierarchy_df)
    # Should not overwrite existing level column
    assert result['level'].iloc[0] == 99


def test_prep_df_no_parent_id_unchanged(hierarchy_df):
    df = pd.DataFrame({'location_id': [1, 2], 'count': [1.0, 2.0]})
    result = prep_df(df, hierarchy_df)
    assert 'parent_id' not in result.columns
    assert 'level' in result.columns


# ---------------------------------------------------------------------------
# make_aa_df_square
# ---------------------------------------------------------------------------

def test_make_aa_df_square_fills_missing(hierarchy_df):
    # Only location 11 has data for year 2020; 12-15 are missing
    df = pd.DataFrame({'location_id': [11], 'year_id': [2020], 'count': [5.0]})
    result = make_aa_df_square('count', df, hierarchy_df, level_start=5, level_end=5)
    locs = set(result['location_id'].tolist())
    assert {11, 12, 13, 14, 15} <= locs


def test_make_aa_df_square_fills_with_zero(hierarchy_df):
    df = pd.DataFrame({'location_id': [11], 'year_id': [2020], 'count': [5.0]})
    result = make_aa_df_square('count', df, hierarchy_df, level_start=5, level_end=5)
    missing = result[result['location_id'] != 11]
    assert (missing['count'] == 0).all()


def test_make_aa_df_square_no_missing_unchanged(hierarchy_df):
    level5_ids = hierarchy_df[hierarchy_df['level'] == 5]['location_id'].tolist()
    rows = [{'location_id': loc, 'year_id': 2020, 'count': 1.0} for loc in level5_ids]
    df = pd.DataFrame(rows)
    result = make_aa_df_square('count', df, hierarchy_df, level_start=5, level_end=5)
    assert len(result) == len(df)


def test_make_aa_df_square_multi_variable(hierarchy_df):
    df = pd.DataFrame({'location_id': [11], 'year_id': [2020], 'a': [1.0], 'b': [2.0]})
    result = make_aa_df_square(['a', 'b'], df, hierarchy_df, level_start=5, level_end=5)
    assert 'a' in result.columns
    assert 'b' in result.columns
