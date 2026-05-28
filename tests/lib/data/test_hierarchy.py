"""
Tests for lib/data/hierarchy.py

Uses synthetic hierarchy DataFrames — no real pipeline data.
"""

import pandas as pd
import pytest
from unittest.mock import patch

from idd_forecast_mbp.lib.data.hierarchy import (
    level_filter,
    get_location_ids,
    load_hierarchy,
    make_location_filter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hierarchy_df():
    """
    Minimal synthetic hierarchy with levels 1–5.
    Each level has a distinct set of location_ids.

    level 1: [1]
    level 2: [2, 3]
    level 3: [4, 5, 6]
    level 4: [7, 8, 9, 10]
    level 5: [11, 12, 13, 14, 15]
    """
    rows = []
    loc = 1
    for level in range(1, 6):
        for _ in range(level):
            rows.append({'location_id': loc, 'level': level, 'parent_id': loc - 1})
            loc += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# level_filter
# ---------------------------------------------------------------------------

def test_level_filter_single_level(hierarchy_df):
    f = level_filter(hierarchy_df, 5)
    assert f[0] == 'location_id'
    assert f[1] == 'in'
    assert set(f[2]) == {11, 12, 13, 14, 15}


def test_level_filter_range(hierarchy_df):
    f = level_filter(hierarchy_df, 3, 5)
    ids = set(f[2])
    # Should include levels 3, 4, 5
    assert {4, 5, 6} <= ids       # level 3
    assert {7, 8, 9, 10} <= ids   # level 4
    assert {11, 12, 13, 14, 15} <= ids  # level 5
    # Should not include levels 1 or 2
    assert 1 not in ids
    assert 2 not in ids


def test_level_filter_return_ids(hierarchy_df):
    f, ids = level_filter(hierarchy_df, 5, return_ids=True)
    assert isinstance(ids, list)
    assert set(ids) == {11, 12, 13, 14, 15}
    assert f == ('location_id', 'in', ids)


def test_level_filter_single_level_no_end(hierarchy_df):
    # end_level defaults to start_level
    f1 = level_filter(hierarchy_df, 3)
    f2 = level_filter(hierarchy_df, 3, 3)
    assert set(f1[2]) == set(f2[2])


def test_level_filter_returns_tuple(hierarchy_df):
    result = level_filter(hierarchy_df, 4)
    assert isinstance(result, tuple)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# get_location_ids
# ---------------------------------------------------------------------------

def test_get_location_ids_single(hierarchy_df):
    ids = get_location_ids(hierarchy_df, 5)
    assert set(ids) == {11, 12, 13, 14, 15}


def test_get_location_ids_list(hierarchy_df):
    ids = get_location_ids(hierarchy_df, [4, 5])
    assert {7, 8, 9, 10} <= set(ids)
    assert {11, 12, 13, 14, 15} <= set(ids)
    assert 1 not in ids


def test_get_location_ids_extra_filter(hierarchy_df):
    # Add a gbd_location_id column for testing extra_filter
    df = hierarchy_df.copy()
    df['gbd_location_id'] = df['location_id']  # identity for simplicity
    allowed_gbd = {11, 12}
    extra = df['gbd_location_id'].isin(allowed_gbd)
    ids = get_location_ids(df, 5, extra_filter=extra)
    assert set(ids) == {11, 12}


def test_get_location_ids_returns_list(hierarchy_df):
    result = get_location_ids(hierarchy_df, 3)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# make_location_filter
# ---------------------------------------------------------------------------

def test_make_location_filter_structure():
    ids = [1, 2, 3]
    f = make_location_filter(ids)
    assert f == ('location_id', 'in', [1, 2, 3])


def test_make_location_filter_empty():
    f = make_location_filter([])
    assert f == ('location_id', 'in', [])


def test_make_location_filter_roundtrip(hierarchy_df):
    # level_filter and make_location_filter should produce the same tuple
    ids = get_location_ids(hierarchy_df, 5)
    f1 = make_location_filter(ids)
    f2 = level_filter(hierarchy_df, 5)
    assert set(f1[2]) == set(f2[2])


# ---------------------------------------------------------------------------
# load_hierarchy
# ---------------------------------------------------------------------------

def test_load_hierarchy_uses_explicit_path(tmp_path, hierarchy_df):
    path = tmp_path / 'hierarchy.parquet'
    hierarchy_df.to_parquet(path, index=False)
    result = load_hierarchy(path)
    assert set(result.columns) >= {'location_id', 'level'}


def test_load_hierarchy_default_path(hierarchy_df, tmp_path):
    """When path=None, load_hierarchy reads from mbpc.HIERARCHY_READ_PATH."""
    path = tmp_path / 'full_hierarchy_lsae_1209.parquet'
    hierarchy_df.to_parquet(path, index=False)

    with patch('idd_forecast_mbp.lib.data.hierarchy.read_parquet_with_integer_ids') as mock_read:
        mock_read.return_value = hierarchy_df
        result = load_hierarchy(path=None)

    assert mock_read.called
    assert isinstance(result, pd.DataFrame)
