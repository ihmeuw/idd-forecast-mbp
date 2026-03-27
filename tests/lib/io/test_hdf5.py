"""
Tests for lib/io/hdf5.py

Uses small synthetic DataFrames — no real pipeline data.
"""

import os
import pytest
import numpy as np
import pandas as pd

from idd_forecast_mbp.lib.io.hdf5 import (
    write_hdf,
    create_hdf_structure,
    write_draw_column,
    read_hdf_metadata,
)


@pytest.fixture
def simple_df():
    return pd.DataFrame({
        'location_id': [1, 2, 3],
        'year_id':     [2030, 2040, 2050],
        'value':       [10.0, 20.0, 30.0],
    })


# ---------------------------------------------------------------------------
# write_hdf / round-trip
# ---------------------------------------------------------------------------

def test_round_trip(tmp_path, simple_df):
    path = tmp_path / 'test.h5'
    write_hdf(simple_df, path)
    result = pd.read_hdf(path, key='df')
    assert len(result) == len(simple_df)
    assert list(result.columns) == list(simple_df.columns)


def test_write_creates_parent_dirs(tmp_path, simple_df):
    path = tmp_path / 'deep' / 'nested' / 'test.h5'
    write_hdf(simple_df, path)
    assert path.exists()


def test_write_sets_permissions(tmp_path, simple_df):
    path = tmp_path / 'test.h5'
    write_hdf(simple_df, path)
    mode = oct(os.stat(path).st_mode)[-3:]
    assert mode == '775'


def test_write_custom_key(tmp_path, simple_df):
    path = tmp_path / 'test.h5'
    write_hdf(simple_df, path, key='mydata')
    result = pd.read_hdf(path, key='mydata')
    assert len(result) == len(simple_df)


# ---------------------------------------------------------------------------
# create_hdf_structure / write_draw_column / read_hdf_metadata
# ---------------------------------------------------------------------------

def test_create_and_write_draw_column(tmp_path):
    meta = pd.DataFrame({'location_id': [1, 2, 3], 'year_id': [2030, 2030, 2030]})
    draws = ['draw_0', 'draw_1']
    path = tmp_path / 'draws.h5'

    create_hdf_structure(path, meta, draws, ['location_id', 'year_id'])

    values = np.array([0.1, 0.2, 0.3])
    write_draw_column(path, 'draw_0', values)

    import h5py
    with h5py.File(path, 'r') as f:
        assert np.allclose(f['draw_0'][:], values)
        assert np.allclose(f['draw_1'][:], np.zeros(3))


def test_write_draw_column_bad_key(tmp_path):
    meta = pd.DataFrame({'location_id': [1]})
    path = tmp_path / 'draws.h5'
    create_hdf_structure(path, meta, ['draw_0'], ['location_id'])

    with pytest.raises(KeyError, match='not found'):
        write_draw_column(path, 'nonexistent_draw', np.array([1.0]))


def test_read_hdf_metadata(tmp_path):
    meta = pd.DataFrame({'location_id': [1, 2], 'year_id': [2030, 2040]})
    path = tmp_path / 'draws.h5'
    create_hdf_structure(path, meta, ['draw_0'], ['location_id', 'year_id'])

    result = read_hdf_metadata(path, ['location_id', 'year_id'])
    assert list(result.columns) == ['location_id', 'year_id']
    assert len(result) == 2


def test_read_hdf_metadata_ignores_missing_columns(tmp_path):
    meta = pd.DataFrame({'location_id': [1]})
    path = tmp_path / 'draws.h5'
    create_hdf_structure(path, meta, [], ['location_id'])

    # Requesting a column that doesn't exist should return only the ones that do
    result = read_hdf_metadata(path, ['location_id', 'nonexistent'])
    assert 'location_id' in result.columns
    assert 'nonexistent' not in result.columns
