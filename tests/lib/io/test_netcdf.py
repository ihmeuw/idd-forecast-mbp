"""
Tests for lib/io/netcdf.py

Uses small synthetic Datasets — no real pipeline data.
"""

import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp.lib.io.netcdf import (
    cast_coordinate_types,
    ensure_id_coordinates_are_integers,
    sort_coordinates,
    read_netcdf_with_integer_ids,
    write_netcdf,
    convert_to_xarray,
    convert_with_preset,
    filter_ds_by_multiple_coords,
    filter_ds_by_range,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_ds():
    """Small aa_variables-shaped Dataset."""
    return xr.Dataset(
        {'value': (['location_id', 'year_id'], np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={'location_id': [2, 1], 'year_id': [2040, 2030]},
    )


@pytest.fixture
def simple_df():
    """Small all-combinations DataFrame for convert_to_xarray tests."""
    rows = []
    for loc in [1, 2]:
        for yr in [2030, 2040]:
            rows.append({'location_id': loc, 'year_id': yr, 'value': float(loc + yr)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# cast_coordinate_types
# ---------------------------------------------------------------------------

def test_cast_coordinate_types_known_coords(simple_ds):
    ds = xr.Dataset(
        {'v': (['location_id', 'year_id'], np.ones((2, 2)))},
        coords={'location_id': np.array([1, 2], dtype='int64'),
                'year_id':     np.array([2030, 2040], dtype='int64')},
    )
    result = cast_coordinate_types(ds)
    assert result['location_id'].dtype == np.int32
    assert result['year_id'].dtype == np.int16


def test_cast_coordinate_types_ignores_unknown(simple_ds):
    # Coordinates not in the map should be untouched
    ds = xr.Dataset(
        {'v': (['x'], np.ones(3))},
        coords={'x': [10, 20, 30]},
    )
    result = cast_coordinate_types(ds)
    assert 'x' in result.coords


# ---------------------------------------------------------------------------
# sort_coordinates
# ---------------------------------------------------------------------------

def test_sort_coordinates_default_priority(simple_ds):
    result = sort_coordinates(simple_ds)
    assert list(result['location_id'].values) == [1, 2]
    assert list(result['year_id'].values) == [2030, 2040]


# ---------------------------------------------------------------------------
# write_netcdf / read_netcdf_with_integer_ids — round-trip
# ---------------------------------------------------------------------------

def test_round_trip(tmp_path, simple_ds):
    path = tmp_path / 'test.nc'
    write_netcdf(simple_ds, path)
    result = read_netcdf_with_integer_ids(path)
    assert set(result.data_vars) == set(simple_ds.data_vars)
    assert set(result.dims) == set(simple_ds.dims)


def test_write_creates_parent_dirs(tmp_path, simple_ds):
    path = tmp_path / 'deep' / 'nested' / 'test.nc'
    write_netcdf(simple_ds, path)
    assert path.exists()


def test_write_sets_permissions(tmp_path, simple_ds):
    path = tmp_path / 'test.nc'
    write_netcdf(simple_ds, path)
    mode = oct(os.stat(path).st_mode)[-3:]
    assert mode == '775'


def test_write_mkdir_false_raises_on_missing_dir(tmp_path, simple_ds):
    path = tmp_path / 'missing_dir' / 'test.nc'
    with pytest.raises(Exception):
        write_netcdf(simple_ds, path, mkdir=False, max_retries=1)


# ---------------------------------------------------------------------------
# convert_to_xarray
# ---------------------------------------------------------------------------

def test_convert_to_xarray_auto_dimensions(simple_df):
    ds = convert_to_xarray(simple_df)
    assert 'location_id' in ds.dims
    assert 'year_id' in ds.dims
    assert 'value' in ds.data_vars


def test_convert_to_xarray_validates_rectangular_grid():
    # 2 locations × 2 years = 4 expected, but only 3 rows — missing (2, 2040)
    df = pd.DataFrame({
        'location_id': [1, 2, 1],
        'year_id':     [2030, 2030, 2040],
        'v':           [1.0, 2.0, 3.0],
    })
    with pytest.raises(ValueError, match='rectangular'):
        convert_to_xarray(df, validate_dimensions=True)


def test_convert_to_xarray_skip_validation():
    df = pd.DataFrame({'location_id': [1, 2], 'year_id': [2030, 2030], 'v': [1.0, 2.0]})
    ds = convert_to_xarray(df, validate_dimensions=False)
    assert 'v' in ds.data_vars


def test_convert_to_xarray_explicit_dimensions(simple_df):
    ds = convert_to_xarray(simple_df, dimensions=['location_id', 'year_id'])
    assert set(ds.dims) == {'location_id', 'year_id'}


def test_convert_to_xarray_missing_dimension_raises(simple_df):
    with pytest.raises(ValueError, match='not found'):
        convert_to_xarray(simple_df, dimensions=['location_id', 'nonexistent_id'])


# ---------------------------------------------------------------------------
# convert_with_preset
# ---------------------------------------------------------------------------

def test_convert_with_preset_aa(simple_df):
    ds = convert_with_preset(simple_df, preset='aa_variables')
    assert ds['location_id'].dtype == np.int32
    assert ds['year_id'].dtype == np.int16


def test_convert_with_preset_unknown_raises():
    df = pd.DataFrame({'location_id': [1], 'year_id': [2030], 'v': [1.0]})
    with pytest.raises(ValueError, match='Unknown preset'):
        convert_with_preset(df, preset='bad_preset')


# ---------------------------------------------------------------------------
# filter_ds_by_multiple_coords
# ---------------------------------------------------------------------------

def test_filter_ds_scalar(simple_ds):
    result = filter_ds_by_multiple_coords(simple_ds, location_id=1)
    assert list(result['location_id'].values) == [1]


def test_filter_ds_list(simple_ds):
    result = filter_ds_by_multiple_coords(simple_ds, location_id=[1, 2])
    assert len(result['location_id']) == 2


def test_filter_ds_bad_coord(simple_ds):
    with pytest.raises(ValueError, match='not found'):
        filter_ds_by_multiple_coords(simple_ds, nonexistent=1)


# ---------------------------------------------------------------------------
# filter_ds_by_range
# ---------------------------------------------------------------------------

def test_filter_ds_by_range_basic(simple_ds):
    result = filter_ds_by_range(simple_ds, year_id=(2030, 2030))
    assert list(result['year_id'].values) == [2030]


def test_filter_ds_by_range_inclusive(simple_ds):
    result = filter_ds_by_range(simple_ds, year_id=(2030, 2040))
    assert len(result['year_id']) == 2


def test_filter_ds_by_range_bad_coord(simple_ds):
    with pytest.raises(ValueError, match='not found'):
        filter_ds_by_range(simple_ds, nonexistent=(0, 1))
