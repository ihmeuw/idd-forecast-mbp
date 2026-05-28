"""
Tests for lib/io/netcdf.py

Uses small synthetic Datasets — no real pipeline data.
"""

import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import patch, MagicMock

from idd_forecast_mbp.lib.io.netcdf import (
    _auto_int_dtype,
    _auto_variable_dtype,
    _build_encoding,
    _fix_nullable_dtypes,
    cast_coordinate_types,
    convert_to_xarray,
    convert_with_preset,
    ensure_id_coordinates_are_integers,
    filter_ds_by_multiple_coords,
    filter_ds_by_range,
    read_netcdf_with_integer_ids,
    sort_coordinates,
    write_netcdf,
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


def test_filter_ds_by_multiple_coords_no_args_returns_ds(simple_ds):
    result = filter_ds_by_multiple_coords(simple_ds)
    assert set(result.dims) == set(simple_ds.dims)


def test_filter_ds_by_range_no_args_returns_ds(simple_ds):
    result = filter_ds_by_range(simple_ds)
    assert set(result.dims) == set(simple_ds.dims)


# ---------------------------------------------------------------------------
# ensure_id_coordinates_are_integers
# ---------------------------------------------------------------------------

def test_ensure_id_coords_non_id_coord_skipped():
    """Coordinates not ending in _id should be left alone."""
    ds = xr.Dataset(
        {'v': (['x'], np.array([1.0, 2.0]))},
        coords={'x': np.array([10, 20], dtype='int64')},
    )
    result = ensure_id_coordinates_are_integers(ds)
    assert 'x' in result.coords  # unchanged, no error


def test_ensure_id_coords_negative_values():
    """Negative coord values should select a signed dtype."""
    ds = xr.Dataset(
        {'v': (['loc_id'], np.array([1.0, 2.0]))},
        coords={'loc_id': np.array([-1, 50], dtype='int64')},
    )
    result = ensure_id_coordinates_are_integers(ds)
    assert result['loc_id'].dtype.kind == 'i'  # signed integer


# ---------------------------------------------------------------------------
# write_netcdf — use_temp_file=False and encoding kwarg
# ---------------------------------------------------------------------------

def test_write_netcdf_no_temp_file(tmp_path, simple_ds):
    path = tmp_path / 'test.nc'
    result = write_netcdf(simple_ds, path, use_temp_file=False)
    assert result is True
    assert path.exists()


def test_write_netcdf_encoding_kwarg_merged(tmp_path, simple_ds):
    """Passing encoding={} kwarg should not crash — it merges with computed encoding."""
    path = tmp_path / 'test.nc'
    write_netcdf(simple_ds, path, encoding={})
    assert path.exists()


def _make_open_dataset_ctx(fake_ds: xr.Dataset):
    """Build a mock context manager for xr.open_dataset that yields fake_ds."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=fake_ds)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def test_write_netcdf_dim_size_mismatch_raises(tmp_path, simple_ds):
    """Validation: dimension size mismatch triggers ValueError."""
    path = tmp_path / 'test.nc'
    wrong_ds = xr.Dataset(
        {'value': (['location_id', 'year_id'], np.ones((3, 2)))},
        coords={'location_id': [1, 2, 3], 'year_id': [2030, 2040]},
    )
    with patch('xarray.open_dataset', return_value=_make_open_dataset_ctx(wrong_ds)):
        with pytest.raises(ValueError, match='Dimension size mismatch'):
            write_netcdf(simple_ds, path, max_retries=1)


def test_write_netcdf_data_var_mismatch_raises(tmp_path, simple_ds):
    """Validation: data_vars mismatch triggers ValueError."""
    path = tmp_path / 'test.nc'
    wrong_ds = xr.Dataset(
        {'other': (['location_id', 'year_id'], np.ones((2, 2)))},
        coords={'location_id': [1, 2], 'year_id': [2030, 2040]},
    )
    with patch('xarray.open_dataset', return_value=_make_open_dataset_ctx(wrong_ds)):
        with pytest.raises(ValueError, match='Data variable mismatch'):
            write_netcdf(simple_ds, path, max_retries=1)


def test_write_netcdf_coord_mismatch_raises(tmp_path, simple_ds):
    """Validation: coordinate mismatch triggers ValueError."""
    path = tmp_path / 'test.nc'
    wrong_ds = xr.Dataset(
        {'value': (['location_id', 'year_id'], np.ones((2, 2)))},
        coords={'location_id': [1, 2], 'year_id': [2030, 2040], 'extra': 99},
    )
    with patch('xarray.open_dataset', return_value=_make_open_dataset_ctx(wrong_ds)):
        with pytest.raises(ValueError, match='Coordinate mismatch'):
            write_netcdf(simple_ds, path, max_retries=1)


def test_write_netcdf_retry_on_transient_failure(tmp_path, simple_ds, capsys):
    """Transient write failure triggers retry; success on second attempt."""
    path = tmp_path / 'test.nc'
    call_count = {'n': 0}
    original = xr.Dataset.to_netcdf

    def flaky_to_netcdf(self, *args, **kwargs):
        call_count['n'] += 1
        if call_count['n'] == 1:
            raise OSError('transient disk error')
        return original(self, *args, **kwargs)

    with patch.object(xr.Dataset, 'to_netcdf', flaky_to_netcdf):
        result = write_netcdf(simple_ds, path, max_retries=3)

    assert result is True
    captured = capsys.readouterr()
    assert 'Retrying' in captured.out


# ---------------------------------------------------------------------------
# _build_encoding variants
# ---------------------------------------------------------------------------

def test_build_encoding_no_compression_no_chunking(simple_ds):
    enc = _build_encoding(
        simple_ds,
        compression=False, compression_level=4,
        chunking=False, chunk_threshold=1_000_000,
        max_chunk_size=1000, manual_chunks=None, chunk_by_dim=None,
    )
    assert enc == {}


def test_build_encoding_manual_chunks_dict_spec():
    ds = xr.Dataset(
        {'v': (['location_id', 'year_id'], np.ones((4, 3)))},
        coords={'location_id': [1, 2, 3, 4], 'year_id': [2030, 2040, 2050]},
    )
    enc = _build_encoding(
        ds,
        compression=False, compression_level=4,
        chunking=True, chunk_threshold=1,
        max_chunk_size=1000,
        manual_chunks={'v': {'location_id': 2, 'year_id': 3}},
        chunk_by_dim=None,
    )
    assert 'chunksizes' in enc.get('v', {})


def test_build_encoding_manual_chunks_tuple_spec():
    ds = xr.Dataset(
        {'v': (['location_id', 'year_id'], np.ones((4, 3)))},
        coords={'location_id': [1, 2, 3, 4], 'year_id': [2030, 2040, 2050]},
    )
    enc = _build_encoding(
        ds,
        compression=False, compression_level=4,
        chunking=True, chunk_threshold=1,
        max_chunk_size=1000, manual_chunks={'v': (2, 3)}, chunk_by_dim=None,
    )
    assert 'chunksizes' in enc.get('v', {})


def test_build_encoding_manual_chunks_all_key():
    ds = xr.Dataset(
        {'v': (['location_id', 'year_id'], np.ones((4, 3)))},
        coords={'location_id': [1, 2, 3, 4], 'year_id': [2030, 2040, 2050]},
    )
    enc = _build_encoding(
        ds,
        compression=False, compression_level=4,
        chunking=True, chunk_threshold=1,
        max_chunk_size=1000,
        manual_chunks={'all': {'location_id': 2, 'year_id': 3}},
        chunk_by_dim=None,
    )
    assert 'chunksizes' in enc.get('v', {})


def test_build_encoding_chunk_by_dim():
    ds = xr.Dataset(
        {'v': (['location_id', 'year_id'], np.ones((4, 3)))},
        coords={'location_id': [1, 2, 3, 4], 'year_id': [2030, 2040, 2050]},
    )
    enc = _build_encoding(
        ds,
        compression=False, compression_level=4,
        chunking=True, chunk_threshold=1,
        max_chunk_size=1000, manual_chunks=None,
        chunk_by_dim={'location_id': 2, 'year_id': 3},
    )
    assert 'chunksizes' in enc.get('v', {})


def test_build_encoding_auto_chunking_large_array():
    big = np.ones((100, 20))
    ds = xr.Dataset(
        {'v': (['location_id', 'year_id'], big)},
        coords={'location_id': list(range(100)), 'year_id': list(range(20))},
    )
    enc = _build_encoding(
        ds,
        compression=False, compression_level=4,
        chunking=True, chunk_threshold=10,  # small threshold to trigger
        max_chunk_size=50, manual_chunks=None, chunk_by_dim=None,
    )
    assert 'chunksizes' in enc.get('v', {})


# ---------------------------------------------------------------------------
# convert_to_xarray — no _id columns raises
# ---------------------------------------------------------------------------

def test_convert_to_xarray_no_id_columns_raises():
    df = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
    with pytest.raises(ValueError, match='No dimensions'):
        convert_to_xarray(df)


# ---------------------------------------------------------------------------
# _fix_nullable_dtypes
# ---------------------------------------------------------------------------

def test_fix_nullable_dtypes_int_types():
    df = pd.DataFrame({
        'a': pd.array([1, 2], dtype='Int8'),
        'b': pd.array([1000, 2000], dtype='Int16'),
        'c': pd.array([1, 2], dtype='Int32'),
        'd': pd.array([1, 2], dtype='Int64'),
    })
    result = _fix_nullable_dtypes(df)
    assert result['a'].dtype == 'int8'
    assert result['b'].dtype == 'int16'
    assert result['c'].dtype == 'int32'
    assert result['d'].dtype == 'int64'


def test_fix_nullable_dtypes_uint_types():
    df = pd.DataFrame({
        'a': pd.array([1, 2], dtype='UInt8'),
        'b': pd.array([1, 2], dtype='UInt16'),
        'c': pd.array([1, 2], dtype='UInt32'),
    })
    result = _fix_nullable_dtypes(df)
    assert result['a'].dtype == 'uint8'
    assert result['b'].dtype == 'uint16'
    assert result['c'].dtype == 'uint32'


def test_fix_nullable_dtypes_float_types():
    df = pd.DataFrame({
        'a': pd.array([1.0, 2.0], dtype='Float32'),
        'b': pd.array([1.0, 2.0], dtype='Float64'),
    })
    result = _fix_nullable_dtypes(df)
    assert result['a'].dtype == 'float32'
    assert result['b'].dtype == 'float64'


def test_fix_nullable_dtypes_boolean():
    df = pd.DataFrame({'a': pd.array([True, None], dtype='boolean')})
    result = _fix_nullable_dtypes(df)
    assert result['a'].dtype == bool


def test_fix_nullable_dtypes_string():
    df = pd.DataFrame({'a': pd.array(['x', 'y'], dtype='string')})
    result = _fix_nullable_dtypes(df)
    assert result['a'].dtype == object


# ---------------------------------------------------------------------------
# _auto_variable_dtype
# ---------------------------------------------------------------------------

def test_auto_int_dtype_negative_values():
    """Negative values select signed integer dtype (line 502 of netcdf.py)."""
    s = pd.Series([-1, 50])
    result = _auto_int_dtype(s)
    assert result == 'int8'


def test_auto_int_dtype_large_negative():
    s = pd.Series([-200, 50])
    result = _auto_int_dtype(s)
    assert result == 'int16'


def test_auto_variable_dtype_non_numeric():
    s = pd.Series(['a', 'b', 'c'])
    result = _auto_variable_dtype(s)
    assert result == s.dtype


def test_auto_variable_dtype_int_with_nan():
    s = pd.Series([1.0, 2.0, float('nan')])
    result = _auto_variable_dtype(s)
    assert result in ('float32', 'float64')


def test_auto_variable_dtype_all_nan():
    # all-NaN passes the `% 1 == 0` check (empty.all() = True) with has_na=True;
    # abs(series).max() = NaN, NaN < 1e6 is False → 'float64'
    s = pd.Series([float('nan'), float('nan')])
    result = _auto_variable_dtype(s)
    assert result == 'float64'


def test_auto_variable_dtype_large_float():
    # Fractional values (not integer-like) with magnitude > 1e6 → float64
    s = pd.Series([1e7 + 0.5, 2e7 + 0.5])
    result = _auto_variable_dtype(s)
    assert result == 'float64'
