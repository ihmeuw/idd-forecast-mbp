"""
NetCDF / xarray I/O utilities for the idd-forecast-mbp pipeline.

Consolidated from: src/idd_forecast_mbp/xarray_functions.py (canonical)
Supersedes:        loading_functions.write_netcdf (simpler, older implementation)

Behavior changes from originals:
  - write_netcdf: gains mkdir parameter (defaults True) — creates parent directories
    automatically, matching write_parquet behavior.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cast_coordinate_types(ds: xr.Dataset) -> xr.Dataset:
    """Cast standard ID coordinates to memory-efficient integer types.

    # Extracted from: xarray_functions.py:33
    """
    coord_type_map = {
        'location_id': 'int32',
        'year_id':     'int16',
        'age_group_id': 'int16',
        'sex_id':      'int8',
    }
    for coord, dtype in coord_type_map.items():
        if coord in ds.coords:
            ds = ds.assign_coords({coord: ds[coord].astype(dtype)})
    return ds


def ensure_id_coordinates_are_integers(ds: xr.Dataset) -> xr.Dataset:
    """Cast all *_id coordinates to the smallest appropriate integer type.

    # Extracted from: xarray_functions.py:259
    """
    ds_copy = ds.copy()
    for coord in list(ds_copy.coords):
        if not coord.endswith('_id'):
            continue
        values = ds_copy.coords[coord].values
        lo, hi = values.min(), values.max()
        if lo >= 0:
            dtype = 'uint8' if hi <= 255 else 'uint16' if hi <= 65535 else 'uint32' if hi <= 4294967295 else 'uint64'
        else:
            dtype = 'int8' if lo >= -128 and hi <= 127 else 'int16' if lo >= -32768 and hi <= 32767 else 'int32' if lo >= -2147483648 and hi <= 2147483647 else 'int64'
        ds_copy.coords[coord] = ds_copy.coords[coord].astype(dtype)
    return ds_copy


def sort_coordinates(
    ds: xr.Dataset,
    coords: list[str] | None = None,
    prioritize: list[str] | None = None,
) -> xr.Dataset:
    """Sort Dataset coordinates; location_id and year_id sort first by default.

    Only 1-D coordinates are used for sorting.

    # Extracted from: xarray_functions.py:236
    """
    if coords is None:
        coords = list(ds.coords)
    if prioritize is None:
        prioritize = ['location_id', 'year_id']

    coords_1d = [c for c in coords if ds.coords[c].ndim == 1]
    ordered = [c for c in prioritize if c in coords_1d]
    remaining = [c for c in coords_1d if c not in ordered]
    sort_order = ordered + remaining

    if sort_order:
        ds = ds.sortby(sort_order)
    return ds


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_netcdf_with_integer_ids(path: str | Path, **kwargs) -> xr.Dataset:
    """Read NetCDF file, cast *_id coordinates to integers, sort coordinates.

    # Extracted from: xarray_functions.py:46
    """
    ds = xr.open_dataset(path, **kwargs)
    ds = cast_coordinate_types(ds)
    ds = sort_coordinates(ds)
    return ensure_id_coordinates_are_integers(ds)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_netcdf(
    ds: xr.Dataset,
    filepath: str | Path,
    max_retries: int = 3,
    engine: str = 'netcdf4',
    compression: bool = True,
    compression_level: int = 4,
    chunking: bool = True,
    chunk_threshold: int = 1_000_000,
    max_chunk_size: int = 1000,
    manual_chunks: dict | None = None,
    chunk_by_dim: dict | None = None,
    use_temp_file: bool = True,
    mkdir: bool = True,
    **kwargs,
) -> bool:
    """Write NetCDF with compression, chunking, atomic rename, chmod 0o775, and validation.

    Supersedes loading_functions.write_netcdf.

    Parameters
    ----------
    ds:
        Dataset to write.
    filepath:
        Destination path.
    max_retries:
        Number of write attempts before raising.
    engine:
        NetCDF engine ('netcdf4', 'h5netcdf', 'scipy').
    compression:
        Apply zlib compression to all variables and ID coordinates.
    compression_level:
        Compression level 1–9 (higher = smaller file, slower write).
    chunking:
        Apply chunking to large arrays.
    chunk_threshold:
        Minimum array size (elements) to trigger auto-chunking.
    max_chunk_size:
        Maximum chunk size per dimension for auto-chunking.
    manual_chunks:
        Per-variable chunk specification. Two forms:
          {var: {dim: size}} or {var: (size1, size2, ...)} or {'all': {dim: size}}
    chunk_by_dim:
        Chunk sizes keyed by dimension name; applied to all variables.
        Example: {'location_id': 1500, 'year_id': 79}
    use_temp_file:
        Write to a temp file then rename atomically (prevents partial writes).
    mkdir:
        Create parent directories if they do not exist. Defaults to True.
        Was absent in xarray_functions.py — added to match write_parquet behavior.

    # Extracted from: xarray_functions.py:53
    # Behavior change: mkdir parameter added (defaults True).
    """
    filepath = str(filepath)

    encoding = _build_encoding(
        ds,
        compression=compression,
        compression_level=compression_level,
        chunking=chunking,
        chunk_threshold=chunk_threshold,
        max_chunk_size=max_chunk_size,
        manual_chunks=manual_chunks,
        chunk_by_dim=chunk_by_dim,
    )

    if 'encoding' in kwargs:
        encoding.update(kwargs.pop('encoding'))
    if encoding:
        kwargs['encoding'] = encoding

    for attempt in range(max_retries):
        try:
            if mkdir:
                os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

            if use_temp_file:
                tmp = tempfile.NamedTemporaryFile(
                    suffix='.nc',
                    dir=os.path.dirname(filepath) or '.',
                    delete=False,
                )
                write_path = tmp.name
                tmp.close()
            else:
                write_path = filepath

            ds.to_netcdf(write_path, engine=engine, **kwargs)

            if use_temp_file:
                with xr.open_dataset(write_path) as test_ds:
                    if test_ds.sizes != ds.sizes:
                        raise ValueError(
                            f'Dimension size mismatch: {test_ds.sizes} vs {ds.sizes}'
                        )
                    if list(test_ds.data_vars) != list(ds.data_vars):
                        raise ValueError(
                            f'Data variable mismatch: '
                            f'{list(test_ds.data_vars)} vs {list(ds.data_vars)}'
                        )
                    if list(test_ds.coords) != list(ds.coords):
                        raise ValueError(
                            f'Coordinate mismatch: '
                            f'{list(test_ds.coords)} vs {list(ds.coords)}'
                        )
                    for var in ds.data_vars:
                        if test_ds[var].shape != ds[var].shape:
                            raise ValueError(
                                f'Shape mismatch for {var}: '
                                f'{test_ds[var].shape} vs {ds[var].shape}'
                            )
                os.rename(write_path, filepath)

            os.chmod(filepath, 0o775)
            return True

        except Exception as e:
            print(f'write_netcdf attempt {attempt + 1} failed: {e}')
            if use_temp_file and 'write_path' in dir() and write_path != filepath:
                if os.path.exists(write_path):
                    os.remove(write_path)
            if attempt == max_retries - 1:
                raise
            print(f'Retrying ({attempt + 1}/{max_retries})...')

    return False


def _build_encoding(
    ds: xr.Dataset,
    compression: bool,
    compression_level: int,
    chunking: bool,
    chunk_threshold: int,
    max_chunk_size: int,
    manual_chunks: dict | None,
    chunk_by_dim: dict | None,
) -> dict:
    """Build the encoding dict for to_netcdf(). Internal helper."""
    encoding = {}
    if not compression and not chunking:
        return encoding

    comp = {'zlib': True, 'complevel': compression_level, 'shuffle': True}

    for coord in ds.coords:
        if coord.endswith('_id') and compression:
            encoding[coord] = dict(comp)

    for var in ds.data_vars:
        var_enc = {}
        if compression:
            var_enc.update(comp)

        chunks = None
        if manual_chunks and var in manual_chunks:
            spec = manual_chunks[var]
            if isinstance(spec, dict):
                chunks = [min(spec.get(d, ds.sizes[d]), ds.sizes[d]) for d in ds[var].dims]
            else:
                chunks = [min(c, ds.sizes[d]) for c, d in zip(spec, ds[var].dims)]
        elif manual_chunks and 'all' in manual_chunks:
            spec = manual_chunks['all']
            chunks = [min(spec.get(d, ds.sizes[d]), ds.sizes[d]) for d in ds[var].dims]
        elif chunk_by_dim:
            chunks = [min(chunk_by_dim.get(d, ds.sizes[d]), ds.sizes[d]) for d in ds[var].dims]
        elif chunking and ds[var].size > chunk_threshold:
            chunks = [min(max_chunk_size, ds.sizes[d]) for d in ds[var].dims]

        if chunks:
            var_enc['chunksizes'] = tuple(chunks)

        if var_enc:
            encoding[var] = var_enc

    return encoding


# ---------------------------------------------------------------------------
# DataFrame ↔ xarray conversion
# ---------------------------------------------------------------------------

def convert_to_xarray(
    df: pd.DataFrame,
    dimensions: list[str] | None = None,
    dimension_dtypes: dict[str, str] | None = None,
    variable_dtypes: dict[str, str] | None = None,
    auto_optimize_dtypes: bool = True,
    validate_dimensions: bool = True,
) -> xr.Dataset:
    """Convert DataFrame to xarray Dataset with configurable dimensions and dtypes.

    Parameters
    ----------
    df:
        Input DataFrame.
    dimensions:
        Columns to use as xarray dimensions. Auto-detects *_id columns if None.
    dimension_dtypes:
        dtype overrides for dimension columns.
    variable_dtypes:
        dtype overrides for data variable columns.
    auto_optimize_dtypes:
        Automatically select memory-efficient dtypes for unspecified columns.
    validate_dimensions:
        Raise if dimension columns do not form a complete rectangular grid.

    # Extracted from: xarray_functions.py:353
    """
    df_work = _fix_nullable_dtypes(df).copy()

    if dimensions is None:
        dimensions = [c for c in df_work.columns if c.endswith('_id')]
        if not dimensions:
            raise ValueError("No dimensions specified and no *_id columns found")

    missing = [d for d in dimensions if d not in df_work.columns]
    if missing:
        raise ValueError(f'Dimensions not found in DataFrame: {missing}')

    variables = [c for c in df_work.columns if c not in dimensions]
    dimension_dtypes = dimension_dtypes or {}
    variable_dtypes = variable_dtypes or {}

    if auto_optimize_dtypes:
        for d in dimensions:
            if d not in dimension_dtypes:
                dimension_dtypes[d] = _auto_int_dtype(df_work[d])
        for v in variables:
            if v not in variable_dtypes:
                variable_dtypes[v] = _auto_variable_dtype(df_work[v])

    for d, dtype in dimension_dtypes.items():
        if d in df_work.columns:
            df_work[d] = df_work[d].astype(dtype)
    for v, dtype in variable_dtypes.items():
        if v in df_work.columns:
            df_work[v] = df_work[v].astype(dtype)

    df_work = _fix_nullable_dtypes(df_work)

    if validate_dimensions:
        expected = 1
        for d in dimensions:
            expected *= df_work[d].nunique()
        if len(df_work) != expected:
            raise ValueError(
                f'Incomplete rectangular grid: expected {expected} rows, got {len(df_work)}. '
                'Pass validate_dimensions=False to skip this check.'
            )

    return df_work.set_index(dimensions).to_xarray()


# Common dimension presets
_DIMENSION_PRESETS = {
    'as_variables': {
        'dimensions': ['location_id', 'year_id', 'age_group_id', 'sex_id'],
        'dimension_dtypes': {
            'location_id':  'int32',
            'year_id':      'int16',
            'age_group_id': 'int16',
            'sex_id':       'int8',
        },
    },
    'aa_variables': {
        'dimensions': ['location_id', 'year_id'],
        'dimension_dtypes': {
            'location_id': 'int32',
            'year_id':     'int16',
        },
    },
}


def convert_with_preset(
    df: pd.DataFrame,
    preset: str = 'as_variables',
    **kwargs,
) -> xr.Dataset:
    """Convert DataFrame to xarray using a named dimension preset.

    Available presets:
      'as_variables' — location, year, age_group, sex (age-sex resolved data)
      'aa_variables' — location, year only (all-age data)

    # Extracted from: xarray_functions.py:657
    """
    if preset not in _DIMENSION_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(_DIMENSION_PRESETS)}")
    config = {**_DIMENSION_PRESETS[preset], **kwargs}
    return convert_to_xarray(df, **config)


# ---------------------------------------------------------------------------
# Dataset filter helpers
# ---------------------------------------------------------------------------

def filter_ds_by_multiple_coords(ds: xr.Dataset, **coord_filters) -> xr.Dataset:
    """Filter xarray Dataset by one or more coordinate values (scalar or list).

    Example
    -------
    filter_ds_by_multiple_coords(ds, location_id=[1, 2], year_id=2050)

    # Extracted from: xarray_functions.py:673
    """
    import operator
    from functools import reduce

    for coord in coord_filters:
        if coord not in ds.coords:
            raise ValueError(f"Coordinate '{coord}' not found in dataset")

    if not coord_filters:
        return ds

    conditions = []
    for coord, values in coord_filters.items():
        if isinstance(values, (int, float, str, np.integer, np.floating)):
            conditions.append(ds[coord] == values)
        else:
            conditions.append(ds[coord].isin(values))

    mask = reduce(operator.and_, conditions)
    return ds.where(mask, drop=True)


def filter_ds_by_range(ds: xr.Dataset, **coord_ranges) -> xr.Dataset:
    """Filter xarray Dataset by inclusive coordinate value ranges.

    Example
    -------
    filter_ds_by_range(ds, year_id=(2020, 2050))

    # Extracted from: xarray_functions.py:843
    """
    import operator
    from functools import reduce

    for coord in coord_ranges:
        if coord not in ds.coords:
            raise ValueError(f"Coordinate '{coord}' not found in dataset")

    if not coord_ranges:
        return ds

    conditions = []
    for coord, (lo, hi) in coord_ranges.items():
        conditions.append((ds[coord] >= lo) & (ds[coord] <= hi))

    mask = reduce(operator.and_, conditions)
    return ds.where(mask, drop=True)


# ---------------------------------------------------------------------------
# Private dtype helpers
# ---------------------------------------------------------------------------

def _fix_nullable_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable dtypes to standard numpy dtypes for xarray compatibility.

    # Extracted from: xarray_functions.py:456
    """
    df = df.copy()
    nullable_int_map = {
        'Int8': ('int8', -1), 'Int16': ('int16', -1),
        'Int32': ('int32', -1), 'Int64': ('int64', -1),
        'UInt8': ('uint8', 255), 'UInt16': ('uint16', 65535),
        'UInt32': ('uint32', 4294967295), 'UInt64': ('uint64', 18446744073709551615),
    }
    for col in df.columns:
        name = getattr(df[col].dtype, 'name', '')
        if name in nullable_int_map:
            target, fill = nullable_int_map[name]
            df[col] = df[col].fillna(fill).astype(target)
        elif name == 'Float32':
            df[col] = df[col].astype('float32')
        elif name == 'Float64':
            df[col] = df[col].astype('float64')
        elif name == 'boolean':
            df[col] = df[col].fillna(False).astype('bool')
        elif name == 'string':
            df[col] = df[col].astype('object')
    return df


def _auto_int_dtype(series: pd.Series) -> str:
    """Select smallest integer dtype for a series based on value range."""
    lo, hi = series.min(), series.max()
    if lo >= 0:
        return 'uint8' if hi <= 255 else 'uint16' if hi <= 65535 else 'uint32' if hi <= 4294967295 else 'uint64'
    return 'int8' if lo >= -128 and hi <= 127 else 'int16' if lo >= -32768 and hi <= 32767 else 'int32' if lo >= -2147483648 and hi <= 2147483647 else 'int64'


def _auto_variable_dtype(series: pd.Series) -> str:
    """Select float32 or float64 based on magnitude; keep original for non-numeric."""
    if not pd.api.types.is_numeric_dtype(series):
        return series.dtype
    if pd.api.types.is_integer_dtype(series) or (series.dropna() % 1 == 0).all():
        return _auto_int_dtype(series)
    if series.isna().all():
        return 'float32'
    return 'float32' if abs(series).max() < 1e6 else 'float64'
