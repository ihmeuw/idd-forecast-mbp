"""
Parquet I/O utilities for the idd-forecast-mbp pipeline.

Consolidated from: src/idd_forecast_mbp/parquet_functions.py (canonical)
Supersedes:        loading_functions.write_parquet (simpler, older implementation)

Behavior changes from originals:
  - write_parquet: use_atomic now defaults to True (was False in parquet_functions.py)
  - write_parquet: only 'metadata' and 'none' validation are supported; 'full' and
    'sample' from parquet_functions.py loaded data back into memory and caused OOM
    at pipeline scale — they are not carried over.
"""

from __future__ import annotations

import operator
import os
import tempfile
from functools import reduce
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# ID column helpers
# ---------------------------------------------------------------------------

def ensure_id_columns_are_integers(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all *_id columns from float to nullable integer (Int64).

    # Extracted from: parquet_functions.py:14
    """
    for col in df.columns:
        if col.endswith('_id') and pd.api.types.is_float_dtype(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df


def sort_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by *_id columns; location_id and year_id sort first.

    # Extracted from: parquet_functions.py:23
    """
    id_columns = [col for col in df.columns if col.endswith('_id')]

    ordered = []
    if 'location_id' in id_columns:
        ordered.append('location_id')
    if 'year_id' in id_columns:
        ordered.append('year_id')
    remaining = [c for c in id_columns if c not in ('location_id', 'year_id')]
    id_columns = ordered + remaining

    if id_columns:
        df = df.sort_values(by=id_columns)
    return df


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_parquet_with_integer_ids(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read parquet file, sort by ID columns, and cast *_id columns to integer.

    # Extracted from: parquet_functions.py:45
    """
    df = pd.read_parquet(path, **kwargs)
    df = sort_id_columns(df)
    return ensure_id_columns_are_integers(df)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_parquet(
    df: pd.DataFrame,
    filepath: str | Path,
    max_retries: int = 3,
    validate: bool = True,
    overwrite: bool = True,
    compression: str = 'lz4',
    index: bool = False,
    use_atomic: bool = True,
    row_group_size: int = 100_000,
    **kwargs,
) -> bool:
    """Write parquet with retry, atomic rename, chmod 0o775, and metadata validation.

    Always creates parent directories. Supersedes loading_functions.write_parquet.

    Parameters
    ----------
    df:
        DataFrame to write.
    filepath:
        Destination path.
    max_retries:
        Number of write attempts before raising.
    validate:
        If True, validates via parquet metadata (row count + column names) after
        writing. This is the only validation mode — 'full' and 'sample' read-back
        methods from parquet_functions.py caused OOM at pipeline scale and are
        not supported here.
    overwrite:
        If True, removes any existing file before writing.
    compression:
        Parquet compression codec.
    index:
        Whether to write the DataFrame index.
    use_atomic:
        Write to a temp file then rename atomically (prevents partial writes visible
        to other processes). Uses 2x disk space temporarily. Defaults to True.
        Was False in parquet_functions.py — changed to match write_netcdf behavior.
    row_group_size:
        Rows per parquet row group. Smaller values improve partial-read performance.

    # Extracted from: parquet_functions.py:52
    # Behavior changes: use_atomic=True default; validation simplified to metadata-only.
    """
    import pyarrow.parquet as pq

    filepath = str(filepath)

    if overwrite and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f'Warning: could not remove existing file {filepath}: {e}')

    for attempt in range(max_retries):
        target_path = filepath
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

            if use_atomic:
                tmp = tempfile.NamedTemporaryFile(
                    suffix='.parquet',
                    dir=os.path.dirname(filepath) or '.',
                    delete=False,
                )
                target_path = tmp.name
                tmp.close()

            df.to_parquet(
                target_path,
                compression=compression,
                index=index,
                row_group_size=row_group_size,
                **kwargs,
            )
            os.chmod(target_path, 0o775)

            if validate:
                meta = pq.read_metadata(target_path)
                if meta.num_rows != len(df):
                    raise ValueError(
                        f'Row count mismatch after write: file has {meta.num_rows} rows, '
                        f'DataFrame has {len(df)}'
                    )
                file_cols = set(pq.read_schema(target_path).names)
                df_cols = set(df.columns)
                if file_cols != df_cols:
                    raise ValueError(
                        f'Column mismatch after write: '
                        f'extra in file={file_cols - df_cols}, '
                        f'missing from file={df_cols - file_cols}'
                    )

            if use_atomic:
                os.rename(target_path, filepath)

            return True

        except Exception as e:
            print(f'write_parquet attempt {attempt + 1} failed: {e}')
            if use_atomic and target_path != filepath and os.path.exists(target_path):
                os.remove(target_path)
            elif not use_atomic and os.path.exists(filepath):
                os.remove(filepath)

            if attempt == max_retries - 1:
                raise
            print(f'Retrying ({attempt + 1}/{max_retries})...')

    return False  # pragma: no cover


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def filter_df(df: pd.DataFrame, **id_filters) -> pd.DataFrame:
    """Filter DataFrame by one or more columns. Accepts scalar or list values.

    Example
    -------
    filter_df(df, location_id=[1, 2, 3], year_id=2050)

    # Extracted from: parquet_functions.py:177
    """
    for col in id_filters:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    if not id_filters:
        return df

    conditions = []
    for col, values in id_filters.items():
        if isinstance(values, (int, float, str)):
            conditions.append(df[col] == values)
        else:
            conditions.append(df[col].isin(values))

    return df[reduce(operator.and_, conditions)]


def filter_df_by_range(df: pd.DataFrame, **column_ranges) -> pd.DataFrame:
    """Filter DataFrame by inclusive value ranges on one or more columns.

    Example
    -------
    filter_df_by_range(df, year_id=(2020, 2050), age_group_id=(5, 15))

    # Extracted from: parquet_functions.py:198
    """
    for col in column_ranges:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    if not column_ranges:
        return df

    conditions = []
    for col, (lo, hi) in column_ranges.items():
        conditions.append((df[col] >= lo) & (df[col] <= hi))

    return df[reduce(operator.and_, conditions)]
