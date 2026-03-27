"""
HDF5 I/O utilities for the idd-forecast-mbp pipeline.

Extracted from: src/idd_forecast_mbp/hd5_functions.py (single implementation — no consolidation needed)

No behavior changes from original.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def write_hdf(
    df: pd.DataFrame,
    filepath: str | Path,
    key: str = 'df',
    max_retries: int = 3,
    validate: bool = True,
    compression: str = 'blosc:zstd',
    complevel: int = 9,
    **kwargs,
) -> bool:
    """Write DataFrame to HDF5 with exponential-backoff retry and read-back validation.

    Creates parent directories. Sets chmod 0o775.

    Parameters
    ----------
    df:
        DataFrame to write.
    filepath:
        Destination path.
    key:
        HDF5 key to store the DataFrame under.
    max_retries:
        Maximum retry attempts with exponential backoff on lock errors.
    validate:
        Read the file back and verify row count and column names.
    compression:
        Compression algorithm (e.g. 'blosc:zstd', 'blosc:lz4').
    complevel:
        Compression level 1–9.

    # Extracted from: hd5_functions.py:8
    """
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    hdf_kwargs = {
        'format': 'table',
        'complib': compression.split(':')[0] if ':' in compression else compression,
        'complevel': complevel,
        **kwargs,
    }

    for attempt in range(max_retries):
        try:
            df.to_hdf(filepath, key=key, mode='w', **hdf_kwargs)

            if validate:
                test_df = pd.read_hdf(filepath, key=key)
                if len(test_df) != len(df):
                    raise ValueError(f'Row count mismatch: {len(test_df)} vs {len(df)}')
                if list(test_df.columns) != list(df.columns):
                    raise ValueError('Column names mismatch')

            os.chmod(filepath, 0o775)
            return True

        except Exception as e:
            if 'Resource temporarily unavailable' in str(e) or 'unable to lock' in str(e):
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # exponential backoff: 1, 2, 4 seconds
                    print(f'File lock error (attempt {attempt + 1}). Retrying in {delay}s...')
                    time.sleep(delay)
                    continue

            if attempt == max_retries - 1:
                raise
            raise

    return False


def create_hdf_structure(
    file_path: str | Path,
    metadata_df: pd.DataFrame,
    draw_columns: list[str],
    metadata_columns: list[str],
) -> None:
    """Create HDF5 file with metadata datasets and pre-allocated draw columns.

    Parameters
    ----------
    file_path:
        Path to HDF5 file to create.
    metadata_df:
        DataFrame containing metadata columns (location_id, year_id, etc.).
    draw_columns:
        Draw column names to pre-allocate (e.g. ['draw_0', ..., 'draw_99']).
    metadata_columns:
        Metadata column names to write from metadata_df.

    # Extracted from: hd5_functions.py:189
    """
    n_rows = len(metadata_df)

    with h5py.File(str(file_path), 'w') as f:
        for col in metadata_columns:
            if col not in metadata_df.columns:
                continue
            data = metadata_df[col].values
            if data.dtype == object:
                max_len = max((len(str(x)) for x in data), default=10)
                data = data.astype(f'S{max_len}')
            f.create_dataset(col, data=data)

        for draw_col in draw_columns:
            f.create_dataset(draw_col, shape=(n_rows,), dtype='float64',
                             fillvalue=0.0, compression='gzip')


def write_draw_column(
    file_path: str | Path,
    draw_column: str,
    values: np.ndarray,
) -> None:
    """Write values to a single draw column in an existing HDF5 file.

    Parameters
    ----------
    file_path:
        Path to existing HDF5 file.
    draw_column:
        Name of the draw column to write.
    values:
        Array of values to write.

    # Extracted from: hd5_functions.py:225
    """
    with h5py.File(str(file_path), 'a') as f:
        if draw_column not in f:
            raise KeyError(f"Draw column '{draw_column}' not found in HDF5 file")
        f[draw_column][:] = values


def read_hdf_metadata(
    file_path: str | Path,
    metadata_columns: list[str],
) -> pd.DataFrame:
    """Read only metadata columns from HDF5 file (avoids loading draw data).

    Parameters
    ----------
    file_path:
        Path to HDF5 file.
    metadata_columns:
        Column names to read.

    # Extracted from: hd5_functions.py:244
    """
    data: dict[str, Any] = {}
    with h5py.File(str(file_path), 'r') as f:
        for col in metadata_columns:
            if col in f:
                values = f[col][:]
                if values.dtype.kind == 'S':
                    values = values.astype(str)
                data[col] = values

    return pd.DataFrame(data)
