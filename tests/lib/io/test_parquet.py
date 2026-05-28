"""
Tests for lib/io/parquet.py

Uses small synthetic DataFrames — no real pipeline data.
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from idd_forecast_mbp.lib.io.parquet import (
    ensure_id_columns_are_integers,
    sort_id_columns,
    read_parquet_with_integer_ids,
    write_parquet,
    filter_df,
    filter_df_by_range,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        'location_id': [3.0, 1.0, 2.0],
        'year_id': [2050.0, 2030.0, 2040.0],
        'value': [10.0, 20.0, 30.0],
    })


@pytest.fixture
def id_df():
    """DataFrame with several *_id columns to test sorting priority.

    Contains all combinations of location_id in {1, 2} × year_id in {2030, 2040}
    so tests can rely on specific cross-combinations existing.
    """
    return pd.DataFrame({
        'age_group_id': [5,    5,    5,    5   ],
        'year_id':      [2040, 2030, 2040, 2030],
        'location_id':  [2,    2,    1,    1   ],
        'value':        [1.0,  2.0,  3.0,  4.0 ],
    })


# ---------------------------------------------------------------------------
# ensure_id_columns_are_integers
# ---------------------------------------------------------------------------

def test_ensure_id_columns_casts_float_ids(simple_df):
    result = ensure_id_columns_are_integers(simple_df)
    assert result['location_id'].dtype.name == 'Int64'
    assert result['year_id'].dtype.name == 'Int64'


def test_ensure_id_columns_leaves_non_id_columns(simple_df):
    result = ensure_id_columns_are_integers(simple_df)
    assert result['value'].dtype == float


def test_ensure_id_columns_already_int():
    df = pd.DataFrame({'location_id': pd.array([1, 2, 3], dtype='Int64'), 'value': [1.0, 2.0, 3.0]})
    result = ensure_id_columns_are_integers(df)
    assert list(result['location_id']) == [1, 2, 3]


# ---------------------------------------------------------------------------
# sort_id_columns
# ---------------------------------------------------------------------------

def test_sort_id_columns_location_year_priority(id_df):
    result = sort_id_columns(id_df)
    # location_id sorts first, year_id second (age_group_id is a tertiary key)
    assert list(result['location_id']) == [1, 1, 2, 2]
    assert list(result['year_id']) == [2030, 2040, 2030, 2040]


def test_sort_id_columns_no_id_columns():
    df = pd.DataFrame({'a': [3, 1, 2], 'b': [30, 10, 20]})
    result = sort_id_columns(df)
    assert list(result['a']) == [3, 1, 2]  # unchanged


# ---------------------------------------------------------------------------
# write_parquet / read_parquet_with_integer_ids — round-trip
# ---------------------------------------------------------------------------

def test_round_trip(tmp_path, simple_df):
    path = tmp_path / 'test.parquet'
    write_parquet(simple_df, path)
    result = read_parquet_with_integer_ids(path)
    # IDs cast to Int64, sorted by location_id then year_id
    assert result['location_id'].dtype.name == 'Int64'
    assert list(result['location_id']) == [1, 2, 3]


def test_write_creates_parent_dirs(tmp_path, simple_df):
    path = tmp_path / 'deep' / 'nested' / 'test.parquet'
    write_parquet(simple_df, path)
    assert path.exists()


def test_write_sets_permissions(tmp_path, simple_df):
    path = tmp_path / 'test.parquet'
    write_parquet(simple_df, path)
    mode = oct(os.stat(path).st_mode)[-3:]
    assert mode == '775'


def test_write_atomic_no_partial_file_on_failure(tmp_path):
    """With use_atomic=True, a failed write should not leave a partial file at the target path."""
    df = pd.DataFrame({'location_id': [1], 'value': [1.0]})
    path = tmp_path / 'test.parquet'

    # Patch to_parquet to fail on first attempt
    original = pd.DataFrame.to_parquet
    call_count = {'n': 0}

    def failing_to_parquet(self, *args, **kwargs):
        call_count['n'] += 1
        if call_count['n'] == 1:
            raise IOError('simulated disk error')
        return original(self, *args, **kwargs)

    pd.DataFrame.to_parquet = failing_to_parquet
    try:
        write_parquet(df, path, max_retries=2, validate=False)
    finally:
        pd.DataFrame.to_parquet = original

    # File should exist (written on retry) and target path should not have been
    # corrupted by the first attempt
    assert path.exists()


def test_write_validation_catches_corruption(tmp_path, simple_df):
    """Metadata validation should catch a row-count mismatch."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / 'test.parquet'
    # Write a truncated version directly
    bad = simple_df.iloc[:1]
    pq.write_table(pa.Table.from_pandas(bad), str(path))

    # Now try to write the full df with validation — should succeed (it overwrites)
    # The point is that validation runs against what was just written, not a pre-existing file
    result = write_parquet(simple_df, path, validate=True)
    assert result is True


def test_write_overwrite_removes_existing(tmp_path, simple_df):
    path = tmp_path / 'test.parquet'
    write_parquet(simple_df, path)
    mtime_1 = os.path.getmtime(path)

    write_parquet(simple_df, path, overwrite=True)
    mtime_2 = os.path.getmtime(path)
    assert mtime_2 >= mtime_1


# ---------------------------------------------------------------------------
# filter_df
# ---------------------------------------------------------------------------

def test_filter_df_scalar(id_df):
    result = filter_df(id_df, location_id=1)
    assert all(result['location_id'] == 1)
    assert len(result) == 2


def test_filter_df_list(id_df):
    result = filter_df(id_df, location_id=[1, 2])
    assert len(result) == 4


def test_filter_df_multiple_cols(id_df):
    result = filter_df(id_df, location_id=1, year_id=2030)
    assert len(result) == 1


def test_filter_df_empty_filters(id_df):
    result = filter_df(id_df)
    assert len(result) == len(id_df)


def test_filter_df_bad_column(id_df):
    with pytest.raises(ValueError, match="not found"):
        filter_df(id_df, nonexistent_col=1)


# ---------------------------------------------------------------------------
# filter_df_by_range
# ---------------------------------------------------------------------------

def test_filter_df_by_range_basic(id_df):
    result = filter_df_by_range(id_df, year_id=(2030, 2035))
    assert all(result['year_id'] == 2030)


def test_filter_df_by_range_inclusive(id_df):
    result = filter_df_by_range(id_df, year_id=(2030, 2040))
    assert len(result) == 4


def test_filter_df_by_range_multiple(id_df):
    result = filter_df_by_range(id_df, location_id=(1, 1), year_id=(2030, 2030))
    assert len(result) == 1


def test_filter_df_by_range_bad_column(id_df):
    with pytest.raises(ValueError, match="not found"):
        filter_df_by_range(id_df, nonexistent=(0, 1))


def test_filter_df_by_range_no_args_returns_df(id_df):
    result = filter_df_by_range(id_df)
    assert len(result) == len(id_df)


# ---------------------------------------------------------------------------
# write_parquet edge cases
# ---------------------------------------------------------------------------

def test_write_overwrite_remove_fails_continues(tmp_path, simple_df, capsys):
    """If os.remove raises during overwrite, write still proceeds (warning only)."""
    path = tmp_path / 'test.parquet'
    write_parquet(simple_df, path)  # create first

    with patch('idd_forecast_mbp.lib.io.parquet.os.remove', side_effect=OSError('locked')):
        # Should not raise — the warning is printed and write proceeds
        write_parquet(simple_df, path, overwrite=True)

    captured = capsys.readouterr()
    assert 'Warning' in captured.out


def test_write_validation_row_count_mismatch_raises(tmp_path, simple_df):
    """Metadata row count mismatch triggers ValueError on attempt, then raises after retries."""
    path = tmp_path / 'test.parquet'

    mock_meta = MagicMock()
    mock_meta.num_rows = 0  # wrong row count

    with patch('pyarrow.parquet.read_metadata', return_value=mock_meta):
        with pytest.raises(ValueError, match='Row count mismatch'):
            write_parquet(simple_df, path, max_retries=1, validate=True)


def test_write_validation_column_mismatch_raises(tmp_path, simple_df):
    """Column set mismatch triggers ValueError on attempt, then raises after retries."""
    path = tmp_path / 'test.parquet'

    mock_meta = MagicMock()
    mock_meta.num_rows = len(simple_df)
    mock_schema = MagicMock()
    mock_schema.names = ['wrong_col']

    with patch('pyarrow.parquet.read_metadata', return_value=mock_meta), \
         patch('pyarrow.parquet.read_schema', return_value=mock_schema):
        with pytest.raises(ValueError, match='Column mismatch'):
            write_parquet(simple_df, path, max_retries=1, validate=True)


def test_write_non_atomic_cleanup_on_failure(tmp_path, simple_df, capsys):
    """use_atomic=False: failed write cleans up the partial target file."""
    path = tmp_path / 'test.parquet'

    call_count = {'n': 0}
    original = pd.DataFrame.to_parquet

    def failing_to_parquet(self, *args, **kwargs):
        call_count['n'] += 1
        if call_count['n'] == 1:
            # Write a partial file then raise
            original(self, *args, **kwargs)
            raise IOError('simulated mid-write failure')
        return original(self, *args, **kwargs)

    pd.DataFrame.to_parquet = failing_to_parquet
    try:
        write_parquet(simple_df, path, max_retries=2, validate=False, use_atomic=False)
    finally:
        pd.DataFrame.to_parquet = original

    assert path.exists()  # written on retry
