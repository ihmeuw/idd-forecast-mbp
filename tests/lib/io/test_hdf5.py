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


# ---------------------------------------------------------------------------
# write_hdf validation failures
# ---------------------------------------------------------------------------

def test_write_hdf_validation_row_count_mismatch_raises(tmp_path, simple_df):
    """Validation row count mismatch should raise ValueError (no retry — not a lock error)."""
    from unittest.mock import patch, MagicMock

    path = tmp_path / 'test.h5'
    bad_df = simple_df.iloc[:1].copy()

    with patch('idd_forecast_mbp.lib.io.hdf5.pd.read_hdf', return_value=bad_df):
        with pytest.raises(ValueError, match='Row count mismatch'):
            write_hdf(simple_df, path, validate=True, max_retries=1)


def test_write_hdf_validation_column_mismatch_raises(tmp_path, simple_df):
    """Validation column name mismatch should raise ValueError."""
    from unittest.mock import patch

    path = tmp_path / 'test.h5'
    wrong_df = simple_df.rename(columns={'value': 'wrong'})

    with patch('idd_forecast_mbp.lib.io.hdf5.pd.read_hdf', return_value=wrong_df):
        with pytest.raises(ValueError, match='Column names mismatch'):
            write_hdf(simple_df, path, validate=True, max_retries=1)


def test_write_hdf_non_lock_error_reraises_immediately(tmp_path, simple_df):
    """Non-lock errors re-raise immediately even on the first attempt."""
    from unittest.mock import patch

    path = tmp_path / 'test.h5'

    def always_fails(self, *args, **kwargs):
        raise RuntimeError('something unrelated to locking')

    with patch.object(pd.DataFrame, 'to_hdf', always_fails):
        with pytest.raises(RuntimeError, match='something unrelated'):
            write_hdf(simple_df, path, max_retries=3, validate=False)


def test_write_hdf_lock_retry(tmp_path, simple_df, capsys):
    """Lock errors trigger exponential-backoff retry; success on second attempt."""
    from unittest.mock import patch

    path = tmp_path / 'test.h5'
    call_count = {'n': 0}
    original_to_hdf = pd.DataFrame.to_hdf

    def flaky_to_hdf(self, *args, **kwargs):
        call_count['n'] += 1
        if call_count['n'] == 1:
            raise OSError('Resource temporarily unavailable')
        return original_to_hdf(self, *args, **kwargs)

    with patch.object(pd.DataFrame, 'to_hdf', flaky_to_hdf), \
         patch('idd_forecast_mbp.lib.io.hdf5.time.sleep'):
        result = write_hdf(simple_df, path, max_retries=3, validate=False)

    assert result is True
    captured = capsys.readouterr()
    assert 'lock' in captured.out.lower() or 'Retry' in captured.out


# ---------------------------------------------------------------------------
# create_hdf_structure — metadata column not in DataFrame
# ---------------------------------------------------------------------------

def test_create_hdf_structure_skips_missing_metadata_col(tmp_path):
    """Columns in metadata_columns but not in metadata_df should be silently skipped."""
    meta = pd.DataFrame({'location_id': [1, 2]})
    path = tmp_path / 'test.h5'
    # 'nonexistent' is in metadata_columns but not in meta — should not raise
    create_hdf_structure(path, meta, [], ['location_id', 'nonexistent'])

    import h5py
    with h5py.File(path, 'r') as f:
        assert 'location_id' in f
        assert 'nonexistent' not in f


def test_create_hdf_structure_object_dtype_column(tmp_path):
    """Object dtype columns should be encoded as fixed-length byte strings."""
    meta = pd.DataFrame({'location_id': [1], 'label': ['region_a']})
    path = tmp_path / 'test.h5'
    create_hdf_structure(path, meta, [], ['location_id', 'label'])

    import h5py
    with h5py.File(path, 'r') as f:
        assert 'label' in f
        assert f['label'].dtype.kind == 'S'  # byte string


# ---------------------------------------------------------------------------
# read_hdf_metadata — byte-string conversion
# ---------------------------------------------------------------------------

def test_read_hdf_metadata_byte_string_decoded(tmp_path):
    """Byte-string (S-kind) datasets should be converted to str."""
    meta = pd.DataFrame({'location_id': [1], 'label': ['region_a']})
    path = tmp_path / 'test.h5'
    create_hdf_structure(path, meta, [], ['location_id', 'label'])

    result = read_hdf_metadata(path, ['label'])
    assert result['label'].dtype == object
    assert result['label'].iloc[0] == 'region_a'
