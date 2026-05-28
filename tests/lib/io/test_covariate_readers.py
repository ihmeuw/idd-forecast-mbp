"""
Tests for lib/io/covariate_readers.py

Uses synthetic DataFrames and tmp_path parquet files — no real pipeline data.
"""

import pytest
import pandas as pd

from idd_forecast_mbp.lib.io.covariate_readers import (
    merge_dataframes,
    read_income_paths,
    read_urban_paths,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_df():
    return pd.DataFrame({
        'location_id': [1, 2, 3],
        'year_id': [2020, 2020, 2020],
        'rate': [0.1, 0.2, 0.3],
    })


@pytest.fixture
def gdppc_df(tmp_path):
    df = pd.DataFrame({
        'location_id': [1, 2, 3],
        'year_id': [2020, 2020, 2020],
        'scenario': ['rcp45', 'rcp45', 'rcp45'],
        'gdppc_mean': [1000.0, 2000.0, 3000.0],
    })
    path = tmp_path / 'gdppc_mean.parquet'
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def urban_df(tmp_path):
    df = pd.DataFrame({
        'location_id': [1, 2, 3],
        'year_id': [2020, 2020, 2020],
        'population': [1000, 2000, 3000],
        'weighted_300.0_simple_mean': [0.1, 0.2, 0.3],
        'weighted_1km_urban': [0.4, 0.5, 0.6],
    })
    path = tmp_path / 'urban.parquet'
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# merge_dataframes
# ---------------------------------------------------------------------------

def test_merge_dataframes_single(base_df, tmp_path):
    extra = pd.DataFrame({
        'location_id': [1, 2, 3],
        'year_id': [2020, 2020, 2020],
        'covariate': [10.0, 20.0, 30.0],
    })
    result = merge_dataframes(base_df.copy(), {'extra': extra})
    assert 'covariate' in result.columns
    assert len(result) == 3


def test_merge_dataframes_multiple(base_df):
    cov_a = pd.DataFrame({
        'location_id': [1, 2, 3],
        'year_id': [2020, 2020, 2020],
        'cov_a': [1.0, 2.0, 3.0],
    })
    cov_b = pd.DataFrame({
        'location_id': [1, 2, 3],
        'year_id': [2020, 2020, 2020],
        'cov_b': [4.0, 5.0, 6.0],
    })
    result = merge_dataframes(base_df.copy(), {'a': cov_a, 'b': cov_b})
    assert 'cov_a' in result.columns
    assert 'cov_b' in result.columns


def test_merge_dataframes_empty_dict(base_df):
    result = merge_dataframes(base_df.copy(), {})
    assert list(result.columns) == list(base_df.columns)


def test_merge_dataframes_left_join_preserves_rows(base_df):
    # Only location 1 in extra — other rows should get NaN
    extra = pd.DataFrame({
        'location_id': [1],
        'year_id': [2020],
        'cov': [99.0],
    })
    result = merge_dataframes(base_df.copy(), {'x': extra})
    assert len(result) == 3
    assert result.loc[result['location_id'] == 1, 'cov'].iloc[0] == 99.0
    assert result.loc[result['location_id'] == 2, 'cov'].isna().iloc[0]


# ---------------------------------------------------------------------------
# read_income_paths
# ---------------------------------------------------------------------------

def test_read_income_paths_filters_scenario(gdppc_df, tmp_path):
    income_paths = {'gdppc': str(gdppc_df)}
    result = read_income_paths(income_paths, rcp_scenario='rcp45', variable_data_path=str(tmp_path))
    assert 'gdppc' in result
    assert 'scenario' not in result['gdppc'].columns
    assert len(result['gdppc']) == 3


def test_read_income_paths_wrong_scenario_returns_empty(gdppc_df, tmp_path):
    income_paths = {'gdppc': str(gdppc_df)}
    result = read_income_paths(income_paths, rcp_scenario='rcp26', variable_data_path=str(tmp_path))
    assert len(result['gdppc']) == 0


def test_read_income_paths_substitutes_path_placeholder(tmp_path):
    df = pd.DataFrame({
        'location_id': [1],
        'year_id': [2020],
        'scenario': ['rcp45'],
        'gdppc_mean': [5000.0],
    })
    path = tmp_path / 'sub' / 'gdppc_mean.parquet'
    path.parent.mkdir()
    df.to_parquet(path, index=False)

    income_paths = {'gdppc': '{VARIABLE_DATA_PATH}/sub/gdppc_mean.parquet'}
    result = read_income_paths(income_paths, rcp_scenario='rcp45', variable_data_path=str(tmp_path))
    assert len(result['gdppc']) == 1


# ---------------------------------------------------------------------------
# read_urban_paths
# ---------------------------------------------------------------------------

def test_read_urban_paths_drops_population(urban_df, tmp_path):
    urban_paths = {'urban': str(urban_df)}
    result = read_urban_paths(urban_paths, variable_data_path=str(tmp_path))
    assert 'population' not in result['urban'].columns


def test_read_urban_paths_renames_300_suffix(urban_df, tmp_path):
    urban_paths = {'urban': str(urban_df)}
    result = read_urban_paths(urban_paths, variable_data_path=str(tmp_path))
    cols = result['urban'].columns.tolist()
    assert any('300' in c and '300.0_simple_mean' not in c for c in cols)


def test_read_urban_paths_renames_1km_suffix(urban_df, tmp_path):
    urban_paths = {'urban': str(urban_df)}
    result = read_urban_paths(urban_paths, variable_data_path=str(tmp_path))
    cols = result['urban'].columns.tolist()
    assert any('urban_1km' in c for c in cols)


def test_read_urban_paths_removes_weighted_prefix(urban_df, tmp_path):
    urban_paths = {'urban': str(urban_df)}
    result = read_urban_paths(urban_paths, variable_data_path=str(tmp_path))
    cols = result['urban'].columns.tolist()
    assert not any(c.startswith('weighted_') for c in cols)
