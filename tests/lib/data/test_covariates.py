"""
Tests for lib/data/covariates.py

Uses temp parquet files to simulate the dict-of-paths loading pattern.
No real pipeline data.
"""

import pytest
import pandas as pd
import numpy as np

from idd_forecast_mbp.lib.data.covariates import (
    UNIVERSAL_COVARIATE_CLIP_RULES,
    load_covariates_for_draw,
    read_income_paths,
    read_urban_paths,
    merge_dataframes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_covariate_parquet(path, location_ids, year_ids, draw, values):
    """Write a minimal covariate parquet with one draw column."""
    rows = []
    for loc, yr, val in zip(location_ids, year_ids, values):
        rows.append({'location_id': loc, 'year_id': yr, draw: val})
    pd.DataFrame(rows).to_parquet(path)


# ---------------------------------------------------------------------------
# UNIVERSAL_COVARIATE_CLIP_RULES
# ---------------------------------------------------------------------------

def test_universal_rules_contains_rh():
    assert 'relative_humidity' in UNIVERSAL_COVARIATE_CLIP_RULES
    lo, hi = UNIVERSAL_COVARIATE_CLIP_RULES['relative_humidity']
    assert lo == 0.001
    assert hi == 99.999


# ---------------------------------------------------------------------------
# load_covariates_for_draw
# ---------------------------------------------------------------------------

def test_load_covariates_merges_columns(tmp_path):
    draw = 'draw_0'
    locs = [1, 2]
    yrs  = [2030, 2030]

    # Write to the *formatted* filenames (ssp245 already substituted)
    write_covariate_parquet(tmp_path / 'precip_ssp245.parquet', locs, yrs, draw, [10.0, 20.0])
    write_covariate_parquet(tmp_path / 'suitability_ssp245.parquet', locs, yrs, draw, [0.5, 0.6])

    paths = {
        'total_precipitation': str(tmp_path / 'precip_{ssp_scenario}.parquet'),
        'malaria_suitability': str(tmp_path / 'suitability_{ssp_scenario}.parquet'),
    }
    result = load_covariates_for_draw(
        paths, draw='draw_0', ssp_scenario='ssp245',
        climate_data_path=str(tmp_path),
        clip_rules={},
    )
    assert 'total_precipitation' in result.columns
    assert 'malaria_suitability' in result.columns
    assert 'location_id' in result.columns
    assert 'year_id' in result.columns
    assert len(result) == 2


def test_load_covariates_applies_clip_rules(tmp_path):
    draw = 'draw_0'
    write_covariate_parquet(tmp_path / 'rh_ssp245.parquet', [1, 2, 3], [2030, 2030, 2030], draw, [0.0, 50.0, 100.0])

    paths = {'relative_humidity': str(tmp_path / 'rh_{ssp_scenario}.parquet')}
    result = load_covariates_for_draw(
        paths, draw=draw, ssp_scenario='ssp245',
        climate_data_path=str(tmp_path),
        clip_rules={'relative_humidity': (0.001, 99.999)},
    )
    assert result['relative_humidity'].min() >= 0.001
    assert result['relative_humidity'].max() <= 99.999


def test_load_covariates_universal_rules_applied_by_default(tmp_path):
    draw = 'draw_0'
    write_covariate_parquet(tmp_path / 'rh_ssp245.parquet', [1], [2030], draw, [0.0])  # below lower bound

    paths = {'relative_humidity': str(tmp_path / 'rh_{ssp_scenario}.parquet')}
    result = load_covariates_for_draw(
        paths, draw=draw, ssp_scenario='ssp245',
        climate_data_path=str(tmp_path),
        # clip_rules defaults to UNIVERSAL_COVARIATE_CLIP_RULES
    )
    assert result['relative_humidity'].iloc[0] == 0.001


def test_load_covariates_extra_clip_rules(tmp_path):
    draw = 'draw_0'
    write_covariate_parquet(tmp_path / 'myvar_ssp245.parquet', [1], [2030], draw, [-5.0])

    paths = {'myvar': str(tmp_path / 'myvar_{ssp_scenario}.parquet')}
    result = load_covariates_for_draw(
        paths, draw=draw, ssp_scenario='ssp245',
        climate_data_path=str(tmp_path),
        clip_rules={},
        extra_clip_rules={'myvar': (0.0, 100.0)},
    )
    assert result['myvar'].iloc[0] == 0.0


def test_load_covariates_empty_paths(tmp_path):
    result = load_covariates_for_draw(
        {}, draw='draw_0', ssp_scenario='ssp245',
        climate_data_path=str(tmp_path),
    )
    assert list(result.columns) == ['location_id', 'year_id']


def test_load_covariates_ssp_substituted(tmp_path):
    draw = 'draw_0'
    # Create file with ssp245 in the name
    path = tmp_path / 'var_ssp245.parquet'
    write_covariate_parquet(path, [1], [2030], draw, [1.0])

    paths = {'myvar': str(tmp_path / 'var_{ssp_scenario}.parquet')}
    result = load_covariates_for_draw(
        paths, draw=draw, ssp_scenario='ssp245',
        climate_data_path=str(tmp_path),
        clip_rules={},
    )
    assert len(result) == 1


# ---------------------------------------------------------------------------
# read_income_paths
# ---------------------------------------------------------------------------

def test_read_income_paths_filters_scenario(tmp_path):
    df = pd.DataFrame({
        'location_id': [1, 1, 2],
        'year_id':     [2030, 2030, 2030],
        'scenario':    ['ssp245', 'ssp585', 'ssp245'],
        'gdppc':       [100.0, 200.0, 150.0],
    })
    path = tmp_path / 'income.parquet'
    df.to_parquet(path)

    paths = {'gdppc': str(tmp_path / 'income.parquet')}
    result = read_income_paths(paths, rcp_scenario='ssp245', variable_data_path=str(tmp_path))

    assert 'gdppc' in result
    assert 'scenario' not in result['gdppc'].columns
    assert len(result['gdppc']) == 2  # only ssp245 rows


def test_read_income_paths_drops_scenario_column(tmp_path):
    df = pd.DataFrame({
        'location_id': [1],
        'year_id':     [2030],
        'scenario':    ['ssp245'],
        'value':       [1.0],
    })
    path = tmp_path / 'income.parquet'
    df.to_parquet(path)

    paths = {'myincome': str(path)}
    result = read_income_paths(paths, rcp_scenario='ssp245', variable_data_path='')
    assert 'scenario' not in result['myincome'].columns


# ---------------------------------------------------------------------------
# read_urban_paths
# ---------------------------------------------------------------------------

def test_read_urban_paths_drops_population(tmp_path):
    df = pd.DataFrame({
        'location_id': [1],
        'year_id':     [2030],
        'population':  [1000.0],
        'urban_frac':  [0.5],
    })
    path = tmp_path / 'urban.parquet'
    df.to_parquet(path)

    paths = {'urban': str(path)}
    result = read_urban_paths(paths, variable_data_path='')
    assert 'population' not in result['urban'].columns
    assert 'urban_frac' in result['urban'].columns


def test_read_urban_paths_column_normalization(tmp_path):
    df = pd.DataFrame({
        'location_id':        [1],
        'year_id':            [2030],
        '300.0_simple_mean':  [0.1],
        '1500.0_simple_mean': [0.2],
        '100m_urban':         [0.3],
        '1km_urban':          [0.4],
        'weighted_urban_frac': [0.5],
    })
    path = tmp_path / 'urban.parquet'
    df.to_parquet(path)

    paths = {'urban': str(path)}
    result = read_urban_paths(paths, variable_data_path='')['urban']

    assert '300' in result.columns
    assert '1500' in result.columns
    assert 'urban_100m' in result.columns
    assert 'urban_1km' in result.columns
    assert 'urban_frac' in result.columns
    assert 'weighted_urban_frac' not in result.columns


# ---------------------------------------------------------------------------
# merge_dataframes
# ---------------------------------------------------------------------------

def test_merge_dataframes_basic():
    model = pd.DataFrame({'location_id': [1, 2], 'year_id': [2030, 2030], 'forecast': [10.0, 20.0]})
    cov1 = pd.DataFrame({'location_id': [1, 2], 'year_id': [2030, 2030], 'precip': [5.0, 6.0]})
    cov2 = pd.DataFrame({'location_id': [1, 2], 'year_id': [2030, 2030], 'rh': [50.0, 60.0]})

    result = merge_dataframes(model, {'precip': cov1, 'rh': cov2})
    assert 'precip' in result.columns
    assert 'rh' in result.columns
    assert 'forecast' in result.columns
    assert len(result) == 2


def test_merge_dataframes_left_join_preserves_model_rows():
    model = pd.DataFrame({'location_id': [1, 2, 3], 'year_id': [2030, 2030, 2030], 'v': [1.0, 2.0, 3.0]})
    cov = pd.DataFrame({'location_id': [1, 2], 'year_id': [2030, 2030], 'c': [10.0, 20.0]})

    result = merge_dataframes(model, {'c': cov})
    assert len(result) == 3  # row for location 3 is kept, c is NaN
    assert pd.isna(result[result['location_id'] == 3]['c'].iloc[0])


def test_merge_dataframes_empty_dict():
    model = pd.DataFrame({'location_id': [1], 'year_id': [2030], 'v': [1.0]})
    result = merge_dataframes(model, {})
    assert list(result.columns) == ['location_id', 'year_id', 'v']
