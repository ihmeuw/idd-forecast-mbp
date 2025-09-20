import pandas as pd
import numpy as np
from pathlib import Path
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import read_income_paths, read_urban_paths
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids
from idd_forecast_mbp.xarray_functions import read_netcdf_with_integer_ids, convert_to_xarray, write_netcdf


ssp_scenarios = rfc.ssp_scenarios
cause_map = rfc.cause_map
draws = rfc.draws
PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
VARIABLE_DATA_PATH = PROCESSED_DATA_PATH / "lsae_1209"
CLIMATE_DATA_PATH = f"/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/lsae_1209"

def convert_draws_to_xarray(df):
    draw_cols = [col for col in df.columns if col in draws]
    coord_cols = [col for col in df.columns if col not in draw_cols]
    df_melted = pd.melt(df, id_vars=coord_cols, value_vars=draw_cols, var_name='draw', value_name='value')
    df_melted['draw'] = df_melted['draw'].astype(int)
    dimensions = ['location_id', 'year_id', 'draw']
    dimension_dtypes = {'location_id': 'uint32', 'year_id': 'uint16', 'draw': 'uint8'}
    variable_dtypes = {'value': 'float32'}
    ds = convert_to_xarray(df_melted, dimensions=dimensions, dimension_dtypes=dimension_dtypes, variable_dtypes=variable_dtypes, validate_dimensions=True)
    return ds


def get_cause_suitability_df(cause, year_ids, location_ids, ssp_scenario):
    coord_cols = ['location_id', 'year_id']
    year_filter = ('year_id', 'in', year_ids)
    location_id_filter = ('location_id', 'in', location_ids)
    columns_to_read = coord_cols + draws
    cause_suitability_path_template = f"{CLIMATE_DATA_PATH}/{cause}_suitability_{ssp_scenario}.parquet"
    df = read_parquet_with_integer_ids(cause_suitability_path_template, filters=[year_filter, location_id_filter], columns=columns_to_read)
    return df

def get_cause_suitability_ds(cause, year_ids, location_ids, ssp_scenarios):
    coord_cols = ['location_id', 'year_id']
    year_filter = ('year_id', 'in', year_ids)
    location_id_filter = ('location_id', 'in', location_ids)
    columns_to_read = coord_cols + draws
    all_dfs = []
    for ssp_scenario in ssp_scenarios:
        cause_suitability_path_template = f"{CLIMATE_DATA_PATH}/{cause}_suitability_{ssp_scenario}.parquet"
        df = read_parquet_with_integer_ids(cause_suitability_path_template, filters=[year_filter, location_id_filter], columns=columns_to_read)
        df['ssp_scenario'] = ssp_scenario
        df['cause'] = cause
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    ds = convert_draws_to_xarray(combined_df)
    return ds

rcp_to_ssp = {config['rcp_scenario']: ssp for ssp, config in ssp_scenarios.items()}
rcp_values = list(rcp_to_ssp.keys())

def get_income_ds(year_ids = None, location_ids = None):
    filters = []
    if year_ids is not None:
        year_filter = ('year_id', 'in', year_ids)
        filters.append(year_filter)
    if location_ids is not None:
        location_id_filter = ('location_id', 'in', location_ids)
        filters.append(location_id_filter)
    if len(filters) > 0:
        df = read_parquet_with_integer_ids(f"{VARIABLE_DATA_PATH}/gdppc_mean.parquet", filters=filters)
    else:
        df = read_parquet_with_integer_ids(f"{VARIABLE_DATA_PATH}/gdppc_mean.parquet")
    df = df[df['scenario'].isin(rcp_values)]
    df['ssp_scenario'] = df['scenario'].map(rcp_to_ssp)
    df = df.drop(columns=['scenario']).rename(columns={'gdppc_mean': 'value'})
    # Convert to xarray
    dimensions = ['location_id', 'year_id', 'ssp_scenario']
    dimension_dtypes = {'location_id': 'uint32', 'year_id': 'uint16', 'ssp_scenario': 'object'}
    variable_dtypes = {'value': 'float32'}
    ds = convert_to_xarray(
        df, 
        dimensions=dimensions, 
        dimension_dtypes=dimension_dtypes, 
        variable_dtypes=variable_dtypes, 
        validate_dimensions=True
        )
    return ds

def get_flooding_ds(year_ids = None, location_ids = None,ssp_scenarios=rfc.ssp_scenarios):
    filters = []
    if year_ids is not None:
        year_filter = ('year_id', 'in', year_ids)
        filters.append(year_filter)
    if location_ids is not None:
        location_id_filter = ('location_id', 'in', location_ids)
        filters.append(location_id_filter)
    
    dfs = []
    for ssp_scenario in ssp_scenarios:
        flooding_path = f"/mnt/team/rapidresponse/pub/flooding/results/output/lsae_1209/fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet"
        if len(filters) > 0:
            df = read_parquet_with_integer_ids(flooding_path, filters=filters)
        else:
            df = read_parquet_with_integer_ids(flooding_path)
        df = df.drop(columns=['population', 'people_flood_days','variant','model'], errors='ignore')
        df = df.rename(columns={'people_flood_days_per_capita': 'value', 'scenario': 'ssp_scenario'})
        df = df[df['year_id'] >= 2000]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # Convert to xarray
    dimensions = ['location_id', 'year_id', 'ssp_scenario']
    dimension_dtypes = {'location_id': 'uint32', 'year_id': 'uint16', 'ssp_scenario': 'object'}
    variable_dtypes = {'value': 'float32'}
    ds = convert_to_xarray(
        df, 
        dimensions=dimensions, 
        dimension_dtypes=dimension_dtypes, 
        variable_dtypes=variable_dtypes, 
        validate_dimensions=True
    )
    return ds

def get_urban_ds(year_ids = None, location_ids = None):
    filters = []
    if year_ids is not None:
        year_filter = ('year_id', 'in', year_ids)
        filters.append(year_filter)
    if location_ids is not None:
        location_id_filter = ('location_id', 'in', location_ids)
        filters.append(location_id_filter)
    if len(filters) > 0:
        df = read_parquet_with_integer_ids(f"{VARIABLE_DATA_PATH}/urban_threshold_300.0_simple_mean.parquet", filters=filters)
    else:
        df = read_parquet_with_integer_ids(f"{VARIABLE_DATA_PATH}/urban_threshold_300.0_simple_mean.parquet")
    df = df.drop(columns=['weighted_100m_urban_threshold_300.0_simple_mean', 'population'], errors='ignore')
    df = df.rename(columns={'weighted_1km_urban_threshold_300.0_simple_mean': 'value'})
    dimensions = ['location_id', 'year_id']
    dimension_dtypes = {'location_id': 'uint32', 'year_id': 'uint16'}
    variable_dtypes = {'value': 'float32'}
    ds = convert_to_xarray(
        df, 
        dimensions=dimensions, 
        dimension_dtypes=dimension_dtypes, 
        variable_dtypes=variable_dtypes, 
        validate_dimensions=True
    )
    return ds


def get_dah_ds(year_ids = None, location_ids = None, dah_scenarios = ['Baseline', 'Constant']):
    filters = []
    if year_ids is not None:
        year_filter = ('year_id', 'in', year_ids)
        filters.append(year_filter)
    if location_ids is not None:
        location_id_filter = ('location_id', 'in', location_ids)
        filters.append(location_id_filter)
    dfs = []
    for dah_scenario in dah_scenarios:
        dah_path = f"{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp245_dah_scenario_{dah_scenario}_draw_000.parquet"
        if len(filters) > 0: 
            df = read_parquet_with_integer_ids(dah_path, filters=filters)
        else:
            df = read_parquet_with_integer_ids(dah_path)
        df = df[['location_id', 'year_id', 'mal_DAH_total_per_capita']]
        df = df.rename(columns={'mal_DAH_total_per_capita': 'value'})
        df['dah_scenario'] = dah_scenario
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    dimensions = ['location_id', 'year_id', 'dah_scenario']
    dimension_dtypes = {'location_id': 'uint32', 'year_id': 'uint16', 'dah_scenario': 'object'}
    variable_dtypes = {'value': 'float32'}
    ds = convert_to_xarray(
        df, 
        dimensions=dimensions, 
        dimension_dtypes=dimension_dtypes, 
        variable_dtypes=variable_dtypes, 
        validate_dimensions=True
    )
    return ds