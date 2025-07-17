import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from affine import Affine # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
from typing import Literal, NamedTuple
import itertools
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.xarray_functions import write_netcdf, convert_with_preset
import argparse

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--cause", type=str, required=True, help="Cause name (e.g., 'malaria')")
parser.add_argument("--measure", type=str, required=False, default="mortality", help="measure (e.g., 'mortality', 'incidence')")
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=False, help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")

# Parse arguments
args = parser.parse_args()

cause = args.cause
measure = args.measure
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
draw = args.draw

measure_map = rfc.measure_map

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data"

if cause == "malaria":
    forecast_df_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.parquet"
    processed_forecast_df_path = f"{UPLOAD_DATA_PATH}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.parquet"
    processed_forecast_ds_path = f"{UPLOAD_DATA_PATH}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
else:   
    forecast_df_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet"
    processed_forecast_df_path = f"{UPLOAD_DATA_PATH}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet"
    processed_forecast_ds_path = f"{UPLOAD_DATA_PATH}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"

# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"
as_merge_variables = rfc.as_merge_variables

def process_forecast_data(forecast_df_path, measure, hierarchy_df):
    """
    Process forecast data, adding necessary columns and formatting.
    
    Parameters:
    -----------
    ssp_scenario : str
        SSP scenario name
    dah_scenario : str
        DAH scenario name
    draw : str
        Draw identifier
    hierarchy_df : pandas.DataFrame
        Hierarchy dataframe for location information
    Returns:
    --------
    pandas.DataFrame
        Full hierarchy forecast dataframe with all necessary columns and aggregations applied.
    """
    df = read_parquet_with_integer_ids(forecast_df_path)
    df = df[df["year_id"] >= 2022]
    short = measure_map[measure]["short"]
    df = df.rename(columns={col: col.replace(f'{cause}_{short}_', '') for col in df.columns if f'{cause}_{short}_' in col})
    drop_cols = [col for col in df.columns if 'rate' in col or 'pop' in col]
    df = df.drop(columns=drop_cols)

    df = df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")

    child_df = df.copy()

    for level in reversed(range(1,6)):
        
        print(f"Processing level {level}...")
        child_df = child_df.merge(hierarchy_df[["location_id", "parent_id"]], on="location_id", how="left")
        print(child_df["level"][0])
        parent_df = child_df.groupby(
            ["parent_id", "year_id", "age_group_id", "sex_id"]).agg({
            "count_pred": "sum"
        }).reset_index()

        parent_df = parent_df.rename(columns={
            "parent_id": "location_id"
        })

        parent_df = parent_df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
        df = pd.concat([df, parent_df], ignore_index=True)

        child_df = parent_df.copy()

    return df

# Process the forecast data
if cause == "malaria":
    print(f"Running for measure: {measure}, ssp_scenario: {ssp_scenario}, dah_scenario: {dah_scenario}, draw: {draw}")
else:
    print(f"Running for measure: {measure}, ssp_scenario: {ssp_scenario}, draw: {draw}")

full_hierarchy_forecast_df = process_forecast_data(forecast_df_path, measure, hierarchy_df)

full_hierarchy_forecast_ds = convert_with_preset(
    full_hierarchy_forecast_df,
    preset='as_variables',
    variable_dtypes={
        'count_pred': 'float32',
        'population': 'float32',
        'level': 'int8'
    },
    validate_dimensions=False  # Skip validation since we may have sparse data after aggregation
)

# Save the processed dataframe
write_parquet(full_hierarchy_forecast_df, processed_forecast_df_path)

write_netcdf(
    full_hierarchy_forecast_ds, 
    processed_forecast_ds_path,
    compression=True,
    compression_level=4,
    chunking=True,
    max_retries=3
)