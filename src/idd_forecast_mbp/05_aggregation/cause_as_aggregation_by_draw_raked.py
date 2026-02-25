import os
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
from idd_forecast_mbp.xarray_functions import read_netcdf_with_integer_ids, write_netcdf, convert_with_preset
import argparse

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--cause", type=str, required=True, help="Cause name (e.g., 'malaria')")
parser.add_argument("--measure", type=str, required=False, default="mortality", help="measure (e.g., 'mortality', 'incidence')")
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=False, help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")
# parser.add_argument("--vaccinate", type=str, required=True, default='None', help="Vaccine status (e.g., 'True' or 'False')")
parser.add_argument("--hold_variable", type=str, required=True, default='None', help="Hold variable (e.g., 'gdppc', 'suitability', 'urban') or None for primary task")
parser.add_argument("--run_hold_variables", type=bool, required=False, default=False, help="Whether to run hold variables")
parser.add_argument("--run_date", type=str, required=True, default='2025_07_08', help="Run date in format YYYY_MM_DD (e.g., '2025_06_25')")

# Parse arguments
args = parser.parse_args()

cause = args.cause
measure = args.measure
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
draw = args.draw
vaccinate = 'None'
hold_variable = args.hold_variable
run_date = args.run_date

measure_map = rfc.measure_map

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data"

raked_base = '/mnt/team/rapidresponse/pub/malaria-denv/deliverables/2025_08_26_admin_2_counts'
folder_template_dict = {
    'dengue': '{direction}/as_cause_dengue_measure_{measure}_metric_count_ssp_scenario_{ssp_scenario}{suffix}',
    'malaria': '{direction}/as_cause_malaria_measure_{measure}_metric_count_ssp_scenario_{ssp_scenario}_dah_scenario_Baseline{suffix}'
}

output_input_matching={
    'incidence':['incidence', 'yld'],
    'mortality':['mortality', 'yll']
}

output_measures = output_input_matching[measure]

for output_measure in output_measures:
    # Calculate multipliers
    
    output_folder = folder_template_dict[cause].format(direction='output/2025_09_08', measure=output_measure, ssp_scenario=ssp_scenario, suffix='_raked')
    draw_int = int(draw)
    output_ds_path = f'{raked_base}/{output_folder}/draw_{draw_int}.nc'
    output_ds = xr.open_dataset(output_ds_path)
    output_ds = output_ds.drop_vars(['draw', 'draw_id', 'scenario']).rename({'value': 'val'})
    # Squeeze only if 'draw' and 'scenario' are coordinates
    for dim in ['draw', 'scenario']:
        if dim in output_ds.coords:
            output_ds = output_ds.squeeze(dim)
    
    input_folder = folder_template_dict[cause].format(direction='input', measure=measure, ssp_scenario=ssp_scenario, suffix='')
    input_ds_path = f'{raked_base}/{input_folder}/draw_{draw_int}.nc'
    input_ds = xr.open_dataset(input_ds_path)
    level_5_location_ids = output_ds['location_id']
    input_ds = input_ds.drop_vars(['draw_id']).sel(location_id=level_5_location_ids)

    ratio = output_ds['val'] / input_ds['val']
    ratio = xr.where(input_ds['val'] == 0, np.nan, ratio)
    ratio_ds = ratio.to_dataset(name='ratio')
    del output_ds, input_ds, ratio

    # Run the rest of things (be sure to multiply by ratio right after loading the draw)

    if cause == "malaria":
        if hold_variable == 'None':
            forecast_ds_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
            processed_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_{output_measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
        else:
            forecast_ds_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
            processed_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_{output_measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
    else:
        if hold_variable == 'None':
            if vaccinate == 'None':
                forecast_ds_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
                processed_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_{output_measure}_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
            else:
                forecast_ds_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions.nc"
                processed_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_{output_measure}_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions.nc"
        else:
            if vaccinate == 'None':
                forecast_ds_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
                processed_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_{output_measure}_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
            else:
                forecast_ds_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
                processed_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_{output_measure}_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions_hold_{hold_variable}.nc"

    # Hierarchy path
    hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
    hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

    def process_forecast_data(forecast_ds_path, measure, hierarchy_df, ratio_ds):
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
        ds = read_netcdf_with_integer_ids(forecast_ds_path)
        vars_to_drop = [v for v in ds.data_vars if 'aa' in v or 'gbd' in v]
        ds = ds.drop_vars(vars_to_drop)

        ds_location_ids = ds["location_id"]
        tmp_ratio_ds = ratio_ds.sel(location_id = ds_location_ids)
        for var in ds.data_vars:
            ds[var] = ds[var] * tmp_ratio_ds['ratio']


        df = ds.to_dataframe().reset_index()

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

        df = df.drop(columns=['level'])
        return df

    # Process the forecast data
    if cause == "malaria":
        print(f"Running for measure: {output_measure}, ssp_scenario: {ssp_scenario}, dah_scenario: {dah_scenario}, draw: {draw}")
    else:
        print(f"Running for measure: {output_measure}, ssp_scenario: {ssp_scenario}, vaccinate: {vaccinate}, draw: {draw}")



    full_hierarchy_forecast_df = process_forecast_data(forecast_ds_path, measure, hierarchy_df, ratio_ds)

    # full_hierarchy_forecast_df = full_hierarchy_forecast_df[['location_id', 'year_id', 'count_pred']].groupby(['location_id', 'year_id']).sum().reset_index()

    full_hierarchy_forecast_ds = convert_with_preset(
        full_hierarchy_forecast_df,
        preset='as_variables',
        variable_dtypes={
            'count_pred': 'float32',
            'level': 'int8'
        },
        validate_dimensions=False  # Skip validation since we may have sparse data after aggregation
    )

    write_netcdf(
        full_hierarchy_forecast_ds, 
        processed_forecast_ds_path,
        compression=True,
        compression_level=4,
        chunking=True,
        max_retries=3
    )