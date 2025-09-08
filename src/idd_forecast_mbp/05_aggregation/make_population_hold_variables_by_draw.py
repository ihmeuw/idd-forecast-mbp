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

parser = argparse.ArgumentParser(description="Add DAH Scenarios and create draw level dataframes for forecasting malaria")

# Define arguments
parser.add_argument("--cause", type=str, required=True, help="Cause name (e.g., 'malaria')")
parser.add_argument("--measure", type=str, required=False, default="mortality", help="measure (e.g., 'mortality', 'incidence')")
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")
parser.add_argument("--run_date", type=str, required=True, help="Run date in format YYYY_MM_DD (e.g., '2025_06_25')")

# Parse arguments
args = parser.parse_args()

cause = args.cause
measure = args.measure
ssp_scenario = args.ssp_scenario
draw = args.draw
run_date = args.run_date
reference_year = 2022

dah_scenario = 'Baseline'

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data"

as_full_population_ds_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population_ds.nc"
aa_full_population_ds_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population_ds.nc"
hierarchy_ds_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.nc'

processed_forecast_ds_path_template = "{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}{dah_text}_draw_{draw}_with_predictions{hold_text}.nc"

def get_draw_path(processed_forecast_ds_path_template, UPLOAD_DATA_PATH, cause, measure, ssp_scenario, hold_variable, dah_scenario, draw, run_date) -> Path:
    if hold_variable is None:
        hold_text = ''
    else:
        hold_text = f'_hold_{hold_variable}'
    if cause == 'dengue':
        dah_text = ''
    else:
        dah_text = f'_dah_scenario_{dah_scenario}'
    processed_forecast_ds_path = processed_forecast_ds_path_template.format(
        UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
        run_date=run_date,
        cause=cause,
        measure=measure,
        ssp_scenario=ssp_scenario,
        dah_text=dah_text,
        draw=draw,
        hold_text=hold_text
    )
    return Path(processed_forecast_ds_path) 

# Make ds paths
base_ds_path = get_draw_path(processed_forecast_ds_path_template, UPLOAD_DATA_PATH, cause, measure, ssp_scenario, 
    hold_variable=None, dah_scenario=dah_scenario, draw=draw, run_date=run_date
)

population_hold_path = get_draw_path(processed_forecast_ds_path_template, UPLOAD_DATA_PATH, cause, measure, ssp_scenario, 
    hold_variable='population', dah_scenario=dah_scenario, draw=draw, run_date=run_date
)

as_structure_hold_path = get_draw_path(processed_forecast_ds_path_template, UPLOAD_DATA_PATH, cause, measure, ssp_scenario, 
    hold_variable='as_structure', dah_scenario=dah_scenario, draw=draw, run_date=run_date
)

# Load hierarchy data

hierarchy_ds = read_netcdf_with_integer_ids(hierarchy_ds_path)
hierarchy_df = hierarchy_ds.to_dataframe().reset_index()
# Find level 5 location ids
level_5_location_ids = hierarchy_ds.location_id.where(hierarchy_ds.level == 5, drop=True).values


# Load population data
aa_full_population_ds = read_netcdf_with_integer_ids(aa_full_population_ds_path)
as_full_population_ds = read_netcdf_with_integer_ids(as_full_population_ds_path)
as_full_population = as_full_population_ds['population']
aa_full_population = aa_full_population_ds['population']

# Get reference year population data
as_pop_reference_year = as_full_population.sel(year_id=reference_year)
aa_pop_reference_year = aa_full_population.sel(year_id=reference_year)

# Get outcome data
base_ds = read_netcdf_with_integer_ids(base_ds_path)
hierarchy_subset = hierarchy_ds[['level']]
base_with_hierarchy = xr.merge([base_ds, hierarchy_subset], join='left')
base_ds = base_with_hierarchy.where(base_with_hierarchy.level == 5, drop=True)


# Calculate ratios once (much more efficient)
as_reference_fraction = as_pop_reference_year / as_full_population
aa_reference_fraction = aa_pop_reference_year / aa_full_population

as_reference_fraction_matched = as_reference_fraction.sel(
    location_id=base_ds.location_id,
    year_id=base_ds.year_id
)
aa_reference_fraction_matched = aa_reference_fraction.sel(
    location_id=base_ds.location_id,
    year_id=base_ds.year_id
)

as_reference_fraction_matched = as_reference_fraction_matched.astype(np.float32)
aa_reference_fraction_matched = aa_reference_fraction_matched.astype(np.float32)

def process_forecast_data(ds, hierarchy_df):
        df = ds.to_dataframe().reset_index()
        df = df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
        child_df = df.copy()
        for level in reversed(range(1,6)):
            child_df = child_df.merge(hierarchy_df[["location_id", "parent_id"]], on="location_id", how="left")
            parent_df = child_df.groupby(["parent_id", "year_id", "age_group_id", "sex_id"]).agg({"count_pred": "sum"}).reset_index()

            parent_df = parent_df.rename(columns={"parent_id": "location_id"})
            parent_df = parent_df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
            df = pd.concat([df, parent_df], ignore_index=True)
            child_df = parent_df.copy()
        df = df.drop(columns=['level'])
        ds = convert_with_preset(
            df, preset='as_variables', variable_dtypes={'count_pred': 'float32'}, validate_dimensions=False)
        return ds

#########################################
## Making hold_variable = 'population' ##
#########################################
hold_population_count = base_ds['count_pred'] * as_reference_fraction_matched
population_hold_ds = base_ds.copy()
population_hold_ds['count_pred'] = hold_population_count
full_population_hold_ds = process_forecast_data(population_hold_ds, hierarchy_df)

write_netcdf(
        full_population_hold_ds, 
        population_hold_path,
        compression=True,
        compression_level=4,
        chunking=True,
        max_retries=3
    )
del population_hold_ds

###################################################
## Making hold_variable = 'as_structure' ##
###################################################
hold_as_structure_count = base_ds['count_pred'] * as_reference_fraction_matched / aa_reference_fraction_matched

# Create dataset for as_structure hold
as_structure_hold_ds = base_ds.copy()
as_structure_hold_ds['count_pred'] = hold_as_structure_count

full_as_structure_hold_ds = process_forecast_data(as_structure_hold_ds, hierarchy_df)

write_netcdf(
        full_as_structure_hold_ds, 
        as_structure_hold_path,
        compression=True,
        compression_level=4,
        chunking=True,
        max_retries=3
    )