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
from idd_forecast_mbp.helper_functions import check_folders_for_files
from idd_forecast_mbp.hd5_functions import write_hdf
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids
from idd_forecast_mbp.xarray_functions import convert_with_preset, write_netcdf, read_netcdf_with_integer_ids
import argparse
import os

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--cause", type=str, required=False, default="malaria", help="Cause (e.g., 'malaria', 'dengue')")
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=False, default="Baseline", help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--measure", type=str, required=False, default="mortality", help="measure (e.g., 'mortality', 'incidence')")
parser.add_argument("--hold_variable", type=str, required=False, default='None', help="Hold variable (e.g., 'gdppc', 'suitability', 'urban') or None for primary task")
parser.add_argument("--run_date", type=str, required=True, default='2025_07_08', help="Run date in format YYYY_MM_DD (e.g., '2025_06_25')")
parser.add_argument("--delete_existing", type=str, required=False, default=True, help="Flag to indicate if existing upload folder should be deleted")


# Parse arguments
args = parser.parse_args()

cause = args.cause
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
vaccinate = 'None'
measure = args.measure
hold_variable = args.hold_variable
run_date = args.run_date
delete_existing = args.delete_existing

# hold_variable = "None"
# vaccinate = "None"
# cause = "malaria"  # or "dengue"
# ssp_scenario = "ssp126"  # or "ssp245", "ssp585"
# dah_scenario = "Baseline"  # or other scenarios
# measure = "mortality"  # or "incidence"
# run_date = '2025_08_11'  # Example run date
# delete_existing = False  # Flag to indicate if existing upload folder should be deleted

metric = "count"

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data"
# FINAL_UPLOAD_DATA_PATH = '/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2025/'
FINAL_UPLOAD_DATA_PATH = UPLOAD_DATA_PATH
FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"

ssp_draws = rfc.draws
measure_map = rfc.measure_map
metric_map = rfc.metric_map
cause_map = rfc.cause_map
ssp_scenarios = rfc.ssp_scenarios
scenario = ssp_scenarios[ssp_scenario]["dhs_scenario"] #  is the DHS scenario name

if cause == "malaria":
    if hold_variable == 'None':
        processed_forecast_ds_path_template = "{UPLOAD_DATA_PATH}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
        as_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}"
        aa_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}"
    else:
        processed_forecast_ds_path_template = "{UPLOAD_DATA_PATH}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
        as_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_hold_variable_{hold_variable}"
        aa_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_hold_variable_{hold_variable}"
    
else:
    if hold_variable == 'None':
        if vaccinate == 'None':
            processed_forecast_ds_path_template = "{UPLOAD_DATA_PATH}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
            as_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}"
            aa_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}"
        else:
            processed_forecast_ds_path_template = "{UPLOAD_DATA_PATH}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions.nc"
            as_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_no_vaccinate"
            aa_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_no_vaccinate"
    else:
        if vaccinate == 'None':
            processed_forecast_ds_path_template = "{UPLOAD_DATA_PATH}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
            as_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_hold_variable_{hold_variable}"
            aa_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_hold_variable_{hold_variable}"
        else:
            processed_forecast_ds_path_template = "{UPLOAD_DATA_PATH}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
            as_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_no_vaccinate_hold_variable_{hold_variable}"
            aa_upload_folder_path = f"{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_no_vaccinate_hold_variable_{hold_variable}"
    
as_upload_draws_file_path = f"{as_upload_folder_path}/draws.nc"
as_upload_mean_file_path = f"{as_upload_folder_path}/mean.nc"
aa_upload_draws_file_path = f"{aa_upload_folder_path}/draws.nc"
aa_upload_mean_file_path = f"{aa_upload_folder_path}/mean.nc"

folders_and_files = {
    as_upload_folder_path: [as_upload_draws_file_path, as_upload_mean_file_path],
    aa_upload_folder_path: [aa_upload_draws_file_path, aa_upload_mean_file_path]
}

# Check all folders and their files at once
folder_results = check_folders_for_files(folders_and_files, delete_existing=delete_existing)
no_need_to_run = all(folder_results.values())

if no_need_to_run:
    print("All required files are already present and you didn't say to delete. No need to run the script.")
    exit(0)
else:
    print("Some files or folders were missing or needed to be deleted. Proceeding with the script.")

age_metadata_path = f"{FHS_DATA_PATH}/age_metadata.parquet"

# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"
aa_full_population_df_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population.parquet"
as_merge_variables = rfc.as_merge_variables
aa_merge_variables = rfc.aa_merge_variables

swap_location_ids = [60908, 95069, 94364]

all_location_ids = hierarchy_df["location_id"].unique().tolist()

year_ids = range(2022, 2101)
# Make filters based on hierarchy_df
all_location_filter = ('location_id', 'in', all_location_ids)

# Use all_location_ids for non-FHS flag
location_filter = all_location_filter


year_filter = ('year_id', 'in', year_ids)

print(f"Processing SSP scenario: {ssp_scenario}")
scenario = ssp_scenarios[ssp_scenario]["dhs_scenario"]
print(f"Scenario: {scenario}")

def get_file_path(draw, cause, measure, ssp_scenario, dah_scenario=None, vaccinate = None, hold_variable=None):
    """Generate file path based on cause type"""
    return processed_forecast_ds_path_template.format(
        UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
        cause=cause,
        measure=measure,
        ssp_scenario=ssp_scenario,
        dah_scenario=dah_scenario,
        vaccinate=vaccinate,
        draw=draw,
        hold_variable=hold_variable
    )

# Generate all file paths for all draws
file_paths = [get_file_path(draw, cause, measure, ssp_scenario, dah_scenario, vaccinate, hold_variable) 
              for draw in ssp_draws]

print(f"Loading {len(file_paths)} files...")

# Open all files as a single dataset with lazy loading
upload_ds = xr.open_mfdataset(
    file_paths,
    combine='nested',
    concat_dim='draw_id',  # This creates a new dimension for the draws
    chunks='auto',  # Enable dask for lazy loading and memory efficiency
    drop_variables=['gbd_location_id', 'aa_count', 'level']
)

# Assign proper draw names to the new dimension
upload_ds = upload_ds.assign_coords(draw_id=ssp_draws)

# The variables will be named after the draw_dim coordinate values (which are the ssp_draws)
# So they should already have the correct names like '000', '001', etc.
print(f"Variable names created: {list(upload_ds.data_vars)}")

print("Data loading complete (lazy - data stays on disk until computed)")
print(f"Dataset variables: {list(upload_ds.data_vars)}")
print(f"Dataset shape: {upload_ds.dims}")

# ULTRA-FAST ALTERNATIVE: Use xr.zeros_like and reindex
import xarray as xr
import numpy as np

# Get the existing coordinates from your dataset
existing_location_ids = upload_ds.coords['location_id'].values
year_ids = upload_ds.coords['year_id'].values  
age_group_ids = upload_ds.coords['age_group_id'].values
sex_ids = upload_ds.coords['sex_id'].values
draw_ids = upload_ds.coords['draw_id'].values

# Load age metadata
age_metadata_df = read_parquet_with_integer_ids(age_metadata_path)
age_group_ids_full = age_metadata_df["age_group_id"].unique()

# Find missing location IDs
sex_ids_full = [1, 2]
missing_location_ids = set(all_location_ids) - set(existing_location_ids)
missing_location_ids.discard(44858)
missing_location_ids = list(missing_location_ids)

# Create the complete coordinate space we want
complete_coords = {
    'location_id': sorted(list(existing_location_ids) + missing_location_ids),
    'year_id': year_ids,
    'age_group_id': age_group_ids_full,
    'sex_id': sex_ids_full,
    'draw_id': ssp_draws
}

# Reindex the original dataset to the complete coordinate space
upload_ds_complete = upload_ds.reindex(complete_coords, fill_value=0.0)

# Update the dataset reference
as_ds = upload_ds_complete
as_ds = as_ds.rename({'count_pred': 'val'})
as_mean_ds = as_ds.to_array().mean(dim='draw_id').to_dataset(name='val')

################ Calculate all-age mean and sum ################
aa_ds = upload_ds.sum(dim=['sex_id', 'age_group_id'])
aa_mean_ds = aa_ds.to_array().mean(dim='draw_id').to_dataset(name='val')

############### Write the datasets to NetCDF files ################
# Write the all-age mean to NetCDF
print(f"Writing to all-age mean to {aa_upload_mean_file_path}")
write_netcdf(
    ds=aa_mean_ds,
    filepath=aa_upload_mean_file_path,
    compression_level=4,  # Good balance for large files
    max_chunk_size=2000,  # Larger chunks for forecast data
    chunk_threshold=500000  # Lower threshold for better chunking
)


# Write the all-age draws to NetCDF
print(f"Writing to all-age draws  to {aa_upload_draws_file_path}")
write_netcdf(
    ds=aa_ds,
    filepath=aa_upload_draws_file_path,
    compression_level=4,  # Good balance for large files
    max_chunk_size=2000,  # Larger chunks for forecast data
    chunk_threshold=500000  # Lower threshold for better chunking
)

# Write the age-specific mean to NetCDF
print(f"Writing to age-specific mean to {as_upload_mean_file_path}")
write_netcdf(
    ds=as_mean_ds,
    filepath=as_upload_mean_file_path,
    compression_level=4,  # Good balance for large files
    max_chunk_size=2000,  # Larger chunks for forecast data
    chunk_threshold=500000  # Lower threshold for better chunking
)

# # Write the age- and sex-specific draws to NetCDF
print(f"Writing to age- and sex-specific draws to {as_upload_draws_file_path}")
write_netcdf(
    ds=as_ds,
    filepath=as_upload_draws_file_path,
    compression_level=1,  # Fastest compression
    chunk_by_dim={
        'location_id': 1500,  # Chunk locations into groups of 1500
        'year_id': 79,        # Keep all years together
        'age_group_id': 25,   # Keep all ages together
        'sex_id': 2           # Keep both sexes together
    },
    use_temp_file=False
)