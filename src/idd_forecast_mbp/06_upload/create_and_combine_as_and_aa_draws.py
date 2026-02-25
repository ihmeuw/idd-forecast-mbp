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

# --- CONFIGURATION AND ARGUMENT PARSING ---

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--cause", type=str, required=False, default="malaria", help="Cause (e.g., 'malaria', 'dengue')")
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=False, default="Baseline", help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--measure", type=str, required=False, default="mortality", help="measure (e.g., 'mortality', 'incidence')")
parser.add_argument("--metric", type=str, required=False, default="count", help="metric (e.g., 'count', 'rate')")
parser.add_argument("--run_date", type=str, required=True, help="Run date in format YYYY_MM_DD (e.g., '2025_06_25')")


# Parse arguments
args = parser.parse_args()

cause = args.cause
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
measure = args.measure
metric = args.metric
hold_variable = 'None'
run_date = args.run_date

# --- PATHS AND CONSTANTS ---

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data"
FINAL_UPLOAD_DATA_PATH = UPLOAD_DATA_PATH
FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"

ssp_draws = rfc.draws
metric_map = rfc.metric_map
cause_map = rfc.cause_map
ssp_scenarios = rfc.ssp_scenarios
full_measure_map = rfc.full_measure_map
scenario = ssp_scenarios[ssp_scenario]["dhs_scenario"]

# Build file path components
hold_text = f'_hold_{hold_variable}' if hold_variable != 'None' else ''
dah_text = f'_dah_scenario_{dah_scenario}' if cause != 'dengue' else ''

processed_forecast_ds_path_template = "{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}{dah_text}_draw_{draw}_with_predictions{hold_text}.nc"
as_upload_folder_path_template = "{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}{dah_text}{hold_text}"
aa_upload_folder_path_template = "{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}{dah_text}{hold_text}"

def get_as_path(cause, measure, metric, ssp_scenario, dah_text, hold_text, run_date):
    as_upload_folder_path = as_upload_folder_path_template.format(FINAL_UPLOAD_DATA_PATH=FINAL_UPLOAD_DATA_PATH, run_date=run_date, cause=cause, measure=measure, metric=metric, ssp_scenario=ssp_scenario, dah_text=dah_text, hold_text=hold_text)
    aa_upload_folder_path = aa_upload_folder_path_template.format(FINAL_UPLOAD_DATA_PATH=FINAL_UPLOAD_DATA_PATH, run_date=run_date, cause=cause, measure=measure, metric=metric, ssp_scenario=ssp_scenario, dah_text=dah_text, hold_text=hold_text)
    return as_upload_folder_path, aa_upload_folder_path

as_upload_folder_path, aa_upload_folder_path = get_as_path(cause=cause, measure=measure, metric=metric, ssp_scenario=ssp_scenario, dah_text=dah_text, hold_text=hold_text, run_date=run_date)

# Ensure AS folder exists
# Ensure AA folder exists
mkdir(as_upload_folder_path, exist_ok=True, parents=True)
mkdir(aa_upload_folder_path, exist_ok=True, parents=True)

as_upload_draws_file_path = f"{as_upload_folder_path}/draws.nc"
as_upload_mean_file_path = f"{as_upload_folder_path}/mean.nc"
aa_upload_draws_file_path = f"{aa_upload_folder_path}/draws.nc"
aa_upload_mean_file_path = f"{aa_upload_folder_path}/mean.nc"
age_metadata_path = f"{FHS_DATA_PATH}/age_metadata.parquet"
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
as_full_population_ds_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population_ds.nc"

# --- HIERARCHY AND FILTER PREP ---

hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)
fhs_hierarchy_df = hierarchy_df[hierarchy_df["in_fhs_hierarchy"] == True].copy()

swap_location_ids = [60908, 95069, 94364]
fhs_location_ids = fhs_hierarchy_df["location_id"].unique().tolist() + swap_location_ids
year_ids = range(2022, 2101)

# --- 1. POPULATION LOADING AND PRE-FILTERING (Efficient) ---

# Load population data early for efficient filtering of the main forecast.
pop_ds = read_netcdf_with_integer_ids(as_full_population_ds_path)

pop_locations = pop_ds.location_id.values
pop_years = pop_ds.year_id.values

# Define the *intersection* of target locations/years and available population data.
locations_to_filter = [loc for loc in fhs_location_ids if loc in pop_locations]
years_to_filter = [yr for yr in year_ids if yr in pop_years]

# Filter population data using the safe lists
pop_ds = pop_ds.sel(location_id=locations_to_filter, year_id=years_to_filter)
print(f"Processing SSP scenario: {ssp_scenario}")

# --- 2. FORECAST LOADING AND EFFICIENT FILTERING (Robust) ---

def get_draw_file_path(UPLOAD_DATA_PATH, run_date, draw, cause, measure, ssp_scenario, dah_text, hold_text):
    """Generate file path based on cause type"""
    return processed_forecast_ds_path_template.format(UPLOAD_DATA_PATH=UPLOAD_DATA_PATH, run_date=run_date, cause=cause, measure=measure, ssp_scenario=ssp_scenario, dah_text=dah_text, draw=draw, hold_text=hold_text)

file_paths = [get_draw_file_path(UPLOAD_DATA_PATH, run_date, draw, cause, measure, ssp_scenario, dah_text, hold_text) for draw in ssp_draws]

print(f"Loading {len(file_paths)} files and applying efficient location/year filter...")

# Define the preprocess function using a lambda. Uses .reindex() to prevent KeyError.
preprocess_func = lambda ds: ds.reindex(
    location_id=locations_to_filter, 
    year_id=years_to_filter
).drop_vars(
    ['gbd_location_id', 'aa_count', 'level'], 
    errors='ignore'
)

# Open files with efficient Dask preprocessing
upload_ds = xr.open_mfdataset(
    file_paths,
    combine='nested',
    concat_dim='draw_id',  
    chunks='auto', 
    drop_variables=['gbd_location_id', 'aa_count', 'level'],
    preprocess=preprocess_func 
)

# --- 3. MERGE POPULATION AND CALCULATE SEX=3 AGGREGATE ---

print("Merging population data and calculating sex=3 aggregates...")

# Merge forecast counts ('count_pred') with population ('population') using Left Join.
upload_ds = xr.merge([upload_ds, pop_ds['population']], join='left')

# Calculate Both Sexes (sex_id=3)
counts_both_sexes = upload_ds['count_pred'].sum(dim='sex_id', skipna=False)
pop_both_sexes = upload_ds['population'].sum(dim='sex_id', skipna=False)

# Re-add the 'sex_id' dimension with the new label (3)
counts_both_sexes = counts_both_sexes.assign_coords(sex_id=3).expand_dims('sex_id')
pop_both_sexes = pop_both_sexes.assign_coords(sex_id=3).expand_dims('sex_id')

# Concatenate the new sex_id=3 slice back onto the main dataset.
upload_ds['count_pred'] = xr.concat([upload_ds['count_pred'], counts_both_sexes], dim='sex_id')
upload_ds['population'] = xr.concat([upload_ds['population'], pop_both_sexes], dim='sex_id')

# Rename raw forecast counts and population columns for clarity through the rest of the pipeline
upload_ds = upload_ds.rename({'count_pred': 'val'})

# --- 4. AS RAW DATASET FINALIZATION (Coordinate Assignment & Re-index) ---

# Assign draw coordinate labels
if isinstance(ssp_draws[0], str):
    ssp_draws_int = [int(x) for x in ssp_draws]
else:
    ssp_draws_int = ssp_draws
    
upload_ds = upload_ds.assign_coords(draw_id=ssp_draws_int)

# Extract existing coordinates for re-indexing logic
existing_location_ids = upload_ds.coords['location_id'].values

# Calculate full coordinate space required
age_metadata_df = read_parquet_with_integer_ids(age_metadata_path)
age_group_ids_full = age_metadata_df["age_group_id"].unique()
sex_ids_full = [1, 2, 3] # Include all 3 sex IDs for final output

# Find missing locations (to fill with 0.0)
missing_location_ids = set(fhs_location_ids) - set(existing_location_ids)
missing_location_ids.discard(44858)
missing_location_ids = list(missing_location_ids)

# Create the complete coordinate space
complete_coords = {
    'location_id': sorted(list(existing_location_ids) + missing_location_ids),
    'year_id': year_ids, # Use full year range
    'age_group_id': age_group_ids_full, # Use full age group range
    'sex_id': sex_ids_full, # Use all sex IDs (1, 2, 3)
    'draw_id': ssp_draws_int
}

print("Reindexing to full AS coordinate space...")
# Reindex and Rechunk for final output optimization
as_ds_raw_counts = upload_ds.reindex(complete_coords, fill_value=0.0)
as_ds_raw_counts = as_ds_raw_counts.chunk({
    'draw_id': 10,  # 10 draws per chunk
    'location_id': -1,  
    'year_id': -1,
    'age_group_id': -1,
    'sex_id': -1
})
as_ds_raw_counts = as_ds_raw_counts.rename({'val': 'val_count', 'population': 'population_count'})


# --- 5. ALL-AGE (AA) AGGREGATION (Must aggregate raw counts) ---

# AA Draw Aggregation: Sum across sex_id (1, 2) and age_group_id
print("Aggregating AS data to create All-Age (AA) data...")

# NOTE: Must explicitly select sex_id=[1, 2] to prevent double-counting sex_id=3.
aa_ds = as_ds_raw_counts.sel(sex_id=[1, 2]).sum(dim=['sex_id', 'age_group_id'], skipna=False)

# Re-add sex_id dimension with label 3 (All Sexes)
aa_ds = aa_ds.assign_coords(sex_id=3).expand_dims('sex_id')


# --- 6. FINAL RATE CALCULATION AND WRITE NETCDF ---

def calculate_rate_and_mean(ds_raw, metric_type, ds_type):
    """
    Calculates rate (if needed), mean, and assigns metadata for a single dataset.
    ds_raw contains 'val_count' and 'population_count'.
    """
    
    ds_out = ds_raw.copy()
    
    # 1. Perform Rate Calculation (if needed)
    if metric_type == 'rate':
        print(f"[{ds_type}] Calculating final rates per 1,000...")
        
        # Robust division: val / population * 1000
        ds_out['val_rate'] = ds_out['val_count'].where(
            ds_out['population_count'] > 0, 
            other=0.0
        )
        ds_out['val_rate'] = (ds_out['val_rate'] / ds_out['population_count']) * 1000
        
        # Rename and cleanup
        ds_out = ds_out.drop_vars(['val_count', 'population_count'])
        ds_out = ds_out.rename({'val_rate': 'val'})
    
    else: # metric_type is 'count'
        print(f"[{ds_type}] Using raw counts.")
        # Rename and cleanup
        ds_out = ds_out.rename({'val_count': 'val'})
        ds_out = ds_out.drop_vars('population_count')
    
    # 2. Calculate Mean Dataset
    ds_mean = ds_out.to_array().mean(dim='draw_id').to_dataset(name='val')

    # 3. Assign Metadata (Scalars)
    ds_out = ds_out.assign_coords(cause_id=cause_map[cause]['cause_id'])
    ds_out = ds_out.assign_coords(metric_id=metric_map[metric_type]['metric_id'])
    ds_out = ds_out.assign_coords(measure_id=full_measure_map[measure]['measure_id'])
    
    ds_mean = ds_mean.assign_coords(cause_id=cause_map[cause]['cause_id'])
    ds_mean = ds_mean.assign_coords(metric_id=metric_map[metric_type]['metric_id'])
    ds_mean = ds_mean.assign_coords(measure_id=full_measure_map[measure]['measure_id'])
    
    # 4. Type Casting for AA output only
    if ds_type == 'AA':
         ds_out = ds_out.assign_coords(
            location_id=ds_out.location_id.astype(np.int32),
            year_id=ds_out.year_id.astype(np.int16),
            draw_id=ds_out.draw_id.astype(np.int8)
        )
         ds_mean = ds_mean.assign_coords(
            location_id=ds_mean.location_id.astype(np.int32),
            year_id=ds_mean.year_id.astype(np.int16)
        )
        
    return ds_out, ds_mean

# Prepare AS data for final output
as_draws, as_mean = calculate_rate_and_mean(as_ds_raw_counts, metric, 'AS')

# Prepare AA data for final output
aa_draws, aa_mean = calculate_rate_and_mean(aa_ds, metric, 'AA')


# Write final NetCDF files
print("Writing final AS and AA files...")
write_netcdf(ds=as_mean, filepath=as_upload_mean_file_path, compression_level=4, max_chunk_size=2000, chunk_threshold=500000)
write_netcdf(ds=as_draws, filepath=as_upload_draws_file_path, compression_level=4, max_chunk_size=2000, chunk_threshold=500000)
write_netcdf(ds=aa_mean, filepath=aa_upload_mean_file_path, compression_level=4, max_chunk_size=2000, chunk_threshold=500000)
write_netcdf(ds=aa_draws, filepath=aa_upload_draws_file_path, compression_level=4, max_chunk_size=2000, chunk_threshold=500000)

del as_ds_raw_counts, aa_ds, as_draws, as_mean, aa_draws, aa_mean