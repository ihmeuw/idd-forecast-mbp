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
from idd_forecast_mbp.hd5_functions import write_hdf
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids
from idd_forecast_mbp.xarray_functions import write_netcdf, read_netcdf_with_integer_ids
import argparse
import os

parser = argparse.ArgumentParser(description="Create summary files")

# Define arguments
parser.add_argument("--cause", type=str, required=False, default="malaria", help="Cause (e.g., 'malaria', 'dengue')")
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=False, default="Baseline", help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--measure", type=str, required=False, default="mortality", help="measure (e.g., 'mortality', 'incidence')")
parser.add_argument("--run_date", type=str, required=True, default=2025_06_25, help="Run date in format YYYY_MM_DD (e.g., '2025_06_25')")


# Parse arguments
args = parser.parse_args()

cause = args.cause
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
measure = args.measure
run_date = args.run_date

UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data"

if cause == "malaria":
    count_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_count_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}"
    rate_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_rate_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}"
    aa_count_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_count_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}"
    aa_rate_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_rate_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}"
    count_upload_file_path = f"{count_upload_folder_path}/draws.nc"
else:
    count_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_count_ssp_scenario_{ssp_scenario}"
    rate_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_rate_ssp_scenario_{ssp_scenario}"
    aa_count_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_count_ssp_scenario_{ssp_scenario}"
    aa_rate_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_rate_ssp_scenario_{ssp_scenario}"
    count_upload_file_path = f"{count_upload_folder_path}/draws.nc"

# Create the aa_ folders if they don't already exist
mkdir(aa_count_upload_folder_path, exist_ok=True, parents=True)
mkdir(aa_rate_upload_folder_path, exist_ok=True, parents=True)

# Read the count dataset
count_ds = read_netcdf_with_integer_ids(count_upload_file_path)

# Get all draw columns
draw_vars = [var for var in count_ds.data_vars if var.startswith('draw_')]

# Aggregate across age and sex for each draw
all_age_ds = count_ds[draw_vars + ['population']].sum(dim=['age_group_id', 'sex_id'])

all_age_count_file_path = f"{count_upload_folder_path}/all_age_draws.nc"
write_netcdf(all_age_ds, all_age_count_file_path)
print(f"Saved all-age count draws to: {all_age_count_file_path}")

# Calculate mean across draws for counts
draw_arrays = [all_age_ds[draw_var] for draw_var in draw_vars]
mean_count = xr.concat(draw_arrays, dim='draw').mean(dim='draw')
# Create dataset with mean count
mean_count_ds = xr.Dataset({
    'mean_count': mean_count,
    'population': all_age_ds['population']
})

# Save mean count file
mean_count_file_path = f"{aa_count_upload_folder_path}/all_age_mean.nc"
write_netcdf(mean_count_ds, mean_count_file_path)
print(f"Saved all-age mean count to: {mean_count_file_path}")

# Calculate draw-specific rates
rate_data = {}
for draw_var in draw_vars:
    rate_data[draw_var] = all_age_ds[draw_var] / all_age_ds['population']

rate_data['population'] = all_age_ds['population']
rate_draws_ds = xr.Dataset(rate_data)

# Save rate draws
all_age_rate_draws_file_path = f"{rate_upload_folder_path}/all_age_draws.nc"
write_netcdf(rate_draws_ds, all_age_rate_draws_file_path)
print(f"Saved all-age rate draws to: {all_age_rate_draws_file_path}")

# Calculate mean rate
mean_rate = mean_count_ds['mean_count'] / mean_count_ds['population']
mean_rate_ds = xr.Dataset({
    'mean_rate': mean_rate,
    'population': mean_count_ds['population']
})

# Save mean rate file
mean_rate_file_path = f"{aa_rate_upload_folder_path}/all_age_mean.nc"
write_netcdf(mean_rate_ds, mean_rate_file_path)
print(f"Saved all-age mean rate to: {mean_rate_file_path}")

print("\nAll aggregation and rate calculations completed!")