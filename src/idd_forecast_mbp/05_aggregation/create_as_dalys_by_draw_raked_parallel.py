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

if cause == "malaria":
    if hold_variable == 'None':
        processed_yld_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yld_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
        processed_yll_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yll_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
        processed_daly_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_daly_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
    else:
        processed_yld_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yld_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
        processed_yll_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yll_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
        processed_daly_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_daly_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
else:
    if hold_variable == 'None':
        if vaccinate == 'None':
            processed_yld_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yld_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
            processed_yll_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yll_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
            processed_daly_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_daly_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
        else:
            processed_yld_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yld_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions.nc"
            processed_yll_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yll_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions.nc"
            processed_daly_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_daly_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions.nc"
    else:
        if vaccinate == 'None':
            processed_yld_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yld_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
            processed_yll_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yll_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
            processed_daly_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_daly_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
        else:
            processed_yld_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yld_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
            processed_yll_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yll_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
            processed_daly_forecast_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_daly_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions_hold_{hold_variable}.nc"

yld_ds = xr.open_dataset(processed_yld_forecast_ds_path)
yll_ds = xr.open_dataset(processed_yll_forecast_ds_path)
yld_aligned, yll_aligned = xr.align(yld_ds['count_pred'], yll_ds['count_pred'], join='outer', fill_value=0)
daly_ds = (yld_aligned + yll_aligned).to_dataset(name='count_pred')
# Clean up immediately
del yld_ds, yll_ds, yld_aligned, yll_aligned

write_netcdf(
        daly_ds, 
        processed_daly_forecast_ds_path,
        compression=True,
        compression_level=4,
        chunking=True,
        max_retries=3
    )