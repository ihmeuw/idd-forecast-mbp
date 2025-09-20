import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from typing import cast, Literal, NamedTuple, List, Dict, Optional
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
import itertools
import matplotlib.pyplot as plt
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.hd5_functions import write_hdf
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids
from idd_forecast_mbp.xarray_functions import convert_to_xarray, write_netcdf, ensure_id_coordinates_are_integers, read_netcdf_with_integer_ids
import os
from pathlib import Path
from idd_forecast_mbp.number_functions import *
from idd_forecast_mbp.fhs_functions import *

best_run_date = "2025_07_08"
pull_date = '2025_07_23'
version = 'v6'

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data"
VISUALIZATION_PATH = rfc.MODEL_ROOT / "06-visualization"
FIGURES_PATH = rfc.MODEL_ROOT / "07-figures"
FHS_RESULTS_PATH = rfc.FHS_RESULTS_PATH

FHS_UPLOAD_DATA_PATH = UPLOAD_DATA_PATH / "fhs_upload_folders"


measure_map = rfc.measure_map
full_measure_map = rfc.full_measure_map
metric_map = rfc.metric_map
cause_map = rfc.cause_map
ssp_scenarios = rfc.ssp_scenarios
aa_merge_variables = rfc.aa_merge_variables
fhs_draws = rfc.fhs_draws


hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

hierarchy_ds_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.nc'
hierarchy_ds = read_netcdf_with_integer_ids(hierarchy_ds_path)

aa_full_population_df_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population.parquet"
as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"


ds_coords = rfc.ds_coords
import pprint

# Your subset dictionary
keys_to_keep = ['cause', 'measure', 'metric', 'ref_ssp_scenario', 'ref_dah_scenario', 
                'alt_ssp_scenario', 'alt_dah_scenario', 'alt_hold_variable', 'location_id', 'location_name']



FINAL_UPLOAD_DATA_PATH = '/mnt/team/idd/pub/forecast-mbp/05-upload_data'
as_upload_folder_path_template = "{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}{dah_text}{hold_text}"
aa_upload_folder_path_template = "{FINAL_UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}{dah_text}{hold_text}"

def get_aa_path(cause, measure, metric, ssp_scenario, dah_scenario = 'dah_2023', hold_variable = 'None', run_date = '2025_08_28'):
    if hold_variable is None:
        hold_text = ''
    else:
        hold_text = f'_hold_{hold_variable}'
    if cause == 'dengue':
        dah_text = ''
    else:
        dah_text = f'_dah_scenario_{dah_scenario}'
    as_upload_folder_path = as_upload_folder_path_template.format(FINAL_UPLOAD_DATA_PATH=FINAL_UPLOAD_DATA_PATH, run_date=run_date, 
                                                                           cause=cause, measure=measure, metric=metric, ssp_scenario=ssp_scenario, dah_text=dah_text, hold_text=hold_text)
    aa_upload_folder_path = aa_upload_folder_path_template.format(FINAL_UPLOAD_DATA_PATH=FINAL_UPLOAD_DATA_PATH, run_date=run_date, 
                                                                           cause=cause, measure=measure, metric=metric, ssp_scenario=ssp_scenario, dah_text=dah_text, hold_text=hold_text)
    return as_upload_folder_path, aa_upload_folder_path

ds_path_template = '{folder_path}/{statistic}.nc'

def load_ref_and_alt_ds(cause, measure, metric, ref_ssp_scenario='ssp245', ref_dah_scenario='Baseline', alt_ssp_scenario=None, 
                        alt_dah_scenario=None, alt_hold_variable = None, year_start = 2022, year_end = 2100, run_date='2025_08_28'):
    if alt_hold_variable is None and alt_ssp_scenario is None and alt_dah_scenario is None:
        raise ValueError("At least one of alt_ssp_scenario, alt_dah_scenario, or alt_hold must be provided.")
    if alt_ssp_scenario is None:
        alt_ssp_scenario = ref_ssp_scenario
    if alt_dah_scenario is None:
        alt_dah_scenario = ref_dah_scenario
    
    _, aa_ref_ds_folder = get_aa_path(cause=cause, measure=measure, metric=metric,
                                           ssp_scenario=ref_ssp_scenario, dah_scenario=ref_dah_scenario,
                                           hold_variable=None, run_date=run_date)
    aa_ref_ds_path = ds_path_template.format(folder_path=aa_ref_ds_folder, statistic='draws')
    aa_ref_ds = read_netcdf_with_integer_ids(aa_ref_ds_path)


    _, aa_alt_ds_folder = get_aa_path(cause=cause, measure=measure, metric=metric,
                                           ssp_scenario=alt_ssp_scenario, dah_scenario=alt_dah_scenario,
                                           hold_variable=alt_hold_variable, run_date=run_date)
    aa_alt_ds_path = ds_path_template.format(folder_path=aa_alt_ds_folder, statistic='draws')
    aa_alt_ds = read_netcdf_with_integer_ids(aa_alt_ds_path)

    data_dict = {
        'cause': cause,
        'measure': measure,
        'metric': metric,
        'year_start': year_start,
        'year_end': year_end,
        'ref_ssp_scenario': ref_ssp_scenario,
        'ref_dah_scenario': ref_dah_scenario,
        'alt_ssp_scenario': alt_ssp_scenario,
        'alt_dah_scenario': alt_dah_scenario,
        'alt_hold_variable': alt_hold_variable,
        'ref_ds': aa_ref_ds,
        'alt_ds': aa_alt_ds
    }

    return data_dict

def calculate_dss_and_statistics(data_dict, location_id=1, val_var='val', draw_dim='draw_id', year_dim='year_id'):
    
    location_name = hierarchy_df.loc[hierarchy_df['location_id'] == location_id, 'location_name'].values[0]
    
    year_start = data_dict['year_start']
    year_end = data_dict['year_end']

    ref_ds = data_dict['ref_ds']
    alt_ds = data_dict['alt_ds']
    
    # Select location and extract value variables
    ref_ds = ref_ds.sel(location_id=location_id, year_id=range(year_start, year_end + 1))[val_var]
    alt_ds = alt_ds.sel(location_id=location_id, year_id=range(year_start, year_end + 1))[val_var]

    # Calculate differences using the data arrays
    diff_data = ref_ds - alt_ds
    rel_diff_data = diff_data / ref_ds
    cum_diff_data = diff_data.cumsum(dim=year_dim)
    rel_diff_cum_data = (ref_ds.cumsum(dim=year_dim) - alt_ds.cumsum(dim=year_dim)) / ref_ds.cumsum(dim=year_dim)
    rel_diff_cum_alt_data = (ref_ds.cumsum(dim=year_dim) - alt_ds.cumsum(dim=year_dim)) / alt_ds.cumsum(dim=year_dim)
    
    # Use the refactored function to calculate summary statistics
    ref_summary_ds = get_summary_ds_from_ds(ref_ds, dim=draw_dim)
    max_ref_value = ref_summary_ds['upper'].max().item()
    min_ref_value = ref_summary_ds['lower'].min().item()
    max_ref_value = max(abs(min_ref_value), abs(max_ref_value))
    
    alt_summary_ds = get_summary_ds_from_ds(alt_ds, dim=draw_dim)
    max_alt_value = alt_summary_ds['upper'].max().item()
    min_alt_value = alt_summary_ds['lower'].min().item()
    max_alt_value = max(abs(min_alt_value), abs(max_alt_value))

    diff_summary_ds = get_summary_ds_from_ds(diff_data, dim=draw_dim)
    max_diff_value = diff_summary_ds['upper'].max().item()
    min_diff_value = diff_summary_ds['lower'].min().item()
    max_diff_value = max(abs(min_diff_value), abs(max_diff_value))

    rel_diff_summary_ds = get_summary_ds_from_ds(rel_diff_data, dim=draw_dim)
    max_rel_diff_value = rel_diff_summary_ds['upper'].max().item()
    min_rel_diff_value = rel_diff_summary_ds['lower'].min().item()
    max_rel_diff_value = max(abs(min_rel_diff_value), abs(max_rel_diff_value))

    cum_diff_summary_ds = get_summary_ds_from_ds(cum_diff_data, dim=draw_dim)
    max_cum_diff_value = cum_diff_summary_ds['upper'].max().item()
    min_cum_diff_value = cum_diff_summary_ds['lower'].min().item()
    max_cum_diff_value = max(abs(min_cum_diff_value), abs(max_cum_diff_value))

    rel_diff_cum_summary_ds = get_summary_ds_from_ds(rel_diff_cum_data, dim=draw_dim)
    max_rel_diff_cum_value = rel_diff_cum_summary_ds['upper'].max().item()
    min_rel_diff_cum_value = rel_diff_cum_summary_ds['lower'].min().item()
    max_rel_diff_cum_value = max(abs(min_rel_diff_cum_value), abs(max_rel_diff_cum_value))

    rel_diff_cum_alt_summary_ds = get_summary_ds_from_ds(rel_diff_cum_alt_data, dim=draw_dim)
    max_rel_diff_cum_alt_value = rel_diff_cum_alt_summary_ds['upper'].max().item()
    min_rel_diff_cum_alt_value = rel_diff_cum_alt_summary_ds['lower'].min().item()
    max_rel_diff_cum_alt_value = max(abs(min_rel_diff_cum_alt_value), abs(max_rel_diff_cum_alt_value))

    plot_dict = {
        'cause': data_dict['cause'],  # Fixed: get from data_dict
        'measure': data_dict['measure'],
        'metric': data_dict['metric'],
        'ref_ssp_scenario': data_dict['ref_ssp_scenario'],
        'ref_dah_scenario': data_dict['ref_dah_scenario'],
        'alt_ssp_scenario': data_dict['alt_ssp_scenario'],
        'alt_dah_scenario': data_dict['alt_dah_scenario'],
        'alt_hold_variable': data_dict['alt_hold_variable'],
        'location_id': location_id,
        'location_name': location_name,
        'ref_ds': ref_ds,
        'alt_ds': alt_ds,
        'diff_ds': diff_data,
        'rel_diff_ds': rel_diff_data,
        'cum_diff_ds': cum_diff_data,
        'rel_diff_cum_ds': rel_diff_cum_data,
        'rel_diff_cum_alt_ds': rel_diff_cum_alt_data,
        'baseline': ref_summary_ds,
        'constant': alt_summary_ds,
        'difference': diff_summary_ds,
        'cumulative_difference': cum_diff_summary_ds,
        'max_ref_value': max_ref_value,
        'max_alt_value': max_alt_value,
        'max_diff_value': max_diff_value,
        'max_rel_diff_value': max_rel_diff_value,
        'max_cum_diff_value': max_cum_diff_value,
        'max_rel_diff_cum_value': max_rel_diff_cum_value,
        'max_rel_diff_cum_alt_value': max_rel_diff_cum_alt_value
    }

    return plot_dict

def print_short_dict(dict):
    short_dict = {k: dict[k] for k in keys_to_keep if k in dict}
    # Pretty print it
    pprint.pprint(short_dict, width=80, indent=2)