import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
from typing import Literal, NamedTuple
import itertools
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import merge_dataframes, read_income_paths, read_urban_paths, level_filter
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet


import argparse
parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--ssp_scenario", type=str, required=True, help="ssp scenario number (ssp16, ssp245, ssp585")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")

# Parse arguments
args = parser.parse_args()


ssp_scenario = args.ssp_scenario
draw = args.draw

# Hierarchy
hierarchy = "lsae_1209"
PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"

# # New DAH data
# new_dah_scenarios = {
#     'reference': {
#         'name': 'reference',
#         'path': f'{PROCESSED_DATA_PATH}/dah_reference_df.parquet'
#     },
#     'better': {
#         'name': 'better',
#         'path': f'{PROCESSED_DATA_PATH}/dah_better_df.parquet'
#     },
#     'worse': {
#         'name': 'worse',
#         'path': f'{PROCESSED_DATA_PATH}/dah_worse_df.parquet'
#     }
# }

# New DAH data
# Created using make_GK_dah_df.py
new_dah_scenarios = {
    'reference': {
        'name': 'GK_reference_2025_11_02',
        'path': f'{PROCESSED_DATA_PATH}/GK_dah_ref_df_2025_07_08.parquet'
    },
    'cut20': {
        'name': 'GK_cut20_2025_11_02',
        'path': f'{PROCESSED_DATA_PATH}/GK_dah_cut20_df_2025_07_08.parquet'
    }
}


base_dah_scenario_df_path_template = "{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_Baseline_draw_{draw}.parquet"
dah_scenario_df_path_template = "{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet"

columns_to_keep = ['location_id', 'year_id', 'people_flood_days_per_capita', 
    'gdppc_mean', 'log_gdppc_mean', 
    'logit_malaria_pfpr',
    'aa_malaria_mort_rate', 'aa_malaria_inc_rate',
    'base_malaria_mort_rate', 'base_malaria_inc_rate',
    'log_aa_malaria_mort_rate', 'log_aa_malaria_inc_rate',
    'log_base_malaria_mort_rate', 'log_base_malaria_inc_rate', 
    'malaria_suitability', 'year_to_rake_to', 'A0_af']

dah_columns_to_keep = ['location_id', 'year_id', 'mal_DAH_total_per_capita']


base_dah_scenario_df_path = base_dah_scenario_df_path_template.format(
    FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
    ssp_scenario=ssp_scenario,
    draw=draw
)
base_dah_scenario_df = read_parquet_with_integer_ids(base_dah_scenario_df_path,
    columns=columns_to_keep
)

for dah_scenario_name, dah_scenario in new_dah_scenarios.items():
    print(f"Processing DAH scenario: {dah_scenario_name}")
    
    # FIX: Create a fresh copy of the base data for the current scenario merge.
    current_scenario_df = base_dah_scenario_df.copy()
    
    # Read the new DAH scenario data
    dah_df = read_parquet_with_integer_ids(dah_scenario['path'],
        columns=dah_columns_to_keep)
    
    print("-" * 50)
    print(f"DIAGNOSTIC: {dah_scenario_name.upper()} Input Check")

    # Check 1: Does the input DAH file (dah_df) have non-zero data post-2025?
    # Assuming 2025 is the start year of divergence
    post_2025_dah = dah_df[dah_df['year_id'] >= 2025]['mal_DAH_total_per_capita']

    if not post_2025_dah.empty and post_2025_dah.abs().sum() > 1e-9:
        print(f"✅ Input DAH data is NOT zero post-2025. Max value: {post_2025_dah.max():.4f}")
    else:
        print("❌ Input DAH data is zero/missing post-2025. The problem is upstream.")
    print("-" * 50)

    current_scenario_df['A0_location_id'] = current_scenario_df['A0_af'].str.extract(r'A0_(\d+)')[0].astype(int)
    
    # Merge the unique DAH data into the fresh copy
    dah_scenario_df = current_scenario_df.merge(
        dah_df, 
        left_on=['A0_location_id', 'year_id'],
        right_on=['location_id', 'year_id'],
        how='left',
        suffixes=('', '_dah')
    )
    dah_scenario_df = dah_scenario_df.drop(columns=['location_id_dah', 'A0_location_id'])

    
    # Add the new DAH column (fillna is still correct)
    dah_scenario_df['mal_DAH_total_per_capita'] = dah_scenario_df['mal_DAH_total_per_capita'].fillna(0)
    # Write the output to a new parquet file
    dah_scenario_df_path = dah_scenario_df_path_template.format(
        FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
        ssp_scenario=ssp_scenario,
        dah_scenario_name=dah_scenario['name'],
        draw=draw
    )

    write_parquet(dah_scenario_df, dah_scenario_df_path)