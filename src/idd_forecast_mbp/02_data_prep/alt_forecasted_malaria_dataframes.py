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
parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating dengue")

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

# New DAH data
new_dah_scenarios = {
    'reference': {
        'name': 'reference',
        'path': f'{PROCESSED_DATA_PATH}/dah_reference_df.parquet'
    },
    'better': {
        'name': 'better',
        'path': f'{PROCESSED_DATA_PATH}/dah_better_df.parquet'
    },
    'worse': {
        'name': 'worse',
        'path': f'{PROCESSED_DATA_PATH}/dah_worse_df.parquet'
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
    
    # Read the new DAH scenario data
    dah_df = read_parquet_with_integer_ids(dah_scenario['path'],
        columns=dah_columns_to_keep)
    
    # Merge with the existing DAH scenario data
    dah_scenario_df = base_dah_scenario_df.merge(dah_df, on=['location_id', 'year_id'], how='left')
    
    # Add the new DAH column
    dah_scenario_df['mal_DAH_total_per_capita'] = dah_scenario_df['mal_DAH_total_per_capita'].fillna(0)
    
    # Write the output to a new parquet file
    dah_scenario_df_path = dah_scenario_df_path_template.format(
        FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
        ssp_scenario=ssp_scenario,
        dah_scenario_name=dah_scenario_name,
        draw=draw
    )

    write_parquet(dah_scenario_df, dah_scenario_df_path)