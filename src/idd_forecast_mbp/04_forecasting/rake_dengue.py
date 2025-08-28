###----------------------------------------------------------###
### 1. Setup and Configuration
###----------------------------------------------------------###

# 1.1 Import and Argument Parsing
# Purpose: Import libraries and parse command-line arguments
# Inputs: Command-line arguments (cause, modeling_measure, ssp_scenario, dah_scenario, draw)
# Creates: Parsed variables for analysis parameters
# Output: Configured analysis parameters
import pandas as pd
import numpy as np
import os
import sys
import itertools
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.xarray_functions import read_netcdf_with_integer_ids, write_netcdf, convert_with_preset
import glob

import argparse

parser = argparse.ArgumentParser(description="Rake base dengue and complete the cfr regression predictions")

# Define arguments
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")
parser.add_argument("--hold_variable", type=str, required=True, default='None', help="Hold variable (e.g., 'gdppc', 'suitability', 'urban') or None for primary task")


# Parse arguments
args = parser.parse_args()

ssp_scenario = args.ssp_scenario
draw = args.draw
hold_variable = args.hold_variable

# For testing purposes, you can uncomment and modify the following lines:
# ssp_scenario = 'ssp126'
# draw = '001'

ssp_scenarios = rfc.ssp_scenarios
dah_scenarios = rfc.dah_scenarios
measure_map = rfc.measure_map
cause_map = rfc.cause_map
modeling_measure_map = rfc.modeling_measure_map

cause = 'dengue'
modeling_measure = 'incidence'

reference_age_group_id = cause_map[cause]['reference_age_group_id']
reference_sex_id = cause_map[cause]['reference_sex_id']

# 1.3 Path Configuration
PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH
MODELING_DATA_PATH = rfc.MODELING_DATA_PATH
FORECASTING_DATA_PATH = rfc.FORECASTING_DATA_PATH
GBD_DATA_PATH = rfc.GBD_DATA_PATH

aa_gbd_cause_df_path_template = "{GBD_DATA_PATH}/gbd_2023_{cause}_aa.parquet"
as_gbd_cause_df_path_template = "{GBD_DATA_PATH}/gbd_2023_{cause}_as.parquet"

as_full_cause_df_path_template = '{PROCESSED_DATA_PATH}/as_full_{cause}_df.parquet'
aa_full_cause_df_path_template = '{PROCESSED_DATA_PATH}/aa_full_{cause}_df.parquet'
as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"
aa_full_population_df_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population.parquet"

full_2023_hierarchy_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet"
hierarchy_df = read_parquet_with_integer_ids(full_2023_hierarchy_path)

age_sex_df_path = f'{PROCESSED_DATA_PATH}/age_sex_df.parquet'
age_sex_df = read_parquet_with_integer_ids(age_sex_df_path)

aa_merge_variables = rfc.aa_merge_variables
as_merge_variables = rfc.as_merge_variables


###----------------------------------------------------------###
### 
###----------------------------------------------------------###

# 2.1 Input/Output Path Logic
if hold_variable == 'None':
    input_cause_draw_path = f"{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet"
    output_cause_draw_path = f"{FORECASTING_DATA_PATH}/raked_{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
else:
    input_cause_draw_path = f"{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
    output_cause_draw_path = f"{FORECASTING_DATA_PATH}/raked_{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"


if hold_variable == 'None':
    reference_df = read_parquet_with_integer_ids(input_cause_draw_path)
else:
    ds = read_netcdf_with_integer_ids(input_cause_draw_path)
    reference_df = ds.to_dataframe().reset_index()

suit_col = [col for col in reference_df.columns if 'suit' in col]
reference_df = reference_df.drop(columns=suit_col + ['as_id', 'age_group_id', 'sex_id', 'log_gdppc_mean', 'logit_urban_1km_threshold_300', 'A0_af', 'logit_dengue_cfr'], errors='ignore')
level_5_location_ids = reference_df['location_id'].unique()

tmp_df = reference_df[reference_df['year_id'] == 2022].copy()
tmp_df['shift'] = tmp_df['base_log_dengue_inc_rate'] - tmp_df['base_log_dengue_inc_rate_pred_raw']
reference_df = reference_df.merge(tmp_df[['location_id', 'shift']], on=['location_id'], how='left')
reference_df['base_log_dengue_inc_rate_pred'] = reference_df['base_log_dengue_inc_rate_pred_raw'] + reference_df['shift']

df_location_filter = ('location_id', 'in', level_5_location_ids)
future_year_ids = list(range(2000, 2101))
year_filter = ('year_id', 'in', future_year_ids)

as_population_df = read_parquet_with_integer_ids(as_full_population_df_path,
                                                 columns = as_merge_variables,
                                                 filters = [df_location_filter, year_filter])

forecast_df = as_population_df.merge(reference_df[['location_id', 'year_id', 'base_log_dengue_inc_rate_pred', 'logit_dengue_cfr_pred_raw']], on=aa_merge_variables, how='left').copy()

# Regression variables
as_id_levels =  pd.read_csv(f'{MODELING_DATA_PATH}/as_id_levels.csv')
mod_cfr_all_coefficients = pd.read_csv(
    f'{MODELING_DATA_PATH}/mod_cfr_all_coefficients.csv',
    names=['variable', 'coefficient'],
    header=0  
)

# Reset index to get level_num as a column
as_id_levels_expanded = as_id_levels.reset_index()
as_id_levels_expanded = as_id_levels_expanded.rename(columns={'index': 'level_num', 'x': 'level'})

# Extract age_group_id and sex_id from the level column
as_id_levels_expanded['age_group_id'] = as_id_levels_expanded['level'].str.extract(r'a(\d+)_s\d+').astype(int)
as_id_levels_expanded['sex_id'] = as_id_levels_expanded['level'].str.extract(r'a\d+_s(\d+)').astype(int)

mod_cfr_coef = mod_cfr_all_coefficients[
    mod_cfr_all_coefficients['variable'].str.contains('as_id', na=False)
].copy()
mod_cfr_coef = mod_cfr_coef.reset_index(drop=True)
# Remove the 'as_id_' prefix from the variable names
mod_cfr_coef['level'] = mod_cfr_coef['variable'].str.replace('as_id', '', regex=False)

as_id_levels_expanded = as_id_levels_expanded.merge(mod_cfr_coef[['level', 'coefficient']], on='level', how='left')
as_id_levels_expanded['coefficient'] = as_id_levels_expanded['coefficient'].fillna(0)

reference_coef = as_id_levels_expanded.loc[
    (as_id_levels_expanded['age_group_id'] == reference_age_group_id) &
    (as_id_levels_expanded['sex_id'] == reference_sex_id), 'coefficient'
].iloc[0]

as_id_levels_expanded['reference_coefficient'] = reference_coef
as_id_levels_expanded['logit_cfr_shift'] = as_id_levels_expanded['coefficient'] - reference_coef

forecast_df = forecast_df.merge(as_id_levels_expanded[['age_group_id', 'sex_id', 'logit_cfr_shift']], on=['age_group_id', 'sex_id'], how='left')

forecast_df['logit_dengue_cfr_pred_raw'] = forecast_df['logit_dengue_cfr_pred_raw'] + forecast_df['logit_cfr_shift']
forecast_df = forecast_df.drop(columns=['logit_cfr_shift'], errors='ignore')
forecast_2022_df = forecast_df[forecast_df['year_id'] == 2022].copy()
forecast_2022_df = forecast_2022_df[as_merge_variables + ['logit_dengue_cfr_pred_raw']]

cfr_rake_year_filter = ('year_id', 'in', [2022])
as_full_cause_df_path = f'{PROCESSED_DATA_PATH}/as_full_{cause}_df.parquet'
as_md_dengue_modeling_df = read_parquet_with_integer_ids(as_full_cause_df_path,
                                                            filters=[cfr_rake_year_filter, df_location_filter])
as_md_dengue_modeling_df["dengue_cfr"] = as_md_dengue_modeling_df["dengue_mort_rate"] / as_md_dengue_modeling_df["dengue_inc_rate"]

covariates_to_logit_transform = ['dengue_cfr']
for col in covariates_to_logit_transform:
    clipped_values = as_md_dengue_modeling_df[col].clip(upper=0.99)
    print(f"Range of {col}: {as_md_dengue_modeling_df[col].min()} to {as_md_dengue_modeling_df[col].max()}")
    as_md_dengue_modeling_df[f"logit_{col}"] = np.log(clipped_values / (1 - clipped_values))

cfr_rake_df = as_md_dengue_modeling_df[as_merge_variables + ['logit_dengue_cfr']].copy()

forecast_2022_df = forecast_2022_df.merge(cfr_rake_df[['location_id', 'age_group_id', 'sex_id', 'logit_dengue_cfr']], on=['location_id', 'age_group_id', 'sex_id'], how='left')

forecast_2022_df['shift'] = forecast_2022_df['logit_dengue_cfr'] - forecast_2022_df['logit_dengue_cfr_pred_raw']
forecast_df = forecast_df.merge(forecast_2022_df[['location_id', 'age_group_id','sex_id', 'shift']], on=['location_id', 'age_group_id', 'sex_id'], how='left')

forecast_df['logit_dengue_cfr_pred'] = forecast_df['logit_dengue_cfr_pred_raw'] + forecast_df['shift']
forecast_df = forecast_df.drop(columns=['shift', 'logit_dengue_cfr_pred_raw'], errors='ignore')

forecast_ds = convert_with_preset(forecast_df, preset='as_variables')
write_netcdf(forecast_ds, output_cause_draw_path)