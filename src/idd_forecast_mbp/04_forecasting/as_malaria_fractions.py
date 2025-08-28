"""
Code Summary:
Input: Forecast data for a specific cause (malaria), measure (mortality), SSP scenario, DAH scenario, and draw number
Output: Age-sex specific forecasts with predicted rates and counts saved as parquet file
Objective: Disaggregate all-age population forecasts into age-sex specific estimates using relative risk patterns from reference data
"""

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
import dask
import dask.dataframe as dd
import time
import psutil
import gc
import sys
import itertools
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.xarray_functions import convert_with_preset, write_netcdf, read_netcdf_with_integer_ids
import glob

# Memory and time tracking function
def log_time_and_memory(message):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
    print(f"{message}: Memory usage: {memory_gb:.2f} GB")
    return time.time()

# Start tracking memory usage
start_time = log_time_and_memory("Starting script")

import argparse

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=False, default='Baseline', help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")
parser.add_argument("--hold_variable", type=str, required=True, default='None', help="Hold variable for the forecast data")

# Parse arguments
args = parser.parse_args()
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
draw = args.draw
hold_variable = args.hold_variable

# ssp_scenario = "ssp126"
# dah_scenario = "Baseline"
# draw = "001"



# 1.2 Constants and Reference Values
# Purpose: Load configuration mappings and extract reference values
# Inputs: Constants from rfc module
# Creates: Reference age/sex IDs, short names, base column names
# Output: Analysis configuration variables
ssp_scenarios = rfc.ssp_scenarios
dah_scenarios = rfc.dah_scenarios
measure_map = rfc.measure_map
cause_map = rfc.cause_map
modeling_measure_map = rfc.modeling_measure_map

cause = "malaria"
reference_age_group_id = cause_map[cause]['reference_age_group_id']
reference_sex_id = cause_map[cause]['reference_sex_id']



# 1.3 Path Configuration
# Purpose: Set up data paths for different data sources
# Creates: Path variables for processed data, modeling data, forecasting data, GBD data
# Output: Standardized path constants
PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH
MODELING_DATA_PATH = rfc.MODELING_DATA_PATH
FORECASTING_DATA_PATH = rfc.FORECASTING_DATA_PATH
GBD_DATA_PATH = rfc.GBD_DATA_PATH


###----------------------------------------------------------###
### 2. File Path Generation
###----------------------------------------------------------###

# 2.1 Input/Output Path Logic
# Purpose: Generate file paths based on cause type (malaria vs others)
# Logic: Malaria includes DAH scenario in filename, others don't
# Creates: input_cause_draw_path, output_cause_draw_path
# Output: Cause-specific file paths

# input_cause_draw_path = f"{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.parquet"
if hold_variable == 'None':
    input_cause_draw_path = f"{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
    output_malaria_incidence_draw_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
    output_malaria_mortality_draw_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
else:
    input_cause_draw_path = f"{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
    output_malaria_incidence_draw_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
    output_malaria_mortality_draw_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"




# 2.2 Template Path Definitions
# Purpose: Define path templates for various data sources
# Creates: Templates for GBD data, processed data, hierarchy files
# Output: Path templates for data loading
aa_gbd_cause_df_path = f"{GBD_DATA_PATH}/gbd_2023_{cause}_aa.parquet"
as_gbd_cause_df_path = f"{GBD_DATA_PATH}/gbd_2023_{cause}_as.parquet"

as_full_cause_df_path = f'{PROCESSED_DATA_PATH}/as_full_{cause}_df.parquet'
aa_full_cause_df_path = f'{PROCESSED_DATA_PATH}/aa_full_{cause}_df.parquet'

as_md_gbd_malaria_df_path = f"{PROCESSED_DATA_PATH}/as_md_gbd_malaria_df.parquet"

as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"
aa_full_population_df_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population.parquet"

full_2023_hierarchy_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet"
age_sex_df_path = f'{PROCESSED_DATA_PATH}/age_sex_df.parquet'


###----------------------------------------------------------###
### 3. Reference Data Loading
###----------------------------------------------------------###

# Purpose: Load static reference datasets needed throughout analysis
# Inputs: Hierarchy file, age-sex mapping file
# Creates: hierarchy_df, age_sex_df
# Output: Reference DataFrames for merging and filtering

hierarchy_df = read_parquet_with_integer_ids(full_2023_hierarchy_path)
age_sex_df = read_parquet_with_integer_ids(age_sex_df_path)

aa_merge_variables = rfc.aa_merge_variables
as_merge_variables = rfc.as_merge_variables

print('Loaded hierarchy')

###----------------------------------------------------------###
### 4. Main Data Loading and Processing
###----------------------------------------------------------###

# 4.1 Forecast Data Loading
# Purpose: Load main forecast data and filter to relevant columns
# Inputs: Forecast parquet file for specific cause/scenario/draw
# Processing: Filter columns containing short name, add basic merge columns
# Creates: Main forecast DataFrame df
# Output: Filtered forecast data ready for processing
forecast_years = list(range(2022, 2101))
forecast_year_filter = ('year_id', 'in', forecast_years)

ds = read_netcdf_with_integer_ids(input_cause_draw_path)
ds = ds.sel(year_id=forecast_years)
df = ds.to_dataframe().reset_index()
# df = read_parquet_with_integer_ids(input_cause_draw_path,
#     filters=[forecast_year_filter])
modeling_location_ids = df['location_id'].unique().tolist()
modeling_location_filter = ('location_id', 'in', modeling_location_ids)

aa_pop_df = read_parquet_with_integer_ids(aa_full_population_df_path,
                                         filters=[modeling_location_filter, forecast_year_filter])
aa_pop_df = aa_pop_df.rename(columns={'population': 'aa_population'})
df = df.merge(aa_pop_df[['location_id', 'year_id', 'aa_population']],
              on=['location_id', 'year_id'], how='left')

df['aa_malaria_mort_rate'] = np.exp(df['log_aa_malaria_mort_rate_pred'])
df['aa_malaria_inc_rate'] = np.exp(df['log_aa_malaria_inc_rate_pred'])
df = df.drop(columns=['log_aa_malaria_mort_rate_pred', 'log_aa_malaria_inc_rate_pred'])
for col in df.columns:
    if 'malaria' in col:
        df[col] = df[col].fillna(0)

df['aa_malaria_mort_count'] = df['aa_malaria_mort_rate'] * df['aa_population']
df['aa_malaria_inc_count'] = df['aa_malaria_inc_rate'] * df['aa_population']


md_gbd_location_df = hierarchy_df[hierarchy_df['most_detailed_gbd'] == True].copy()

last_year = min(forecast_years)


md_gbd_location_filter = ('location_id', 'in', md_gbd_location_df['location_id'].unique().tolist())

last_year_filter = ('year_id', '==', last_year)

as_md_gbd_malaria_df = read_parquet_with_integer_ids(as_md_gbd_malaria_df_path)
as_md_gbd_malaria_df = as_md_gbd_malaria_df.rename(columns={'location_id': 'gbd_location_id'})
as_md_gbd_malaria_df = as_md_gbd_malaria_df[(as_md_gbd_malaria_df['rr_inc_as'] > 0) | (as_md_gbd_malaria_df['rr_mort_as'] > 0)].copy()
gbd_location_ids = as_md_gbd_malaria_df['gbd_location_id'].unique().tolist()

level_5_location_ids = hierarchy_df[(hierarchy_df['level'] == 5) & (hierarchy_df['gbd_location_id'].isin(gbd_location_ids))]['location_id'].unique().tolist()
level_5_location_filter = ('location_id', 'in', level_5_location_ids)

forecast_columns_to_read = as_merge_variables + ['population']
forecast_df = read_parquet_with_integer_ids(as_full_population_df_path,
                                            columns = forecast_columns_to_read,
                                            filters = [level_5_location_filter, forecast_year_filter])



forecast_df = forecast_df.merge(df[['location_id', 'year_id', 'aa_malaria_mort_count', 'aa_malaria_inc_count']],
                                on=['location_id', 'year_id'], how='left')


# Clean up to free memory
del df
gc.collect()


forecast_df = forecast_df.merge(hierarchy_df[['location_id', 'gbd_location_id']],
                                on='location_id', how='left')


forecast_df = forecast_df.merge(as_md_gbd_malaria_df[['gbd_location_id', 'age_group_id', 'sex_id', 'rr_inc_as', 'rr_mort_as']],
                                on=['gbd_location_id', 'age_group_id', 'sex_id'], how='left')


# Clean up to free memory
del as_md_gbd_malaria_df
gc.collect()


forecast_df['rr_inc_as_pop'] = forecast_df['rr_inc_as'] * forecast_df['population']
forecast_df['rr_mort_as_pop'] = forecast_df['rr_mort_as'] * forecast_df['population']


forecast_df['sum_rr_inc_as_pop'] = forecast_df.groupby(['location_id', 'year_id'])['rr_inc_as_pop'].transform('sum')
forecast_df['sum_rr_mort_as_pop'] = forecast_df.groupby(['location_id', 'year_id'])['rr_mort_as_pop'].transform('sum')


forecast_df['inc_fraction'] = forecast_df['rr_inc_as_pop'] / forecast_df['sum_rr_inc_as_pop']
forecast_df['mort_fraction'] = forecast_df['rr_mort_as_pop'] / forecast_df['sum_rr_mort_as_pop']


forecast_df.drop(columns=['rr_inc_as_pop', 'rr_mort_as_pop', 'sum_rr_inc_as_pop', 'sum_rr_mort_as_pop', 'rr_inc_as', 'rr_mort_as','population'], inplace=True)


forecast_df['malaria_inc_count_pred'] = forecast_df['inc_fraction'] * forecast_df['aa_malaria_inc_count']
forecast_df['malaria_mort_count_pred'] = forecast_df['mort_fraction'] * forecast_df['aa_malaria_mort_count']
# Set any row with age_group_id 2 to zero for incidence and mortality counts
forecast_df.loc[forecast_df['age_group_id'] == 2, 'malaria_inc_count_pred'] = 0
forecast_df.loc[forecast_df['age_group_id'] == 2, 'malaria_mort_count_pred'] = 0



# Drop fraction columns as they are no longer needed

forecast_df.drop(columns=['inc_fraction', 'mort_fraction'], inplace=True)
#

non_measure_columns = [col for col in forecast_df.columns if 'inc' not in col and 'mort' not in col]
incidence_columns = [col for col in forecast_df.columns if 'inc' in col]
mortality_columns = [col for col in forecast_df.columns if 'mort' in col]


# Save output files
incidence_df = forecast_df[non_measure_columns + incidence_columns]
incidence_ds = convert_with_preset(incidence_df, preset='as_variables')
write_netcdf(incidence_ds, output_malaria_incidence_draw_path)


mortality_df = forecast_df[non_measure_columns + mortality_columns]
mortality_ds = convert_with_preset(mortality_df, preset='as_variables')
write_netcdf(mortality_ds, output_malaria_mortality_draw_path)



# Final cleanup
del forecast_df
gc.collect()
