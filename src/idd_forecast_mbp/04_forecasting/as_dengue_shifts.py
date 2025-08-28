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
import sys
import itertools
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.xarray_functions import convert_with_preset, write_netcdf, read_netcdf_with_integer_ids
import glob

import argparse

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

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

cause = 'dengue'
modeling_measure = 'incidence'

short = modeling_measure_map[cause][modeling_measure]["short"]
base_col = f"base_{short}"

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

include_no_vaccinate = False

if hold_variable == 'None':
    input_cause_draw_path = f"{FORECASTING_DATA_PATH}/raked_{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
    output_dengue_incidence_draw_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
    output_dengue_mortality_draw_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
    output_dengue_incidence_draw_path_no_vaccinate = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions.nc"
    output_dengue_mortality_draw_path_no_vaccinate = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions.nc"
else:
    input_cause_draw_path = f"{FORECASTING_DATA_PATH}/raked_{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
    output_dengue_incidence_draw_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
    output_dengue_mortality_draw_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
    output_dengue_incidence_draw_path_no_vaccinate = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
    output_dengue_mortality_draw_path_no_vaccinate = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_no_vaccinate_draw_{draw}_with_predictions_hold_{hold_variable}.nc"

###----------------------------------------------------------###
### 2. File Path Generation
###----------------------------------------------------------###

# 2.1 Input/Output Path Logic
# Purpose: Generate file paths based on cause type (malaria vs others)
# Logic: Malaria includes DAH scenario in filename, others don't
# Creates: input_cause_draw_path, output_cause_draw_path
# Output: Cause-specific file paths


# 2.2 Template Path Definitions
# Purpose: Define path templates for various data sources
# Creates: Templates for GBD data, processed data, hierarchy files
# Output: Path templates for data loading
aa_gbd_cause_df_path_template = "{GBD_DATA_PATH}/gbd_2023_{cause}_aa.parquet"
as_gbd_cause_df_path_template = "{GBD_DATA_PATH}/gbd_2023_{cause}_as.parquet"

as_full_cause_df_path_template = '{PROCESSED_DATA_PATH}/as_full_{cause}_df.parquet'
aa_full_cause_df_path_template = '{PROCESSED_DATA_PATH}/aa_full_{cause}_df.parquet'
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

###----------------------------------------------------------###
### 4. Main Data Loading and Processing
###----------------------------------------------------------###

# 4.1 Forecast Data Loading
# Purpose: Load main forecast data and filter to relevant columns
# Inputs: Forecast parquet file for specific cause/scenario/draw
# Processing: Filter columns containing short name, add basic merge columns
# Creates: Main forecast DataFrame df
# Output: Filtered forecast data ready for processing
columns_to_read = as_merge_variables + ['logit_dengue_cfr_pred', 'base_log_dengue_inc_rate_pred']
future_year_ids = list(range(2022, 2101))
year_filter = ('year_id', 'in', future_year_ids)

ds = read_netcdf_with_integer_ids(input_cause_draw_path)
ds = ds.sel(year_id = future_year_ids)
df = ds.to_dataframe().reset_index()
df = df[columns_to_read].copy()

df['dengue_cfr_pred'] = 1 / (1 + np.exp(-df['logit_dengue_cfr_pred']))
df = df.drop(columns=['logit_dengue_cfr_pred'])

# df = df.merge(hierarchy_df[['location_id', 'fhs_location_id']], on='location_id', how='left').copy()

# 4.2 Data Preparation and Filtering
# Purpose: Prepare data for GBD data loading by creating filters
# Processing: Extract location IDs, year IDs, create parquet filters
# Creates: Location filters, year filters, measure/metric filters
# Output: Filter tuples for efficient parquet reading
df_location_ids = df['location_id'].unique().tolist()
df_fhs_ids = hierarchy_df[hierarchy_df['most_detailed_fhs'] == True]['location_id'].unique().tolist()
df_location_filter = ('location_id', 'in', df_location_ids)

measure_id_value = modeling_measure_map[cause][modeling_measure]['gbd_measure_id']
measure_id_filter = ('measure_id', 'in', [measure_id_value] if isinstance(measure_id_value, int) else measure_id_value)

metric_id_value = modeling_measure_map[cause][modeling_measure]['gbd_metric_id']
metric_id_filter = ('metric_id', 'in', [metric_id_value] if isinstance(metric_id_value, int) else metric_id_value)

# 4.3 Population Data Loading
# Purpose: Load age-sex population data with filters
# Inputs: Age-sex population parquet file
# Creates: as_population_df with filtered population data
# Output: Population DataFrame for final calculations
pop_cols_to_read = as_merge_variables + ['population']
as_population_df = read_parquet_with_integer_ids(as_full_population_df_path,
                                                 columns = pop_cols_to_read,
                                                 filters = [df_location_filter, year_filter])
as_population_df = as_population_df.merge(hierarchy_df[['location_id', 'gbd_location_id']], on='location_id', how='left')

forecast_df = as_population_df.merge(df, on=as_merge_variables, how='left').copy()

# 6.3 Data Consolidation
# Purpose: Combine all batch results into single DataFrame
# Creates: Complete as_fhs_df with all reference data
# Output: Consolidated age-sex reference data
as_md_gbd_dengue_df_path = f"{PROCESSED_DATA_PATH}/as_md_gbd_dengue_df.parquet"
columns_to_read = ['location_id', 'sex_id', 'age_group_id', 'rr_inc_as']
as_md_gbd_dengue_df = read_parquet_with_integer_ids(as_md_gbd_dengue_df_path,
                                                    columns = columns_to_read).rename(columns={
    'location_id': 'gbd_location_id'
})

forecast_df = forecast_df.merge(as_md_gbd_dengue_df, on=['gbd_location_id', 'age_group_id', 'sex_id'], how='left')

# Replace all NaN in any column of forecast_df with 0
forecast_df['base_log_dengue_inc_rate_pred'] = forecast_df['base_log_dengue_inc_rate_pred'].fillna(0)
forecast_df['dengue_cfr_pred'] = forecast_df['dengue_cfr_pred'].fillna(0)

forecast_df['dengue_inc_count_pred'] = forecast_df['population'] * np.exp(forecast_df['base_log_dengue_inc_rate_pred']) * forecast_df['rr_inc_as']
forecast_df['dengue_mort_count_pred'] = forecast_df['dengue_inc_count_pred'] * forecast_df['dengue_cfr_pred']

keep_columns = as_merge_variables + ['population', 'dengue_inc_count_pred', 'dengue_mort_count_pred']
forecast_df = forecast_df[keep_columns]


##################################################################
##################################################################
##### Save, vaccinate, and save again
##################################################################
################################################################## 
non_measure_columns = [col for col in forecast_df.columns if 'inc' not in col and 'mort' not in col]
incidence_columns = [col for col in forecast_df.columns if 'inc' in col]
mortality_columns = [col for col in forecast_df.columns if 'mort' in col]

if include_no_vaccinate:
    incidence_df = forecast_df[non_measure_columns + incidence_columns]
    incidence_ds = convert_with_preset(incidence_df, preset='as_variables')
    write_netcdf(incidence_ds, output_dengue_incidence_draw_path)

    mortality_df = forecast_df[non_measure_columns + mortality_columns]
    mortality_ds = convert_with_preset(mortality_df, preset='as_variables')
    write_netcdf(mortality_ds, output_dengue_mortality_draw_path)

##### Vaccination

locations = ['Singapore', 'Brazil', 'Indonesia', 'Thailand']
location_ids = hierarchy_df[hierarchy_df['location_name'].isin(locations)]['location_id'].unique()
children_ids = hierarchy_df[hierarchy_df['parent_id'].isin(location_ids)]['location_id'].unique()
grand_children_ids = hierarchy_df[hierarchy_df['parent_id'].isin(children_ids)]['location_id'].unique()

dengue_vaccine_df_path = f"{FORECASTING_DATA_PATH}/dengue_vaccine_df.parquet"
dengue_vaccine_df = read_parquet_with_integer_ids(dengue_vaccine_df_path)

locations = ['Singapore', 'Brazil', 'Indonesia', 'Thailand']
location_ids = hierarchy_df[hierarchy_df['location_name'].isin(locations)]['location_id'].unique()
children_ids = hierarchy_df[hierarchy_df['parent_id'].isin(location_ids)]['location_id'].unique()
grand_children_ids = hierarchy_df[hierarchy_df['parent_id'].isin(children_ids)]['location_id'].unique()

dengue_vaccine_df_path = f"{FORECASTING_DATA_PATH}/dengue_vaccine_df.parquet"
dengue_vaccine_df = read_parquet_with_integer_ids(dengue_vaccine_df_path)

# Create vaccination lookup table
vaccine_lookup = dengue_vaccine_df.set_index('age_group_id')

# Filter to vaccination locations and years
vaccine_mask = (
    forecast_df['location_id'].isin(grand_children_ids) &
    (forecast_df['year_id'] >= 2023) &
    (forecast_df['year_id'] <= 2100)
)

# Apply vaccination effects using vectorized operations
for year in range(2023, 2101):
    year_col = f'year_{year}'
    if year_col in vaccine_lookup.columns:
        # Create mask for this specific year
        year_mask = vaccine_mask & (forecast_df['year_id'] == year)
        
        if year_mask.any():
            # Get reduction values for all age groups in this year
            age_group_reductions = vaccine_lookup[year_col]
            
            # Map reductions to the mortality forecasts
            reductions = forecast_df.loc[year_mask, 'age_group_id'].map(age_group_reductions)
            
            # Apply reductions (fill NaN with 1.0 for no reduction)
            reductions = reductions.fillna(1.0)
            
            # Apply vaccine effectiveness
            forecast_df.loc[year_mask, 'dengue_mort_count_pred'] *= reductions


mortality_df = forecast_df[non_measure_columns + mortality_columns]
mortality_ds = convert_with_preset(mortality_df, preset='as_variables')
write_netcdf(mortality_ds, output_dengue_mortality_draw_path)

incidence_df = forecast_df[non_measure_columns + incidence_columns]
incidence_ds = convert_with_preset(incidence_df, preset='as_variables')
write_netcdf(incidence_ds, output_dengue_incidence_draw_path)
