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
import glob

import argparse

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--modeling_measure", type=str, required=False, default="incidence", help="Measure (e.g., 'mortality', 'incidence', or 'cfr')")
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=False, default='Baseline', help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")

# Parse arguments
args = parser.parse_args()

modeling_measure = args.modeling_measure
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
draw = args.draw

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

short = modeling_measure_map[cause][modeling_measure]["short"]
base_col = f"base_{short}"

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

input_cause_draw_path = f"{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.parquet"
output_cause_draw_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{modeling_measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.parquet"


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
df = read_parquet_with_integer_ids(input_cause_draw_path)

df_columns = [col for col in df.columns if short in col]
df_columns = ['location_id', 'year_id', 'population', 'year_to_rake_to'] + df_columns

df = df[df_columns].copy()
df['year_to_rake_to'] = df['year_to_rake_to'].astype(int)
df = df.merge(hierarchy_df[['location_id', 'fhs_location_id']], on='location_id', how='left')
df = df[df['year_id'] >= df['year_to_rake_to'].min()].copy()

# Extract column names for predictions and observations
pred_raw_col = [col for col in df.columns if "pred_raw" in col][0]
obs_col = pred_raw_col.replace("_pred_raw", "")
pred_col = pred_raw_col.replace("_pred_raw", "_pred")

# 4.2 Data Preparation and Filtering
# Purpose: Prepare data for GBD data loading by creating filters
# Processing: Extract location IDs, year IDs, create parquet filters
# Creates: Location filters, year filters, measure/metric filters
# Output: Filter tuples for efficient parquet reading
df_location_ids = df['location_id'].unique().tolist()
df_fhs_ids = hierarchy_df[hierarchy_df['most_detailed_fhs'] == True]['location_id'].unique().tolist()
df_year_ids = df['year_id'].unique().tolist()
df_location_filter = ('location_id', 'in', df_location_ids)

df_year_filter = ('year_id', 'in', df_year_ids)
measure_id_value = modeling_measure_map[cause][modeling_measure]['gbd_measure_id']
measure_id_filter = ('measure_id', 'in', [measure_id_value] if isinstance(measure_id_value, int) else measure_id_value)

metric_id_value = modeling_measure_map[cause][modeling_measure]['gbd_metric_id']
metric_id_filter = ('metric_id', 'in', [metric_id_value] if isinstance(metric_id_value, int) else metric_id_value)

# 4.3 Population Data Loading
# Purpose: Load age-sex population data with filters
# Inputs: Age-sex population parquet file
# Creates: as_population_df with filtered population data
# Output: Population DataFrame for final calculations
as_population_df = read_parquet_with_integer_ids(as_full_population_df_path,
                                               filters = [df_location_filter, df_year_filter])
as_population_df = as_population_df.merge(hierarchy_df[['location_id', 'fhs_location_id']], on='location_id', how='left')
###----------------------------------------------------------###
### 5. GBD Reference Data Processing
###----------------------------------------------------------###

# 5.1 Year-Location Grouping
# Purpose: Group locations by year_to_rake_to for efficient data loading
# Processing: Create sub-DataFrames mapping FHS locations to rake years
# Creates: sub_dfs list containing location-year mappings
# Output: Organized location groups for batch processing
sub_dfs = []
for year_to_rake_to in df['year_to_rake_to'].unique():
    temp_df = df[df['year_to_rake_to'] == year_to_rake_to]['location_id'].unique()
    sub_fhs_location_ids = hierarchy_df[hierarchy_df['location_id'].isin(temp_df)]['fhs_location_id'].unique()
    sub_df = pd.DataFrame({
        'fhs_location_id': sub_fhs_location_ids,
        'year_to_rake_to': year_to_rake_to
    })
    sub_dfs.append(sub_df)

###----------------------------------------------------------###
### 6. Age-Sex GBD Data Loading and Processing
###----------------------------------------------------------###

# 6.1 Batch Data Loading Loop
# Purpose: Load GBD cause data in batches by year_to_rake_to
# Inputs: Age-sex GBD cause parquet files
# Processing: Filter by location, year, measure, and metric IDs
# Creates: Individual as_fhs_df for each batch
# Output: Filtered GBD data for specific year-location combinations
as_columns_to_read = ['location_id', 'sex_id', 'age_group_id', 'measure_id', 'metric_id', 'val', 'population']
as_fhs_dfs = []
for i, sub_df in enumerate(sub_dfs):
    sub_fhs_location_ids = sub_df['fhs_location_id'].unique()
    year_to_rake_to = sub_df['year_to_rake_to'].unique()[0]
    sex_ids = [1,2]
    year_to_rake_to_filter = ('year_id', '=', year_to_rake_to)
    sex_filter = ('sex_id', 'in', sex_ids)
    print(f"Processing subset {i+1} of {len(sub_dfs)}")
    fhs_location_filter = ('location_id', 'in', sub_fhs_location_ids.tolist())
    as_fhs_df = read_parquet_with_integer_ids(
        as_gbd_cause_df_path_template.format(
            GBD_DATA_PATH=GBD_DATA_PATH,
            cause=cause
        ),
        filters=[fhs_location_filter, year_to_rake_to_filter, measure_id_filter, metric_id_filter,sex_filter],
        columns=as_columns_to_read
    )

    as_fhs_df = as_fhs_df.rename(columns={'val': short}).drop(columns=['measure_id', 'metric_id']).copy()
    as_fhs_df['year_to_rake_to'] = year_to_rake_to
    as_fhs_dfs.append(as_fhs_df)

# 6.3 Data Consolidation
# Purpose: Combine all batch results into single DataFrame
# Creates: Complete as_fhs_df with all reference data
# Output: Consolidated age-sex reference data
as_fhs_df = pd.concat(as_fhs_dfs, ignore_index=True)
as_fhs_df = as_fhs_df.merge(hierarchy_df[['location_id', 'fhs_location_id']], on='location_id', how='left')

###----------------------------------------------------------###
### 7. Relative Risk Calculation
###----------------------------------------------------------###

# 7.1 Reference Rate Extraction
# Purpose: Extract baseline rates for reference age-sex group
# Processing: Filter to reference age/sex, rename columns
# Creates: base_fhs_df with reference rates
# Output: Baseline rates for relative risk calculation
base_fhs_df = as_fhs_df[(as_fhs_df['age_group_id'] == reference_age_group_id) & (as_fhs_df['sex_id'] == reference_sex_id)].drop(columns=['population', 'year_to_rake_to']).copy()
base_fhs_df = base_fhs_df.rename(columns={
    short: base_col,
    'sex_id': 'base_sex_id',
    'age_group_id': 'base_age_group_id'})

# 7.2 Relative Risk Computation
# Purpose: Calculate relative risks and prepare for merging
# Processing: Merge baseline rates, calculate ratios, rename columns
# Creates: Relative risk values for each age-sex-location combination
# Output: as_fhs_df with relative risks ready for forecasting
as_fhs_df = as_fhs_df.merge(base_fhs_df, on=['location_id'], how='left')

as_fhs_df['relative_risk'] = as_fhs_df[short] / as_fhs_df[base_col]
as_fhs_df.loc[as_fhs_df[short] == 0, 'relative_risk'] = 0.0
as_fhs_df = as_fhs_df.rename(columns={
    'location_id': 'fhs_location_id',
    short: 'fhs_' + short,
    base_col: 'fhs_' + base_col})

as_fhs_df = as_fhs_df.drop(columns=['population', 'base_sex_id', 'base_age_group_id', 'year_to_rake_to'])

###----------------------------------------------------------###
### 8. Final Forecasting Calculations
###----------------------------------------------------------###

# 8.1 Data Merging
# Purpose: Combine population, forecast, and relative risk data
# Processing: Sequential merges on location-year and location-age-sex
# Creates: forecasting_df with all necessary components
# Output: Complete dataset for final calculations
forecasting_df = as_population_df.merge(df[['location_id', 'year_id', 'year_to_rake_to', pred_col, obs_col]], on=['location_id', 'year_id'], how='left').copy()

forecasting_df = forecasting_df.merge(as_fhs_df, on=['fhs_location_id', 'age_group_id', 'sex_id'], how='left')

# 8.2 Transformation and Rate Calculation
# Purpose: Apply inverse transformations and calculate final rates
# Processing: Exponential , multiply by relative risks
# Creates: Base rates and age-sex-specific predicted rates
# Output: Final forecasted rates by age-sex-location-year
forecasting_df['base_' + short] = np.exp(forecasting_df[pred_col])


# 8.3 Special Cases and Count Calculation
# Purpose: Handle special cases and calculate counts from rates
# Processing: Set age group 2 to zero, multiply rates by population
# Creates: Final rate and count predictions
# Output: Complete forecasted dataset
# Hard code age_group_id 2 to have no dengue or malaria
forecasting_df.loc[forecasting_df["age_group_id"] == 2, "relative_risk"] = 0.0
forecasting_df[short + '_pred'] = forecasting_df['base_' + short] * forecasting_df['relative_risk']

forecasting_df = forecasting_df[as_merge_variables + ['population' , short + '_pred']].copy()


count_name = modeling_measure_map[cause][modeling_measure]['count_name']
forecasting_df[count_name + '_pred'] = forecasting_df[short + '_pred'] * forecasting_df['population']

###----------------------------------------------------------###
### 9. Output Generation
###----------------------------------------------------------###

# Purpose: Filter to final columns and save results
# Processing: Select final output columns, calculate counts if needed
# Output: Write final parquet file with age-sex-specific forecasts
write_parquet(forecasting_df, output_cause_draw_path)