"""
Code Summary:
Input: Forecast data for a specific cause (malaria), measure (mortality), SSP scenario, DAH scenario, and draw number
Output: Age-sex specific forecasts with predicted rates and counts saved as parquet file
Objective: Disaggregate all-age population forecasts into age-sex specific estimates using relative risk patterns from reference data
"""

###----------------------------------------------------------###
### 1. Setup and Configuration
###----------------------------------------------------------###

# 1.1 Import and Parameter Definition
# Imports necessary libraries and sets up analysis parameters (cause, measure, scenarios, draw number)
import pandas as pd
import numpy as np
import os
import sys
import itertools
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import write_parquet, read_parquet_with_integer_ids
import glob

import argparse

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--cause", type=str, required=False, default="malaria", help="Cause (e.g., 'malaria', 'dengue')")
parser.add_argument("--measure", type=str, required=False, default="mortality", help="Measure (e.g., 'mortality', 'incidence')")
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=True, help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")


# Parse arguments
args = parser.parse_args()

cause = args.cause
measure = args.measure
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
draw = args.draw

# cause = "malaria"
# measure = "mortality"
# ssp_scenario = "ssp245"
# dah_scenario = "Baseline"
# draw = "001"

hierarchy = "lsae_1209"

ssp_scenarios = rfc.ssp_scenarios
dah_scenarios = rfc.dah_scenarios
measure_map = rfc.measure_map

# 1.2 Path Configuration
# Establishes standardized paths for processed data, modeling data, and forecasting outputs using predefined constants
PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"

FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
age_metadata_path = f"{FHS_DATA_PATH}/age_metadata.parquet"

as_fhs_df_template = "{MODELING_DATA_PATH}/fhs_{cause}_{measure}_{metric}_modeling_df.parquet"
cause_draw_path_template = "{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}_with_predictions.parquet"
output_cause_draw_path_template = "{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}_with_predictions.parquet"

###----------------------------------------------------------###
### 2. Age Metadata Processing
### 2.1 Age-Sex Combination Generation
### Loads age group metadata and creates all possible age-sex combinations
### for disaggregation of all-age population values.
###----------------------------------------------------------###
age_metadata_df = read_parquet_with_integer_ids(age_metadata_path)
age_group_ids = age_metadata_df["age_group_id"].unique()
sex_ids = [1, 2]  # 1
combinations = list(itertools.product(age_group_ids, sex_ids))
as_df = pd.DataFrame(combinations, columns=['age_group_id', 'sex_id'])

###----------------------------------------------------------###
### 3. Main Data Loading and Processing
###----------------------------------------------------------###

# 3.1 Forecast Data Loading
# Loads the main forecast dataframe for the specified cause, scenario, and draw, then filters to relevant columns
df = read_parquet_with_integer_ids(
    cause_draw_path_template.format(
        FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
        cause=cause,
        ssp_scenario=ssp_scenario,
        dah_scenario_name=dah_scenario,
        draw=draw
    )
)

short = measure_map[measure]["short"]

reference_rate_col = f"{cause}_{short}_rate_reference"

df_columns = [col for col in df.columns if short in col]
df_columns = ['location_id', 'year_id', 'population', 'location_name',
       'fhs_location_id'] + df_columns
df = df[df_columns].copy()

# 3.2 Column Name Standardization
# Identifies prediction columns and creates standardized naming conventions for observed values, raw predictions, and reference predictions
# Find the column with "pred" in it and save that column name
pred_raw_col = [col for col in df.columns if "pred_raw" in col][0]
# Set _pred_raw to ""
obs_col = pred_raw_col.replace("_pred_raw", "")
pred_col = pred_raw_col.replace("_pred_raw", "_reference_pred")

###----------------------------------------------------------###
### 4. Reference Data Integration
###----------------------------------------------------------###

# 4.1 Age-Specific FHS Data Loading
# Loads age-specific Forecasting Health Scenarios (FHS) reference data and extracts reference age group and sex identifiers
as_fhs_df = read_parquet_with_integer_ids(
    as_fhs_df_template.format(
        MODELING_DATA_PATH=MODELING_DATA_PATH,
        cause=cause,
        measure=measure,
        metric="count"
    )
)

reference_age_group_id = as_fhs_df["reference_age_group_id"].unique()[0]
reference_sex_id = as_fhs_df["reference_sex_id"].unique()[0]

# 4.2 Reference Rate Calculation
# Creates reference rates from the baseline age-sex group and merges with main forecast data to establish prediction baselines
as_fhs_0_df = as_fhs_df[
    (as_fhs_df["age_group_id"] == reference_age_group_id) &
    (as_fhs_df["sex_id"] == reference_sex_id)].copy()
as_fhs_0_df = as_fhs_0_df[["location_id", "risk_0", "year_id"]]
as_fhs_0_df = as_fhs_0_df.rename(columns={"location_id": "fhs_location_id"})
as_fhs_0_df = as_fhs_0_df.rename(columns={"risk_0": f"{cause}_{short}_rate_reference"})

as_fhs_0_df[obs_col] = np.log(as_fhs_0_df[reference_rate_col] + 1e-6)

df = df.merge(
    as_fhs_0_df,
    on=["fhs_location_id", "year_id"],
    how="left")

###----------------------------------------------------------###
### 5. Prediction Adjustment
###----------------------------------------------------------###

# 5.1 Shift Calculation
# Calculates the difference between observed and predicted values in 2022 to create adjustment factors for predictions
df_2022 = df[df["year_id"] == 2022].copy()

df_2022["shift"] = df_2022[obs_col] - df_2022[pred_raw_col]
df_2022["shift"] = df_2022["shift"].fillna(0)

df = df.merge(
    df_2022[["location_id", "shift"]],
    on=["location_id"],
    how="left"
)

# 5.2 Prediction Correction
# Applies the calculated shift to adjust raw predictions and generates corrected reference prediction rates
df[pred_col] = df[pred_raw_col] + df["shift"]
df[f"{cause}_{short}_rate_reference_pred"] = np.exp(df[pred_col])

# Drop all columns that have "log" in them
df = df.drop(columns=[col for col in df.columns if "log" in col])
# Drop all columns that have shift in them
df = df.drop(columns=[col for col in df.columns if "shift" in col])

###----------------------------------------------------------###
### 6. Age-Sex Disaggregation
###----------------------------------------------------------###

# 6.1 Relative Risk Integration
# Loads 2022 age-sex specific relative risks and applies special handling for malaria age group 2 (setting relative risk to 0)
as_fhs_2022_df = as_fhs_df[as_fhs_df["year_id"] == 2022].copy()

# Set as_fhs_2022_df relatve_risk_as that is NaN to 1
as_fhs_2022_df["relative_risk_as"] = as_fhs_2022_df["relative_risk_as"].fillna(0)
as_fhs_2022_df = as_fhs_2022_df.rename(columns={"location_id": "fhs_location_id"})
if cause == "malaria":
    as_fhs_2022_df.loc[as_fhs_2022_df["age_group_id"] == 2, "relative_risk_as"] = 0.0

# 6.2 Cross-Product Expansion
# Creates a full cross-product of locations/years with all age-sex combinations and merges relative risk data
full_df = df.merge(as_df, how = "cross")
# Drop population column
full_df = full_df.drop(columns=["population"])

full_df["reference_age_group_id"] = reference_age_group_id
full_df["reference_sex_id"] = reference_sex_id

full_df = full_df.merge(
    as_fhs_2022_df[["fhs_location_id", "age_group_id", "sex_id", "relative_risk_as"]],
    on=["fhs_location_id", "age_group_id", "sex_id"],
    how="left"
)

# 6.3 Rate Prediction
# Calculates age-sex specific rates by multiplying reference predictions with relative risk factors
full_df["rate_pred"] = full_df[f"{cause}_{short}_rate_reference_pred"] * full_df["relative_risk_as"]

###----------------------------------------------------------###
### 7. Population Integration and Final Calculations
###----------------------------------------------------------###

# 7.1 Population Data Merging
# Loads and merges age-sex specific population data filtered to match the forecast locations and years
# Make filters based on FHS hierarchy
location_ids = df["location_id"].unique()
year_ids = df["year_id"].unique()
location_filter = ('location_id', 'in', location_ids)
year_filter = ('year_id', 'in', year_ids)

as_lsae_population_df_path = f"{MODELING_DATA_PATH}/as_lsae_population_df.parquet"

# Read FHS population data with filters
as_lsae_population_df = read_parquet_with_integer_ids(
    as_lsae_population_df_path,
    filters=[[location_filter, year_filter]]  # Combining with AND logic
)

full_df = full_df.merge(
    as_lsae_population_df[["location_id", "year_id", "population_aa", "age_group_id", "sex_id", "pop_fraction_aa", "population"]],
    on=["location_id", "year_id", "age_group_id", "sex_id"],
    how="left")

# 7.2 Count Calculation
# Converts predicted rates to predicted counts by multiplying with corresponding population values
full_df["count_pred"] = full_df["rate_pred"] * full_df["population"]
full_df["count_pred"] = full_df["count_pred"].fillna(0)

###----------------------------------------------------------###
### 8. Output Generation
### 8.1 Final Data Export
### Saves the complete age-sex disaggregated forecast data to a parquet file with the standardized naming convention
###----------------------------------------------------------###
# Write the final DataFrame to a parquet file
# Don't make any new changes, just write it out
# Don't run ensure 
output_path = output_cause_draw_path_template.format(
    FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
    cause=cause,
    measure=measure,
    ssp_scenario=ssp_scenario,
    dah_scenario_name=dah_scenario,
    draw=draw
)

write_parquet(full_df, output_path, max_retries=3, compression="snappy", index=False)