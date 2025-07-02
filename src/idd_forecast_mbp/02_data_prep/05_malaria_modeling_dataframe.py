################################################################
### MALARIA MODELING DATA PREPARATION
################################################################

###----------------------------------------------------------###
### 1. Setup and Configuration
### Sets up the environment with necessary libraries, constants, and path definitions.
### Establishes thresholds and directory structures for the modeling pipeline.
###----------------------------------------------------------###
import pandas as pd
import numpy as np
import os
import sys
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import merge_dataframes, read_income_paths, read_urban_paths, level_filter
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet
import glob

malaria_mortality_threshold = 1

# Hierarchy
hierarchy = "lsae_1209"

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
# Scenarios

cause_map = rfc.cause_map
cause = 'malaria'
reference_age_group_id = cause_map[cause]['reference_age_group_id']
reference_sex_id = cause_map[cause]['reference_sex_id']

###----------------------------------------------------------###
### 2. Path Configuration and Data Sources
### Defines all file paths for input data including hierarchy, climate variables,
### economic indicators, and health assistance data needed for modeling.
###----------------------------------------------------------###
aa_ge3_malaria_stage_1_modeling_df_path = f"{MODELING_DATA_PATH}/aa_ge3_{cause}_stage_1_modeling_df.parquet"
aa_md_malaria_pfpr_modeling_df_path = f"{MODELING_DATA_PATH}/aa_md_{cause}_pfpr_modeling_df.parquet"
as_md_modeling_df_path = f"{MODELING_DATA_PATH}/as_md_{cause}_modeling_df.parquet"
base_md_modeling_df_path = f"{MODELING_DATA_PATH}/base_md_{cause}_modeling_df.parquet"
rest_md_modeling_df_path = f"{MODELING_DATA_PATH}/rest_md_{cause}_modeling_df.parquet"



aa_full_cause_df_path_template = f'{PROCESSED_DATA_PATH}/aa_full_{cause}_df.parquet'
as_full_cause_df_path_template = f'{PROCESSED_DATA_PATH}/as_full_{cause}_df.parquet'

full_2023_hierarchy_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet"
age_sex_df_path = f'{PROCESSED_DATA_PATH}/age_sex_df.parquet'

hierarchy_df = read_parquet_with_integer_ids(full_2023_hierarchy_path)
age_sex_df = read_parquet_with_integer_ids(age_sex_df_path)

# LSAE 1209 variable path
VARIABLE_DATA_PATH = f"{PROCESSED_DATA_PATH}/{hierarchy}"
# CLIMATE 1209 variable path
CLIMATE_DATA_PATH = f"/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/{hierarchy}"

# ppp
income_paths = {
    "gdppc":                   "{VARIABLE_DATA_PATH}/gdppc_mean.parquet",
    "ldipc":                   "{VARIABLE_DATA_PATH}/ldipc_mean.parquet"
}
# DAH
dah_df_path = f"{VARIABLE_DATA_PATH}/dah_df.parquet"
# Population
population_path = "{VARIABLE_DATA_PATH}/population.parquet"
# Urban paths
urban_paths = {
    "urban_threshold_300":      "{VARIABLE_DATA_PATH}/urban_threshold_300.0_simple_mean.parquet",
    "urban_threshold_1500":     "{VARIABLE_DATA_PATH}/urban_threshold_1500.0_simple_mean.parquet",
}
# Climate variables
cc_sensitive_paths = {
    "total_precipitation":      "{CLIMATE_DATA_PATH}/total_precipitation_{ssp_scenario}.parquet",
    "precipitation_days":       "{CLIMATE_DATA_PATH}/precipitation_days_{ssp_scenario}.parquet",
    "relative_humidity":        "{CLIMATE_DATA_PATH}/relative_humidity_{ssp_scenario}.parquet",
    "wind_speed":               "{CLIMATE_DATA_PATH}/wind_speed_{ssp_scenario}.parquet",
    "mean_temperature":         "{CLIMATE_DATA_PATH}/mean_temperature_{ssp_scenario}.parquet",
    "mean_low_temperature":     "{CLIMATE_DATA_PATH}/mean_low_temperature_{ssp_scenario}.parquet",
    "mean_high_temperature":    "{CLIMATE_DATA_PATH}/mean_high_temperature_{ssp_scenario}.parquet",
    "malaria_suitability":      "{CLIMATE_DATA_PATH}/malaria_suitability_{ssp_scenario}.parquet",
    "flooding":                 "/mnt/team/rapidresponse/pub/flooding/results/output/lsae_1209/fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet"
}

## Past data
ssp_scenarios =  rfc.ssp_scenarios
ssp_scenario = list(ssp_scenarios.keys())[0]
rcp_scenario = ssp_scenarios[ssp_scenario]["rcp_scenario"]


years = list(range(2000, 2023))
year_filter = ('year_id', 'in', years)
sex_ids = [1, 2]
sex_filter = ('sex_id', 'in', sex_ids)

age_group_ids = age_sex_df['age_group_id'].unique().tolist()
age_filter = ('age_group_id', 'in', age_group_ids)

aa_merge_variables = rfc.aa_merge_variables
as_merge_variables = rfc.as_merge_variables

###----------------------------------------------------------###
### 4. Data Loading and Integration
### Loads the base malaria dataset and integrates various predictor variables
### including development assistance, urbanization metrics, income data,
### and climate variables from different sources.
###----------------------------------------------------------###


# Load core malaria data
malaria_df = read_parquet_with_integer_ids(aa_full_cause_df_path_template,
filters=[year_filter, level_filter(hierarchy_df, start_level = 3, end_level = 5)])
# Read and merge development assistance data
dah_df = read_parquet_with_integer_ids(dah_df_path)
malaria_df = pd.merge(malaria_df, 
                      dah_df[aa_merge_variables + ['mal_DAH_total','mal_DAH_total_per_capita']], 
                      on = aa_merge_variables, how="left")

# Load and merge urbanization metrics
urban_dfs = read_urban_paths(urban_paths, VARIABLE_DATA_PATH)
malaria_df = merge_dataframes(malaria_df, urban_dfs)
# Set the max value of any urban threshold to 1
for col in [c for c in malaria_df.columns if "urban" in c]:
    malaria_df[col] = malaria_df[col].clip(upper=1)

# Load and merge income metrics
income_dfs = read_income_paths(income_paths, rcp_scenario, VARIABLE_DATA_PATH)
malaria_df = merge_dataframes(malaria_df, income_dfs)

# Load and merge climate variables
for key, path_template in cc_sensitive_paths.items():
    # Replace {ssp_scenario} in the path with the current ssp_scenario
    path = path_template.format(CLIMATE_DATA_PATH=CLIMATE_DATA_PATH, ssp_scenario=ssp_scenario)
    print(f"Reading {key} data from {path}")
    # Read the parquet file
    if key == "flooding":
        df = read_parquet_with_integer_ids(path, filters=[[level_filter(hierarchy_df, start_level = 3, end_level = 5), year_filter]])
        df = df.drop(columns=["model", "scenario", "variant", "population"], errors='ignore')
    else:
        # Select only the relevant columns
        columns_to_read = ["location_id", "year_id", "000"]
        df = read_parquet_with_integer_ids(path, columns=columns_to_read, filters=[[level_filter(hierarchy_df, start_level = 3, end_level = 5), year_filter]])
        # Rename the 000 column to the key
        df = df.rename(columns={"000": key})
    # Merge the file with malaria_df
    malaria_df = pd.merge(malaria_df, df, on=["location_id", "year_id"], how="left")

###----------------------------------------------------------###
### 5. Stage 1 Modeling Data
### Completes the comprehensive data integration and saves the full dataset
### for initial modeling before further refinement.
###----------------------------------------------------------###
# Save stage 1 malaria_df to a parquet file
write_parquet(malaria_df, aa_ge3_malaria_stage_1_modeling_df_path)

###----------------------------------------------------------###
### 6. Stage 2 Data Filtering and Selection
### Filters the dataset to focus on high-burden malaria areas and 
### the most detailed geographic units with sufficient data for
### meaningful analysis and prediction.
###----------------------------------------------------------###
malaria_stage_2_df = malaria_df.copy()
malaria_stage_2_df = malaria_stage_2_df.merge(hierarchy_df[["location_id", "level", "A0_location_id", "most_detailed_lsae"]], on=["location_id"], how="left")

# Remove rows with missing mortality data
malaria_stage_2_df = malaria_stage_2_df[malaria_stage_2_df["malaria_mort_count"].notna()]

# Filter out zero-value or invalid data points
malaria_stage_2_df = malaria_stage_2_df[
    (malaria_stage_2_df["malaria_pfpr"] > 0) &
    (malaria_stage_2_df["malaria_mort_count"] > 0) &
    (malaria_stage_2_df["malaria_inc_count"] >= 0)
]
malaria_stage_2_df = malaria_stage_2_df.copy()

# Select countries with significant malaria burden (mortality > threshold)
A0_malaria_stage_2_df = malaria_stage_2_df[(malaria_stage_2_df["location_id"] == malaria_stage_2_df["A0_location_id"]) & (malaria_stage_2_df["year_id"] == 2022)].copy()
A0_malaria_stage_2_df = A0_malaria_stage_2_df.rename(columns={
    "malaria_pfpr": "A0_malaria_pfpr",
    "malaria_mort_count": "A0_malaria_mort_count",
    "malaria_inc_count": "A0_malaria_inc_count"})

A0_malaria_stage_2_df = A0_malaria_stage_2_df[A0_malaria_stage_2_df["A0_malaria_mort_count"] >= malaria_mortality_threshold]
phase_2_A0_location_ids = A0_malaria_stage_2_df["A0_location_id"].unique()

# Subset to high-burden countries and most detailed geographic units
malaria_stage_2_df = malaria_stage_2_df[malaria_stage_2_df["A0_location_id"].isin(phase_2_A0_location_ids)]
malaria_stage_2_df = malaria_stage_2_df.merge(A0_malaria_stage_2_df[["A0_location_id", "A0_malaria_pfpr", "A0_malaria_mort_count", "A0_malaria_inc_count"]], on=["A0_location_id"], how="left")

###----------------------------------------------------------###
### 7. Feature Engineering and Transformations
### Applies appropriate transformations to variables to improve model fit
### and statistical properties, creating derived features needed for modeling.
###----------------------------------------------------------###
# Create country-level factor variable for fixed effects modeling
malaria_stage_2_df["A0_location_id"] = malaria_stage_2_df["A0_location_id"].astype(int)
malaria_stage_2_df['A0_af'] = 'A0_' + malaria_stage_2_df['A0_location_id'].astype(str)
malaria_stage_2_df['A0_af'] = malaria_stage_2_df['A0_af'].astype('category')
# Remove rows with missing economic data
malaria_stage_2_df = malaria_stage_2_df[malaria_stage_2_df["gdppc_mean"].notna()]
malaria_stage_2_df = malaria_stage_2_df[malaria_stage_2_df["most_detailed_lsae"] == 1]
# Define variables that need log transformation
covariates_to_log_transform = [
    "mal_DAH_total_per_capita",
    "gdppc_mean",
    "ldipc_mean",
]

# Apply log transformations
for col in covariates_to_log_transform:
    malaria_stage_2_df[f"log_{col}"] = np.log(malaria_stage_2_df[col] + 1e-6)

# Find urban variables for logit transformation
covariates_to_logit_transform = [col for col in malaria_stage_2_df.columns if "urban" in col]

# Apply logit transformations to urbanization variables
for col in covariates_to_logit_transform:
    clipped_values = malaria_stage_2_df[col].clip(lower=0.001, upper=0.999)
    malaria_stage_2_df[f"logit_{col}"] = np.log(clipped_values / (1 - clipped_values))

# Apply logit transform to malaria prevalence (PfPR)
malaria_stage_2_df[f"logit_malaria_pfpr"] = np.log(0.999 * malaria_stage_2_df["malaria_pfpr"] / (1 - 0.999 * malaria_stage_2_df["malaria_pfpr"]))
# Save stage 1 malaria_df to a parquet file
write_parquet(malaria_stage_2_df, aa_md_malaria_pfpr_modeling_df_path)
###----------------------------------------------------------###
### 8. Final Modeling Dataset Preparation
### Prepares the final dataset for modeling by selecting relevant columns,
### merging to the age-sex-location-year level, and saving the final dataset.
###----------------------------------------------------------###
aa_md_malaria_pfpr_modeling_df_path = f"{MODELING_DATA_PATH}/aa_md_malaria_pfpr_modeling_df.parquet"

stage_2_df_columns_to_keep = ['location_id', 'year_id', 'malaria_pfpr', 
    'mal_DAH_total', 'mal_DAH_total_per_capita', 'urban_1km_threshold_300',
    'urban_100m_threshold_300', 'urban_1km_threshold_1500', 'urban_100m_threshold_1500', 'gdppc_mean', 
    'total_precipitation', 'relative_humidity', 'mean_temperature',
    'mean_high_temperature', 'malaria_suitability', 'people_flood_days_per_capita', 'A0_location_id',
    "A0_malaria_pfpr", "A0_malaria_mort_count", "A0_malaria_inc_count",
    'A0_af', 'log_mal_DAH_total_per_capita', 'log_gdppc_mean', 'logit_urban_1km_threshold_300',
    'logit_urban_100m_threshold_300', 'logit_urban_1km_threshold_1500', 'logit_urban_100m_threshold_1500', 'logit_malaria_pfpr']
malaria_stage_3_df = malaria_stage_2_df[stage_2_df_columns_to_keep].copy()

md_location_ids = malaria_stage_3_df["location_id"].unique().tolist()
md_location_filter = ('location_id', 'in', md_location_ids)

as_md_df = read_parquet_with_integer_ids(as_full_cause_df_path_template,
columns=as_merge_variables + ["malaria_mort_rate","malaria_inc_rate"],
filters=[year_filter, md_location_filter, age_filter, sex_filter])

covariates_to_log_transform = [
    "malaria_mort_rate",
    "malaria_inc_rate"
]
# Apply log transformations
for col in covariates_to_log_transform:
    as_md_df[f"log_{col}"] = np.log(as_md_df[col] + 1e-6)

as_md_modeling_df = as_md_df.merge(malaria_stage_3_df, on=["location_id", "year_id"], how="left")
as_md_modeling_df = as_md_modeling_df[~as_md_modeling_df["A0_af"].isna()]

as_md_modeling_df = as_md_modeling_df[~(as_md_modeling_df["age_group_id"] == 2)]
as_md_modeling_df["as_id"] = "a" + as_md_modeling_df["age_group_id"].astype(str) + "_s" + as_md_modeling_df["sex_id"].astype(str)

write_parquet(as_md_modeling_df, as_md_modeling_df_path)

cause_columns = list([col for col in as_md_modeling_df.columns if cause in col and "suit" not in col])
base_md_modeling_df = as_md_modeling_df[(as_md_modeling_df['age_group_id'] == reference_age_group_id) & (as_md_modeling_df['sex_id'] == reference_sex_id)].copy()
# Add 'base_' prefix to every cause_column in base_md_modeling_df
base_column_mapping = {col: f'base_{col}' for col in cause_columns}
base_md_modeling_df = base_md_modeling_df.rename(columns=base_column_mapping)

rest_md_modeling_df = as_md_modeling_df[~((as_md_modeling_df['age_group_id'] == reference_age_group_id) & (as_md_modeling_df['sex_id'] == reference_sex_id))].copy()
# Merge rest_md_modeling_df with the base data
rest_md_modeling_df = rest_md_modeling_df.merge(base_md_modeling_df[aa_merge_variables + [f'base_{col}' for col in cause_columns]],
on=aa_merge_variables,
how='left')

rest_md_modeling_df["as_id"] = "a" + rest_md_modeling_df["age_group_id"].astype(str) + "_s" + rest_md_modeling_df["sex_id"].astype(str)
rest_md_modeling_df['as_id'] = rest_md_modeling_df['as_id'].astype('category')

write_parquet(base_md_modeling_df, base_md_modeling_df_path)
write_parquet(rest_md_modeling_df, rest_md_modeling_df_path)
