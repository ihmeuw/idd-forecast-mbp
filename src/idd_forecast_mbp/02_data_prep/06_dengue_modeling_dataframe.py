################################################################
### DENGUE MODELING DATA PREPARATION
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

dengue_mortality_theshold = 1
dengue_mortality_rate_theshold = 1e-7


# Hierarchy
hierarchy = "lsae_1209"

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"

cause_map = rfc.cause_map
cause = 'dengue'
reference_age_group_id = cause_map[cause]['reference_age_group_id']
reference_sex_id = cause_map[cause]['reference_sex_id']
###----------------------------------------------------------###
### 2. Path Configuration and Data Sources
### Defines all file paths for input data including hierarchy, climate variables,
### economic indicators, and health assistance data needed for modeling.
###----------------------------------------------------------###
aa_ge3_dengue_stage_1_modeling_df_path = f"{MODELING_DATA_PATH}/aa_ge3_{cause}_stage_1_modeling_df.parquet"
as_md_dengue_modeling_df_path = f"{MODELING_DATA_PATH}/as_md_{cause}_modeling_df.parquet"
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

aa_full_dengue_df_path = PROCESSED_DATA_PATH / "aa_full_dengue_df.parquet"
dengue_df_path = PROCESSED_DATA_PATH / "aa_full_dengue_df.parquet"
# ppp
income_paths = {
    "gdppc":                   "{VARIABLE_DATA_PATH}/gdppc_mean.parquet"
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
    "relative_humidity":        "{CLIMATE_DATA_PATH}/relative_humidity_{ssp_scenario}.parquet",
    "mean_temperature":         "{CLIMATE_DATA_PATH}/mean_temperature_{ssp_scenario}.parquet",
    "mean_low_temperature":     "{CLIMATE_DATA_PATH}/mean_low_temperature_{ssp_scenario}.parquet",
    "mean_high_temperature":    "{CLIMATE_DATA_PATH}/mean_high_temperature_{ssp_scenario}.parquet",
    "dengue_suitability":       "{CLIMATE_DATA_PATH}/dengue_suitability_{ssp_scenario}.parquet",
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

aa_merge_variables = ["location_id", "year_id"]
as_merge_variables = ["location_id", "year_id", "age_group_id", "sex_id"]

###----------------------------------------------------------###
### 3. Data Loading and Integration
### Loads the base dengue dataset and integrates various predictor variables
### including urbanization metrics, income data,
### and climate variables from different sources.
###----------------------------------------------------------###
# Load core dengue data
dengue_df = read_parquet_with_integer_ids(aa_full_cause_df_path_template,
                                           filters=[year_filter, level_filter(hierarchy_df, start_level = 3, end_level = 5)])

dengue_df = dengue_df.rename(columns={
    "dengue_mort_count": "aa_dengue_mort_count",
    "dengue_inc_count": "aa_dengue_inc_count",
    "dengue_mort_rate": "aa_dengue_mort_rate",
    "dengue_inc_rate": "aa_dengue_inc_rate",
})
# Make CFR
dengue_df["aa_dengue_cfr"] = dengue_df["aa_dengue_mort_count"] / dengue_df["aa_dengue_inc_count"]
# Set to 0 if inc count is 0
dengue_df.loc[dengue_df["aa_dengue_inc_count"] == 0, "aa_dengue_cfr"] = 0

dengue_df = pd.merge(dengue_df, 
                     hierarchy_df[["location_id", "level", "A0_location_id"]], 
                     on=["location_id"], how="left")

# Load and merge urbanization metrics
urban_dfs = read_urban_paths(urban_paths, VARIABLE_DATA_PATH)
dengue_df = merge_dataframes(dengue_df, urban_dfs)
# Set the max value of any urban threshold to 1
for col in [c for c in dengue_df.columns if "urban" in c]:
    dengue_df[col] = dengue_df[col].clip(upper=1)

# Load and merge income metrics
income_dfs = read_income_paths(income_paths, rcp_scenario, VARIABLE_DATA_PATH)
dengue_df = merge_dataframes(dengue_df, income_dfs)

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
    # Merge the file with dengue_df
    dengue_df = pd.merge(dengue_df, df, on=["location_id", "year_id"], how="left")

covariates_to_log_transform = [
    "gdppc_mean",
]

# Log transform the covariates and save them as new columns with "log_" prefix
for col in covariates_to_log_transform:
    # Create a new column with the log transformed value
    dengue_df[f"log_{col}"] = np.log(dengue_df[col] + 1e-6)

# Covariates to logit transform
covariates_to_logit_transform = [col for col in dengue_df.columns if "urban" in col]

# Logit transform the covariates and save them as new columns with "logit_" prefix
for col in covariates_to_logit_transform:
    # Create a new column with the logit transformed value
    # print range of the column
    print(f"Range of {col}: {dengue_df[col].min()} to {dengue_df[col].max()}")
    # Clip values to be strictly between 0 and 1
    clipped_values = dengue_df[col].clip(lower=0.001, upper=0.999)
    dengue_df[f"logit_{col}"] = np.log(clipped_values / (1 - clipped_values))

aa_A0_dengue_df = dengue_df[(dengue_df["location_id"] == dengue_df["A0_location_id"]) & (dengue_df["year_id"] == 2022)].copy()
aa_A0_dengue_df = aa_A0_dengue_df[aa_A0_dengue_df['aa_dengue_mort_count'] > 1].copy()
A0_dengue_ids = aa_A0_dengue_df['A0_location_id'].unique()


# Make the yn variable. This will be used as the response in the phase 1 model and to trim the data in the phase 2 model
# dengue_df$yn[which(dengue_df$dengue_mort_rate > dengue_mortality_rate_theshold & dengue_df$dengue_mort_count > dengue_mortality_theshold & dengue_df$dengue_suitability > 0)] <- 1
dengue_df["yn"] = 0
dengue_df.loc[
    (dengue_df["aa_dengue_mort_rate"] > 1/100000) &
    (dengue_df["aa_dengue_mort_count"] > 0) &
    (dengue_df["aa_dengue_inc_count"] > 0) &
    (dengue_df["dengue_suitability"] > 0),
    "yn"
] = 1

write_parquet(dengue_df, aa_ge3_dengue_stage_1_modeling_df_path)

dengue_df = dengue_df[dengue_df['A0_location_id'].isin(A0_dengue_ids)].copy() 



###----------------------------------------------------------###
### 8. Final Modeling Dataset Preparation
### Prepares the final dataset for modeling by selecting relevant columns,
### merging to the age-sex-location-year level, and saving the final dataset.
###----------------------------------------------------------###
dengue_stage_2_df = dengue_df.copy()
# Drop any columns that have yn = 0
dengue_stage_2_df = dengue_stage_2_df[dengue_stage_2_df["yn"] == 1].drop(columns=["yn"])
# Subset down to level 5 locations
dengue_stage_2_df = dengue_stage_2_df[dengue_stage_2_df["level"] == 5].drop(columns=["level"])
# Create the A0_af factor variable
dengue_stage_2_df["A0_location_id"] = dengue_stage_2_df["A0_location_id"].astype(int)
dengue_stage_2_df['A0_af'] = 'A0_' + dengue_stage_2_df['A0_location_id'].astype(str)
dengue_stage_2_df['A0_af'] = dengue_stage_2_df['A0_af'].astype('category')
# dengue_stage_2_df = dengue_stage_2_df.drop(columns=['aa_dengue_inc_count', 'aa_dengue_inc_rate', 'aa_dengue_mort_count', 'aa_dengue_mort_rate', 'aa_dengue_cfr'])
# Get the as data
md_location_ids = dengue_stage_2_df["location_id"].unique().tolist()
md_location_filter = ('location_id', 'in', md_location_ids)

as_md_df = read_parquet_with_integer_ids(as_full_cause_df_path_template,
                                         columns=as_merge_variables + ["dengue_mort_rate","dengue_inc_rate","dengue_mort_count","dengue_inc_count","population","aa_population"],
                                         filters=[year_filter, md_location_filter, age_filter, sex_filter])


as_md_df["dengue_cfr"] = as_md_df["dengue_mort_rate"] / as_md_df["dengue_inc_rate"]
as_md_df.loc[as_md_df["dengue_inc_rate"] == 0, "dengue_cfr"] = 0

covariates_to_log_transform = [
    "dengue_mort_rate",
    "dengue_inc_rate"
]
for col in covariates_to_log_transform:
    as_md_df[f"log_{col}"] = np.log(as_md_df[col] + 1e-6)

covariates_to_logit_transform = ['dengue_cfr']
for col in covariates_to_logit_transform:
    clipped_values = as_md_df[col].clip(lower=0.001, upper=0.999)
    as_md_df[f"logit_{col}"] = np.log(clipped_values / (1 - clipped_values))

dengue_stage_2_df = dengue_stage_2_df.drop(columns=["population"])
as_md_modeling_df = as_md_df.merge(dengue_stage_2_df, on=["location_id", "year_id"], how="left")
as_md_modeling_df = as_md_modeling_df[~as_md_modeling_df["mean_high_temperature"].isna()]

as_md_modeling_df = as_md_modeling_df[~(as_md_modeling_df["age_group_id"] == 2)]
as_md_modeling_df["as_id"] = "a" + as_md_modeling_df["age_group_id"].astype(str) + "_s" + as_md_modeling_df["sex_id"].astype(str)

write_parquet(as_md_modeling_df, as_md_dengue_modeling_df_path)

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


write_parquet(base_md_modeling_df, base_md_modeling_df_path)
write_parquet(rest_md_modeling_df, rest_md_modeling_df_path)