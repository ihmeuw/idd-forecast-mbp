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

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import ensure_id_columns_are_integers, read_parquet_with_integer_ids, read_income_paths, merge_dataframes, write_parquet, read_urban_paths


hierarchy = "lsae_1209"
ssp_scenarios = rfc.ssp_scenarios
draws = rfc.draws
years = rfc.model_years
year_filter = ('year_id', 'in', years)

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
VARIABLE_DATA_PATH = f"{PROCESSED_DATA_PATH}/{hierarchy}"
# CLIMATE 1209 variable path
CLIMATE_DATA_PATH = f"/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/{hierarchy}"

# Hierarchy
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

aa_full_population_df_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population.parquet"
aa_full_population_df = read_parquet_with_integer_ids(aa_full_population_df_path)
aa_merge_variables = rfc.aa_merge_variables
# ppp
income_paths = {
    "gdppc": "{VARIABLE_DATA_PATH}/gdppc_mean.parquet",
}

# DAH
dah_df_path = f"{VARIABLE_DATA_PATH}/dah_df.parquet"

urban_paths = {
    "urban_threshold_300":      "{VARIABLE_DATA_PATH}/urban_threshold_300.0_simple_mean.parquet",
    "urban_threshold_1500":     "{VARIABLE_DATA_PATH}/urban_threshold_1500.0_simple_mean.parquet",
}

# Climate variables
cc_sensitive_paths = {
    "flooding": "/mnt/team/rapidresponse/pub/flooding/results/output/lsae_1209/fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet"
}


for ssp_scenario in ssp_scenarios:
    print(f"Processing SSP scenario: {ssp_scenario}")
    rcp_scenario = ssp_scenarios[ssp_scenario]['rcp_scenario']

    flooding_df_path_template = cc_sensitive_paths['flooding']
    flooding_df_path = flooding_df_path_template.format(ssp_scenario=ssp_scenario)

    forecast_df = read_parquet_with_integer_ids(flooding_df_path,
                                                filters = [year_filter])
    forecast_df = forecast_df.drop(
        columns=["model", "variant", 'population']
    )

    urban_dfs = read_urban_paths(urban_paths, VARIABLE_DATA_PATH)
    forecast_df = merge_dataframes(forecast_df, urban_dfs)

    covariates_to_logit_transform = [col for col in forecast_df.columns if "urban" in col]

    # Logit transform the covariates and save them as new columns with "logit_" prefix
    for col in covariates_to_logit_transform:
        # Create a new column with the logit transformed value
        # print range of the column
        print(f"Range of {col}: {forecast_df[col].min()} to {forecast_df[col].max()}")
        # Clip values to be strictly between 0 and 1
        clipped_values = forecast_df[col].clip(lower=0.001, upper=0.999)
        forecast_df[f"logit_{col}"] = np.log(clipped_values / (1 - clipped_values))

    forecast_df = forecast_df.merge(
        aa_full_population_df,
        on = aa_merge_variables,
        how = "left")

    # Merge in the hierarchy_df
    forecast_df = forecast_df.merge(
        hierarchy_df[['location_id', 'A0_location_id', 'most_detailed_lsae']],
        how="left",
        left_on="location_id",
        right_on="location_id"
    )

    # Drop rows where A0_location_id is NaN
    forecast_df = forecast_df.dropna(subset=["A0_location_id"])
    forecast_df = forecast_df[forecast_df["most_detailed_lsae"] == 1]
    
    forecast_df = ensure_id_columns_are_integers(forecast_df)

    print("Reading income paths...")
    income_dfs = read_income_paths(income_paths, rcp_scenario, VARIABLE_DATA_PATH)
    forecast_df = merge_dataframes(forecast_df, income_dfs)

    print("Writing dengue forecast non-draw part...")
    cause = "dengue"
    write_parquet(forecast_df, f"{FORECASTING_DATA_PATH}/{cause}_forecast_scenario_{ssp_scenario}_non_draw_part.parquet")

    print("Reading DAH data...")
    dah_df = read_parquet_with_integer_ids(dah_df_path)
    dah_df = dah_df.filter(regex="location_id|year_id|total")
    forecast_df = forecast_df.merge(dah_df, on=["location_id", "year_id"], how = "left")

    print("Writing malaria forecast non-draw part...")
    cause = "malaria"
    write_parquet(forecast_df, f"{FORECASTING_DATA_PATH}/{cause}_forecast_scenario_{ssp_scenario}_non_draw_part.parquet")