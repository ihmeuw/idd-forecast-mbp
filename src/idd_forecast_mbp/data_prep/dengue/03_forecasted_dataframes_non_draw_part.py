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
from idd_forecast_mbp.helper_functions import ensure_id_columns_are_integers, read_parquet_with_integer_ids, read_income_paths, merge_dataframes
import glob

hierarchy = "lsae_1209"

ssp_scenarios = rfc.ssp_scenarios
draws = rfc.draws

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
VARIABLE_DATA_PATH = f"{PROCESSED_DATA_PATH}/{hierarchy}"
# CLIMATE 1209 variable path
CLIMATE_DATA_PATH = f"/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/{hierarchy}"


# Hierarchy
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

# ppp
income_paths = {
    "gdppc": "{VARIABLE_DATA_PATH}/gdppc_mean.parquet",
}

# Climate variables
cc_sensitive_paths = {
    "total_precipitation":      "{CLIMATE_DATA_PATH}/total_precipitation_{ssp_scenario}.parquet",
    # "precipitation_days":       "{CLIMATE_DATA_PATH}/precipitation_days_{ssp_scenario}.parquet",
    # "relative_humidity":        "{CLIMATE_DATA_PATH}/relative_humidity_{ssp_scenario}.parquet",
    # "wind_speed":               "{CLIMATE_DATA_PATH}/wind_speed_{ssp_scenario}.parquet",
    # "mean_temperature":         "{CLIMATE_DATA_PATH}/mean_temperature_{ssp_scenario}.parquet",
    # "mean_low_temperature":     "{CLIMATE_DATA_PATH}/mean_low_temperature_{ssp_scenario}.parquet",
    # "mean_high_temperature":    "{CLIMATE_DATA_PATH}/mean_high_temperature_{ssp_scenario}.parquet",
    "malaria_suitability":      "{CLIMATE_DATA_PATH}/malaria_suitability_{ssp_scenario}.parquet",
    "flooding":                 "/mnt/team/rapidresponse/pub/flooding/results/output/lsae_1209/fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet"
}


for ssp_scenario in ssp_scenarios:
    rcp_scenario = ssp_scenarios[ssp_scenario]['rcp_scenario']

    flooding_df_path_template = cc_sensitive_paths['flooding']
    flooding_df_path = flooding_df_path_template.format(ssp_scenario=ssp_scenario)

    forecast_df = read_parquet_with_integer_ids(flooding_df_path)
    forecast_df = forecast_df.drop(
        columns=["model", "variant"]
    )
    # Merge in the hierarchy_df
    forecast_df = forecast_df.merge(
        hierarchy_df,
        how="left",
        left_on="location_id",
        right_on="location_id"
    )

    # Drop rows where A0_location_id is NaN
    forecast_df = forecast_df.dropna(subset=["A0_location_id"])
    forecast_df = forecast_df[forecast_df["most_detailed_lsae"] == 1]
    
    forecast_df = ensure_id_columns_are_integers(forecast_df)

    income_dfs = read_income_paths(income_paths, rcp_scenario, VARIABLE_DATA_PATH)
    forecast_df = merge_dataframes(forecast_df, income_dfs)

    # Write the malaria_stage_2_modeling_df to a parquet file
    forecast_df.to_parquet(f"{FORECASTING_DATA_PATH}/dengue_forecast_scenario_{ssp_scenario}_non_draw_part.parquet", index=False)