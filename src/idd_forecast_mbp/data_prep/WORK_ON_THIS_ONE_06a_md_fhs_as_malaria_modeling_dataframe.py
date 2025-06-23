
# To DO
# - Make this work
# - Make the same work for dengue
# - Finalize models
# - Forecast all the things
# - Aggregate all the things
# - Rake all the things
# - Create code that checks the raking
# - Agregate to upload files
# - Upload





################################################################
### FHS Malaria Mortality Data Preparation
### This script prepares the FHS malaria mortality data for modeling.
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
from idd_forecast_mbp.helper_functions import read_parquet_with_integer_ids
import glob

malaria_mortality_threshold = 1
# Hierarchy
hierarchy = "lsae_1209"

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"

cause_map = rfc.cause_map
measure_map = rfc.measure_map
metric_map = rfc.metric_map
age_type_map = rfc.age_type_map

cause = "malaria"

###----------------------------------------------------------###
### 2. Path Configuration and Data Sources
### Defines all file paths for input data including hierarchy, climate variables,
### economic indicators, and health assistance data needed for modeling.
###----------------------------------------------------------###
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)
# LSAE 1209 variable path
VARIABLE_DATA_PATH = f"{PROCESSED_DATA_PATH}/{hierarchy}"
# CLIMATE 1209 variable path
CLIMATE_DATA_PATH = f"/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/{hierarchy}"

# Malaria data
malaria_df_path = MODELING_DATA_PATH / "malaria_stage_2_modeling_full_df.parquet"
as_fhs_df_path = "{MODELING_DATA_PATH}/fhs_{cause}_{measure}_{metric}_df.parquet"

# Read malaria data
malaria_df = read_parquet_with_integer_ids(malaria_df_path)
malaria_df = malaria_df.merge(hierarchy_df[["location_id", "fhs_location_id"]], on=["location_id"], how="left")


md_malaria_df = malaria_df.copy()
md_malaria_df = md_malaria_df[md_malaria_df["most_detailed_lsae"] == 1].copy()
md_malaria_df["gdp"] = md_malaria_df["gdppc_mean"] * md_malaria_df["population"]
fhs_gdp_df = md_malaria_df.groupby(["fhs_location_id", "year_id"]).agg({
    "gdp": "sum",
    "population": "sum"
}).reset_index()
fhs_gdp_df["gdppc_mean"] = fhs_gdp_df["gdp"] / fhs_gdp_df["population"]
# Rename fhs_location_id to location_id for consistency
fhs_gdp_df = fhs_gdp_df.rename(columns={"fhs_location_id": "location_id"})
# Update gdppc_mean in malaria_df with FHS-aggregated values without changing other values
malaria_df = malaria_df.merge(
    fhs_gdp_df[["location_id", "year_id", "gdppc_mean"]].rename(columns={"gdppc_mean": "gdppc_mean_fhs"}),
    on=["location_id", "year_id"],
    how="left"
)

# Replace the original gdppc_mean with the FHS-aggregated values where available
malaria_df["gdppc_mean"] = malaria_df["gdppc_mean_fhs"].fillna(malaria_df["gdppc_mean"])

# Drop the temporary column
malaria_df = malaria_df.drop(columns=["gdppc_mean_fhs"])

malaria_df["log_gdppc_mean"] = np.log(malaria_df["gdppc_mean"] + 1e-6)  # Adding a small constant to avoid log(0)


###----------------------------------------------------------###
### 3. Data Preparation
### Reads the hierarchy data and prepares the malaria data for modeling.
### It merges the malaria data with the hierarchy and filters it based on reference
###----------------------------------------------------------###
# measure = "mortality"
# metric = "count"
for measure in measure_map:
    for metric in metric_map:
        as_fhs_df = read_parquet_with_integer_ids(as_fhs_df_path.format(MODELING_DATA_PATH=MODELING_DATA_PATH, cause=cause, measure=measure, metric=metric))
        as_fhs_df = as_fhs_df.merge(hierarchy_df[["location_id", "most_detailed_fhs"]], on="location_id", how="left")

        reference_age_group_id = as_fhs_df["reference_age_group_id"].unique()[0]
        reference_sex_id = as_fhs_df["reference_sex_id"].unique()[0]

        as_fhs_0_df = as_fhs_df[
            (as_fhs_df["age_group_id"] == reference_age_group_id) &
            (as_fhs_df["sex_id"] == reference_sex_id)].copy()

        # Drop reference_age_group_id and reference_sex_id columns
        as_fhs_0_df = as_fhs_0_df.drop(columns=["reference_age_group_id", "reference_sex_id"])

        # as_fhs_df = as_fhs_df.merge(malaria_df, on=["location_id", "year_id"], how="left")
        as_fhs_0_df = as_fhs_0_df.merge(malaria_df, on=["location_id", "year_id"], how="left")
        # Write both dataframes to parquet
        as_fhs_df.to_parquet(
            MODELING_DATA_PATH / f"fhs_{cause}_{measure}_{metric}_modeling_df.parquet",
            index=False)
        as_fhs_0_df.to_parquet(
            MODELING_DATA_PATH / f"fhs_{cause}_{measure}_{metric}_modeling_df_0.parquet",
            index=False)