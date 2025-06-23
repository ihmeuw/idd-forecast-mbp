import pandas as pd
import numpy as np
import os
import sys
import glob

# Scenarios
ssp_scenarios = ["ssp126", "ssp245", "ssp585"]
rcp_scenarios = [2.6, 4.5, 8.5]
# Draws
draws = [f"{i:03d}" for i in range(100)]

# Hierarchy
hierarchy = "lsae_1209"

# Hierarchy path
HIERARCHY_PATH = f"/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_{hierarchy}.parquet"

#
OUTCOME_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/02-processed_data/gbd"
# LSAE 1209 variable path
VARIABLE_DATA_PATH = f"/mnt/team/idd/pub/forecast-mbp/02-processed_data/{hierarchy}"
# CLIMATE 1209 variable path
CLIMATE_DATA_PATH = f"/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/{hierarchy}"
# MODELING_DATA_PATH 
MODELING_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"
FORECASTING_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data"

# Population
FHS_population_path = "/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/fhs_2021/population.parquet"
LSAE_population_path = "/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/lsae_1209/population.parquet"

# Dengue modeling dataframes
dengue_stage_1_modeling_df_path = f"{MODELING_DATA_PATH}/dengue_stage_1_modeling_df.parquet"
dengue_stage_2_modeling_df_path = f"{MODELING_DATA_PATH}/dengue_stage_2_modeling_df.parquet"


# Raked DENV
dengue_df_path = f"{OUTCOME_DATA_PATH}/raked_dengue_aa.parquet"

## Covariates
population_path = "{VARIABLE_DATA_PATH}/population.parquet"
urban_paths = {
    "urban_threshold_300":      "{VARIABLE_DATA_PATH}/urban_threshold_300.0_simple_mean.parquet",
    "urban_threshold_1500":     "{VARIABLE_DATA_PATH}/urban_threshold_1500.0_simple_mean.parquet",
}
income_paths = {
    "gdppc":                   "{VARIABLE_DATA_PATH}/gdppc_mean.parquet",
    "ldipc":                   "{VARIABLE_DATA_PATH}/ldipc_mean.parquet"
}
cc_sensitive_paths = {
    "total_precipitation":      "{CLIMATE_DATA_PATH}/total_precipitation_{ssp_scenario}.parquet",
    # "precipitation_days":       "{CLIMATE_DATA_PATH}/precipitation_days_{ssp_scenario}.parquet",
    "relative_humidity":        "{CLIMATE_DATA_PATH}/relative_humidity_{ssp_scenario}.parquet",
    # "wind_speed":               "{CLIMATE_DATA_PATH}/wind_speed_{ssp_scenario}.parquet",
    "mean_temperature":         "{CLIMATE_DATA_PATH}/mean_temperature_{ssp_scenario}.parquet",
    # "mean_low_temperature":     "{CLIMATE_DATA_PATH}/mean_low_temperature_{ssp_scenario}.parquet",
    "mean_high_temperature":    "{CLIMATE_DATA_PATH}/mean_high_temperature_{ssp_scenario}.parquet",
    "dengue_suitability":       "{CLIMATE_DATA_PATH}/dengue_suitability_{ssp_scenario}.parquet",
    "flooding":                 "/mnt/team/rapidresponse/pub/flooding/results/output/lsae_1209/fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet"
}


####################################

def merge_dataframes(model_df, dfs):
    for key, df in dfs.items():
        model_df = pd.merge(model_df, df, on=["location_id", "year_id"], how="left", suffixes=("", f"_{key}"))
    return model_df

def read_income_paths(income_paths):
    income_dfs = {}
    for key, path in income_paths.items():
        path = path.format(VARIABLE_DATA_PATH=VARIABLE_DATA_PATH)
        income_dfs[key] = pd.read_parquet(path)
        income_dfs[key] = income_dfs[key][income_dfs[key]["scenario"] == rcp_scenario]
        # Drop scenario   
        income_dfs[key] = income_dfs[key].drop(columns=["scenario"], errors='ignore')
    return income_dfs 

def read_urban_paths(urban_paths):
    urban_dfs = {}
    for key, path in urban_paths.items():
        path = path.format(VARIABLE_DATA_PATH=VARIABLE_DATA_PATH)
        urban_dfs[key] = pd.read_parquet(path)
        # Drop population
        urban_dfs[key] = urban_dfs[key].drop(columns=["population"], errors='ignore')
        urban_dfs[key] = urban_dfs[key].rename(columns=lambda x: x.replace("300.0_simple_mean", "300") if "300.0_simple_mean" in x else x)
        urban_dfs[key] = urban_dfs[key].rename(columns=lambda x: x.replace("1500.0_simple_mean", "1500") if "1500.0_simple_mean" in x else x)
        urban_dfs[key] = urban_dfs[key].rename(columns=lambda x: x.replace("100m_urban", "urban_100m") if "100m_urban" in x else x)
        urban_dfs[key] = urban_dfs[key].rename(columns=lambda x: x.replace("1km_urban", "urban_1km") if "1km_urban" in x else x)
        # Remove every instance of "weighted_" from the column names
        urban_dfs[key] = urban_dfs[key].rename(columns=lambda x: x.replace("weighted_", "") if "weighted_" in x else x)
    return urban_dfs

###################################

hierarchy_df = pd.read_parquet(HIERARCHY_PATH)
hierarchy_df = hierarchy_df[
    [
        "location_id",
        "parent_id",
        "level",
        "location_name",
        "path_to_top_parent"
    ]
]
# Subset to level 3 locations and call it A0_hierarchy_df
A0_hierarchy_df = hierarchy_df.loc[hierarchy_df["level"] == 1]
# Find all the unique location_ids in the A0_hierarchy_df
A0_location_ids = A0_hierarchy_df["location_id"].unique()
# Create a new column in hierarchy_df called A0_location_id
hierarchy_df["A0_location_id"] = np.nan
# For every row of hierarchy_df, look at the numbers in the path_to_top_parent, find which one of them is an element of A0_location_ids and place that in the A0_location_id column
# Note: and example path_to_top_parent is 1,311,61467,69489
# In this case, 311 is an element of the A0_location_ids
# So the A0_location_id column will be 311
for i, row in hierarchy_df.iterrows():
    # Split the path_to_top_parent by ","
    path = row["path_to_top_parent"].split(",")
    # Find the first element of path that is in A0_location_ids
    for loc in path:
        if int(loc) in A0_location_ids:
            hierarchy_df.at[i, "A0_location_id"] = int(loc)
            break


# Read the malaria_df parquet file
dengue_df = pd.read_parquet(dengue_stage_2_modeling_df_path)
# Set A0_location_id to integer
dengue_df["A0_location_id"] = dengue_df["A0_location_id"].astype(int)
# Draws

for scenario_number in range(3):
    ssp_scenario = ssp_scenarios[scenario_number]
    rcp_scenario = rcp_scenarios[scenario_number]

    flooding_df_path_template = cc_sensitive_paths['flooding']
    flooding_df_path = flooding_df_path_template.format(ssp_scenario=ssp_scenario)
    # Read in the flooding dataframe
    forecast_df = pd.read_parquet(flooding_df_path)

    # Drop the model, variant, and people_flood_days columns
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
    # Set A0_loctation_id to integer
    forecast_df["A0_location_id"] = forecast_df["A0_location_id"].astype(int)

    # Only keep rows with level == 3
    forecast_df = forecast_df[forecast_df["level"] == 3]
    # Drop the level column
    forecast_df = forecast_df.drop(columns=["level"])

    urban_dfs = read_urban_paths(urban_paths)
    # Merge urban dataframes
    forecast_df = merge_dataframes(forecast_df, urban_dfs)

    income_dfs = read_income_paths(income_paths)
    forecast_df = merge_dataframes(forecast_df, income_dfs)

    covariates_to_log_transform = [
        "gdppc_mean",
        # "ldipc_mean"
    ]
    # Log transform the covariates and save them as new columns with "log_" prefix
    for col in covariates_to_log_transform:
        # Create a new column with the log transformed value
        forecast_df[f"log_{col}"] = np.log(forecast_df[col] + 1e-6)

    # Covariates to logit transform
    covariates_to_logit_transform = [col for col in forecast_df.columns if "urban" in col]

    # Logit transform the covariates and save them as new columns with "logit_" prefix
    for col in covariates_to_logit_transform:
        # Create a new column with the logit transformed value
        # print range of the column
        print(f"Range of {col}: {forecast_df[col].min()} to {forecast_df[col].max()}")
        # Clip values to be strictly between 0 and 1
        clipped_values = forecast_df[col].clip(lower=0.001, upper=0.999)
        forecast_df[f"logit_{col}"] = np.log(clipped_values / (1 - clipped_values))


    # Write the malaria_stage_2_modeling_df to a parquet file
    forecast_df.to_parquet(f"{FORECASTING_DATA_PATH}/dengue_forecast_scenario_{ssp_scenario}_non_draw_part.parquet", index=False)