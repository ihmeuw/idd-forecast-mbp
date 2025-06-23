import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from affine import Affine # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
from shapely import MultiPolygon, Polygon # type: ignore
from typing import Literal, NamedTuple
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import read_parquet_with_integer_ids
import argparse


parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating dengue")

# Define arguments
parser.add_argument("--ssp_scenario", type=str, required=True, help="ssp scenario number (ssp16, ssp245, ssp585")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")

# Parse arguments
args = parser.parse_args()

ssp_scenarios = rfc.ssp_scenarios
ssp_scenario = args.ssp_scenario
draw = args.draw

rcp_scenario = ssp_scenarios[ssp_scenario]["rcp_scenario"]

# Hierarchy
hierarchy = "lsae_1209"
PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"

# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

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

# Draws

ssp_scenario = ssp_scenarios[scenario_number]
rcp_scenario = rcp_scenarios[scenario_number]

forecast_by_draw_df = pd.read_parquet(f"{FORECASTING_DATA_PATH}/dengue_forecast_scenario_{ssp_scenario}_non_draw_part.parquet")

# Add the draw column
forecast_by_draw_df["draw"] = draw

for key, path_template in cc_sensitive_paths.items():
    # Skip the flooding key, as it has already been read
    if key == "flooding":
        continue
    # Replace {ssp_scenario} in the path with the current ssp_scenario
    path = path_template.format(CLIMATE_DATA_PATH=CLIMATE_DATA_PATH, ssp_scenario=ssp_scenario)
    # Read the parquet file
    columns_to_read = ["location_id", "year_id", draw]
    df = pd.read_parquet(path, columns=columns_to_read)
    df = df.rename(columns={draw: key})
    # Merge the file with forecast_by_draw_df
    forecast_by_draw_df = pd.merge(forecast_by_draw_df, df, on=["location_id", "year_id"], how="left")



# Read in dengue_stage_2_modeling_df
dengue_stage_2_modeling_df = pd.read_parquet(dengue_stage_2_modeling_df_path)
# Get the unique values of A0_location_id

columns_to_keep = [
    'location_id',
    'year_id',
    # Add PF columns dynamically
    *[col for col in dengue_stage_2_modeling_df.columns if 'dengue' in col and 'suitability' not in col],
]
dengue_stage_2_modeling_df = dengue_stage_2_modeling_df[columns_to_keep]
# Make a "stage_2" column that is all 1s
dengue_stage_2_modeling_df["stage_2"] = 1

# Merge in the dengue_stage_2_modeling_df
forecast_by_draw_df = forecast_by_draw_df.merge(
    dengue_stage_2_modeling_df,
    how="left",
    on=["location_id", "year_id"]
)

# Write the dengue_stage_2_modeling_df to a parquet file
forecast_by_draw_df.to_parquet(f"{FORECASTING_DATA_PATH}/dengue_forecast_scenario_{ssp_scenario}_draw_{draw}.parquet", index=False)