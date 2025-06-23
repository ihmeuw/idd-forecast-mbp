import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from affine import Affine # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
from typing import Literal, NamedTuple
import itertools
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import read_parquet_with_integer_ids
import argparse

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=True, help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--measure", type=str, required=False, default="mortality", help="measure (e.g., 'mortality', 'incidence')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")

# Parse arguments
args = parser.parse_args()

measure = args.measure
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
draw = args.draw

measure_map = rfc.measure_map

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data"

forecast_df_path = f"{FORECASTING_DATA_PATH}/as_malaria_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.parquet"
processed_forecast_df_path_template = f"{UPLOAD_DATA_PATH}/full_as_malaria_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.parquet"


# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)



high_level_hierarchy_df = hierarchy_df[hierarchy_df["level"] > 3].copy()
low_level_hierarchy_df = hierarchy_df[hierarchy_df["level"] <= 3].copy()




def process_forecast_data(forecast_df_path, measure, hierarchy_df):
    """
    Process forecast data, adding necessary columns and formatting.
    
    Parameters:
    -----------
    ssp_scenario : str
        SSP scenario name
    dah_scenario : str
        DAH scenario name
    draw : str
        Draw identifier
    hierarchy_df : pandas.DataFrame
        Hierarchy dataframe for location information
    Returns:
    --------
    pandas.DataFrame
        Full hierarchy forecast dataframe with all necessary columns and aggregations applied.
    """
    df = read_parquet_with_integer_ids(forecast_df_path)

    short = measure_map[measure]["short"]

    df = df[df["year_id"] >= 2022]
    # Drop all columns with short in their name
    df = df.drop(columns=[col for col in df.columns if short in col], errors='ignore')
    df = df.drop(columns=["reference_age_group_id", "reference_sex_id", "relative_risk_as", "rate_pred",
                                            "population_aa", "pop_fraction_aa", "fhs_location_id", "location_name"])


    df = df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")

    child_df = df.copy()

    for level in reversed(range(1,6)):
        
        print(f"Processing level {level}...")
        child_df = child_df.merge(hierarchy_df[["location_id", "parent_id"]], on="location_id", how="left")
        print(child_df["level"][0])
        parent_df = child_df.groupby(
            ["parent_id", "year_id", "age_group_id", "sex_id"]).agg({
            "count_pred": "sum"
        }).reset_index()

        parent_df = parent_df.rename(columns={
            "parent_id": "location_id"
        })

        parent_df = parent_df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
        df = pd.concat([df, parent_df], ignore_index=True)

        child_df = parent_df.copy()

    return df

# Process the forecast data
print(f"Running for measure: {measure}, ssp_scenario: {ssp_scenario}, dah_scenario: {dah_scenario}, draw: {draw}")

full_hierarchy_forecast_df = process_forecast_data(forecast_df_path, measure, hierarchy_df)

low_level_df = full_hierarchy_forecast_df[full_hierarchy_forecast_df["level"] <= 3].copy()
high_level_df = full_hierarchy_forecast_df[full_hierarchy_forecast_df["level"] > 3].copy()

########## Merge populations
##### Low level populations
future_fhs_population_path = "/mnt/share/forecasting/data/9/future/population/20250606_first_sub_rcp45_climate_ref_100d_hiv_shocks_covid_all/summary/summary.nc"
future_fhs_population = xr.open_dataset(future_fhs_population_path)
# The actual data is in the 'draws' variable, select the 'mean' statistic
fhs_pop_data = future_fhs_population['draws'].sel(statistic='mean')
future_fhs_population_df = fhs_pop_data.to_dataframe().reset_index()
future_fhs_population_df = future_fhs_population_df.rename(columns={"draws": "population"})
future_fhs_population_df = future_fhs_population_df.drop(columns=["scenario"], errors='ignore')
# Drop the 'statistic' column as it is no longer needed
future_fhs_population_df = future_fhs_population_df.drop(columns=["statistic"], errors='ignore')
future_fhs_population_df = future_fhs_population_df[future_fhs_population_df["location_id"].isin(low_level_hierarchy_df["location_id"])].copy()

low_level_df = low_level_df.merge(
    future_fhs_population_df[["location_id", "age_group_id", "sex_id", "year_id", "population"]],
    on=["location_id", "age_group_id", "sex_id", "year_id"],
    how="left"
)

# Remove future_fhs_population_df as it is no longer needed
del future_fhs_population_df

high_level_location_filter = ('location_id', 'in', high_level_hierarchy_df['location_id'].unique().tolist())
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
as_lsae_population_df_path = f"{MODELING_DATA_PATH}/as_lsae_population_df.parquet"
as_lsae_population_df = read_parquet_with_integer_ids(as_lsae_population_df_path,
                                                       filters=[[high_level_location_filter]])


high_level_df = high_level_df.merge(
    as_lsae_population_df[["location_id", "age_group_id", "sex_id", "year_id", "population"]],
    on=["location_id", "age_group_id", "sex_id", "year_id"],
    how="left"
)

# Remove as_lsae_population_df as it is no longer needed
del as_lsae_population_df

full_hierarchy_forecast_df = pd.concat([high_level_df, low_level_df], ignore_index=True)

# Sort to maintain original order if needed
full_hierarchy_forecast_df = full_hierarchy_forecast_df.sort_values(
    ["location_id", "year_id", "age_group_id", "sex_id"]
).reset_index(drop=True)

# Save the processed dataframe

full_hierarchy_forecast_df.to_parquet(
    processed_forecast_df_path_template, compression="snappy",
    index=False
)