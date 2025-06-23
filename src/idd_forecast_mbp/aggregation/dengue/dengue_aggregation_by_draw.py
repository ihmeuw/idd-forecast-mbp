import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from affine import Affine # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
from shapely import MultiPolygon, Polygon # type: ignore
from typing import Literal, NamedTuple
import itertools
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import load_yaml_dictionary, parse_yaml_dictionary
import argparse

#
coverage = 0.9
efficacy = 0.844

parser = argparse.ArgumentParser(description="Aggregate dengue up to full hierarchy by draw")

# Define arguments
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")

# Parse arguments
args = parser.parse_args()

ssp_scenario = args.ssp_scenario
draw = args.draw

# ssp_sceanro = "ssp245"
# dah_scenario = "Baseline"
# draw = "000"


# Hierarchy
lsae_hierarchy = "lsae_1209"
fhs_hierarchy = "fhs_2021"

# MODELING_DATA_PATH 
MODELING_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"
FORECASTING_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data"
UPLOAD_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/05-upload_data"

# Population
FHS_population_path = "/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/fhs_2021/population.parquet"
LSAE_population_path = "/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/lsae_1209/population.parquet"

# Hierarchy path
HIERARCHY_PATH = "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_{hierarchy}.parquet"
hierarchy_df_path = f"{FORECASTING_DATA_PATH}/hierarchy_{lsae_hierarchy}_full.parquet"

forecast_df_path = "{FORECASTING_DATA_PATH}/dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet"
processed_forecast_df_path = "{UPLOAD_DATA_PATH}/full_hierarchy_dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet"


hierarchy_df = pd.read_parquet(hierarchy_df_path)

locations = ['Singapore', 'Brazil', 'Indonesia', 'Thailand']
location_rows = hierarchy_df[hierarchy_df['location_name'].isin(locations)]
children_rows = hierarchy_df[hierarchy_df['parent_id'].isin(location_rows['location_id'].unique())]
grand_children_rows = hierarchy_df[hierarchy_df['parent_id'].isin(children_rows['location_id'].unique())]

location_rows = location_rows[['location_id', 'location_name']]
location_rows.rename(columns={'location_id': 'grandparent_id', 'location_name': 'grandparent_name'}, inplace=True)

children_rows = children_rows[['location_id', 'location_name', 'parent_id']]
children_rows.rename(columns={'location_id': 'parent_id', 'location_name': 'parent_name', 'parent_id': 'grandparent_id'}, inplace=True)
children_rows = pd.merge(children_rows, location_rows, on='grandparent_id', how='left')

grand_children_rows = grand_children_rows[['location_id', 'location_name', 'parent_id']]
grand_children_rows = pd.merge(grand_children_rows, children_rows[['parent_id', 'parent_name', 'grandparent_id', 'grandparent_name']], on='parent_id', how='left')

# Combine the location IDs and their children IDs
cohort_data_path = f"{FORECASTING_DATA_PATH}/expanding_cohort_fraction_all_locations.csv"
cohort_df = pd.read_csv(cohort_data_path)

# Population
lsae_population_df = pd.read_parquet(LSAE_population_path)
# Rename population to population_total
lsae_population_df.rename(columns={'population': 'population_total'}, inplace=True)
fhs_population_df = pd.read_parquet(FHS_population_path)
# Rename population to population_total
fhs_population_df.rename(columns={'population': 'population_total'}, inplace=True)

def process_forecast_data(ssp_scenario, draw, hierarchy_df, lsae_population_df, fhs_population_df):
    """
    Process forecast data, adding necessary columns and formatting.
    
    Parameters:
    -----------
    ssp_scenario : str
        SSP scenario name
    draw : str
        Draw identifier
    hierarchy_df : pandas.DataFrame
        Hierarchy dataframe for location information
    lsae_population_df : pandas.DataFrame
        Population dataframe for LSAE hierarchy
    fhs_population_df : pandas.DataFrame
        Population dataframe for FHS hierarchy
    Returns:
    --------
    pandas.DataFrame
        Full hierarchy forecast dataframe with all necessary columns and aggregations applied.
    """
    # # Load the forecast data
    df = pd.read_parquet(forecast_df_path.format(
        FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
        ssp_scenario=ssp_scenario,
        draw=draw
    ))

    df['ssp_scenario'] = ssp_scenario
    ## Process the children for the first time
    # Get natural space outcomes
    df['dengue_inc_rate_pred'] = np.exp(df['log_dengue_inc_rate_pred'])
    df['dengue_cfr_pred'] = df['logit_dengue_cfr_pred'].apply(lambda x: 1 / (1 + np.exp(-x)))
    # Get counts
    df['dengue_inc_count_pred'] = df['dengue_inc_rate_pred'] * df['population']
    df['dengue_mort_count_pred'] = df['dengue_cfr_pred'] * df['dengue_inc_count_pred']

    # Drop all the columns that are not needed for the children dataframe
    df = df.drop(['scenario', 'parent_id', 'location_name', 'path_to_top_parent', 'A0_af',
                'people_flood_days','ldipc_mean','gdppc_mean','dengue_mort_rate',
                'people_flood_days_per_capita','A0_location_id','urban_1km_threshold_300', 'urban_100m_threshold_300',
                'urban_1km_threshold_1500', 'urban_100m_threshold_1500','total_precipitation',
                'relative_humidity', 'mean_temperature', 'mean_high_temperature','dengue_suitability',
                'log_gdppc_mean', 'logit_urban_1km_threshold_300', 'logit_urban_100m_threshold_300', 
                'dengue_inc_rate_pred', 'dengue_cfr_pred','dengue_inc_rate',
                'logit_urban_1km_threshold_1500', 'logit_urban_100m_threshold_1500',
                'dengue_cfr', 'log_dengue_mort_rate', 'log_dengue_inc_rate', 'logit_dengue_cfr',
                'stage_2', 'in_model', 'log_dengue_inc_rate_pred', 'logit_dengue_cfr_pred'], axis=1)
    # Update forecast_columns list
    df_columns = list(df.columns)
    df_count_columns = [col for col in df_columns if 'count' in col]
    df[df_count_columns] = df[df_count_columns].fillna(0)

    # Merge with hierarchy info
    df = df.merge(hierarchy_df, how="left", on=["location_id"])
    df = df.dropna(subset=['level'])
    df['level'] = df['level'].astype('int')
    df['population_total'] = df['population']
    df_columns = [col for col in df_columns if 'population' not in col]
    
    print("Processing grandchildren...")
    # Vaccinate grandchildren
    for grandchild_row in grand_children_rows.itertuples():
        grandchild = grandchild_row.location_id
        grandparent_id = grandchild_row.grandparent_id
        # Get the correct rows of cohort_df
        cohort_rows = cohort_df[cohort_df['location_id'] == grandparent_id]
        tmp_years = df['year_id'].unique()
        tmp_years = tmp_years[tmp_years >= 2026]
        
        for vac_year in tmp_years:
            # Find the row of df that corresponds to the grandchild location and year
            grandchild_row_df = df[(df['location_id'] == grandchild) & (df['year_id'] == vac_year)]
            if grandchild_row_df.empty:
                # If no row exists for this grandchild and year, skip to the next iteration
                continue
            
            # Find the year of cohort_rows where year = vac_year - 2026 + 1
            year = vac_year - 2026 + 1
            
            # Get the cohort fraction for the grandparent location and year
            cohort_fraction = cohort_rows[cohort_rows['year'] == year]['fraction_of_total'].values
            
            if len(cohort_fraction) > 0:
                cohort_fraction = cohort_fraction[0] * coverage * efficacy
                
                # Update the SPECIFIC grandchild rows in df (both location AND year)
                mask = (df['location_id'] == grandchild) & (df['year_id'] == vac_year)
                df.loc[mask, 'dengue_mort_count_pred'] *= (1 - cohort_fraction)

    print("Done vaccinating grandchildren.")

    # Start with children dataframe and get the top level (assumes sorted levels)
    children_df = df.copy()
    children_level = children_df['level'].iloc[0]

    # Loop from highest children level down to 0
    print("Processing level: ", end="")
    for level in reversed(range(0, children_level)):
        if level > 0:
            print(f"{level}, ", end = "")
        else:
            print("0")

        parent_df = children_df.groupby(['year_id', 'parent_id']).agg(
            {col: 'sum' for col in df_count_columns}
        ).reset_index()
       
        # Add extra columns
        parent_df['draw'] = draw
        parent_df['ssp_scenario'] = ssp_scenario
        
        # Rename parent_id to location_id
        parent_df.rename(columns={'parent_id': 'location_id'}, inplace=True)
        
        # Merge with hierarchy_df to add location_name and path_to_top_parent
        parent_df = parent_df.merge(hierarchy_df, how="left", on=["location_id"])
        
        # Merge with population dataframe based on level
        if parent_df['level'].max() >= 3:
            parent_df = parent_df.merge(lsae_population_df, how="left", on=["location_id", "year_id"])
        else:
            parent_df = parent_df.merge(fhs_population_df, how="left", on=["location_id", "year_id"])
        
        # Add missing columns with the correct dtype from df
        missing_columns = [col for col in df.columns if col not in parent_df.columns]
        for col in missing_columns:
            col_dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(col_dtype):
                col_dtype = 'Int64'
            parent_df[col] = pd.Series(np.nan, index=parent_df.index, dtype=col_dtype)
        # Reorder columns to match df
        parent_df = parent_df[df.columns]
        
        # Update children and concatenated df
        children_df = parent_df.copy()
        df = pd.concat([df, parent_df], ignore_index=True)
    
    # Reset index and ensure all columns are present
    df.reset_index(drop=True, inplace=True)
    
    return df


print(f"Processing forecast data for SSP scenario: {ssp_scenario}, Draw: {draw}")
# Process the forecast data
full_hierarchy_forecast_df = process_forecast_data(ssp_scenario, draw, hierarchy_df, lsae_population_df, fhs_population_df)


# Save the processed dataframe
full_hierarchy_forecast_df.to_parquet(
    processed_forecast_df_path.format(
        UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
        ssp_scenario=ssp_scenario,
        draw=draw
    ),
    index=False
)
