import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
from typing import Literal, NamedTuple
import itertools
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import merge_dataframes, read_income_paths, read_urban_paths, level_filter
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet

import argparse
parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--ssp_scenario", type=str, required=True, help="ssp scenario number (ssp16, ssp245, ssp585")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")

# Parse arguments
args = parser.parse_args()


ssp_scenario = args.ssp_scenario
draw = args.draw
# ssp_scenario = "ssp245"
# draw = "001"

ssp_scenarios = rfc.ssp_scenarios
rcp_scenario = ssp_scenarios[ssp_scenario]["rcp_scenario"]

malaria_mortality_threshold = 1

# Hierarchy
hierarchy = "lsae_1209"
PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"


cause = "malaria"

cause_map = rfc.cause_map
reference_age_group_id = cause_map[cause]['reference_age_group_id']
reference_sex_id = cause_map[cause]['reference_sex_id']


forecast_non_draw_df_path = f"{FORECASTING_DATA_PATH}/{cause}_forecast_scenario_{ssp_scenario}_non_draw_part.parquet"
forecast_by_draw_df_path_template = "{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}.parquet"
dah_scenario_df_path_template = "{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet"
aa_merge_variables = rfc.aa_merge_variables

# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

# LSAE 1209 variable path
VARIABLE_DATA_PATH = f"{PROCESSED_DATA_PATH}/{hierarchy}"
# CLIMATE 1209 variable path
CLIMATE_DATA_PATH = f"/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/{hierarchy}"

# Malaria modeling dataframes
aa_full_cause_df_path_template = f'{PROCESSED_DATA_PATH}/aa_full_{cause}_df.parquet'
as_full_cause_df_path_template = f'{PROCESSED_DATA_PATH}/as_full_{cause}_df.parquet'

# Climate variables
cc_sensitive_paths = {
    "total_precipitation":      "{CLIMATE_DATA_PATH}/total_precipitation_{ssp_scenario}.parquet",
    "relative_humidity":        "{CLIMATE_DATA_PATH}/relative_humidity_{ssp_scenario}.parquet",
    "malaria_suitability":      "{CLIMATE_DATA_PATH}/malaria_suitability_{ssp_scenario}.parquet",
}

####################################

def generate_dah_scenarios(
    baseline_df,
    ssp_scenario,
    year_start = 2000,
    reference_year=2023,
    modification_start_year=2026,
    dah_scenario_names=None
):
    """
    Generate DAH funding scenarios for malaria forecasting.
    
    This function creates four scenarios:
    1. Baseline: Original DAH projections
    2. Constant: Holds DAH at reference_year levels for future years
    3. Increasing: Progressively increases DAH (1.2, 1.4, 1.6, 1.8, 2.0)
    4. Decreasing: Progressively decreases DAH (0.8, 0.6, 0.4, 0.2, 0.0)
    
    Parameters
    ----------
    ssp_scenarios : list
        List of SSP climate scenario names
    draw : str
        Draw identifier
    forecasting_data_path : str
        Path to forecasting data
    malaria_forecasting_df_path : str
        Template for malaria forecasting data path with format parameters
    hierarchy_df : pandas.DataFrame
        Hierarchy data with location and super region information
    year_start : int
        Starting year for filtering data
    reference_year : int, optional
        Year to use as reference for constant scenario, default 2023
    modification_start_year : int, optional
        Year to start the increasing/decreasing changes, default 2026
    scenario_names : list, optional
        Custom names for the scenarios, default ['Baseline', 'Constant', 'Increasing', 'Decreasing']
    
    Returns
    -------
    tuple
        (dah_scenarios, dah_scenario_names) - List of scenario DataFrame lists and scenario names
    """
    # Set default scenario names if not provided
    if dah_scenario_names is None:
        dah_scenario_names = ['Baseline', 'Constant', 'Increasing', 'Decreasing']
    
    # Define modification schedules
    increasing_factors = {
        modification_start_year: 1.2,
        modification_start_year + 1: 1.4,
        modification_start_year + 2: 1.6,
        modification_start_year + 3: 1.8,
        modification_start_year + 4: 2.0
    }
    
    decreasing_factors = {
        modification_start_year: 0.8,
        modification_start_year + 1: 0.6,
        modification_start_year + 2: 0.4,
        modification_start_year + 3: 0.2,
        modification_start_year + 4: 0.0
    }

    # Process baseline data
    baseline_df = baseline_df.copy() 
    baseline_df = baseline_df[baseline_df['year_id'] >= year_start]
    baseline_df["A0_location_id"] = baseline_df["A0_location_id"].astype(int)
    baseline_df['A0_af'] = 'A0_' + baseline_df['A0_location_id'].astype(str)
      
    baseline_df['ssp_scenario'] = ssp_scenario
    baseline_df['dah_scenario'] = 'Baseline'
    
    # Create Scenario 1: Constant DAH at reference year level
    print(f"  Scenario 1: Constant DAH")
    scenario_1_df = baseline_df.copy()
    scenario_1_df['mal_DAH_total'] = scenario_1_df['mal_DAH_total_per_capita'] * scenario_1_df['aa_population']
    
    # Get reference year values and merge
    values_ref_year = scenario_1_df[scenario_1_df['year_id'] == reference_year][['location_id', 'mal_DAH_total']]
    scenario_1_df = scenario_1_df.merge(values_ref_year, on='location_id', suffixes=('', f'_{reference_year}'))
    
    # Replace future values with reference year values
    mask = scenario_1_df['year_id'] >= reference_year + 1
    scenario_1_df.loc[mask, 'mal_DAH_total'] = scenario_1_df.loc[mask, f'mal_DAH_total_{reference_year}']
    
    # Drop temporary column
    scenario_1_df = scenario_1_df.drop(columns=f'mal_DAH_total_{reference_year}')
    
    # Recalculate per-capita values
    scenario_1_df['mal_DAH_total_per_capita'] = scenario_1_df['mal_DAH_total'] / scenario_1_df['aa_population']

    scenario_1_df['log_mal_DAH_total_per_capita'] = np.log(scenario_1_df['mal_DAH_total_per_capita'] + 1e-6)
    scenario_1_df['ssp_scenario'] = ssp_scenario
    scenario_1_df['dah_scenario'] = 'Constant'
    
    # Create Scenario 2: Increasing DAH
    print(f"  Scenario 2: Increasing DAH")
    scenario_2_df = baseline_df.copy()
    
    # Apply increasing factors to specific years
    for year, factor in increasing_factors.items():
        mask = scenario_2_df['year_id'] == year
        scenario_2_df.loc[mask, 'mal_DAH_total_per_capita'] *= factor
        scenario_2_df.loc[mask, 'mal_DAH_total'] *= factor
    
    # Apply maximum factor to all later years
    max_factor = max(increasing_factors.values())
    max_year = max(increasing_factors.keys())
    mask = scenario_2_df['year_id'] > max_year
    scenario_2_df.loc[mask, 'mal_DAH_total_per_capita'] *= max_factor
    scenario_2_df.loc[mask, 'mal_DAH_total'] = scenario_2_df.loc[mask, 'mal_DAH_total_per_capita'] * scenario_2_df.loc[mask, 'aa_population']
    
    # Recalculate derived values
    scenario_2_df['log_mal_DAH_total_per_capita'] = np.log(scenario_2_df['mal_DAH_total_per_capita'] + 1e-6)
    scenario_2_df['ssp_scenario'] = ssp_scenario
    scenario_2_df['dah_scenario'] = 'Increasing'
    
    # Create Scenario 3: Decreasing DAH
    print(f"  Scenario 3: Decreasing DAH")
    scenario_3_df = baseline_df.copy()
    
    # Apply decreasing factors to specific years
    for year, factor in decreasing_factors.items():
        mask = scenario_3_df['year_id'] == year
        scenario_3_df.loc[mask, 'mal_DAH_total_per_capita'] *= factor
        scenario_3_df.loc[mask, 'mal_DAH_total'] *= factor
    
    # Apply minimum factor to all later years
    min_factor = min(decreasing_factors.values())
    max_year = max(decreasing_factors.keys())
    mask = scenario_3_df['year_id'] > max_year
    scenario_3_df.loc[mask, 'mal_DAH_total_per_capita'] = min_factor  # Use 0 or min_factor
    scenario_3_df.loc[mask, 'mal_DAH_total'] = min_factor * scenario_3_df.loc[mask, 'aa_population']
    
    # Recalculate derived values
    scenario_3_df['log_mal_DAH_total_per_capita'] = np.log(scenario_3_df['mal_DAH_total_per_capita'] + 1e-6)
    scenario_3_df['ssp_scenario'] = ssp_scenario
    scenario_3_df['dah_scenario'] = 'Decreasing'
    
    # Group all scenarios
    dah_scenarios = [baseline_df, scenario_1_df, scenario_2_df, scenario_3_df]    
    return dah_scenarios, dah_scenario_names

aa_malaria_df = read_parquet_with_integer_ids(aa_full_cause_df_path_template,
    filters=[level_filter(hierarchy_df, start_level = 3, end_level = 5)])

aa_malaria_df = aa_malaria_df.merge(hierarchy_df[['location_id', 'A0_location_id', 'level']],
    how="left",
    on="location_id")

aa_A0_malaria_df = aa_malaria_df[(aa_malaria_df["location_id"] == aa_malaria_df["A0_location_id"]) & (aa_malaria_df["year_id"] == 2022)].copy()
aa_A0_malaria_df = aa_A0_malaria_df[aa_A0_malaria_df['malaria_mort_count'] > malaria_mortality_threshold].copy()
A0_malaria_ids = aa_A0_malaria_df['A0_location_id'].unique()

aa_malaria_df = aa_malaria_df[aa_malaria_df['A0_location_id'].isin(A0_malaria_ids)].copy()
aa_malaria_df = aa_malaria_df[
    (aa_malaria_df["malaria_pfpr"] > 0) &
    (aa_malaria_df["malaria_mort_count"] > 0) &
    (aa_malaria_df["malaria_inc_count"] >= 0) &
    (aa_malaria_df["level"] == 5)].copy()

aa_malaria_ids = aa_malaria_df['location_id'].unique()
aa_malaria_filter = ('location_id', 'in', aa_malaria_ids.tolist())

reference_age_group_filter = ('age_group_id', '==', reference_age_group_id)
reference_sex_filter = ('sex_id', '==', reference_sex_id)
as_base_malaria_df = read_parquet_with_integer_ids(as_full_cause_df_path_template,
    filters=[reference_age_group_filter, reference_sex_filter, aa_malaria_filter]).drop(columns=['age_group_id', 'sex_id', 'aa_population'])

as_base_malaria_df = as_base_malaria_df.rename(columns=lambda x: f"base_{x}" if (x.startswith('malaria_') or x.startswith('pop_')) else x)
as_base_malaria_df = as_base_malaria_df.merge(
    aa_malaria_df[aa_merge_variables + ['malaria_pfpr']],
    how="left",
    on=aa_merge_variables)

covariates_to_log_transform = [col for col in as_base_malaria_df.columns if 'rate' in col]
for col in covariates_to_log_transform:
    # Create a new column with the log transformed value
    as_base_malaria_df[f"log_{col}"] = np.log(as_base_malaria_df[col])

as_base_malaria_df[f"logit_malaria_pfpr"] = np.log(0.999 * as_base_malaria_df["malaria_pfpr"] / (1 - 0.999 * as_base_malaria_df["malaria_pfpr"]))

forecast_by_draw_df = read_parquet_with_integer_ids(forecast_non_draw_df_path,
    filters=[aa_malaria_filter])

# Add the draw column
forecast_by_draw_df["draw"] = draw
forecast_by_draw_df = forecast_by_draw_df.rename(columns={
    'population': 'aa_population'})

forecast_by_draw_df = forecast_by_draw_df.merge(as_base_malaria_df, 
    how='left',
    on=['location_id','year_id'])

for key, path_template in cc_sensitive_paths.items():
    # Replace {ssp_scenario} in the path with the current ssp_scenario
    path = path_template.format(CLIMATE_DATA_PATH=CLIMATE_DATA_PATH, ssp_scenario=ssp_scenario)
    # Read the parquet file
    columns_to_read = ["location_id", "year_id", draw]
    df = read_parquet_with_integer_ids(path, columns=columns_to_read,
        filters=[aa_malaria_filter])
    df = df.rename(columns={draw: key})
    # Merge the file with forecast_by_draw_df
    forecast_by_draw_df = pd.merge(forecast_by_draw_df, df, on=["location_id", "year_id"], how="left")

covariates_to_log_transform = [
    "mal_DAH_total_per_capita",
    "gdppc_mean",
]

for col in covariates_to_log_transform:
    # Create a new column with the log transformed value
    forecast_by_draw_df[f"log_{col}"] = np.log(forecast_by_draw_df[col] + 1e-6)

pakistan_id = 165
pakistan_children_ids = hierarchy_df[hierarchy_df['parent_id'] == pakistan_id]['location_id'].tolist()
pakistan_grandchildren_ids = hierarchy_df[hierarchy_df['parent_id'].isin(pakistan_children_ids)]['location_id'].tolist()
# Combine all Pakistan-related location IDs
all_pakistan_locations = [pakistan_id] + pakistan_children_ids + pakistan_grandchildren_ids

forecast_by_draw_df['year_to_rake_to'] = 2022
forecast_by_draw_df.loc[forecast_by_draw_df['location_id'].isin(all_pakistan_locations), 'year_to_rake_to'] = 2021

dah_scenarios, dah_scenario_names = generate_dah_scenarios(
    baseline_df=forecast_by_draw_df,
    ssp_scenario=ssp_scenario
)

for dah_scenario_df, dah_scenario_name in zip(dah_scenarios, dah_scenario_names):
    # Write each scenario to a parquet file
    scenario_df = dah_scenario_df.copy()

    dah_scenario_df_path = dah_scenario_df_path_template.format(
        FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
        cause=cause,
        ssp_scenario=ssp_scenario,
        dah_scenario_name=dah_scenario_name,
        draw=draw
    )
    write_parquet(dah_scenario_df, dah_scenario_df_path)

# Write the malaria_stage_2_modeling_df to a parquet file
forecast_by_draw_df_path = forecast_by_draw_df_path_template.format(
    FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
    cause=cause,
    ssp_scenario=ssp_scenario,
    draw=draw
)
write_parquet(forecast_by_draw_df, forecast_by_draw_df_path)