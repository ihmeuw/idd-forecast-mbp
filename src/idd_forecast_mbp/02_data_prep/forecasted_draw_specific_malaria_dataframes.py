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
from idd_forecast_mbp.helper_functions import read_parquet_with_integer_ids, merge_dataframes, read_income_paths, write_parquet
import argparse


parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

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


cause = "malaria"
aa_md_modeling_df_path = f"{MODELING_DATA_PATH}/aa_md_malaria_pfpr_modeling_df.parquet"
base_md_modeling_df_path = f"{MODELING_DATA_PATH}/base_md_{cause}_modeling_df.parquet"
forecast_non_draw_df_path = f"{FORECASTING_DATA_PATH}/{cause}_forecast_scenario_{ssp_scenario}_non_draw_part.parquet"
forecast_by_draw_df_path_template = "{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}.parquet"
dah_scenario_df_path_template = "{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet"
aa_merge_variables = ["location_id", "year_id"]

# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

# LSAE 1209 variable path
VARIABLE_DATA_PATH = f"{PROCESSED_DATA_PATH}/{hierarchy}"
# CLIMATE 1209 variable path
CLIMATE_DATA_PATH = f"/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/{hierarchy}"

# Malaria modeling dataframes
malaria_stage_2_modeling_df_path = f"{MODELING_DATA_PATH}/malaria_stage_2_modeling_df.parquet"


# Climate variables
cc_sensitive_paths = {
    "total_precipitation":      "{CLIMATE_DATA_PATH}/total_precipitation_{ssp_scenario}.parquet",
    "malaria_suitability":      "{CLIMATE_DATA_PATH}/malaria_suitability_{ssp_scenario}.parquet",
}

####################################

def generate_dah_scenarios(
    baseline_df,
    ssp_scenario,
    malaria_stage_2_modeling_df_path,
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
    malaria_location_ids : list or array
        Location IDs to include
    malaria_df : pandas.DataFrame
        Reference malaria dataframe for category alignment
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
    
    # Read the malaria_df parquet file
    malaria_df = read_parquet_with_integer_ids(malaria_stage_2_modeling_df_path)
    # Set A0_location_id to integer
    malaria_df["A0_location_id"] = malaria_df["A0_location_id"].astype(int)
    # Create the A0_af column as categorical
    malaria_df['A0_af'] = 'A0_' + malaria_df['A0_location_id'].astype(str)
    malaria_df['A0_af'] = malaria_df['A0_af'].astype('category')
    year_start = malaria_df["year_id"].min()
    year_end = malaria_df["year_id"].max()
    malaria_location_ids = malaria_df["location_id"].unique()

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
    baseline_df = baseline_df[baseline_df['location_id'].isin(malaria_location_ids)]

    # Create factor variables - ensure both are categorical before setting categories
    baseline_df['A0_af'] = 'A0_' + baseline_df['A0_location_id'].astype(str)
    baseline_df['A0_af'] = baseline_df['A0_af'].astype('category')
    
    # Get the categories from malaria_df and apply to baseline_df
    malaria_categories = malaria_df['A0_af'].cat.categories
    baseline_df['A0_af'] = baseline_df['A0_af'].cat.set_categories(malaria_categories)
      
    baseline_df['ssp_scenario'] = ssp_scenario
    baseline_df['dah_scenario'] = 'Baseline'
    
    # Create Scenario 1: Constant DAH at reference year level
    print(f"  Scenario 1: Constant DAH")
    scenario_1_df = baseline_df.copy()
    scenario_1_df['mal_DAH_total'] = scenario_1_df['mal_DAH_total_per_capita'] * scenario_1_df['population']
    
    # Get reference year values and merge
    values_ref_year = scenario_1_df[scenario_1_df['year_id'] == reference_year][['location_id', 'mal_DAH_total']]
    scenario_1_df = scenario_1_df.merge(values_ref_year, on='location_id', suffixes=('', f'_{reference_year}'))
    
    # Replace future values with reference year values
    mask = scenario_1_df['year_id'] >= reference_year + 1
    scenario_1_df.loc[mask, 'mal_DAH_total'] = scenario_1_df.loc[mask, f'mal_DAH_total_{reference_year}']
    
    # Drop temporary column
    scenario_1_df = scenario_1_df.drop(columns=f'mal_DAH_total_{reference_year}')
    
    # Recalculate per-capita values
    scenario_1_df['mal_DAH_total_per_capita'] = scenario_1_df['mal_DAH_total'] / scenario_1_df['population']

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
    scenario_2_df.loc[mask, 'mal_DAH_total'] = scenario_2_df.loc[mask, 'mal_DAH_total_per_capita'] * scenario_2_df.loc[mask, 'population']
    
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
    scenario_3_df.loc[mask, 'mal_DAH_total'] = min_factor * scenario_3_df.loc[mask, 'population']
    
    # Recalculate derived values
    scenario_3_df['log_mal_DAH_total_per_capita'] = np.log(scenario_3_df['mal_DAH_total_per_capita'] + 1e-6)
    scenario_3_df['ssp_scenario'] = ssp_scenario
    scenario_3_df['dah_scenario'] = 'Decreasing'
    
    # Group all scenarios
    dah_scenarios = [baseline_df, scenario_1_df, scenario_2_df, scenario_3_df]    
    return dah_scenarios, dah_scenario_names

###################################

forecast_by_draw_df = read_parquet_with_integer_ids(forecast_non_draw_df_path)

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
    df = read_parquet_with_integer_ids(path, columns=columns_to_read)
    df = df.rename(columns={draw: key})
    # Merge the file with forecast_by_draw_df
    forecast_by_draw_df = pd.merge(forecast_by_draw_df, df, on=["location_id", "year_id"], how="left")

covariates_to_log_transform = [
    "mal_DAH_total_per_capita",
    "gdppc_mean",
]
# Log transform the covariates and save them as new columns with "log_" prefix
for col in covariates_to_log_transform:
    # Create a new column with the log transformed value
    forecast_by_draw_df[f"log_{col}"] = np.log(forecast_by_draw_df[col] + 1e-6)
# Read in malaria_stage_2_modeling_df


years = list(range(2021, 2023))
year_filter = ('year_id', 'in', years)
pakistan_id = 165
pakistan_children_ids = hierarchy_df[hierarchy_df['parent_id'] == pakistan_id]['location_id'].tolist()
pakistan_grandchildren_ids = hierarchy_df[hierarchy_df['parent_id'].isin(pakistan_children_ids)]['location_id'].tolist()
# Combine all Pakistan-related location IDs
all_pakistan_locations = [pakistan_id] + pakistan_children_ids + pakistan_grandchildren_ids




aa_columns_to_read = ['location_id', 'year_id', 'logit_malaria_pfpr']
aa_md_modeling_df = read_parquet_with_integer_ids(aa_md_modeling_df_path,
                                                  columns = aa_columns_to_read,
                                                  filters=[year_filter])

# Split the base_md_modeling_df based on Pakistan locations
pakistan_df = aa_md_modeling_df[aa_md_modeling_df['location_id'].isin(all_pakistan_locations)].copy()
non_pakistan_df = aa_md_modeling_df[~aa_md_modeling_df['location_id'].isin(all_pakistan_locations)].copy()

# Filter by years and set year_to_rake
pakistan_2021 = pakistan_df[pakistan_df['year_id'] == 2021].copy()
non_pakistan_2022 = non_pakistan_df[non_pakistan_df['year_id'] == 2022].copy()

aa_md_modeling_df = pd.concat([pakistan_2021, non_pakistan_2022], ignore_index=True)
aa_md_modeling_df = aa_md_modeling_df.drop(columns=['year_id'])

# Continue with existing merge logic
forecast_by_draw_df = forecast_by_draw_df.merge(
    aa_md_modeling_df,
    how="left",
    on=["location_id"]
)



base_columns_to_read = ['location_id', 'year_id', 'base_logit_malaria_pfpr', 'base_log_malaria_mort_rate', 'base_log_malaria_inc_rate']
base_md_modeling_df = read_parquet_with_integer_ids(base_md_modeling_df_path,
                                                    columns = base_columns_to_read,
                                                    filters=[year_filter])

# Split the base_md_modeling_df based on Pakistan locations
pakistan_df = base_md_modeling_df[base_md_modeling_df['location_id'].isin(all_pakistan_locations)].copy()
non_pakistan_df = base_md_modeling_df[~base_md_modeling_df['location_id'].isin(all_pakistan_locations)].copy()

# Filter by years and set year_to_rake
pakistan_2021 = pakistan_df[pakistan_df['year_id'] == 2021].copy()
pakistan_2021['year_to_rake'] = 2021
non_pakistan_2022 = non_pakistan_df[non_pakistan_df['year_id'] == 2022].copy()
non_pakistan_2022['year_to_rake'] = 2022

base_md_modeling_df = pd.concat([pakistan_2021, non_pakistan_2022], ignore_index=True)
base_md_modeling_df = base_md_modeling_df.drop(columns=['year_id'])
base_md_modeling_df["stage_2"] = 1

# Continue with existing merge logic
forecast_by_draw_df = forecast_by_draw_df.merge(
    base_md_modeling_df,
    how="left",
    on=["location_id"]
)



dah_scenarios, dah_scenario_names = generate_dah_scenarios(
    baseline_df=forecast_by_draw_df,
    ssp_scenario=ssp_scenario,
    malaria_stage_2_modeling_df_path=aa_md_modeling_df_path
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