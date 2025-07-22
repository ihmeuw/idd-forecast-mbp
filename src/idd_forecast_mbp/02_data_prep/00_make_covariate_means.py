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
from idd_forecast_mbp.xarray_functions import write_netcdf

ssp_scenarios = rfc.ssp_scenarios
years = rfc.model_years
year_filter = ('year_id', 'in', years)
draws = rfc.draws


# Hierarchy
hierarchy = "lsae_1209"
PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
VARIABLE_DATA_PATH = f"{PROCESSED_DATA_PATH}/{hierarchy}"

covariate_ds_path = f"{MODELING_DATA_PATH}/covariate_means.nc"

# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

md_location_ids = hierarchy_df[hierarchy_df['level'] == 5]['location_id'].unique().tolist()
md_location_filter = ('location_id', 'in', md_location_ids)

aa_full_population_df_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population.parquet"
aa_full_population_df = read_parquet_with_integer_ids(aa_full_population_df_path)
aa_merge_variables = rfc.aa_merge_variables

# LSAE 1209 variable path
# DAH
# dah_df_path = f"{VARIABLE_DATA_PATH}/dah_df.parquet"
dah_df_path = f"{PROCESSED_DATA_PATH}/dah_df_2025_07_08.parquet"

# ppp
income_paths = {
    "gdppc": "{VARIABLE_DATA_PATH}/gdppc_mean.parquet",
}

# urbanization
urban_paths = {
    "urban_threshold_300":      "{VARIABLE_DATA_PATH}/urban_threshold_300.0_simple_mean.parquet",
    "urban_threshold_1500":     "{VARIABLE_DATA_PATH}/urban_threshold_1500.0_simple_mean.parquet",
}

# Climate variables
CLIMATE_DATA_PATH = f"/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/{hierarchy}"

flooding_path_template = "/mnt/team/rapidresponse/pub/flooding/results/output/lsae_1209/fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet"

cc_sensitive_draw_path_templates = {
    "mean_temperature":         "{CLIMATE_DATA_PATH}/mean_temperature_{ssp_scenario}.parquet",
    "dengue_suitability":       "{CLIMATE_DATA_PATH}/dengue_suitability_{ssp_scenario}.parquet",
    "malaria_suitability":       "{CLIMATE_DATA_PATH}/malaria_suitability_{ssp_scenario}.parquet"
}

dah_scenario_df_path_template = "{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}.parquet"
dah_scenarios = ["Baseline", "Constant"]

covariate_dfs = []
for ssp_scenario in ssp_scenarios:
    print(f"Processing SSP scenario: {ssp_scenario}")
    rcp_scenario = ssp_scenarios[ssp_scenario]['rcp_scenario']
    print("  Reading population")
    covariate_df = aa_full_population_df
    covariate_df = covariate_df.rename(columns={'population':'aa_population'})
    covariate_df['ssp_scenario'] = ssp_scenario
    print("  Reading flooding")
    flooding_df_path = flooding_path_template.format(ssp_scenario=ssp_scenario)

    flooding_df = read_parquet_with_integer_ids(flooding_df_path,
                                                filters = [year_filter, md_location_filter])
    flooding_df = flooding_df.drop(
        columns=["model", "variant", 'population', 'people_flood_days', 'scenario']
    )
    flooding_df = flooding_df.rename(columns={'fldfrc_shifted0.1_sum': 'flooding'})
    covariate_df = covariate_df.merge(flooding_df,
                                        on = aa_merge_variables,
                                        how = 'left')

    print("  Reading urbanization")
    urban_dfs = read_urban_paths(urban_paths, VARIABLE_DATA_PATH)
    covariate_df = merge_dataframes(covariate_df, urban_dfs)

    print("  Reading income")
    income_dfs = read_income_paths(income_paths, rcp_scenario, VARIABLE_DATA_PATH)
    covariate_df = merge_dataframes(covariate_df, income_dfs)

    print("  Reading DAH scenarios")
    for dah_scenario in dah_scenarios:
        dah_scenario_df_path = dah_scenario_df_path_template.format(
            FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
            ssp_scenario=ssp_scenario,
            dah_scenario=dah_scenario,
            draw = "000"
        )

        columns_to_keep = ["year_id", "location_id", "mal_DAH_total_per_capita"]
        dah_scenario_df = read_parquet_with_integer_ids(dah_scenario_df_path,
                                                        columns = columns_to_keep)
        
        dah_scenario_df = dah_scenario_df.rename(columns={'mal_DAH_total_per_capita': f'mal_DAH_total_per_capita_{dah_scenario}'})
        
        covariate_df = covariate_df.merge(dah_scenario_df,
                                        on = aa_merge_variables,
                                        how = 'left')
        
        # set NaN values of mal_DAH_total_per_capita to 0
        dah_col = f'mal_DAH_total_per_capita_{dah_scenario}'
        covariate_df[dah_col] = covariate_df[dah_col].fillna(0)

    # Draw-level covariates
    for cc_sensitive_variable in cc_sensitive_draw_path_templates:
        print(f"  Reading {cc_sensitive_variable}")
        cc_sensitive_draw_path_template = cc_sensitive_draw_path_templates[cc_sensitive_variable]

        cc_sensitive_draw_path = cc_sensitive_draw_path_template.format(
            CLIMATE_DATA_PATH = CLIMATE_DATA_PATH,
            ssp_scenario = ssp_scenario
        )

        tmp_df = read_parquet_with_integer_ids(cc_sensitive_draw_path,
                                                    filters = [year_filter, md_location_filter])
        tmp_df[cc_sensitive_variable] = tmp_df[draws].mean(axis=1)
        tmp_df = tmp_df.drop(columns = draws)

        covariate_df = covariate_df.merge(tmp_df,
                                            on = aa_merge_variables,
                                            how = 'left')
        
    covariate_df = covariate_df.merge(hierarchy_df[['location_id', 'level']],
                                        on = ['location_id'], how = 'left')

    cols = list(covariate_df.columns)
    # Specify your desired order
    first_cols = ["location_id", "year_id", "ssp_scenario", "aa_population", 'level']
    # Get the rest of the columns, excluding those already in first_cols
    other_cols = [col for col in cols if col not in first_cols]
    # Combine for new order
    new_order = first_cols + other_cols
    # Reindex the DataFrame
    covariate_df = covariate_df[new_order]

    print("  Aggregating to parent levels")
    for level in list(range(4, -1, -1)):
        child_df = covariate_df[covariate_df['level'] == level + 1]
        child_df = child_df.merge(hierarchy_df[['location_id', 'parent_id']],
                                    on = ['location_id'], how = 'left')
        child_df = child_df.drop(columns=['level', 'ssp_scenario', 'location_id'])
        child_df[other_cols] = child_df[other_cols].fillna(0)

        child_agg_df = (child_df[other_cols].mul(child_df['aa_population'], axis=0)
                        .assign(year_id=child_df['year_id'], 
                            parent_id=child_df['parent_id'],
                            aa_population=child_df['aa_population'])
                        .groupby(['year_id', 'parent_id'], sort=False)
                        .sum()
                        .reset_index())
        # Divide by group weights to get averages
        child_agg_df[other_cols] = child_agg_df[other_cols].div(child_agg_df['aa_population'], axis=0)
        parent_df = child_agg_df.rename(columns={'parent_id': 'location_id'}).copy()
        parent_df['level'] = level
        parent_df['ssp_scenario'] = ssp_scenario
        # Remove existing parent level rows
        covariate_df = covariate_df[~(covariate_df['level'] == level)]
        # Add the new aggregated rows
        covariate_df = pd.concat([covariate_df, parent_df], ignore_index=True)

    print("  Remerging in full population")
    covariate_df = covariate_df.drop(columns=['aa_population', 'level']).merge(
        aa_full_population_df,
        on = aa_merge_variables,
        how = 'left'
    )
    covariate_dfs.append(covariate_df)

print("Concatenating all covariate DataFrames")
# Concatenate all covariate DataFrames
covariate_df = pd.concat(covariate_dfs, ignore_index=True)

print("Converting to xarray Dataset")
# Convert to xarray Dataset
covariate_df = covariate_df.set_index(['location_id', 'year_id', 'ssp_scenario'])
covariate_ds = covariate_df.to_xarray()

print("Saving xarray Dataset to NetCDF file")
# Save the xarray Dataset to a NetCDF file

write_netcdf(covariate_ds, covariate_ds_path)
