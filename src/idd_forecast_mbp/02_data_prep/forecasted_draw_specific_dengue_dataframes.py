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
parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating dengue")

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


cause = "dengue"


cause_map = rfc.cause_map
reference_age_group_id = cause_map[cause]['reference_age_group_id']
reference_sex_id = cause_map[cause]['reference_sex_id']

forecast_non_draw_df_path = f"{FORECASTING_DATA_PATH}/{cause}_forecast_scenario_{ssp_scenario}_non_draw_part.parquet"
forecast_by_draw_df_path_template = "{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}.parquet"

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

aa_merge_variables = rfc.aa_merge_variables
as_merge_variables = rfc.as_merge_variables

aa_ge3_dengue_stage_1_modeling_df_path = f"{MODELING_DATA_PATH}/aa_ge3_{cause}_stage_1_modeling_df.parquet"
as_md_dengue_modeling_df_path = f"{MODELING_DATA_PATH}/as_md_{cause}_modeling_df.parquet"
base_md_modeling_df_path = f"{MODELING_DATA_PATH}/base_md_{cause}_modeling_df.parquet"

cc_sensitive_paths = {
    "total_precipitation":      "{CLIMATE_DATA_PATH}/total_precipitation_{ssp_scenario}.parquet",
    "relative_humidity":        "{CLIMATE_DATA_PATH}/relative_humidity_{ssp_scenario}.parquet",
    "dengue_suitability":       "{CLIMATE_DATA_PATH}/dengue_suitability_{ssp_scenario}.parquet"
}


# Get the unique values of A0_location_id
years = list(range(2022, 2023))
year_filter = ('year_id', '==', 2022)

dengue_df = read_parquet_with_integer_ids(aa_full_cause_df_path_template,
                                           filters=[year_filter, level_filter(hierarchy_df, start_level = 3)])

dengue_df = dengue_df[dengue_df['dengue_inc_count'] > 100].copy()
dengue_df = dengue_df.rename(columns={'location_id':'A0_location_id'})
A0_location_ids = dengue_df['A0_location_id'].unique()

age_filter = ('age_group_id', '==', reference_age_group_id)
sex_filter = ('sex_id', '==', reference_sex_id)

md_dengue_df = read_parquet_with_integer_ids(as_full_cause_df_path_template,
                                           filters=[year_filter, age_filter, sex_filter, level_filter(hierarchy_df, start_level = 5)])

md_dengue_df = md_dengue_df.merge(hierarchy_df[['location_id', 'A0_location_id']],
                                  on='location_id', how='left')

md_dengue_df = md_dengue_df[md_dengue_df['A0_location_id'].isin(A0_location_ids)].copy()

md_dengue_df['base_log_dengue_inc_rate'] = np.log(md_dengue_df['dengue_inc_rate'])


# base_md_modeling_df = read_parquet_with_integer_ids(base_md_modeling_df_path,
#                                                     filters = [year_filter])

dengue_modeling_location_ids = md_dengue_df['location_id'].unique()
dengue_modeling_location_filter = ('location_id', 'in', dengue_modeling_location_ids)


forecast_by_draw_df = read_parquet_with_integer_ids(forecast_non_draw_df_path,
                                                    filters = [dengue_modeling_location_filter])
# Add the draw column

forecast_by_draw_df["draw"] = draw
forecast_by_draw_df['A0_af'] = 'A0_' + forecast_by_draw_df['A0_location_id'].astype(str)

for key, path_template in cc_sensitive_paths.items():
    path = path_template.format(CLIMATE_DATA_PATH=CLIMATE_DATA_PATH, ssp_scenario=ssp_scenario)
    columns_to_read = ["location_id", "year_id", draw]
    df = read_parquet_with_integer_ids(path, columns=columns_to_read)
    df = df.rename(columns={draw: key})
    # Merge the file with forecast_by_draw_df
    if key == 'relative_humidity':
        df[key] = df[key].clip(lower=0.001, upper=99.999)
    forecast_by_draw_df = pd.merge(forecast_by_draw_df, df, on=["location_id", "year_id"], how="left")





md_dengue_df = md_dengue_df[['location_id', 'base_log_dengue_inc_rate']].copy()


# Merge in the dengue_stage_2_modeling_df
forecast_by_draw_df = forecast_by_draw_df.merge(
    md_dengue_df,
    how="left",
    on=["location_id"]
)

covariates_to_log_transform = [
    "gdppc_mean",
]

# Log transform the covariates and save them as new columns with "log_" prefix
for col in covariates_to_log_transform:
    # Create a new column with the log transformed value
    forecast_by_draw_df[f"log_{col}"] = np.log(forecast_by_draw_df[col] + 1e-6)

as_sex_filter = ('sex_id',  '==', reference_sex_id)
as_age_filter = ('age_group_id', '==', reference_age_group_id)


as_md_dengue_modeling_df = read_parquet_with_integer_ids(as_full_cause_df_path_template,
                                                            filters=[year_filter, dengue_modeling_location_filter, as_age_filter, as_sex_filter])

as_md_dengue_modeling_df["as_id"] = "a" + as_md_dengue_modeling_df["age_group_id"].astype(str) + "_s" + as_md_dengue_modeling_df["sex_id"].astype(str)

as_md_dengue_modeling_df["dengue_cfr"] = as_md_dengue_modeling_df["dengue_mort_rate"] / as_md_dengue_modeling_df["dengue_inc_rate"]

covariates_to_logit_transform = ['dengue_cfr']
for col in covariates_to_logit_transform:
    clipped_values = as_md_dengue_modeling_df[col].clip(upper=0.99)
    print(f"Range of {col}: {as_md_dengue_modeling_df[col].min()} to {as_md_dengue_modeling_df[col].max()}")
    as_md_dengue_modeling_df[f"logit_{col}"] = np.log(clipped_values / (1 - clipped_values))




columns_to_keep = as_merge_variables + ['logit_dengue_cfr', 'as_id']
as_md_dengue_modeling_df = as_md_dengue_modeling_df[columns_to_keep].copy()
as_md_dengue_modeling_df['year_to_rake'] = 2022
as_md_dengue_modeling_df = as_md_dengue_modeling_df.drop(columns=['year_id'])

forecast_by_draw_df = forecast_by_draw_df.merge(as_md_dengue_modeling_df,
                                                on=["location_id"],
                                                how="left")

# Write the dengue_stage_2_modeling_df to a parquet file
forecast_by_draw_df_path = forecast_by_draw_df_path_template.format(
    FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
    cause=cause,
    ssp_scenario=ssp_scenario,
    draw=draw
)

columns_to_keep = as_merge_variables + ['logit_dengue_cfr', 'log_gdppc_mean', 'base_log_dengue_inc_rate', 'dengue_suitability', 'logit_urban_1km_threshold_300', 'as_id', 'A0_af']
forecast_by_draw_df = forecast_by_draw_df[columns_to_keep].copy()



write_parquet(forecast_by_draw_df, forecast_by_draw_df_path)
