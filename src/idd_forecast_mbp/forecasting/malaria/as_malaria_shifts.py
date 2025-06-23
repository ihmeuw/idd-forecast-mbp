import pandas as pd
import numpy as np
import os
import sys
import itertools
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import ensure_id_columns_are_integers, read_parquet_with_integer_ids, read_income_paths, merge_dataframes
import glob
import argparse

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=True, help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")


# Parse arguments
args = parser.parse_args()

ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
draw = args.draw

hierarchy = "lsae_1209"

ssp_scenarios = rfc.ssp_scenarios
dah_scenarios = rfc.dah_scenarios
measure_map = rfc.measure_map

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"

FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
age_metadata_path = f"{FHS_DATA_PATH}/age_metadata.parquet"

as_fhs_df = "{MODELING_DATA_PATH}/fhs_{cause}_{measure}_{metric}_modeling_df.parquet"
malaria_draw_path = "{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}_with_predictions.parquet"
output_malaria_draw_path = "{FORECASTING_DATA_PATH}/as_malaria_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}_with_predictions.parquet"

###----------------------------------------------------------###
### 5. Age Metadata Processing
### Loads age group metadata and creates all possible age-sex combinations
### for disaggregation of all-age population values.
###----------------------------------------------------------###
age_metadata_df = read_parquet_with_integer_ids(age_metadata_path)
age_group_ids = age_metadata_df["age_group_id"].unique()
sex_ids = [1, 2]  # 1
combinations = list(itertools.product(age_group_ids, sex_ids))
as_df = pd.DataFrame(combinations, columns=['age_group_id', 'sex_id'])

malaria_df = read_parquet_with_integer_ids(
    malaria_draw_path.format(
        FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
        ssp_scenario=ssp_scenario,
        dah_scenario_name=dah_scenario,
        draw=draw
    )
)

malaria_df = malaria_df.drop(columns=["people_flood_days", "people_flood_days_per_capita", "location_set_version_id", "scenario","fhs_level",
                                      "location_set_id","is_estimate","sort_order",'location_type', 'map_id', 'super_region_id', 'super_region_name',
       'region_id', 'region_name', 'ihme_loc_id', 'local_id', 'path_to_top_parent', 'parent_id', 'logit_malaria_pfpr_pred_raw',
       'most_detailed_lsae', 'in_fhs_hierarchy', 'in_lsae_hierarchy', 'logit_malaria_pfpr_pred','logit_malaria_pfpr_obs',
       'in_gbd_hierarchy', 'most_detailed_gbd', 'most_detailed_fhs', 'malaria_pfpr', 'A0_location_id','level',
       'gbd_location_id', 'gbd_level', 'mal_DAH_total',
       'mal_DAH_total_per_capita', 'gdppc_mean', 'total_precipitation',
       'malaria_suitability', 'log_mal_DAH_total_per_capita', 'log_gdppc_mean',
       'stage_2', 'A0_af'])


cause = "malaria"
measure = "mortality"  # or "incidence", depending on the measure you want to process

malaria_df_columns = [col for col in malaria_df.columns if "mort" in col]
malaria_df_columns = ['location_id', 'year_id', 'population', 'location_name',
    'fhs_location_id'] + malaria_df_columns
malaria_measure_df = malaria_df[malaria_df_columns].copy()

metric = "count"
as_fhs_df = read_parquet_with_integer_ids(
    as_fhs_df.format(
        MODELING_DATA_PATH=MODELING_DATA_PATH,
        cause=cause,
        measure=measure,
        metric=metric
    )
)

reference_age_group_id = as_fhs_df["reference_age_group_id"].unique()[0]
reference_sex_id = as_fhs_df["reference_sex_id"].unique()[0]

as_fhs_0_df = as_fhs_df[
    (as_fhs_df["age_group_id"] == reference_age_group_id) &
    (as_fhs_df["sex_id"] == reference_sex_id)].copy()
as_fhs_0_df = as_fhs_0_df[["location_id", "risk_0", "year_id"]]
as_fhs_0_df = as_fhs_0_df.rename(columns={"location_id": "fhs_location_id"})
as_fhs_0_df = as_fhs_0_df.rename(columns={"risk_0": f"malaria_mort_rate_baseline"})
as_fhs_0_df[f"log_malaria_mort_rate"] = np.log(as_fhs_0_df[f"malaria_mort_rate_baseline"] + 1e-6)

malaria_measure_df = malaria_measure_df.merge(
    as_fhs_0_df,
    on=["fhs_location_id", "year_id"],
    how="left")

malaria_measure_2022_df = malaria_measure_df[malaria_measure_df["year_id"] == 2022].copy()

malaria_measure_2022_df["shift"] = malaria_measure_2022_df[f"log_malaria_mort_rate"] - malaria_measure_2022_df[f"log_malaria_mort_rate_pred_raw"]
malaria_measure_2022_df["shift"] = malaria_measure_2022_df["shift"].fillna(0)


malaria_measure_df = malaria_measure_df.merge(
    malaria_measure_2022_df[["location_id", "shift"]],
    on=["location_id"],
    how="left"
)

malaria_measure_df[f"log_malaria_mort_rate_pred"] = malaria_measure_df[f"log_malaria_mort_rate_pred_raw"] + malaria_measure_df["shift"]


malaria_measure_df[f"malaria_mort_rate_baseline_pred"] = np.exp(malaria_measure_df[f"log_malaria_mort_rate_pred"])


# Drop all columns that have "log" in them
malaria_measure_df = malaria_measure_df.drop(columns=[col for col in malaria_measure_df.columns if "log" in col])
# Drop all columns that have shift in them
malaria_measure_df = malaria_measure_df.drop(columns=[col for col in malaria_measure_df.columns if "shift" in col])

as_fhs_2022_df = as_fhs_df[as_fhs_df["year_id"] == 2022].copy()

# Set as_fhs_2022_df relatve_risk_as that is NaN to 1
as_fhs_2022_df["relative_risk_as"] = as_fhs_2022_df["relative_risk_as"].fillna(1)
as_fhs_2022_df = as_fhs_2022_df.rename(columns={"location_id": "fhs_location_id"})

malaria_measure_full_df = malaria_measure_df.merge(as_df, how = "cross")
# Drop population column
malaria_measure_full_df = malaria_measure_full_df.drop(columns=["population"])

malaria_measure_full_df["reference_age_group_id"] = reference_age_group_id
malaria_measure_full_df["reference_sex_id"] = reference_sex_id

malaria_measure_full_df = malaria_measure_full_df.merge(
    as_fhs_2022_df[["fhs_location_id", "age_group_id", "sex_id", "relative_risk_as"]],
    on=["fhs_location_id", "age_group_id", "sex_id"],
    how="left"
)

malaria_measure_full_df["rate_pred"] = malaria_measure_full_df[f"malaria_mort_rate_baseline_pred"] * malaria_measure_full_df["relative_risk_as"]

# Make filters based on FHS hierarchy
malaria_location_ids = malaria_df["location_id"].unique()
year_ids = malaria_df["year_id"].unique()
malaria_location_filter = ('location_id', 'in', malaria_location_ids)
year_filter = ('year_id', 'in', year_ids)

as_lsae_population_df_path = f"{MODELING_DATA_PATH}/as_lsae_population_df.parquet"

# Read FHS population data with filters
as_lsae_population_df = read_parquet_with_integer_ids(
    as_lsae_population_df_path,
    filters=[[malaria_location_filter, year_filter]]  # Combining with AND logic
)

malaria_measure_full_df = malaria_measure_full_df.merge(
    as_lsae_population_df[["location_id", "year_id", "population_aa", "age_group_id", "sex_id", "pop_fraction_aa", "population"]],
    on=["location_id", "year_id", "age_group_id", "sex_id"],
    how="left")

malaria_measure_full_df["count_pred"] = malaria_measure_full_df["rate_pred"] * malaria_measure_full_df["population"]
malaria_measure_full_df["count_pred"] = malaria_measure_full_df["count_pred"].fillna(0)

# Write the final DataFrame to a parquet file
# Don't make any new changes, just write it out
# Don't run ensure 
output_path = output_malaria_draw_path.format(
    FORECASTING_DATA_PATH=FORECASTING_DATA_PATH,
    measure=measure,
    ssp_scenario=ssp_scenario,
    dah_scenario_name=dah_scenario,
    draw=draw
)
malaria_measure_full_df.to_parquet(output_path, compression="snappy", index=False)