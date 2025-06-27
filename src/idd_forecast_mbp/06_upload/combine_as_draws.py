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
from idd_forecast_mbp.helper_functions import read_parquet_with_integer_ids, write_hdf
import argparse
import os

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--cause", type=str, required=False, default="malaria", help="Cause (e.g., 'malaria', 'dengue')")
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=False, default="Baseline", help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--measure", type=str, required=False, default="mortality", help="measure (e.g., 'mortality', 'incidence')")
parser.add_argument("--metric", type=str, required=False, default="rate", help="metric (e.g., 'rate', 'count')")
parser.add_argument("--fhs_flag", type=int, required=False, default=0, help="Flag to indicate if output will follow FHS format")
parser.add_argument("--run_date", type=str, required=True, default=2025_06_25, help="Run date in format YYYY_MM_DD (e.g., '2025_06_25')")


# Parse arguments
args = parser.parse_args()

cause = args.cause
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
measure = args.measure
metric = args.metric
fhs_flag = args.fhs_flag
run_date = args.run_date


PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data"
FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"



ssp_draws = rfc.draws
measure_map = rfc.measure_map
metric_map = rfc.metric_map
cause_map = rfc.cause_map
ssp_scenarios = rfc.ssp_scenarios
fhs_draws = rfc.fhs_draws
scenario = ssp_scenarios[ssp_scenario]["dhs_scenario"] #  is the DHS scenario name

release_id = 9

cause_id = cause_map[cause]["cause_id"]
measure_id = measure_map[measure]["measure_id"]
metric_id = metric_map[metric]["metric_id"]





if cause == "malaria":
    processed_forecast_df_path_template = "{UPLOAD_DATA_PATH}/full_as_malaria_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.parquet"
    if fhs_flag == 1:
        upload_folder_path = f"{UPLOAD_DATA_PATH}/fhs_upload_folders/cause_id_{cause_id}_measure_id_{measure_id}_scenario_{scenario}_{run_date}"
        upload_file_path = f"{upload_folder_path}/draws.h5"
    else:
        upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}"
        upload_file_path = f"{upload_folder_path}/draws.h5"
else:
    processed_forecast_df_path_template = "{UPLOAD_DATA_PATH}/full_as_dengue_measure_{measure}_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet"
    if fhs_flag == 1:
        upload_folder_path = f"{UPLOAD_DATA_PATH}/fhs_upload_folders/cause_id_{cause_id}_measure_id_{measure_id}_scenario_{scenario}_{run_date}"
        upload_file_path = f"{upload_folder_path}/draws.h5"
    else:
        upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}"
        upload_file_path = f"{upload_folder_path}/draws.h5"



print(f"Upload folder: {upload_folder_path}")

age_metadata_path = f"{FHS_DATA_PATH}/age_metadata.parquet"

# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"
as_merge_variables = rfc.as_merge_variables
fhs_hierarchy_df = hierarchy_df[hierarchy_df["in_fhs_hierarchy"] == True].copy()

all_location_ids = hierarchy_df["location_id"].unique().tolist()
fhs_location_ids = fhs_hierarchy_df["location_id"].unique().tolist()

year_ids = range(2022, 2101)
# Make filters based on fhs_hierarchy_df and hierarchy_df
fhs_location_filter = ('location_id', 'in', fhs_location_ids)
all_location_filter = ('location_id', 'in', all_location_ids)

if fhs_flag == 1:
    location_filter = fhs_location_filter
else:
    location_filter = all_location_filter

year_filter = ('year_id', 'in', year_ids)

print(f"Processing SSP scenario: {ssp_scenario}")
scenario = ssp_scenarios[ssp_scenario]["dhs_scenario"]
print(f"Scenario: {scenario}")

# Make the template with the first draw
if cause == "malaria":
    processed_forecast_df_path = processed_forecast_df_path_template.format(
        UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
        measure=measure,
        ssp_scenario=ssp_scenario,
        dah_scenario=dah_scenario,
        draw="000"
    )
else:
    processed_forecast_df_path = processed_forecast_df_path_template.format(
        UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
        measure=measure,
        ssp_scenario=ssp_scenario,
        draw="000"
    )

upload_df = read_parquet_with_integer_ids(processed_forecast_df_path,
        filters=[[location_filter, year_filter]]  # Combining with AND logic
    )

fhs_draw_name = "draw_0"
if metric == "rate":
    upload_df[fhs_draw_name] = upload_df["count_pred"] / upload_df["population"]
else:
    upload_df[fhs_draw_name] = upload_df["count_pred"]

# if fhs_flag:
#     upload_df = upload_df.drop(columns=["population"])

upload_df = upload_df.drop(columns=["count_pred", "level"])
upload_df["measure_id"] = measure_id
upload_df["metric_id"] = metric_id
upload_df["cause_id"] = cause_id
upload_df["release_id"] = release_id
upload_df["scenario"] = scenario
upload_df["ssp_scenario"] = ssp_scenario
upload_df["dah_scenario"] = dah_scenario

upload_df = upload_df[["measure_id", "metric_id", "cause_id", "age_group_id", "sex_id", "location_id", "year_id", "ssp_scenario", "dah_scenario", "population", fhs_draw_name]]

for ssp_draw in ssp_draws[1:]:
    if cause == "malaria":
        draw_df_path = processed_forecast_df_path_template.format(
            UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
            measure=measure,
            ssp_scenario=ssp_scenario,
            dah_scenario=dah_scenario,
            draw=ssp_draw
        )
    else:
        draw_df_path = processed_forecast_df_path_template.format(
            UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
            measure=measure,
            ssp_scenario=ssp_scenario,
            draw=ssp_draw
        )
    draw_int = int(ssp_draw)
    if draw_int % 10 == 0:
        print(f"Processing draw {draw_int}")

    draw_df = read_parquet_with_integer_ids(
        draw_df_path,
        filters=[[location_filter, year_filter]]  # Combining with AND logic
    )

    fhs_draw_name = f"draw_{draw_int}"
    if metric == "rate":
        draw_df[fhs_draw_name] = draw_df["count_pred"] / draw_df["population"]
    else:
        draw_df[fhs_draw_name] = draw_df["count_pred"]
    
    draw_df = draw_df.drop(columns=["count_pred", "level", "population"])

    upload_df = upload_df.merge(
        draw_df,
        on = ["location_id", "year_id", "age_group_id", "sex_id"],
        how = "left",
    )

######
## Complete df with zero malaria data
######
age_metadata_df = read_parquet_with_integer_ids(age_metadata_path)
age_group_ids = age_metadata_df["age_group_id"].unique()
sex_ids = [1, 2]  # 1
missing_location_ids = set(fhs_location_ids) - set(upload_df["location_id"].unique())
combinations = list(itertools.product(age_group_ids, sex_ids, year_ids, missing_location_ids))
zero_df = pd.DataFrame(combinations, columns=['age_group_id', 'sex_id', "year_id", "location_id"])
zero_df["measure_id"] = measure_id
zero_df["metric_id"] = metric_id
zero_df["cause_id"] = cause_id
zero_df["ssp_scenario"] = ssp_scenario
zero_df["dah_scenario"] = dah_scenario

# Create a DataFrame with all draw columns initialized to 0.0
draw_columns_df = pd.DataFrame(0.0, 
    index=range(len(zero_df)),
    columns=fhs_draws)
    
# Then concatenate horizontally with your original DataFrame
zero_df = pd.concat([zero_df, draw_columns_df], axis=1)

missing_location_filter = ('location_id', 'in', list(missing_location_ids))
year_filter = ('year_id', 'in', year_ids)
as_population_df = read_parquet_with_integer_ids(as_full_population_df_path,
                                               filters = [missing_location_filter, year_filter])

zero_df = zero_df.merge(
    as_population_df,
    on=as_merge_variables,
    how="left"
)

columns_to_keep = ["measure_id", "metric_id", "cause_id", "age_group_id", "sex_id", "location_id", "year_id", "ssp_scenario", "dah_scenario", "population"] + fhs_draws
zero_df = zero_df[columns_to_keep]
upload_df = pd.concat([upload_df, zero_df], ignore_index=True)

######
## Fix Ethiopian location
######

swap_location_ids = [60908, 95069, 94364]
replace_location_id = [44858]

# Remove rows with swap_location_ids and store them separately
rows_to_aggregate = upload_df[upload_df['location_id'].isin(swap_location_ids)].copy()
remaining_df = upload_df[~upload_df['location_id'].isin(swap_location_ids)].copy()

if metric == "rate":
    for col in fhs_draws:
        rows_to_aggregate[col] = rows_to_aggregate[col] * rows_to_aggregate['population']


# Aggregate the removed rows by year, age_group_id, and sex_id
agg_dict = {
    'population': 'sum',
    **{col: 'sum' for col in fhs_draws}
}

# For categorical columns, take the first value (they should be the same within groups)
if cause == "malaria":
    categorical_cols = ['measure_id', 'metric_id', 'cause_id', 'ssp_scenario', 'dah_scenario']
else:
    categorical_cols = ['measure_id', 'metric_id', 'cause_id', 'ssp_scenario']


for col in categorical_cols:
    agg_dict[col] = 'first'


aggregated_data = rows_to_aggregate.groupby(['year_id', 'age_group_id', 'sex_id']).agg(agg_dict).reset_index()

if metric == "rate":
    for col in fhs_draws:
        aggregated_data[col] = aggregated_data[col] / aggregated_data['population']

# Add the new location_id
aggregated_data = aggregated_data.assign(location_id=replace_location_id[0])
aggregated_data = aggregated_data[remaining_df.columns].copy()

upload_df = pd.concat([remaining_df, aggregated_data], ignore_index=True)



######
## Upload
######
print(f"Saving upload data for {ssp_scenario} ssp_scenario")
if cause == "malaria":
    print(f"Saving upload data for {dah_scenario} dah_scenario")

print(f"Creating upload folder: {upload_folder_path}")
mkdir(upload_folder_path, exist_ok=True, parents=True)

if fhs_flag == 1:
    draw_cols = upload_df.columns[upload_df.columns.str.startswith('draw_')].tolist()
    upload_df["release_id"] = release_id
    upload_df['scenario'] = scenario
    columns_to_select = ["measure_id", "metric_id", "cause_id", "location_id", "year_id", "age_group_id", "sex_id", "release_id", "scenario"] + draw_cols
    upload_df = upload_df[columns_to_select]
else:
    upload_df["measure"] = measure
    upload_df["metric"] = metric
    upload_df["cause"] = cause
    upload_df = upload_df.drop(columns=["measure_id", "metric_id", "cause_id", "release_id", "scenario"])


write_hdf(upload_df, upload_file_path, 
        data_columns=['location_id', 'sex_id', 'age_group_id', 'year_id', 'measure_id', 'metric_id', 'cause_id'])


print("Saved the age-sex-specific file")
# Stopped here!
if fhs_flag == 0:


    upload_df = upload_df.copy()
    # Get draw columns efficiently
    draw_cols = upload_df.columns[upload_df.columns.str.startswith('draw_')].tolist()

    # Handle rate conversion BEFORE aggregation if needed
    if metric_id == 3 and 'population' in upload_df.columns:
        # Vectorized operation across all draw columns
        upload_df[draw_cols] = upload_df[draw_cols].multiply(upload_df['population'], axis=0)

    # Use categorical grouping for speed
    upload_df['location_id'] = upload_df['location_id'].astype('category')
    upload_df['year_id'] = upload_df['year_id'].astype('category')

    # Create aggregation dict efficiently
    agg_dict = dict.fromkeys(draw_cols, 'sum')
    if 'population' in upload_df.columns:
        agg_dict['population'] = 'sum'

    # Fast aggregation
    df_agg = upload_df.groupby(['location_id', 'year_id'], as_index=False, observed=True).agg(agg_dict)

    # Convert back to rates AFTER aggregation if needed
    if metric_id == 3 and 'population' in df_agg.columns:
        # Divide aggregated counts by aggregated population to get rates
        df_agg[draw_cols] = df_agg[draw_cols].div(df_agg['population'], axis=0)

    print(f"Aggregated from {len(upload_df)} rows to {len(df_agg)} rows")

    if cause == 'malaria':
        aa_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_id_{cause_id}_measure_id_{measure_id}_metric_id_{metric_id}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}"
    else:
        aa_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/aa_cause_id_{cause_id}_measure_id_{measure_id}_metric_id_{metric_id}_ssp_scenario_{ssp_scenario}"

    aa_upload_file_path = f"{aa_upload_folder_path}/draws.h5"

    df_agg['measure'] = measure
    df_agg['metric'] = metric
    df_agg['cause'] = cause
    df_agg['ssp_scenario'] = ssp_scenario
    df_agg['dah_scenario'] = dah_scenario

    if cause == "malaria":
        columns_to_select = ["measure", "metric", "cause", "location_id", "year_id", "ssp_scenario", "dah_scenario", "population"] + draw_cols
    else:
        columns_to_select = ["measure", "metric", "cause", "location_id", "year_id", "ssp_scenario", "population"] + draw_cols
    df_agg = df_agg[columns_to_select]
    mkdir(aa_upload_folder_path, exist_ok=True, parents=True)
    # Set file permissions to be world-writable and deletable after saving
    write_hdf(df_agg, aa_upload_file_path, 
          data_columns=['location_id', 'year_id', 'measure', 'metric', 'cause'])
