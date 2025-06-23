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
import os

parser = argparse.ArgumentParser(description="Add DAH Sceanrios and create draw level dataframes for forecating malaria")

# Define arguments
parser.add_argument("--cause", type=str, required=False, default="malaria", help="Cause (e.g., 'malaria', 'dengue')")
parser.add_argument("--ssp_scenario", type=str, required=True, help="SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--dah_scenario", type=str, required=False, default="Baseline", help="DAH scenario (e.g., 'Baseline')")
parser.add_argument("--measure", type=str, required=False, default="mortality", help="measure (e.g., 'mortality', 'incidence')")
parser.add_argument("--metric", type=str, required=False, default="rate", help="metric (e.g., 'rate', 'count')")
parser.add_argument("--fhs_flag", type=int, required=False, default=0, help="Flag to indicate if output will follow FHS format")

# Parse arguments
args = parser.parse_args()

cause = args.cause
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
measure = args.measure
metric = args.metric
fhs_flag = args.fhs_flag


PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data"
FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"

run_date = "2025_06_13"

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

future_fhs_population_path = "/mnt/share/forecasting/data/9/future/population/20250606_first_sub_rcp45_climate_ref_100d_hiv_shocks_covid_all/summary/summary.nc"
future_fhs_population = xr.open_dataset(future_fhs_population_path)
# The actual data is in the 'draws' variable, select the 'mean' statistic
fhs_pop_data = future_fhs_population['draws'].sel(statistic='mean')
future_fhs_population_df = fhs_pop_data.to_dataframe().reset_index()
future_fhs_population_df = future_fhs_population_df.rename(columns={"draws": "population"})
future_fhs_population_df = future_fhs_population_df.drop(columns=["scenario"], errors='ignore')
# Drop the 'statistic' column as it is no longer needed
future_fhs_population_df = future_fhs_population_df.drop(columns=["statistic"], errors='ignore')

processed_forecast_df_path = "{UPLOAD_DATA_PATH}/full_as_malaria_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.parquet"

if fhs_flag == 1:
    upload_folder_path = f"{UPLOAD_DATA_PATH}/fhs_upload_folders/cause_id_{cause_id}_measure_id_{measure_id}_scenario_{scenario}_{run_date}"
    upload_file_path = f"{upload_folder_path}/draws.h5"
else:
    upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/cause_id_{cause_id}_measure_id_{measure_id}_metric_id_{metric_id}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_{run_date}"
    upload_file_path = f"{upload_folder_path}/draws.h5"



print(f"Upload folder: {upload_folder_path}")

age_metadata_path = f"{FHS_DATA_PATH}/age_metadata.parquet"
# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)
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
upload_df = read_parquet_with_integer_ids(
        processed_forecast_df_path.format(
            UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
            measure=measure,
            ssp_scenario=ssp_scenario,
            dah_scenario=dah_scenario,
            draw="000"
        ),
        filters=[[location_filter, year_filter]]  # Combining with AND logic
    )

if "population_x" in upload_df.columns:
    upload_df = upload_df.rename(columns={"population_y": "population"})
    upload_df = upload_df.drop(columns=["population_x"])

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
    
    draw_df_path = processed_forecast_df_path.format(
        UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
        measure=measure,
        ssp_scenario=ssp_scenario,
        dah_scenario=dah_scenario,
        draw=ssp_draw
    )
    draw_int = int(ssp_draw)
    if draw_int % 10 == 0:
        print(f"Processing draw {draw_int}")

    draw_df = read_parquet_with_integer_ids(
            draw_df_path,
            filters=[[location_filter, year_filter]]  # Combining with AND logic
        )
    if "population_x" in draw_df.columns:
        draw_df = draw_df.rename(columns={"population_y": "population"})
        draw_df = draw_df.drop(columns=["population_x"])

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
zero_df["release_id"] = release_id
zero_df["scenario"] = scenario
zero_df["ssp_scenario"] = ssp_scenario
zero_df["dah_scenario"] = dah_scenario

# Create a DataFrame with all draw columns initialized to 0.0
draw_columns_df = pd.DataFrame(0.0, 
    index=range(len(zero_df)),
    columns=fhs_draws)
    
# Then concatenate horizontally with your original DataFrame
zero_df = pd.concat([zero_df, draw_columns_df], axis=1)

zero_df = zero_df.merge(
    future_fhs_population_df,
    on=["location_id", "year_id","age_group_id", "sex_id"],
    how="left"
)
zero_df = zero_df[["measure_id", "metric_id", "cause_id", "age_group_id", "sex_id", "location_id", "year_id", "ssp_scenario", "dah_scenario", "population"] + fhs_draws]

upload_df = pd.concat([upload_df, zero_df], ignore_index=True)

######
## Upload
######
print(f"Saveing upload data for {ssp_scenario} ssp_scenario")
print(f"Saveing upload data for {dah_scenario} dah_scenario")

# upload_folder_path = upload_folder_path_template.format(
#     UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
#     cause_id=cause_id,
#     measure_id=measure_id,
#     metric_id=metric_id,
#     scenario=scenario,
#     ssp_scenario=ssp_scenario,
#     dah_scenario=dah_scenario,
#     run_date=run_date
# )
print(f"Creating upload folder: {upload_folder_path}")
mkdir(upload_folder_path, exist_ok=True, parents=True)
# upload_file_path = upload_file_path_template.format(
#     upload_folder_path=upload_folder_path
# )

if fhs_flag == 1:
    draw_cols = upload_df.columns[upload_df.columns.str.startswith('draw_')].tolist()
    upload_df["release_id"] = release_id
    upload_df['scenario'] = scenario
    columns_to_select = ["measure_id", "metric_id", "cause_id", "location_id", "year_id", "age_group_id", "sex_id", "release_id", "scenario"] + draw_cols
    upload_df = upload_df[columns_to_select]

upload_df.to_hdf(
    upload_file_path,
    key='df', 
    mode='w', 
    format='table',
    data_columns=['location_id', 'sex_id', 'age_group_id', 'year_id', 'measure_id', 'metric_id', 'cause_id']
)
os.chmod(upload_file_path, 0o775)
print("Saved the age-sex-specific file")

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

    aa_upload_folder_path = f"{UPLOAD_DATA_PATH}/upload_folders/aa_cause_id_{cause_id}_measure_id_{measure_id}_metric_id_{metric_id}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_{run_date}"
    aa_upload_file_path = f"{aa_upload_folder_path}/draws.h5"

    df_agg['measure_id'] = measure_id
    df_agg['metric_id'] = metric_id
    df_agg['cause_id'] = cause_id
    df_agg['ssp_scenario'] = ssp_scenario
    df_agg['dah_scenario'] = dah_scenario

    columns_to_select = ["measure_id", "metric_id", "cause_id", "location_id", "year_id", "ssp_scenario", "dah_scenario", "population"] + draw_cols
    df_agg = df_agg[columns_to_select]
    mkdir(aa_upload_folder_path, exist_ok=True, parents=True)
    # Set file permissions to be world-writable and deletable after saving
    df_agg.to_hdf(
        aa_upload_file_path,
        key='df', 
        mode='w', 
        format='table',
        data_columns=['location_id', 'year_id', 'measure_id', 'metric_id', 'cause_id']
    )
    os.chmod(aa_upload_file_path, 0o775)
