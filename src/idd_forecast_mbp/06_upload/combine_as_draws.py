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
from idd_forecast_mbp.hd5_functions import write_hdf
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids
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
parser.add_argument("--delete_existing", type=str, required=False, default=True, help="Flag to indicate if existing upload folder should be deleted")


# Parse arguments
args = parser.parse_args()

cause = args.cause
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
measure = args.measure
metric = args.metric
fhs_flag = args.fhs_flag
run_date = args.run_date
delete_existing = args.delete_existing


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

# Delete any contents of upload_folder_path if it exists
if delete_existing and os.path.exists(upload_folder_path):
    print(f"Removing existing contents of {upload_folder_path}")
    for file in os.listdir(upload_folder_path):
        file_path = os.path.join(upload_folder_path, file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            os.rmdir(file_path)

print(f"Upload folder: {upload_folder_path}")

age_metadata_path = f"{FHS_DATA_PATH}/age_metadata.parquet"

# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"
as_merge_variables = rfc.as_merge_variables
fhs_hierarchy_df = hierarchy_df[hierarchy_df["in_fhs_hierarchy"] == True].copy()

swap_location_ids = [60908, 95069, 94364]

all_location_ids = hierarchy_df["location_id"].unique().tolist()
fhs_location_ids = fhs_hierarchy_df["location_id"].unique().tolist() + swap_location_ids

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
    upload_df = upload_df.drop(columns=["count_pred", "level"])
else:
    upload_df[fhs_draw_name] = upload_df["count_pred"]
    upload_df = upload_df.drop(columns=["level"])

# if fhs_flag:
#     upload_df = upload_df.drop(columns=["population"])

upload_df = upload_df[as_merge_variables + ["population"] + [fhs_draw_name]]

# Pre-compute paths
draw_paths = []
for ssp_draw in ssp_draws[1:]:
    if cause == "malaria":
        path = processed_forecast_df_path_template.format(
            UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
            measure=measure,
            ssp_scenario=ssp_scenario,
            dah_scenario=dah_scenario,
            draw=ssp_draw
        )
    else:
        path = processed_forecast_df_path_template.format(
            UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
            measure=measure,
            ssp_scenario=ssp_scenario,
            draw=ssp_draw
        )
    draw_paths.append(path)

needed_cols = ["count_pred"]
if metric == "rate":
    needed_cols.append("population")

draw_data = []
for ssp_draw, draw_path in zip(ssp_draws[1:], draw_paths):
    draw_df = read_parquet_with_integer_ids(
        draw_path,
        filters=[[location_filter, year_filter]],
        columns=needed_cols  # Only read what you need
    )
    
    if metric == "rate":
        draw_data.append(draw_df["count_pred"] / draw_df["population"])
    else:
        draw_data.append(draw_df["count_pred"])

# Create all draw columns at once
draw_names = [f"draw_{int(draw)}" for draw in ssp_draws[1:]]
new_columns = pd.DataFrame(dict(zip(draw_names, draw_data)))
# Concatenate the new columns to the upload_df
print("Concatenating new columns to upload_df")
upload_df = pd.concat([upload_df, new_columns], axis=1)

######
## Complete df with zero malaria data
######
age_metadata_df = read_parquet_with_integer_ids(age_metadata_path)
age_group_ids = age_metadata_df["age_group_id"].unique()
sex_ids = [1, 2]  # 1
missing_location_ids = set(fhs_location_ids) - set(upload_df["location_id"].unique())
missing_location_ids.discard(44858)

combinations = list(itertools.product(missing_location_ids, year_ids, age_group_ids, sex_ids))
zero_df = pd.DataFrame(combinations, columns=as_merge_variables)

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

columns_to_keep = as_merge_variables + ['population'] + fhs_draws
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
    # Vectorized: multiply all draw columns at once
    draw_cols = rows_to_aggregate[fhs_draws]
    pop_col = rows_to_aggregate['population']
    rows_to_aggregate[fhs_draws] = draw_cols.multiply(pop_col, axis=0)


# Aggregate the removed rows by year, age_group_id, and sex_id
agg_dict = {
    'population': 'sum',
    **{col: 'sum' for col in fhs_draws}
}

aggregated_data = rows_to_aggregate.groupby(['year_id', 'age_group_id', 'sex_id']).agg(agg_dict).reset_index()

if metric == "rate":
    # Vectorized: divide all draw columns at once
    draw_cols = aggregated_data[fhs_draws] 
    pop_col = aggregated_data['population']
    aggregated_data[fhs_draws] = draw_cols.div(pop_col, axis=0)

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

# Get draw columns before the conditional block
draw_cols = upload_df.columns[upload_df.columns.str.startswith('draw_')].tolist()

if fhs_flag == 1:
    upload_df["release_id"] = release_id
    upload_df['scenario'] = scenario
    upload_df["measure_id"] = measure_id
    upload_df["metric_id"] = metric_id
    upload_df["cause_id"] = cause_id
    columns_to_select = ["measure_id", "metric_id", "cause_id", "location_id", "year_id", "age_group_id", "sex_id", "release_id", "scenario"] + draw_cols
    upload_df = upload_df[columns_to_select]
    write_hdf(upload_df, upload_file_path, 
        data_columns= as_merge_variables)
else:
    upload_df['measure'] = measure
    upload_df['metric'] = metric
    upload_df['cause'] = cause
    upload_df['ssp_scenario'] = ssp_scenario
    if cause == "malaria":
        upload_df['dah_scenario'] = dah_scenario
        columns_to_select = ['location_id', 'year_id', 'age_group_id', 'sex_id', 'measure', 'metric', 'cause', 'ssp_scenario', 'dah_scenario', 'population'] + draw_cols
        upload_df = upload_df[columns_to_select]
        write_hdf(upload_df, upload_file_path, 
            data_columns=as_merge_variables + ['population'])
    else:
        columns_to_select = ['location_id', 'year_id', 'age_group_id', 'sex_id', 'measure', 'metric', 'cause', 'ssp_scenario', 'population'] + draw_cols
        upload_df = upload_df[columns_to_select]
        write_hdf(upload_df, upload_file_path, 
            data_columns=as_merge_variables + ['population'])

print("Saved the age-sex-specific file")

if fhs_flag == 0:
    print(f"Starting aggregation from {len(upload_df)} rows...")

    # 2. Get draw columns more efficiently
    draw_cols = [col for col in upload_df.columns if col.startswith('draw_')]
    
    # 3. Early check - exit if no population column when needed
    has_population = 'population' in upload_df.columns
    if metric_id == 3 and not has_population:
        raise ValueError("Rate conversion requires population column")
    
    # 4. Pre-create constant columns to avoid repeated operations
    constant_cols = {
        'measure': measure,
        'metric': metric, 
        'cause': cause,
        'ssp_scenario': ssp_scenario
    }
    if cause == "malaria":
        constant_cols['dah_scenario'] = dah_scenario
    
    # 5. Create working DataFrame with only needed columns
    groupby_cols = ['location_id', 'year_id']
    needed_cols = groupby_cols + draw_cols
    if has_population:
        needed_cols.append('population')
    
    work_df = upload_df[needed_cols].copy()  # Only copy what we need
    
    # 6. Handle rate conversion BEFORE aggregation
    if metric_id == 3 and has_population:
        # Vectorized operation across all draw columns at once
        work_df[draw_cols] = work_df[draw_cols].multiply(work_df['population'], axis=0)
    
    # 7. Optimize groupby with categoricals (only if many unique values)
    if work_df['location_id'].nunique() > 100:  # Only categorize if beneficial
        work_df['location_id'] = work_df['location_id'].astype('category')
    if work_df['year_id'].nunique() > 10:
        work_df['year_id'] = work_df['year_id'].astype('category')
    
    # 8. Fast aggregation with pre-built dict
    agg_dict = {col: 'sum' for col in draw_cols}
    if has_population:
        agg_dict['population'] = 'sum'
    
    df_agg = work_df.groupby(groupby_cols, as_index=False, observed=True).agg(agg_dict)
    
    # 9. Convert back to rates AFTER aggregation
    if metric_id == 3 and has_population:
        df_agg[draw_cols] = df_agg[draw_cols].div(df_agg['population'], axis=0)
    
    print(f"Aggregated to {len(df_agg)} rows")
    
    # 10. Add constant columns efficiently
    for col, value in constant_cols.items():
        df_agg[col] = value
    
    # 11. Select columns in final order (avoid reordering)
    if cause == "malaria":
        final_cols = ["measure", "metric", "cause", "location_id", "year_id", 
                     "ssp_scenario", "dah_scenario", "population"] + draw_cols
    else:
        final_cols = ["measure", "metric", "cause", "location_id", "year_id", 
                     "ssp_scenario", "population"] + draw_cols
    
    df_agg = df_agg[final_cols]
    
    # 12. Build path more efficiently
    path_parts = [
        UPLOAD_DATA_PATH, "upload_folders", run_date,
        f"aa_cause_id_{cause_id}_measure_id_{measure_id}_metric_id_{metric_id}_ssp_scenario_{ssp_scenario}"
    ]
    if cause == 'malaria':
        path_parts[-1] += f"_dah_scenario_{dah_scenario}"
    
    aa_upload_folder_path = "/".join(path_parts)
    aa_upload_file_path = f"{aa_upload_folder_path}/draws.h5"
    
    # 13. Create directory and save
    mkdir(aa_upload_folder_path, exist_ok=True, parents=True)
    
    # 14. Only index columns that vary (skip constants)
    variable_data_cols = []
    for col in ['location_id', 'year_id', 'measure', 'metric', 'cause']:
        if df_agg[col].nunique() > 1:
            variable_data_cols.append(col)
    
    write_hdf(df_agg, aa_upload_file_path, data_columns=variable_data_cols)
