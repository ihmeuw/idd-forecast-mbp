import argparse

import numpy as np
import xarray as xr
from rra_tools.shell_tools import mkdir

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.hdf5 import write_hdf
from idd_forecast_mbp.lib.io.netcdf import read_netcdf_with_integer_ids
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.processing.fhs_format import aggregate_locations, counts_to_rates

# --- ARGUMENT PARSING ---

parser = argparse.ArgumentParser(
    description="Produce FHS-format AS draws (.h5) for upload to the FHS pipeline."
)
parser.add_argument("--cause", type=str, default="malaria")
parser.add_argument("--ssp_scenario", type=str, required=True)
parser.add_argument("--dah_scenario", type=str, default="Baseline")
parser.add_argument("--measure", type=str, default="mortality")
parser.add_argument("--run_date", type=str, required=True)
parser.add_argument("--release_id", type=int, default=9)
args = parser.parse_args()

cause = args.cause
ssp_scenario = args.ssp_scenario
dah_scenario = args.dah_scenario
measure = args.measure
run_date = args.run_date
release_id = args.release_id

# --- PATHS AND CONSTANTS ---

PROCESSED_DATA_PATH = mbpc.MODEL_ROOT / "02-processed_data"
UPLOAD_DATA_PATH = mbpc.MODEL_ROOT / "05-upload_data"
FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"

ssp_draws = mbpc.draws
fhs_draws = mbpc.fhs_draws  # ["draw_0", ..., "draw_99"]
cause_map = mbpc.cause_map
measure_map = mbpc.measure_map
metric_map = mbpc.metric_map
ssp_scenarios = mbpc.ssp_scenarios

scenario = ssp_scenarios[ssp_scenario]["dhs_scenario"]
cause_id = cause_map[cause]["cause_id"]
measure_id = measure_map[measure]["measure_id"]
metric_id = metric_map["rate"]["metric_id"]

dah_text = f"_dah_scenario_{dah_scenario}" if cause != "dengue" else ""

upload_folder_path = (
    UPLOAD_DATA_PATH
    / "fhs_upload_folders"
    / run_date
    / f"cause_id_{cause_id}_measure_id_{measure_id}_scenario_{scenario}_{run_date}"
)
upload_file_path = upload_folder_path / "draws.h5"
mkdir(upload_folder_path, exist_ok=True, parents=True)
print(f"Output: {upload_file_path}")

# --- HIERARCHY: FHS LOCATIONS ---

hierarchy_df = read_parquet_with_integer_ids(
    f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet"
)
fhs_hierarchy_df = hierarchy_df[hierarchy_df["in_fhs_hierarchy"] == True]
# These three Ethiopian sub-nationals are aggregated into 44858 for FHS.
swap_location_ids = [60908, 95069, 94364]
eth_agg_loc = 44858
fhs_location_ids = sorted(
    set(fhs_hierarchy_df["location_id"].unique().tolist() + swap_location_ids)
)

year_ids = list(range(2022, 2101))

# --- POPULATION ---

pop_ds = read_netcdf_with_integer_ids(
    f"{PROCESSED_DATA_PATH}/as_2023_full_population_ds.nc"
)
pop_locs = list(pop_ds.location_id.values)
pop_years = list(pop_ds.year_id.values)

loc_filter = sorted(set(loc for loc in fhs_location_ids if loc in pop_locs))
year_filter = sorted(set(yr for yr in year_ids if yr in pop_years))
pop_ds = pop_ds.sel(location_id=loc_filter, year_id=year_filter)

# --- AGE GROUPS ---

age_metadata_df = read_parquet_with_integer_ids(f"{FHS_DATA_PATH}/age_metadata.parquet")
age_group_ids_full = sorted(age_metadata_df["age_group_id"].unique().tolist())

# --- LOAD ALL DRAWS ---

draw_path_template = (
    "{UPLOAD_DATA_PATH}/upload_folders/{run_date}/"
    "full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}"
    "{dah_text}_draw_{draw}_with_predictions.nc"
)
file_paths = [
    draw_path_template.format(
        UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
        run_date=run_date,
        cause=cause,
        measure=measure,
        ssp_scenario=ssp_scenario,
        dah_text=dah_text,
        draw=draw,
    )
    for draw in ssp_draws
]

print(f"Loading {len(file_paths)} draw files...")

preprocess_func = lambda ds: ds.reindex(
    location_id=loc_filter, year_id=year_filter
).drop_vars(["gbd_location_id", "aa_count", "level"], errors="ignore")

count_ds = xr.open_mfdataset(
    file_paths,
    combine="nested",
    concat_dim="draw_id",
    chunks="auto",
    drop_variables=["gbd_location_id", "aa_count", "level"],
    preprocess=preprocess_func,
)
count_ds = count_ds.assign_coords(draw_id=[int(d) for d in ssp_draws])

# Fill missing FHS locations (no modeled malaria/dengue) with zero counts.
existing_locs = list(count_ds.location_id.values)
missing_locs = sorted(set(fhs_location_ids) - set(existing_locs) - {eth_agg_loc})
complete_locs = sorted(existing_locs + missing_locs)

complete_coords = {
    "location_id": complete_locs,
    "year_id": year_filter,
    "age_group_id": age_group_ids_full,
    "sex_id": [1, 2],
    "draw_id": [int(d) for d in ssp_draws],
}
count_ds = (
    count_ds.reindex(complete_coords, fill_value=0.0)
    .rename({"count_pred": "val"})
    .chunk({"draw_id": 10, "location_id": -1, "year_id": -1, "age_group_id": -1, "sex_id": -1})
)

pop_full = pop_ds.reindex(
    location_id=complete_locs,
    year_id=year_filter,
    age_group_id=age_group_ids_full,
    sex_id=[1, 2],
    fill_value=0.0,
)

# --- ETHIOPIAN AGGREGATION (count space) ---
# The FHS system expects location 44858 (Ethiopia national) rather than the
# three sub-nationals 60908, 95069, 94364.

print(f"Aggregating Ethiopian sub-nationals {swap_location_ids} → {eth_agg_loc}...")
count_ds, pop_full = aggregate_locations(
    count_ds.rename({"val": "count"}),
    pop_full,
    source_locs=swap_location_ids,
    target_loc=eth_agg_loc,
)
count_ds = count_ds.rename({"count": "val"})

# --- RATE CALCULATION ---
# Divide counts by population. Where population is zero, rate is zero.

print("Computing rates (count / population)...")
rate_da = counts_to_rates(count_ds["val"], pop_full["population"])

# Force compute before pivoting.
rate_da = rate_da.compute()

# --- CONVERT TO WIDE DATAFRAME ---
# Rows: (location_id, year_id, age_group_id, sex_id)
# Columns: draw_0 .. draw_99

print("Pivoting to wide format...")
rate_df = rate_da.to_series().unstack("draw_id")
rate_df.columns = [f"draw_{i}" for i in rate_df.columns]
rate_df = rate_df.reset_index()

# Add FHS upload metadata columns.
rate_df["measure_id"] = measure_id
rate_df["metric_id"] = metric_id
rate_df["cause_id"] = cause_id
rate_df["release_id"] = release_id
rate_df["scenario"] = scenario

cols = (
    ["measure_id", "metric_id", "cause_id", "location_id", "year_id",
     "age_group_id", "sex_id", "release_id", "scenario"]
    + fhs_draws
)
rate_df = rate_df[cols]

# --- WRITE ---

print(f"Writing {len(rate_df):,} rows to {upload_file_path}...")
write_hdf(
    rate_df,
    upload_file_path,
    data_columns=["location_id", "year_id", "age_group_id", "sex_id"],
)
print("Done.")
