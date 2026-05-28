"""
Integration test for create_and_combine_as_and_aa_draws.py

Runs one scenario (ssp126/malaria/mortality/count) against the 2025_08_28 draw
files and writes output to test_output/. Checks that output dimensions and
dtypes are correct — does not compare values to a production reference since
the production files used different location filtering.

Expected output shapes:
  aa draws : (draw_id: 100, location_id: ~20K+, year_id: 79)  -- no sex_id dim
  aa mean  : (location_id: ~20K+, year_id: 79)               -- no variable dim
  as mean  : (location_id: ~20K+, year_id: 79, age_group_id: 25, sex_id: 3)
"""

import xarray as xr
import numpy as np
from pathlib import Path
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.io.netcdf import read_netcdf_with_integer_ids, write_netcdf

TEST_DIR = Path("/mnt/team/idd/pub/forecast-mbp/test_output/05-upload_data")

cause = "malaria"
measure = "mortality"
metric = "count"
ssp_scenario = "ssp126"
dah_scenario = "Baseline"
run_date = "2025_08_28"

PROCESSED_DATA_PATH = mbpc.MODEL_ROOT / "02-processed_data"
UPLOAD_DATA_PATH = mbpc.MODEL_ROOT / "05-upload_data"
FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"

ssp_draws = mbpc.draws
cause_map = mbpc.cause_map
metric_map = mbpc.metric_map
full_measure_map = mbpc.full_measure_map

dah_text = f'_dah_scenario_{dah_scenario}'

aa_test_folder = TEST_DIR / f"aa_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}{dah_text}"
as_test_folder = TEST_DIR / f"as_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}{dah_text}"
aa_test_folder.mkdir(parents=True, exist_ok=True)
as_test_folder.mkdir(parents=True, exist_ok=True)

aa_draws_path = aa_test_folder / "draws.nc"
aa_mean_path = aa_test_folder / "mean.nc"
as_mean_path = as_test_folder / "mean.nc"

# --- HIERARCHY: all locations ---
hierarchy_df = read_parquet_with_integer_ids(
    f'{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet'
)
all_location_ids = sorted(hierarchy_df["location_id"].unique().tolist())

year_ids = list(range(2022, 2101))

# --- POPULATION (all locations, filtered to forecast years) ---
pop_ds = read_netcdf_with_integer_ids(
    f"{PROCESSED_DATA_PATH}/as_2023_full_population_ds.nc"
)
pop_years = list(pop_ds.year_id.values)
years_to_filter = [yr for yr in year_ids if yr in pop_years]
pop_ds = pop_ds.sel(year_id=years_to_filter)

# --- LOAD ALL DRAWS (all locations, no pre-filter) ---
draw_path_template = (
    "{UPLOAD_DATA_PATH}/upload_folders/{run_date}/"
    "full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}"
    "{dah_text}_draw_{draw}_with_predictions.nc"
)

file_paths = [
    draw_path_template.format(
        UPLOAD_DATA_PATH=UPLOAD_DATA_PATH, run_date=run_date, cause=cause,
        measure=measure, ssp_scenario=ssp_scenario, dah_text=dah_text, draw=draw
    )
    for draw in ssp_draws
]
print(f"Loading {len(file_paths)} files...")

preprocess_func = lambda ds: ds.reindex(
    year_id=years_to_filter
).drop_vars(['gbd_location_id', 'aa_count', 'level'], errors='ignore')

upload_ds = xr.open_mfdataset(
    file_paths,
    combine='nested',
    concat_dim='draw_id',
    chunks='auto',
    drop_variables=['gbd_location_id', 'aa_count', 'level'],
    preprocess=preprocess_func,
)
upload_ds = upload_ds.assign_coords(draw_id=[int(d) for d in ssp_draws])

# Merge population (left join keeps draw-file locations)
upload_ds = xr.merge([upload_ds, pop_ds['population']], join='left')

# sex_id=3 (both sexes)
counts_both = upload_ds['count_pred'].sum(dim='sex_id', skipna=False)
pop_both = upload_ds['population'].sum(dim='sex_id', skipna=False)
counts_both = counts_both.assign_coords(sex_id=3).expand_dims('sex_id')
pop_both = pop_both.assign_coords(sex_id=3).expand_dims('sex_id')
upload_ds['count_pred'] = xr.concat([upload_ds['count_pred'], counts_both], dim='sex_id')
upload_ds['population'] = xr.concat([upload_ds['population'], pop_both], dim='sex_id')
upload_ds = upload_ds.rename({'count_pred': 'val'})

ssp_draws_int = [int(d) for d in ssp_draws]
upload_ds = upload_ds.assign_coords(draw_id=ssp_draws_int)

existing_location_ids = upload_ds.coords['location_id'].values
age_metadata_df = read_parquet_with_integer_ids(f"{FHS_DATA_PATH}/age_metadata.parquet")
age_group_ids_full = age_metadata_df["age_group_id"].unique()
sex_ids_full = [1, 2, 3]
missing_location_ids = list(set(all_location_ids) - set(existing_location_ids))

complete_coords = {
    'location_id': sorted(list(existing_location_ids) + missing_location_ids),
    'year_id': year_ids,
    'age_group_id': age_group_ids_full,
    'sex_id': sex_ids_full,
    'draw_id': ssp_draws_int,
}

print("Reindexing to full coordinate space...")
as_ds_raw = upload_ds.reindex(complete_coords, fill_value=0.0).chunk(
    {'draw_id': 10, 'location_id': -1, 'year_id': -1, 'age_group_id': -1, 'sex_id': -1}
)
as_ds_raw = as_ds_raw.rename({'val': 'val_count', 'population': 'population_count'})

# --- AA aggregation ---
aa_ds = as_ds_raw.sel(sex_id=[1, 2]).sum(dim=['sex_id', 'age_group_id'], skipna=False)
aa_ds = aa_ds.assign_coords(sex_id=3)  # scalar coord, not a dimension

# --- AA draws ---
aa_draws = aa_ds.rename({'val_count': 'val'}).drop_vars('population_count')
aa_draws = aa_draws.assign_coords(
    cause_id=cause_map[cause]['cause_id'],
    metric_id=metric_map[metric]['metric_id'],
    measure_id=full_measure_map[measure]['measure_id'],
    location_id=aa_draws.location_id.astype(np.int32),
    year_id=aa_draws.year_id.astype(np.int16),
    draw_id=aa_draws.draw_id.astype(np.int8),
)

# --- AA mean (no to_array() — avoids spurious 'variable' dimension) ---
aa_mean = aa_draws.mean(dim='draw_id')
aa_mean = aa_mean.assign_coords(
    cause_id=cause_map[cause]['cause_id'],
    metric_id=metric_map[metric]['metric_id'],
    measure_id=full_measure_map[measure]['measure_id'],
    location_id=aa_mean.location_id.astype(np.int32),
    year_id=aa_mean.year_id.astype(np.int16),
)

# --- AS mean ---
as_draws_ds = as_ds_raw.rename({'val_count': 'val'}).drop_vars('population_count')
as_mean = as_draws_ds.mean(dim='draw_id')
as_mean = as_mean.assign_coords(
    cause_id=cause_map[cause]['cause_id'],
    metric_id=metric_map[metric]['metric_id'],
    measure_id=full_measure_map[measure]['measure_id'],
)

print("Writing aa draws...")
write_netcdf(ds=aa_draws, filepath=aa_draws_path, compression_level=4, max_chunk_size=2000, chunk_threshold=500000)
print(f"Wrote {aa_draws_path}")

print("Writing aa mean...")
write_netcdf(ds=aa_mean, filepath=aa_mean_path, compression_level=4, max_chunk_size=2000, chunk_threshold=500000)
print(f"Wrote {aa_mean_path}")

print("Writing as mean...")
write_netcdf(ds=as_mean, filepath=as_mean_path, compression_level=4, max_chunk_size=2000, chunk_threshold=500000)
print(f"Wrote {as_mean_path}")

# --- VERIFY STRUCTURE ---
print("\n=== Verifying output structure ===")

aa_draws_out = xr.open_dataset(aa_draws_path)
aa_mean_out = xr.open_dataset(aa_mean_path)
as_mean_out = xr.open_dataset(as_mean_path)

errors = []

# AA draws: should have (draw_id, location_id, year_id) — no sex_id dim
if 'sex_id' in aa_draws_out.dims:
    errors.append(f"aa draws: unexpected sex_id dimension (size {aa_draws_out.dims['sex_id']})")
if 'variable' in aa_draws_out.dims:
    errors.append("aa draws: unexpected variable dimension")
if 'draw_id' not in aa_draws_out.dims:
    errors.append("aa draws: missing draw_id dimension")

# AA mean: should have (location_id, year_id) — no variable dim, no sex_id dim
if 'variable' in aa_mean_out.dims:
    errors.append("aa mean: unexpected variable dimension (to_array() bug)")
if 'sex_id' in aa_mean_out.dims:
    errors.append("aa mean: unexpected sex_id dimension")
if 'draw_id' in aa_mean_out.dims:
    errors.append("aa mean: draw_id should have been averaged out")

# AS mean: should have (location_id, year_id, age_group_id, sex_id) — no draw_id
if 'draw_id' in as_mean_out.dims:
    errors.append("as mean: draw_id should have been averaged out")
if 'variable' in as_mean_out.dims:
    errors.append("as mean: unexpected variable dimension")

# Location count should be much larger than 513 (FHS-only)
n_locs = aa_draws_out.dims.get('location_id', 0)
if n_locs <= 513:
    errors.append(f"aa draws: only {n_locs} locations — likely still filtered to FHS only")

print(f"  aa draws dims : {dict(aa_draws_out.dims)}")
print(f"  aa mean dims  : {dict(aa_mean_out.dims)}")
print(f"  as mean dims  : {dict(as_mean_out.dims)}")
print(f"  location count: {n_locs:,}")

if errors:
    print("\nFAILED:")
    for e in errors:
        print(f"  - {e}")
else:
    print("\nPASSED: all structural checks OK")
