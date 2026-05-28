import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.netcdf import read_netcdf_with_integer_ids, write_netcdf, convert_with_preset
from idd_forecast_mbp.lib.processing.aggregation import aggregate_to_parent

TEST_DIR = Path("/mnt/team/idd/pub/forecast-mbp/test_output/05-upload_data")

cause = "malaria"
measure = "mortality"
ssp_scenario = "ssp126"
dah_scenario = "Baseline"
draw = "001"
run_date = "2025_08_28"
reference_year = 2022

PROCESSED_DATA_PATH = mbpc.MODEL_ROOT / "02-processed_data"
UPLOAD_DATA_PATH = mbpc.MODEL_ROOT / "05-upload_data"

as_full_population_ds_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population_ds.nc"
aa_full_population_ds_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population_ds.nc"
hierarchy_ds_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.nc"

# Read base_ds from production run_date (same input production used)
base_ds_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"

population_hold_path = TEST_DIR / f"full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_population.nc"
as_structure_hold_path = TEST_DIR / f"full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_as_structure.nc"

hierarchy_ds = read_netcdf_with_integer_ids(hierarchy_ds_path)
hierarchy_df = hierarchy_ds.to_dataframe().reset_index()
level_5_location_ids = hierarchy_ds.location_id.where(hierarchy_ds.level == 5, drop=True).values

aa_full_population_ds = read_netcdf_with_integer_ids(aa_full_population_ds_path)
as_full_population_ds = read_netcdf_with_integer_ids(as_full_population_ds_path)
as_full_population = as_full_population_ds['population']
aa_full_population = aa_full_population_ds['population']

as_pop_reference_year = as_full_population.sel(year_id=reference_year)
aa_pop_reference_year = aa_full_population.sel(year_id=reference_year)

base_ds = read_netcdf_with_integer_ids(base_ds_path)
hierarchy_subset = hierarchy_ds[['level']]
base_with_hierarchy = xr.merge([base_ds, hierarchy_subset], join='left')
base_ds = base_with_hierarchy.where(base_with_hierarchy.level == 5, drop=True)

as_reference_fraction = as_pop_reference_year / as_full_population
aa_reference_fraction = aa_pop_reference_year / aa_full_population

as_reference_fraction_matched = as_reference_fraction.sel(
    location_id=base_ds.location_id,
    year_id=base_ds.year_id
).astype(np.float32)
aa_reference_fraction_matched = aa_reference_fraction.sel(
    location_id=base_ds.location_id,
    year_id=base_ds.year_id
).astype(np.float32)

def process_forecast_data(ds, hierarchy_df):
    df = ds.to_dataframe().reset_index()
    df = df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
    child_df = df.copy()
    for level in reversed(range(1, 6)):
        child_df = child_df.merge(hierarchy_df[["location_id", "parent_id"]], on="location_id", how="left")
        parent_df = child_df.groupby(["parent_id", "year_id", "age_group_id", "sex_id"]).agg({"count_pred": "sum"}).reset_index()
        parent_df = parent_df.rename(columns={"parent_id": "location_id"})
        parent_df = parent_df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
        df = pd.concat([df, parent_df], ignore_index=True)
        child_df = parent_df.copy()
    df = df.drop(columns=[col for col in df.columns if col.startswith('level')])
    ds = convert_with_preset(
        df, preset='as_variables', variable_dtypes={'count_pred': 'float32'}, validate_dimensions=False)
    return ds

# population hold
hold_population_count = base_ds['count_pred'] * as_reference_fraction_matched
population_hold_ds = base_ds.copy()
population_hold_ds['count_pred'] = hold_population_count
full_population_hold_ds = process_forecast_data(population_hold_ds, hierarchy_df)
write_netcdf(full_population_hold_ds, population_hold_path, compression=True, compression_level=4, chunking=True, max_retries=3)
del population_hold_ds
print(f"Wrote {population_hold_path}")

# as_structure hold
hold_as_structure_count = base_ds['count_pred'] * as_reference_fraction_matched / aa_reference_fraction_matched
as_structure_hold_ds = base_ds.copy()
as_structure_hold_ds['count_pred'] = hold_as_structure_count
full_as_structure_hold_ds = process_forecast_data(as_structure_hold_ds, hierarchy_df)
write_netcdf(full_as_structure_hold_ds, as_structure_hold_path, compression=True, compression_level=4, chunking=True, max_retries=3)
print(f"Wrote {as_structure_hold_path}")
