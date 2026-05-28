import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.io.netcdf import read_netcdf_with_integer_ids, write_netcdf, convert_with_preset

TEST_DIR = Path("/mnt/team/idd/pub/forecast-mbp/test_output/05-upload_data")

cause = "malaria"
measure = "mortality"
ssp_scenario = "ssp126"
dah_scenario = "Baseline"
draw = "000"
hold_variable = "None"

measure_map = mbpc.measure_map

PROCESSED_DATA_PATH = mbpc.MODEL_ROOT / "02-processed_data"
FORECASTING_DATA_PATH = mbpc.MODEL_ROOT / "04-forecasting_data"

raked_base = '/mnt/team/rapidresponse/pub/malaria-denv/deliverables/2025_08_26_admin_2_counts'
folder_template_dict = {
    'dengue': '{direction}/as_cause_dengue_measure_{measure}_metric_count_ssp_scenario_{ssp_scenario}{suffix}',
    'malaria': '{direction}/as_cause_malaria_measure_{measure}_metric_count_ssp_scenario_{ssp_scenario}_dah_scenario_Baseline{suffix}'
}

output_input_matching = {
    'incidence': ['incidence', 'yld'],
    'mortality': ['mortality', 'yll']
}

output_measures = output_input_matching[measure]

hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

for output_measure in output_measures:
    output_folder = folder_template_dict[cause].format(
        direction='output/2025_09_08', measure=output_measure, ssp_scenario=ssp_scenario, suffix='_raked')
    draw_int = int(draw)
    output_ds_path = f'{raked_base}/{output_folder}/draw_{draw_int}.nc'
    output_ds = xr.open_dataset(output_ds_path)
    output_ds = output_ds.drop_vars(['draw', 'draw_id', 'scenario']).rename({'value': 'val'})
    for dim in ['draw', 'scenario']:
        if dim in output_ds.coords:
            output_ds = output_ds.squeeze(dim)

    input_folder = folder_template_dict[cause].format(
        direction='input', measure=measure, ssp_scenario=ssp_scenario, suffix='')
    input_ds_path = f'{raked_base}/{input_folder}/draw_{draw_int}.nc'
    input_ds = xr.open_dataset(input_ds_path)
    level_5_location_ids = output_ds['location_id']
    input_ds = input_ds.drop_vars(['draw_id']).sel(location_id=level_5_location_ids)

    ratio = output_ds['val'] / input_ds['val']
    ratio = xr.where(input_ds['val'] == 0, np.nan, ratio)
    ratio_ds = ratio.to_dataset(name='ratio')
    del output_ds, input_ds, ratio

    forecast_ds_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
    processed_forecast_ds_path = TEST_DIR / f"full_as_{cause}_measure_{output_measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_raked_test.nc"

    def process_forecast_data(forecast_ds_path, measure, hierarchy_df, ratio_ds):
        ds = read_netcdf_with_integer_ids(forecast_ds_path)
        vars_to_drop = [v for v in ds.data_vars if 'aa' in v or 'gbd' in v]
        ds = ds.drop_vars(vars_to_drop)

        ds_location_ids = ds["location_id"]
        tmp_ratio_ds = ratio_ds.sel(location_id=ds_location_ids)
        for var in ds.data_vars:
            ds[var] = ds[var] * tmp_ratio_ds['ratio']

        df = ds.to_dataframe().reset_index()
        df = df[df["year_id"] >= 2022]
        short = measure_map[measure]["short"]
        df = df.rename(columns={col: col.replace(f'{cause}_{short}_', '') for col in df.columns if f'{cause}_{short}_' in col})
        drop_cols = [col for col in df.columns if 'rate' in col or 'pop' in col]
        df = df.drop(columns=drop_cols)
        df = df[mbpc.as_merge_variables + ['count_pred']]

        df = df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
        child_df = df.copy()

        for level in reversed(range(1, 6)):
            print(f"  [{output_measure}] Processing level {level}...")
            child_df = child_df.merge(hierarchy_df[["location_id", "parent_id"]], on="location_id", how="left")
            parent_df = child_df.groupby(
                ["parent_id", "year_id", "age_group_id", "sex_id"]).agg({"count_pred": "sum"}).reset_index()
            parent_df = parent_df.rename(columns={"parent_id": "location_id"})
            parent_df = parent_df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
            df = pd.concat([df, parent_df], ignore_index=True)
            child_df = parent_df.copy()

        df = df.drop(columns=['level'])
        return df

    print(f"Running output_measure={output_measure}...")
    full_hierarchy_forecast_df = process_forecast_data(forecast_ds_path, measure, hierarchy_df, ratio_ds)

    full_hierarchy_forecast_ds = convert_with_preset(
        full_hierarchy_forecast_df,
        preset='as_variables',
        variable_dtypes={'count_pred': 'float32'},
        validate_dimensions=False
    )

    write_netcdf(
        full_hierarchy_forecast_ds,
        processed_forecast_ds_path,
        compression=True,
        compression_level=4,
        chunking=True,
        max_retries=3
    )
    print(f"Wrote {processed_forecast_ds_path}")
