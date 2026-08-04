import pandas as pd
from pathlib import Path
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.io.netcdf import read_netcdf_with_integer_ids, write_netcdf, convert_with_preset
from idd_forecast_mbp.lib.processing.aggregation import aggregate_to_parent

TEST_DIR = Path("/mnt/team/idd/pub/forecast-mbp/test_output/05-upload_data")

cause = "malaria"
measure = "mortality"
ssp_scenario = "ssp126"
dah_scenario = "Baseline"
draw = "001"
hold_variable = "None"

measure_map = mbpc.measure_map

PROCESSED_DATA_PATH = mbpc.MODEL_ROOT / "02-processed_data"
FORECASTING_DATA_PATH = mbpc.MODEL_ROOT / "04-forecasting_data"

forecast_ds_path = f"{FORECASTING_DATA_PATH}/as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
output_path = TEST_DIR / f"full_as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"

hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

as_merge_variables = mbpc.as_merge_variables

def process_forecast_data(forecast_ds_path, measure, hierarchy_df):
    ds = read_netcdf_with_integer_ids(forecast_ds_path)
    df = ds.to_dataframe().reset_index()
    df = df[df["year_id"] >= 2022]
    short = measure_map[measure]["short"]
    df = df.rename(columns={col: col.replace(f'{cause}_{short}_', '') for col in df.columns if f'{cause}_{short}_' in col})
    drop_cols = [col for col in df.columns if 'rate' in col or 'pop' in col]
    df = df.drop(columns=drop_cols)
    df = df[as_merge_variables + ['count_pred']]

    df = df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")

    child_df = df.copy()

    for level in reversed(range(1, 6)):
        print(f"Processing level {level}...")
        parent_df = aggregate_to_parent(child_df, hierarchy_df, 'count_pred', preserve_age_sex=True)
        parent_df = parent_df.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
        df = pd.concat([df, parent_df], ignore_index=True)
        child_df = parent_df.copy()

    df = df.drop(columns=['level'])
    return df

full_hierarchy_forecast_df = process_forecast_data(forecast_ds_path, measure, hierarchy_df)

full_hierarchy_forecast_ds = convert_with_preset(
    full_hierarchy_forecast_df,
    preset='as_variables',
    variable_dtypes={
        'count_pred': 'float32',
    },
    validate_dimensions=False
)

write_netcdf(
    full_hierarchy_forecast_ds,
    output_path,
    compression=True,
    compression_level=4,
    chunking=True,
    max_retries=3
)
print(f"Wrote {output_path}")
