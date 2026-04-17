################################################################
### MALARIA MODELING DATA PREPARATION
################################################################

###----------------------------------------------------------###
### 1. Setup and Configuration
### Sets up the environment with necessary libraries, constants, and path definitions.
### Establishes thresholds and directory structures for the modeling pipeline.
###----------------------------------------------------------###
import pandas as pd
import numpy as np
from pathlib import Path

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet, ensure_id_columns_are_integers, sort_id_columns
from idd_forecast_mbp.lib.io.covariate_readers import read_income_paths, merge_dataframes, read_urban_paths


def main(
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    population_read_path: Path = mbpc.POPULATION_READ_PATH,
    lsae_input_path: Path = mbpc.LSAE_INPUT_PATH,
    dah_read_path: Path = mbpc.DAH_READ_PATH,
    forecasting_data_write_path: Path = mbpc.FORECASTING_DATA_PATH,
) -> None:
    forecasting_data_write_path = Path(forecasting_data_write_path)
    forecasting_data_write_path.mkdir(parents=True, exist_ok=True)

    ssp_scenarios = mbpc.ssp_scenarios
    years = mbpc.model_years
    year_filter = ('year_id', 'in', years)

    VARIABLE_DATA_PATH = str(lsae_input_path)
    CLIMATE_DATA_PATH = f"/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/{lsae_hierarchy}"

    hierarchy_df_path = Path(hierarchy_read_path) / f"full_hierarchy_2023_{lsae_hierarchy}.parquet"
    hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

    md_location_ids = hierarchy_df[hierarchy_df['level'] == 5]['location_id'].unique().tolist()
    md_location_filter = ('location_id', 'in', md_location_ids)

    aa_full_population_df_path = Path(population_read_path) / "aa_2023_full_population_df.parquet"
    aa_full_population_df = read_parquet_with_integer_ids(aa_full_population_df_path)
    aa_merge_variables = mbpc.aa_merge_variables

    income_paths = {
        "gdppc": "{VARIABLE_DATA_PATH}/gdppc_mean.parquet",
    }

    dah_df_path = Path(dah_read_path) / "dah_df_2025_07_08.parquet"

    urban_paths = {
        "urban_threshold_300":      "{VARIABLE_DATA_PATH}/urban_threshold_300.0_simple_mean.parquet",
        "urban_threshold_1500":     "{VARIABLE_DATA_PATH}/urban_threshold_1500.0_simple_mean.parquet",
    }

    cc_sensitive_paths = {
        "flooding": f"/mnt/team/rapidresponse/pub/flooding/results/output/{lsae_hierarchy}/fldfrc_shifted0.1_sum_{{ssp_scenario}}_mean_r1i1p1f1.parquet"
    }

    for ssp_scenario in ssp_scenarios:
        print(f"Processing SSP scenario: {ssp_scenario}")
        rcp_scenario = ssp_scenarios[ssp_scenario]['rcp_scenario']

        flooding_df_path_template = cc_sensitive_paths['flooding']
        flooding_df_path = flooding_df_path_template.format(ssp_scenario=ssp_scenario)

        forecast_df = read_parquet_with_integer_ids(flooding_df_path,
                                                    filters = [year_filter, md_location_filter])
        forecast_df = forecast_df.drop(
            columns=["model", "variant", 'population']
        )

        urban_dfs = read_urban_paths(urban_paths, VARIABLE_DATA_PATH)
        forecast_df = merge_dataframes(forecast_df, urban_dfs)

        covariates_to_logit_transform = [col for col in forecast_df.columns if "urban" in col]

        for col in covariates_to_logit_transform:
            print(f"Range of {col}: {forecast_df[col].min()} to {forecast_df[col].max()}")
            clipped_values = forecast_df[col].clip(lower=0.001, upper=0.999)
            forecast_df[f"logit_{col}"] = np.log(clipped_values / (1 - clipped_values))

        forecast_df = forecast_df.merge(
            aa_full_population_df,
            on = aa_merge_variables,
            how = "left")

        forecast_df = forecast_df.merge(
            hierarchy_df[['location_id', 'A0_location_id']],
            how="left",
            left_on="location_id",
            right_on="location_id"
        )

        forecast_df = forecast_df.dropna(subset=["A0_location_id"])

        forecast_df = ensure_id_columns_are_integers(forecast_df)

        print("Reading income paths...")
        income_dfs = read_income_paths(income_paths, rcp_scenario, VARIABLE_DATA_PATH)
        forecast_df = merge_dataframes(forecast_df, income_dfs)

        print("Writing dengue forecast non-draw part...")
        cause = "dengue"
        write_parquet(forecast_df, forecasting_data_write_path / f"{cause}_forecast_scenario_{ssp_scenario}_non_draw_part.parquet")

        print("Reading DAH data...")
        dah_df = read_parquet_with_integer_ids(dah_df_path)
        dah_df = dah_df.rename(columns={'location_id': 'A0_location_id'})
        dah_df = dah_df.drop(columns=['population', 'location_name', 'iso3'], errors='ignore')
        forecast_df = forecast_df.merge(dah_df, on=["A0_location_id", "year_id"], how = "left")
        forecast_df['mal_DAH_total'] = forecast_df['mal_DAH_total'].fillna(0)
        forecast_df['mal_DAH_total_per_capita'] = forecast_df['mal_DAH_total_per_capita'].fillna(0)

        print("Writing malaria forecast non-draw part...")
        cause = "malaria"
        write_parquet(forecast_df, forecasting_data_write_path / f"{cause}_forecast_scenario_{ssp_scenario}_non_draw_part.parquet")


if __name__ == "__main__":
    main()