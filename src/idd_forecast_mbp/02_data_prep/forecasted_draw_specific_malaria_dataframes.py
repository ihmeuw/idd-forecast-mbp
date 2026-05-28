import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
from typing import Literal, NamedTuple
import itertools
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.processing.helpers import level_filter
from idd_forecast_mbp.lib.processing.scenarios import generate_dah_scenarios


def main(
    ssp_scenario: str,
    draw: str,
    suitability_variant: str = mbpc.MALARIA_SUITABILITY_VARIANT,
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    mal_raked_aa_read_path: Path = mbpc.MAL_RAKED_AA_READ_PATH,
    mal_raked_as_read_path: Path = mbpc.MAL_RAKED_AS_READ_PATH,
    forecasting_data_read_path: Path = mbpc.FORECASTING_DATA_PATH,
    forecasting_data_write_path: Path = mbpc.FORECASTING_DATA_PATH,
) -> None:
    forecasting_data_read_path = Path(forecasting_data_read_path)
    forecasting_data_write_path = Path(forecasting_data_write_path) / suitability_variant
    forecasting_data_write_path.mkdir(parents=True, exist_ok=True)

    ssp_scenarios = mbpc.ssp_scenarios
    rcp_scenario = ssp_scenarios[ssp_scenario]["rcp_scenario"]

    malaria_mortality_threshold = 1

    cause = "malaria"
    cause_map = mbpc.cause_map
    reference_age_group_id = cause_map[cause]['reference_age_group_id']
    reference_sex_id = cause_map[cause]['reference_sex_id']

    forecast_non_draw_df_path = forecasting_data_read_path / f"{cause}_forecast_scenario_{ssp_scenario}_non_draw_part.parquet"
    aa_merge_variables = mbpc.aa_merge_variables

    hierarchy_df_path = Path(hierarchy_read_path) / f"full_hierarchy_2023_{lsae_hierarchy}.parquet"
    hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

    CLIMATE_DATA_PATH = str(mbpc.CLIMATE_AGGREGATES_PATH / lsae_hierarchy)

    aa_full_cause_df_path = Path(mal_raked_aa_read_path) / f"aa_full_{cause}_df.parquet"
    as_full_cause_df_path = Path(mal_raked_as_read_path) / f"as_full_{cause}_df.parquet"

    cc_sensitive_paths = {
        "total_precipitation":      "{CLIMATE_DATA_PATH}/total_precipitation_{ssp_scenario}.parquet",
        "relative_humidity":        "{CLIMATE_DATA_PATH}/relative_humidity_{ssp_scenario}.parquet",
        "malaria_suitability":      mbpc.get_malaria_suitability_path(suitability_variant, ssp_scenario, lsae_hierarchy),
    }

    aa_malaria_df = read_parquet_with_integer_ids(aa_full_cause_df_path,
        filters=[level_filter(hierarchy_df, start_level=3, end_level=5)])

    aa_malaria_df = aa_malaria_df.merge(hierarchy_df[['location_id', 'A0_location_id', 'level']],
        how="left",
        on="location_id")

    aa_A0_malaria_df = aa_malaria_df[(aa_malaria_df["location_id"] == aa_malaria_df["A0_location_id"]) & (aa_malaria_df["year_id"] == 2022)].copy()
    aa_A0_malaria_df = aa_A0_malaria_df[aa_A0_malaria_df['malaria_mort_count'] > malaria_mortality_threshold].copy()
    A0_malaria_ids = aa_A0_malaria_df['A0_location_id'].unique()

    aa_malaria_df = aa_malaria_df[aa_malaria_df['A0_location_id'].isin(A0_malaria_ids)].copy()
    aa_malaria_df = aa_malaria_df[
        (aa_malaria_df["malaria_pfpr"] > 0) &
        (aa_malaria_df["malaria_mort_count"] > 0) &
        (aa_malaria_df["malaria_inc_count"] >= 0) &
        (aa_malaria_df["level"] == 5)].copy()

    aa_malaria_ids = aa_malaria_df['location_id'].unique()
    aa_malaria_filter = ('location_id', 'in', aa_malaria_ids.tolist())

    reference_age_group_filter = ('age_group_id', '==', reference_age_group_id)
    reference_sex_filter = ('sex_id', '==', reference_sex_id)
    as_base_malaria_df = read_parquet_with_integer_ids(as_full_cause_df_path,
        filters=[reference_age_group_filter, reference_sex_filter, aa_malaria_filter]).drop(columns=['age_group_id', 'sex_id', 'aa_population'])

    as_base_malaria_df = as_base_malaria_df.rename(columns=lambda x: f"base_{x}" if (x.startswith('malaria_') or x.startswith('pop_')) else x)
    as_base_malaria_df = as_base_malaria_df.merge(
        aa_malaria_df[aa_merge_variables + ['malaria_pfpr']],
        how="left",
        on=aa_merge_variables)

    covariates_to_log_transform = [col for col in as_base_malaria_df.columns if 'rate' in col]
    for col in covariates_to_log_transform:
        as_base_malaria_df[f"log_{col}"] = np.log(as_base_malaria_df[col])

    as_base_malaria_df["logit_malaria_pfpr"] = np.log(0.999 * as_base_malaria_df["malaria_pfpr"] / (1 - 0.999 * as_base_malaria_df["malaria_pfpr"]))

    forecast_by_draw_df = read_parquet_with_integer_ids(forecast_non_draw_df_path,
        filters=[aa_malaria_filter])

    forecast_by_draw_df["draw"] = draw
    forecast_by_draw_df = forecast_by_draw_df.rename(columns={
        'population': 'aa_population'})

    forecast_by_draw_df = forecast_by_draw_df.merge(as_base_malaria_df,
        how='left',
        on=['location_id', 'year_id'])

    for key, path_template in cc_sensitive_paths.items():
        path = path_template.format(CLIMATE_DATA_PATH=CLIMATE_DATA_PATH, ssp_scenario=ssp_scenario)
        columns_to_read = ["location_id", "year_id", draw]
        df = read_parquet_with_integer_ids(path, columns=columns_to_read,
            filters=[aa_malaria_filter])
        df = df.rename(columns={draw: key})
        forecast_by_draw_df = pd.merge(forecast_by_draw_df, df, on=["location_id", "year_id"], how="left")

    covariates_to_log_transform = [
        "mal_DAH_total_per_capita",
        "gdppc_mean",
    ]

    for col in covariates_to_log_transform:
        forecast_by_draw_df[f"log_{col}"] = np.log(forecast_by_draw_df[col] + 1e-6)

    pakistan_id = 165
    pakistan_children_ids = hierarchy_df[hierarchy_df['parent_id'] == pakistan_id]['location_id'].tolist()
    pakistan_grandchildren_ids = hierarchy_df[hierarchy_df['parent_id'].isin(pakistan_children_ids)]['location_id'].tolist()
    all_pakistan_locations = [pakistan_id] + pakistan_children_ids + pakistan_grandchildren_ids

    forecast_by_draw_df['year_to_rake_to'] = 2022
    forecast_by_draw_df.loc[forecast_by_draw_df['location_id'].isin(all_pakistan_locations), 'year_to_rake_to'] = 2021

    dah_scenarios, dah_scenario_names = generate_dah_scenarios(
        baseline_df=forecast_by_draw_df,
        ssp_scenario=ssp_scenario
    )

    for dah_scenario_df, dah_scenario_name in zip(dah_scenarios, dah_scenario_names):
        dah_scenario_df_path = forecasting_data_write_path / f"malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet"
        write_parquet(dah_scenario_df, dah_scenario_df_path)

    forecast_by_draw_df_path = forecasting_data_write_path / f"{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}.parquet"
    write_parquet(forecast_by_draw_df, forecast_by_draw_df_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add DAH Scenarios and create draw level dataframes for forecasting malaria")
    parser.add_argument("--ssp_scenario", type=str, required=True, help="ssp scenario number (ssp126, ssp245, ssp585)")
    parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")
    parser.add_argument("--suitability_variant", type=str, default=mbpc.MALARIA_SUITABILITY_VARIANT)
    args = parser.parse_args()
    main(ssp_scenario=args.ssp_scenario, draw=args.draw, suitability_variant=args.suitability_variant)
