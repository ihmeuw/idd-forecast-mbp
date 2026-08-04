"""
Input: Forecast data for a specific cause (malaria), SSP scenario, DAH scenario, and draw number
Output: Age-sex specific forecasts with predicted rates and counts saved as netCDF files
Objective: Disaggregate all-age population forecasts into age-sex specific estimates using relative risk patterns from reference data
"""
import numpy as np
import gc
import time
import psutil
from pathlib import Path
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.io.netcdf import convert_with_preset, write_netcdf, read_netcdf_with_integer_ids
from idd_forecast_mbp.lib.processing.disaggregation import disaggregate_age_sex_malaria


def log_time_and_memory(message):
    process = psutil.Process()
    memory_gb = process.memory_info().rss / (1024 ** 3)
    print(f"{message}: Memory usage: {memory_gb:.2f} GB")
    return time.time()


def main(
    ssp_scenario: str,
    draw: str,
    dah_scenario: str = "Baseline",
    hold_variable: str = "None",
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    processed_data_path: Path = mbpc.MODEL_ROOT / "02-processed_data",
    forecasting_data_read_path: Path = mbpc.FORECASTING_DATA_PATH,
    forecasting_data_write_path: Path = mbpc.FORECASTING_DATA_PATH,
) -> None:
    hierarchy_read_path = Path(hierarchy_read_path)
    processed_data_path = Path(processed_data_path)
    forecasting_data_read_path = Path(forecasting_data_read_path)
    forecasting_data_write_path = Path(forecasting_data_write_path)
    forecasting_data_write_path.mkdir(parents=True, exist_ok=True)

    log_time_and_memory("Starting script")

    cause = "malaria"
    cause_map = mbpc.cause_map
    as_merge_variables = mbpc.as_merge_variables

    if hold_variable == "None":
        input_cause_draw_path = forecasting_data_read_path / f"{cause}_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
        output_malaria_incidence_draw_path = forecasting_data_write_path / f"as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
        output_malaria_mortality_draw_path = forecasting_data_write_path / f"as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
    else:
        input_cause_draw_path = forecasting_data_read_path / f"{cause}_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
        output_malaria_incidence_draw_path = forecasting_data_write_path / f"as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
        output_malaria_mortality_draw_path = forecasting_data_write_path / f"as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"

    aa_full_population_df_path = processed_data_path / "aa_2023_full_population.parquet"
    as_full_population_df_path = processed_data_path / "as_2023_full_population.parquet"
    as_md_gbd_malaria_df_path = processed_data_path / "as_md_gbd_malaria_df.parquet"
    full_2023_hierarchy_path = hierarchy_read_path / "full_hierarchy_2023_lsae_1209.parquet"

    hierarchy_df = read_parquet_with_integer_ids(full_2023_hierarchy_path)
    print("Loaded hierarchy")

    forecast_years = list(range(2022, 2101))
    forecast_year_filter = ("year_id", "in", forecast_years)

    ds = read_netcdf_with_integer_ids(input_cause_draw_path)
    ds = ds.sel(year_id=forecast_years)
    df = ds.to_dataframe().reset_index()
    modeling_location_ids = df["location_id"].unique().tolist()
    modeling_location_filter = ("location_id", "in", modeling_location_ids)

    aa_pop_df = read_parquet_with_integer_ids(
        aa_full_population_df_path,
        filters=[modeling_location_filter, forecast_year_filter],
    )
    aa_pop_df = aa_pop_df.rename(columns={"population": "aa_population"})
    df = df.merge(aa_pop_df[["location_id", "year_id", "aa_population"]],
                  on=["location_id", "year_id"], how="left")

    df["aa_malaria_mort_rate"] = np.exp(df["log_aa_malaria_mort_rate_pred"])
    df["aa_malaria_inc_rate"] = np.exp(df["log_aa_malaria_inc_rate_pred"])
    df = df.drop(columns=["log_aa_malaria_mort_rate_pred", "log_aa_malaria_inc_rate_pred"])
    for col in df.columns:
        if "malaria" in col:
            df[col] = df[col].fillna(0)

    df["aa_malaria_mort_count"] = df["aa_malaria_mort_rate"] * df["aa_population"]
    df["aa_malaria_inc_count"] = df["aa_malaria_inc_rate"] * df["aa_population"]

    as_md_gbd_malaria_df = read_parquet_with_integer_ids(as_md_gbd_malaria_df_path)
    as_md_gbd_malaria_df = as_md_gbd_malaria_df.rename(columns={"location_id": "gbd_location_id"})
    as_md_gbd_malaria_df = as_md_gbd_malaria_df[
        (as_md_gbd_malaria_df["rr_inc_as"] > 0) | (as_md_gbd_malaria_df["rr_mort_as"] > 0)
    ].copy()
    gbd_location_ids = as_md_gbd_malaria_df["gbd_location_id"].unique().tolist()

    level_5_location_ids = hierarchy_df[
        (hierarchy_df["level"] == 5) & (hierarchy_df["gbd_location_id"].isin(gbd_location_ids))
    ]["location_id"].unique().tolist()
    level_5_location_filter = ("location_id", "in", level_5_location_ids)

    forecast_columns_to_read = as_merge_variables + ["population"]
    forecast_df = read_parquet_with_integer_ids(
        as_full_population_df_path,
        columns=forecast_columns_to_read,
        filters=[level_5_location_filter, forecast_year_filter],
    )

    forecast_df = forecast_df.merge(
        df[["location_id", "year_id", "aa_malaria_mort_count", "aa_malaria_inc_count"]],
        on=["location_id", "year_id"], how="left",
    )

    del df
    gc.collect()

    forecast_df = forecast_df.merge(
        hierarchy_df[["location_id", "gbd_location_id"]],
        on="location_id", how="left",
    )
    forecast_df = forecast_df.merge(
        as_md_gbd_malaria_df[["gbd_location_id", "age_group_id", "sex_id", "rr_inc_as", "rr_mort_as"]],
        on=["gbd_location_id", "age_group_id", "sex_id"], how="left",
    )

    del as_md_gbd_malaria_df
    gc.collect()

    forecast_df = disaggregate_age_sex_malaria(forecast_df)
    forecast_df = forecast_df.drop(columns=["rr_inc_as", "rr_mort_as", "population"])

    non_measure_columns = [col for col in forecast_df.columns if "inc" not in col and "mort" not in col]
    incidence_columns = [col for col in forecast_df.columns if "inc" in col]
    mortality_columns = [col for col in forecast_df.columns if "mort" in col]

    incidence_df = forecast_df[non_measure_columns + incidence_columns]
    incidence_ds = convert_with_preset(incidence_df, preset="as_variables")
    write_netcdf(incidence_ds, output_malaria_incidence_draw_path)

    mortality_df = forecast_df[non_measure_columns + mortality_columns]
    mortality_ds = convert_with_preset(mortality_df, preset="as_variables")
    write_netcdf(mortality_ds, output_malaria_mortality_draw_path)

    del forecast_df
    gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Disaggregate all-age malaria forecasts into age-sex specific estimates"
    )
    parser.add_argument("--ssp_scenario", type=str, required=True)
    parser.add_argument("--dah_scenario", type=str, default="Baseline")
    parser.add_argument("--draw", type=str, required=True)
    parser.add_argument("--hold_variable", type=str, default="None")
    parser.add_argument("--hierarchy_read_path", type=str, default=str(mbpc.HIERARCHY_READ_PATH))
    parser.add_argument("--processed_data_path", type=str, default=str(mbpc.MODEL_ROOT / "02-processed_data"))
    parser.add_argument("--forecasting_data_read_path", type=str, default=str(mbpc.FORECASTING_DATA_PATH))
    parser.add_argument("--forecasting_data_write_path", type=str, default=str(mbpc.FORECASTING_DATA_PATH))

    args = parser.parse_args()
    main(
        ssp_scenario=args.ssp_scenario,
        draw=args.draw,
        dah_scenario=args.dah_scenario,
        hold_variable=args.hold_variable,
        hierarchy_read_path=args.hierarchy_read_path,
        processed_data_path=args.processed_data_path,
        forecasting_data_read_path=args.forecasting_data_read_path,
        forecasting_data_write_path=args.forecasting_data_write_path,
    )
