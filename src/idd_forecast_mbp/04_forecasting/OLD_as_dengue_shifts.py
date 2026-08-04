"""
Input: Raked dengue forecast NC for a specific SSP scenario and draw
Output: Age-sex specific dengue incidence and mortality forecasts saved as netCDF files
Objective: Disaggregate all-age dengue forecasts into age-sex specific estimates and apply vaccination effects
"""
import numpy as np
from pathlib import Path
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.io.netcdf import convert_with_preset, write_netcdf, read_netcdf_with_integer_ids
from idd_forecast_mbp.lib.processing.disaggregation import disaggregate_age_sex_dengue


def main(
    ssp_scenario: str,
    draw: str,
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

    cause = "dengue"
    as_merge_variables = mbpc.as_merge_variables

    if hold_variable == "None":
        input_cause_draw_path = forecasting_data_read_path / f"raked_{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
        output_dengue_incidence_draw_path = forecasting_data_write_path / f"as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
        output_dengue_mortality_draw_path = forecasting_data_write_path / f"as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
    else:
        input_cause_draw_path = forecasting_data_read_path / f"raked_{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
        output_dengue_incidence_draw_path = forecasting_data_write_path / f"as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
        output_dengue_mortality_draw_path = forecasting_data_write_path / f"as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"

    as_full_population_df_path = processed_data_path / "as_2023_full_population.parquet"
    as_md_gbd_dengue_df_path = processed_data_path / "as_md_gbd_dengue_df.parquet"
    full_2023_hierarchy_path = hierarchy_read_path / "full_hierarchy_2023_lsae_1209.parquet"
    dengue_vaccine_df_path = forecasting_data_read_path / "dengue_vaccine_df.parquet"

    hierarchy_df = read_parquet_with_integer_ids(full_2023_hierarchy_path)

    future_year_ids = list(range(2022, 2101))
    year_filter = ("year_id", "in", future_year_ids)

    ds = read_netcdf_with_integer_ids(input_cause_draw_path)
    ds = ds.sel(year_id=future_year_ids)
    df = ds.to_dataframe().reset_index()
    df = df[as_merge_variables + ["logit_dengue_cfr_pred", "base_log_dengue_inc_rate_pred"]].copy()

    df["dengue_cfr_pred"] = 1 / (1 + np.exp(-df["logit_dengue_cfr_pred"]))
    df = df.drop(columns=["logit_dengue_cfr_pred"])

    df_location_ids = df["location_id"].unique().tolist()
    df_location_filter = ("location_id", "in", df_location_ids)

    pop_cols_to_read = as_merge_variables + ["population"]
    as_population_df = read_parquet_with_integer_ids(
        as_full_population_df_path,
        columns=pop_cols_to_read,
        filters=[df_location_filter, year_filter],
    )
    as_population_df = as_population_df.merge(
        hierarchy_df[["location_id", "gbd_location_id"]], on="location_id", how="left"
    )

    forecast_df = as_population_df.merge(df, on=as_merge_variables, how="left").copy()

    dengue_rr_cols = ["location_id", "sex_id", "age_group_id", "rr_inc_as"]
    as_md_gbd_dengue_df = read_parquet_with_integer_ids(
        as_md_gbd_dengue_df_path, columns=dengue_rr_cols
    ).rename(columns={"location_id": "gbd_location_id"})

    forecast_df = forecast_df.merge(
        as_md_gbd_dengue_df, on=["gbd_location_id", "age_group_id", "sex_id"], how="left"
    )

    forecast_df["base_log_dengue_inc_rate_pred"] = forecast_df["base_log_dengue_inc_rate_pred"].fillna(0)
    forecast_df["dengue_cfr_pred"] = forecast_df["dengue_cfr_pred"].fillna(0)

    forecast_df = disaggregate_age_sex_dengue(forecast_df)

    keep_columns = as_merge_variables + ["population", "dengue_inc_count_pred", "dengue_mort_count_pred"]
    forecast_df = forecast_df[keep_columns]

    non_measure_columns = [col for col in forecast_df.columns if "inc" not in col and "mort" not in col]
    incidence_columns = [col for col in forecast_df.columns if "inc" in col]
    mortality_columns = [col for col in forecast_df.columns if "mort" in col]

    # Apply vaccination effects to Singapore, Brazil, Indonesia, Thailand and their sub-locations
    vax_locations = ["Singapore", "Brazil", "Indonesia", "Thailand"]
    location_ids = hierarchy_df[hierarchy_df["location_name"].isin(vax_locations)]["location_id"].unique()
    children_ids = hierarchy_df[hierarchy_df["parent_id"].isin(location_ids)]["location_id"].unique()
    grand_children_ids = hierarchy_df[hierarchy_df["parent_id"].isin(children_ids)]["location_id"].unique()

    dengue_vaccine_df = read_parquet_with_integer_ids(dengue_vaccine_df_path)
    vaccine_lookup = dengue_vaccine_df.set_index("age_group_id")

    vaccine_mask = (
        forecast_df["location_id"].isin(grand_children_ids) &
        (forecast_df["year_id"] >= 2023) &
        (forecast_df["year_id"] <= 2100)
    )

    for year in range(2023, 2101):
        year_col = f"year_{year}"
        if year_col in vaccine_lookup.columns:
            year_mask = vaccine_mask & (forecast_df["year_id"] == year)
            if year_mask.any():
                age_group_reductions = vaccine_lookup[year_col]
                reductions = forecast_df.loc[year_mask, "age_group_id"].map(age_group_reductions).fillna(1.0)
                forecast_df.loc[year_mask, "dengue_mort_count_pred"] *= reductions

    mortality_df = forecast_df[non_measure_columns + mortality_columns]
    mortality_ds = convert_with_preset(mortality_df, preset="as_variables")
    write_netcdf(mortality_ds, output_dengue_mortality_draw_path)

    incidence_df = forecast_df[non_measure_columns + incidence_columns]
    incidence_ds = convert_with_preset(incidence_df, preset="as_variables")
    write_netcdf(incidence_ds, output_dengue_incidence_draw_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Disaggregate raked dengue forecasts into age-sex specific estimates with vaccination effects"
    )
    parser.add_argument("--ssp_scenario", type=str, required=True)
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
        hold_variable=args.hold_variable,
        hierarchy_read_path=args.hierarchy_read_path,
        processed_data_path=args.processed_data_path,
        forecasting_data_read_path=args.forecasting_data_read_path,
        forecasting_data_write_path=args.forecasting_data_write_path,
    )
