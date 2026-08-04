"""
Input: Dengue forecast NC for a specific SSP scenario and draw
Output: Raked dengue forecast with CFR predictions saved as netCDF
Objective: Rake base dengue incidence forecasts and complete CFR regression predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.io.netcdf import read_netcdf_with_integer_ids, write_netcdf, convert_with_preset
from idd_forecast_mbp.lib.utils.transforms import logit


def main(
    ssp_scenario: str,
    draw: str,
    hold_variable: str = "None",
    dengue_raked_as_read_path: Path = mbpc.DEN_RAKED_AS_READ_PATH,
    modeling_data_path: Path = mbpc.MODEL_ROOT / "03-modeling_data" / "dengue_cfr_model" / "lsae_1209" / "current",
    processed_data_path: Path = mbpc.MODEL_ROOT / "02-processed_data",
    forecasting_data_read_path: Path = mbpc.FORECASTING_DATA_PATH,
    forecasting_data_write_path: Path = mbpc.FORECASTING_DATA_PATH,
) -> None:
    dengue_raked_as_read_path = Path(dengue_raked_as_read_path)
    modeling_data_path = Path(modeling_data_path)
    processed_data_path = Path(processed_data_path)
    forecasting_data_read_path = Path(forecasting_data_read_path)
    forecasting_data_write_path = Path(forecasting_data_write_path)
    forecasting_data_write_path.mkdir(parents=True, exist_ok=True)

    cause = "dengue"
    cause_map = mbpc.cause_map
    reference_age_group_id = cause_map[cause]["reference_age_group_id"]
    reference_sex_id = cause_map[cause]["reference_sex_id"]

    aa_merge_variables = mbpc.aa_merge_variables
    as_merge_variables = mbpc.as_merge_variables

    if hold_variable == "None":
        input_cause_draw_path = forecasting_data_read_path / f"{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
        output_cause_draw_path = forecasting_data_write_path / f"raked_{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
    else:
        input_cause_draw_path = forecasting_data_read_path / f"{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"
        output_cause_draw_path = forecasting_data_write_path / f"raked_{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{hold_variable}.nc"

    as_full_population_df_path = processed_data_path / "as_2023_full_population.parquet"
    as_full_dengue_df_path = dengue_raked_as_read_path / "as_full_dengue_df.parquet"

    ds = read_netcdf_with_integer_ids(input_cause_draw_path)
    reference_df = ds.to_dataframe().reset_index()

    suit_col = [col for col in reference_df.columns if "suit" in col]
    reference_df = reference_df.drop(
        columns=suit_col + ["as_id", "age_group_id", "sex_id", "log_gdppc_mean",
                            "logit_urban_1km_threshold_300", "A0_af", "logit_dengue_cfr"],
        errors="ignore",
    )
    level_5_location_ids = reference_df["location_id"].unique()

    tmp_df = reference_df[reference_df["year_id"] == 2022].copy()
    tmp_df["shift"] = tmp_df["base_log_dengue_inc_rate"] - tmp_df["base_log_dengue_inc_rate_pred_raw"]
    reference_df = reference_df.merge(tmp_df[["location_id", "shift"]], on=["location_id"], how="left")
    reference_df["base_log_dengue_inc_rate_pred"] = reference_df["base_log_dengue_inc_rate_pred_raw"] + reference_df["shift"]

    df_location_filter = ("location_id", "in", level_5_location_ids)
    future_year_ids = list(range(2000, 2101))
    year_filter = ("year_id", "in", future_year_ids)

    as_population_df = read_parquet_with_integer_ids(
        as_full_population_df_path,
        columns=as_merge_variables,
        filters=[df_location_filter, year_filter],
    )

    forecast_df = as_population_df.merge(
        reference_df[["location_id", "year_id", "base_log_dengue_inc_rate_pred", "logit_dengue_cfr_pred_raw"]],
        on=aa_merge_variables, how="left",
    ).copy()

    as_id_levels = pd.read_csv(modeling_data_path / "as_id_levels.csv")
    mod_cfr_all_coefficients = pd.read_csv(
        modeling_data_path / "mod_cfr_all_coefficients.csv",
        names=["variable", "coefficient"],
        header=0,
    )

    as_id_levels_expanded = as_id_levels.reset_index()
    as_id_levels_expanded = as_id_levels_expanded.rename(columns={"index": "level_num", "x": "level"})
    as_id_levels_expanded["age_group_id"] = as_id_levels_expanded["level"].str.extract(r"a(\d+)_s\d+").astype(int)
    as_id_levels_expanded["sex_id"] = as_id_levels_expanded["level"].str.extract(r"a\d+_s(\d+)").astype(int)

    mod_cfr_coef = mod_cfr_all_coefficients[
        mod_cfr_all_coefficients["variable"].str.contains("as_id", na=False)
    ].copy()
    mod_cfr_coef = mod_cfr_coef.reset_index(drop=True)
    mod_cfr_coef["level"] = mod_cfr_coef["variable"].str.replace("as_id", "", regex=False)

    as_id_levels_expanded = as_id_levels_expanded.merge(
        mod_cfr_coef[["level", "coefficient"]], on="level", how="left"
    )
    as_id_levels_expanded["coefficient"] = as_id_levels_expanded["coefficient"].fillna(0)

    reference_coef = as_id_levels_expanded.loc[
        (as_id_levels_expanded["age_group_id"] == reference_age_group_id) &
        (as_id_levels_expanded["sex_id"] == reference_sex_id),
        "coefficient",
    ].iloc[0]

    as_id_levels_expanded["reference_coefficient"] = reference_coef
    as_id_levels_expanded["logit_cfr_shift"] = as_id_levels_expanded["coefficient"] - reference_coef

    forecast_df = forecast_df.merge(
        as_id_levels_expanded[["age_group_id", "sex_id", "logit_cfr_shift"]],
        on=["age_group_id", "sex_id"], how="left",
    )
    forecast_df["logit_dengue_cfr_pred_raw"] = forecast_df["logit_dengue_cfr_pred_raw"] + forecast_df["logit_cfr_shift"]
    forecast_df = forecast_df.drop(columns=["logit_cfr_shift"], errors="ignore")

    forecast_2022_df = forecast_df[forecast_df["year_id"] == 2022].copy()
    forecast_2022_df = forecast_2022_df[as_merge_variables + ["logit_dengue_cfr_pred_raw"]]

    cfr_rake_year_filter = ("year_id", "in", [2022])
    as_md_dengue_modeling_df = read_parquet_with_integer_ids(
        as_full_dengue_df_path,
        filters=[cfr_rake_year_filter, df_location_filter],
    )
    as_md_dengue_modeling_df["dengue_cfr"] = (
        as_md_dengue_modeling_df["dengue_mort_rate"] / as_md_dengue_modeling_df["dengue_inc_rate"]
    )

    for col in ["dengue_cfr"]:
        print(f"Range of {col}: {as_md_dengue_modeling_df[col].min()} to {as_md_dengue_modeling_df[col].max()}")
        as_md_dengue_modeling_df[f"logit_{col}"] = logit(as_md_dengue_modeling_df[col])

    cfr_rake_df = as_md_dengue_modeling_df[as_merge_variables + ["logit_dengue_cfr"]].copy()

    forecast_2022_df = forecast_2022_df.merge(
        cfr_rake_df[["location_id", "age_group_id", "sex_id", "logit_dengue_cfr"]],
        on=["location_id", "age_group_id", "sex_id"], how="left",
    )
    forecast_2022_df["shift"] = forecast_2022_df["logit_dengue_cfr"] - forecast_2022_df["logit_dengue_cfr_pred_raw"]

    forecast_df = forecast_df.merge(
        forecast_2022_df[["location_id", "age_group_id", "sex_id", "shift"]],
        on=["location_id", "age_group_id", "sex_id"], how="left",
    )
    forecast_df["logit_dengue_cfr_pred"] = forecast_df["logit_dengue_cfr_pred_raw"] + forecast_df["shift"]
    forecast_df = forecast_df.drop(columns=["shift", "logit_dengue_cfr_pred_raw"], errors="ignore")

    forecast_ds = convert_with_preset(forecast_df, preset="as_variables")
    write_netcdf(forecast_ds, output_cause_draw_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rake base dengue forecasts and complete CFR regression predictions")
    parser.add_argument("--ssp_scenario", type=str, required=True)
    parser.add_argument("--draw", type=str, required=True)
    parser.add_argument("--hold_variable", type=str, default="None")
    parser.add_argument("--dengue_raked_as_read_path", type=str, default=str(mbpc.DEN_RAKED_AS_READ_PATH))
    parser.add_argument("--modeling_data_path", type=str, default=str(mbpc.MODEL_ROOT / "03-modeling_data" / "dengue_cfr_model" / "lsae_1209" / "current"))
    parser.add_argument("--processed_data_path", type=str, default=str(mbpc.MODEL_ROOT / "02-processed_data"))
    parser.add_argument("--forecasting_data_read_path", type=str, default=str(mbpc.FORECASTING_DATA_PATH))
    parser.add_argument("--forecasting_data_write_path", type=str, default=str(mbpc.FORECASTING_DATA_PATH))

    args = parser.parse_args()
    main(
        ssp_scenario=args.ssp_scenario,
        draw=args.draw,
        hold_variable=args.hold_variable,
        dengue_raked_as_read_path=args.dengue_raked_as_read_path,
        modeling_data_path=args.modeling_data_path,
        processed_data_path=args.processed_data_path,
        forecasting_data_read_path=args.forecasting_data_read_path,
        forecasting_data_write_path=args.forecasting_data_write_path,
    )
