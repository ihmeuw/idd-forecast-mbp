"""Part A: FHS population — past + future, combined.

Reads GBD past FHS population and FHS future forecast, applies the
44858→{60908,95069,94364} location split, and writes combined all-age and
age-specific FHS population files.

Outputs (in processed_data_path):
    aa_2023_fhs_population_df.parquet
    as_2023_fhs_population_df.parquet
    aa_2023_fhs_population_ds.nc
    as_2023_fhs_population_ds.nc
"""
import pandas as pd
import xarray as xr
from pathlib import Path

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.io.netcdf import write_netcdf, convert_with_preset
from idd_forecast_mbp.lib.versioning import finalize_artifact


_FUTURE_FHS_POP_PATH = Path(
    "/mnt/share/forecasting/data/32/future/population"
    "/future_population_s130v41/population_agg.nc"
)


def main(
    population_write_path: Path = mbpc.POPULATION_WRITE_PATH,
    raw_data_path: Path = mbpc.RAW_DATA_PATH,
) -> None:
    """Build combined past+future FHS population files."""
    population_write_path = Path(population_write_path)
    population_write_path.mkdir(parents=True, exist_ok=True)

    gbd_data_path = Path(raw_data_path) / "gbd"
    fhs_data_path = mbpc.AGE_SPECIFIC_FHS_PATH

    # Output paths
    aa_fhs_population_df_path = population_write_path / "aa_2023_fhs_population_df.parquet"
    as_fhs_population_df_path = population_write_path / "as_2023_fhs_population_df.parquet"
    aa_fhs_population_ds_path = population_write_path / "aa_2023_fhs_population_ds.nc"
    as_fhs_population_ds_path = population_write_path / "as_2023_fhs_population_ds.nc"

    # Age metadata drives which age groups are selected from FHS
    age_metadata_df = read_parquet_with_integer_ids(fhs_data_path / "age_metadata.parquet")
    age_group_ids = age_metadata_df["age_group_id"].unique().tolist()
    age_group_filter = ('age_group_id', 'in', age_group_ids)
    aa_age_group_filter = ('age_group_id', 'in', [22])
    sex_ids = [1, 2]
    sex_filter = ('sex_id', 'in', sex_ids)
    all_sex_filter = ('sex_id', 'in', [3])

    # ── Past FHS ──────────────────────────────────────────────────────────────
    past_fhs_population_path = gbd_data_path / "fhs_2023_population.parquet"
    as_past_fhs_population_df = read_parquet_with_integer_ids(
        past_fhs_population_path,
        filters=[[age_group_filter, sex_filter]],
    )
    aa_past_fhs_population_df = read_parquet_with_integer_ids(
        past_fhs_population_path,
        filters=[[aa_age_group_filter, all_sex_filter]],
    )

    # ── Future FHS ────────────────────────────────────────────────────────────
    as_future_fhs_population_df = (
        xr.open_dataset(_FUTURE_FHS_POP_PATH)
        .sel(age_group_id=age_metadata_df["age_group_id"].unique(), sex_id=sex_ids)
        .draws.mean(dim="draw")
        .to_dataframe()
        .reset_index()
        .drop(columns=["scenario"])
        .rename(columns={"draws": "population"})
    )
    aa_future_fhs_population_df = (
        xr.open_dataset(_FUTURE_FHS_POP_PATH)
        .sel(age_group_id=22, sex_id=3)
        .draws.mean(dim="draw")
        .to_dataframe()
        .reset_index()
        .drop(columns=["scenario"])
        .rename(columns={"draws": "population"})
    )

    # ── Fix: location 44858 → {60908, 95069, 94364} ──────────────────────────
    # The future FHS file has 44858 (a merged location), but the past has it
    # split into three sub-locations. Distribute the 44858 future population
    # using the 2023 age-sex fractions from the three sub-locations.
    df_2023 = as_past_fhs_population_df[
        (as_past_fhs_population_df["year_id"] == 2023)
        & (as_past_fhs_population_df["location_id"].isin([60908, 95069, 94364]))
    ].copy()
    df_44858 = (
        as_future_fhs_population_df[as_future_fhs_population_df["location_id"] == 44858]
        .copy()
        .drop(columns=["location_id"])
        .rename(columns={"population": "population_44858"})
    )
    df_2023 = df_2023.drop(columns=["year_id"])
    df_2023_sum = (
        df_2023.groupby(["age_group_id", "sex_id"])
        .agg({"population": "sum"})
        .reset_index()
        .rename(columns={"population": "total_as_population"})
    )
    df_2023 = df_2023.merge(df_2023_sum, on=["age_group_id", "sex_id"], how="left")
    df_2023["as_fraction"] = df_2023["population"] / df_2023["total_as_population"]
    df_2023 = df_2023[["age_group_id", "sex_id", "location_id", "as_fraction"]]

    df_as_new_locations = df_2023.merge(df_44858, on=["age_group_id", "sex_id"], how="left")
    df_as_new_locations["population"] = (
        df_as_new_locations["as_fraction"] * df_as_new_locations["population_44858"]
    )
    df_as_new_locations = df_as_new_locations.drop(columns=["population_44858", "as_fraction"])

    df_aa_new_locations = (
        df_as_new_locations.groupby(["location_id", "year_id"])
        .agg({"population": "sum"})
        .reset_index()
        .copy()
    )
    df_aa_new_locations["age_group_id"] = 22
    df_aa_new_locations["sex_id"] = 3

    as_future_fhs_population_df = as_future_fhs_population_df[
        as_future_fhs_population_df["location_id"] != 44858
    ]
    aa_future_fhs_population_df = aa_future_fhs_population_df[
        aa_future_fhs_population_df["location_id"] != 44858
    ]
    as_future_fhs_population_df = pd.concat(
        [as_future_fhs_population_df, df_as_new_locations], ignore_index=True
    )
    aa_future_fhs_population_df = pd.concat(
        [aa_future_fhs_population_df, df_aa_new_locations], ignore_index=True
    )

    # ── Combine past + future ─────────────────────────────────────────────────
    # Future file starts at 2023 which overlaps with past; drop the overlap.
    last_past_year = aa_past_fhs_population_df["year_id"].max()
    aa_future_fhs_population_df = aa_future_fhs_population_df[
        aa_future_fhs_population_df["year_id"] > last_past_year
    ]
    as_future_fhs_population_df = as_future_fhs_population_df[
        as_future_fhs_population_df["year_id"] > last_past_year
    ]

    aa_fhs_population_df = pd.concat(
        [aa_past_fhs_population_df, aa_future_fhs_population_df], ignore_index=True
    ).rename(columns={"population": "aa_population"})

    as_fhs_population_df = pd.concat(
        [as_past_fhs_population_df, as_future_fhs_population_df], ignore_index=True
    )
    as_fhs_population_df = as_fhs_population_df.merge(
        aa_fhs_population_df[["location_id", "year_id", "aa_population"]],
        on=["location_id", "year_id"],
        how="left",
    )
    as_fhs_population_df["as_population_fraction"] = (
        as_fhs_population_df["population"] / as_fhs_population_df["aa_population"]
    )

    # ── Write ─────────────────────────────────────────────────────────────────
    write_parquet(aa_fhs_population_df, aa_fhs_population_df_path)
    write_parquet(as_fhs_population_df, as_fhs_population_df_path)
    write_netcdf(convert_with_preset(aa_fhs_population_df, preset="aa_variables"), aa_fhs_population_ds_path)
    write_netcdf(convert_with_preset(as_fhs_population_df, preset="as_variables"), as_fhs_population_ds_path)
    finalize_artifact(mbpc._A02_POPULATION)


if __name__ == "__main__":
    main()
