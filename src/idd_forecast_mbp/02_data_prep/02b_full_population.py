"""Part B: Full population — LSAE historical + raked + future projection, age-specific.

Reads Part A outputs (aa/as FHS population) and the LSAE pixel-model CSV aggregations,
rakes subnational (levels 4–5) population to match FHS totals, projects future years
using 2023 FHS fractions, and disaggregates all-age population into age-specific estimates.

Inputs (from processed_data_path, written by Part A and script 01):
    full_hierarchy_2023_{lsae_hierarchy}.parquet
    aa_2023_fhs_population_df.parquet
    as_2023_fhs_population_df.parquet

Outputs (in processed_data_path):
    aa_2023_full_population_df.parquet
    aa_2023_full_population_ds.nc
    age_sex_df.parquet
    as_2023_full_population_df.parquet
    as_2023_full_population_ds.nc

Also updates {lsae_hierarchy} hierarchy file in-place with missing-location flags
when gaps between LSAE and GBD locations are found.
"""
import itertools
import pandas as pd
from pathlib import Path

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.io.netcdf import write_netcdf, convert_with_preset


_LSAE_POP_ROOT = Path("/mnt/share/geospatial/ihmepop/gbd_release_id_16")
_LSAE_YEARS = list(range(2000, 2024))
_FUTURE_YEARS = list(range(2024, 2101))


def main(
    population_write_path: Path = mbpc.POPULATION_WRITE_PATH,
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    raw_data_path: Path = mbpc.RAW_DATA_PATH,
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    population_read_path: Path = mbpc.POPULATION_WRITE_PATH,
) -> None:
    """Build full (all locations, all years) population files from LSAE + FHS."""
    population_write_path = Path(population_write_path)
    population_write_path.mkdir(parents=True, exist_ok=True)

    gbd_data_path = Path(raw_data_path) / "gbd"
    fhs_data_path = mbpc.AGE_SPECIFIC_FHS_PATH

    # ── Input paths ───────────────────────────────────────────────────────────
    hierarchy_df_path = Path(hierarchy_read_path) / f"full_hierarchy_2023_{lsae_hierarchy}.parquet"
    aa_fhs_population_df_path = Path(population_read_path) / "aa_2023_fhs_population_df.parquet"
    as_fhs_population_df_path = Path(population_read_path) / "as_2023_fhs_population_df.parquet"

    # ── Output paths ──────────────────────────────────────────────────────────
    aa_full_population_df_path = population_write_path / "aa_2023_full_population_df.parquet"
    as_full_population_df_path = population_write_path / "as_2023_full_population_df.parquet"
    aa_full_population_ds_path = population_write_path / "aa_2023_full_population_ds.nc"
    as_full_population_ds_path = population_write_path / "as_2023_full_population_ds.nc"
    age_sex_df_path = population_write_path / "age_sex_df.parquet"
    missing_level_4_location_path = population_write_path / "missing_level_4_location_ids.parquet"
    missing_level_5_location_path = population_write_path / "missing_level5_location_ids.parquet"

    # ── Reference data ────────────────────────────────────────────────────────
    hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

    age_metadata_df = read_parquet_with_integer_ids(fhs_data_path / "age_metadata.parquet")
    age_group_ids = age_metadata_df["age_group_id"].unique().tolist()
    sex_ids = [1, 2]

    gbd_population_df = read_parquet_with_integer_ids(gbd_data_path / "gbd_2023_population.parquet")
    # Restrict to locations present in this hierarchy
    gbd_population_df = gbd_population_df[
        gbd_population_df["location_id"].isin(hierarchy_df["location_id"])
    ]

    # Part A outputs
    aa_fhs_population_df = read_parquet_with_integer_ids(aa_fhs_population_df_path)
    as_fhs_population_df = read_parquet_with_integer_ids(as_fhs_population_df_path)

    # ── Load LSAE historical all-age population ────────────────────────────────
    lsae_population_dfs = []
    for year in _LSAE_YEARS:
        for level in ("adm1", "adm2"):
            csv_path = (
                _LSAE_POP_ROOT / lsae_hierarchy / f"pop_agg/{year}q1/aggregations/{level}.csv"
            )
            df = pd.read_csv(csv_path)
            df = df.drop(columns=["count", "var", "location_name"], errors="ignore")
            df = df.rename(columns={"Location ID": "location_id", "sum": "population"})
            df["year_id"] = year
            lsae_population_dfs.append(df)

    aa_full_population_df = pd.concat(lsae_population_dfs, ignore_index=True)

    # ── Identify locations in hierarchy but absent from LSAE CSVs ─────────────
    missing_level_4_location_ids = hierarchy_df[
        (hierarchy_df["level"] == 4)
        & (~hierarchy_df["location_id"].isin(aa_full_population_df["location_id"]))
    ]["location_id"].unique().tolist()
    missing_level_5_location_ids = hierarchy_df[
        (hierarchy_df["level"] == 5)
        & (~hierarchy_df["location_id"].isin(aa_full_population_df["location_id"]))
    ]["location_id"].unique().tolist()
    missing_location_ids = missing_level_4_location_ids + missing_level_5_location_ids

    if not missing_location_ids:
        print("No missing locations found in LSAE population data.")
    else:
        print(f"Missing locations: {len(missing_location_ids)}")

        if missing_level_4_location_ids:
            print(f"  Level 4: {len(missing_level_4_location_ids)}")
            write_parquet(
                hierarchy_df[hierarchy_df["location_id"].isin(missing_level_4_location_ids)],
                missing_level_4_location_path,
            )
        if missing_level_5_location_ids:
            print(f"  Level 5: {len(missing_level_5_location_ids)}")
            write_parquet(
                hierarchy_df[hierarchy_df["location_id"].isin(missing_level_5_location_ids)],
                missing_level_5_location_path,
            )

        hierarchy_df["in_gbd_not_lsae"] = hierarchy_df["location_id"].isin(missing_location_ids)

        # Fill from GBD population first
        missing_aa_population_df = gbd_population_df[
            (gbd_population_df["location_id"].isin(missing_location_ids))
            & (gbd_population_df["year_id"].isin(_LSAE_YEARS))
            & (gbd_population_df["age_group_id"] == 22)
            & (gbd_population_df["sex_id"] == 3)
        ].drop(columns=["age_group_id", "sex_id"]).copy()
        aa_full_population_df = pd.concat(
            [aa_full_population_df, missing_aa_population_df], ignore_index=True
        )

        # Zero-fill anything still absent
        still_missing = hierarchy_df[
            hierarchy_df["level"].isin([4, 5])
            & (~hierarchy_df["location_id"].isin(aa_full_population_df["location_id"]))
        ]
        if still_missing.empty:
            print("  No locations still missing after GBD fill.")
        else:
            print(f"  {len(still_missing)} locations zero-filled.")
            hierarchy_df["no_info"] = hierarchy_df["location_id"].isin(still_missing)
            zero_fill = pd.DataFrame(
                list(itertools.product(still_missing["location_id"].unique(), _LSAE_YEARS)),
                columns=["location_id", "year_id"],
            )
            zero_fill["population"] = 0
            aa_full_population_df = pd.concat(
                [aa_full_population_df, zero_fill], ignore_index=True
            )

        write_parquet(hierarchy_df, hierarchy_df_path)

    # ── GBD population for raking and age-sex fractions ───────────────────────
    aa_gbd_population_df = (
        gbd_population_df[
            (gbd_population_df["age_group_id"] == 22) & (gbd_population_df["sex_id"] == 3)
        ]
        .copy()
        .drop(columns=["age_group_id", "sex_id"])
        .rename(columns={"population": "aa_population"})
    )
    as_gbd_population_df = gbd_population_df[
        gbd_population_df["age_group_id"].isin(age_group_ids)
        & gbd_population_df["sex_id"].isin(sex_ids)
    ].copy()
    as_gbd_population_df = as_gbd_population_df.merge(
        aa_gbd_population_df, on=["location_id", "year_id"], how="left"
    )
    as_gbd_population_df["as_population_fraction"] = (
        as_gbd_population_df["population"] / as_gbd_population_df["aa_population"]
    )

    # ── Step 0: Replace LSAE population with GBD all-age where available ──────
    # GBD is preferred over the raw LSAE CSV sum for GBD-modelled locations.
    aa_full_population_df = aa_full_population_df.merge(
        aa_gbd_population_df[["location_id", "year_id", "aa_population"]],
        on=["location_id", "year_id"],
        how="left",
    )
    aa_full_population_df["population"] = aa_full_population_df["aa_population"].fillna(
        aa_full_population_df["population"]
    )
    aa_full_population_df = aa_full_population_df.drop(columns=["aa_population"])

    # ── Step 1: Replace with FHS all-age for FHS-modelled locations ───────────
    aa_full_population_df = aa_full_population_df.merge(
        aa_fhs_population_df[["location_id", "year_id", "aa_population"]],
        on=["location_id", "year_id"],
        how="left",
    )
    aa_full_population_df["population"] = aa_full_population_df["aa_population"].fillna(
        aa_full_population_df["population"]
    )
    aa_full_population_df = aa_full_population_df.drop(columns=["aa_population"])
    aa_full_population_df = aa_full_population_df.merge(
        hierarchy_df[["location_id", "level", "parent_id"]],
        on="location_id",
        how="left",
    )

    # ── Step 2: Rake level 4 (admin-2) to FHS level 3 ────────────────────────
    aa_lsae_level_4_population_df = aa_full_population_df[
        aa_full_population_df["level"] == 4
    ].copy()
    tmp_agg = (
        aa_lsae_level_4_population_df.groupby(["parent_id", "year_id"])
        .agg({"population": "sum"})
        .reset_index()
        .rename(columns={"parent_id": "location_id"})
        .merge(
            aa_fhs_population_df[["location_id", "year_id", "aa_population"]],
            on=["location_id", "year_id"],
            how="left",
        )
    )
    tmp_agg["raking_factor"] = tmp_agg["aa_population"] / tmp_agg["population"]
    tmp_agg.loc[tmp_agg["aa_population"] == 0, "raking_factor"] = 1
    tmp_agg = tmp_agg.rename(columns={"location_id": "parent_id"})
    aa_lsae_level_4_population_df = aa_lsae_level_4_population_df.merge(
        tmp_agg[["parent_id", "year_id", "raking_factor"]],
        on=["parent_id", "year_id"],
        how="left",
    )
    aa_lsae_level_4_population_df["population"] = (
        aa_lsae_level_4_population_df["population"]
        * aa_lsae_level_4_population_df["raking_factor"]
    )
    aa_lsae_level_4_population_df = aa_lsae_level_4_population_df.drop(
        columns=["raking_factor", "level", "parent_id"]
    )

    # ── Step 3: Rake level 5 (admin-3) to raked level 4 ──────────────────────
    aa_lsae_level_5_population_df = aa_full_population_df[
        aa_full_population_df["level"] == 5
    ].copy()
    tmp_agg = (
        aa_lsae_level_5_population_df.groupby(["parent_id", "year_id"])
        .agg({"population": "sum"})
        .reset_index()
        .rename(columns={"parent_id": "location_id"})
        .merge(
            aa_lsae_level_4_population_df.rename(columns={"population": "aa_population"})[
                ["location_id", "year_id", "aa_population"]
            ],
            on=["location_id", "year_id"],
            how="left",
        )
    )
    tmp_agg["raking_factor"] = tmp_agg["aa_population"] / tmp_agg["population"]
    tmp_agg.loc[tmp_agg["aa_population"] == 0, "raking_factor"] = 1
    tmp_agg = tmp_agg.rename(columns={"location_id": "parent_id"})
    aa_lsae_level_5_population_df = aa_lsae_level_5_population_df.merge(
        tmp_agg[["parent_id", "year_id", "raking_factor"]],
        on=["parent_id", "year_id"],
        how="left",
    )
    aa_lsae_level_5_population_df["population"] = (
        aa_lsae_level_5_population_df["population"]
        * aa_lsae_level_5_population_df["raking_factor"]
    )
    aa_lsae_level_5_population_df = aa_lsae_level_5_population_df.drop(
        columns=["raking_factor", "level", "parent_id"]
    )

    # ── Step 4: FHS levels 0–3 (global, super-region, region, national) ───────
    aa_fhs_level_0_3_population_df = aa_fhs_population_df.merge(
        hierarchy_df[["location_id", "level"]], on="location_id", how="left"
    )
    aa_fhs_level_0_3_population_df = aa_fhs_level_0_3_population_df[
        aa_fhs_level_0_3_population_df["level"] <= 3
    ].copy()
    aa_fhs_level_0_3_population_df = aa_fhs_level_0_3_population_df.drop(
        columns=["age_group_id", "sex_id", "level"]
    ).rename(columns={"aa_population": "population"})

    # ── Step 5: Future subnational — scale by 2023 FHS fractions ─────────────
    # Compute each subnational location's share of its FHS parent in 2023
    subnat_last_df = pd.concat(
        [
            aa_lsae_level_4_population_df[aa_lsae_level_4_population_df["year_id"] == 2023],
            aa_lsae_level_5_population_df[aa_lsae_level_5_population_df["year_id"] == 2023],
        ],
        ignore_index=True,
    ).merge(hierarchy_df[["location_id", "fhs_location_id"]], on="location_id", how="left")

    aa_fhs_df = (
        aa_fhs_population_df.copy()
        .rename(columns={"location_id": "fhs_location_id", "aa_population": "fhs_population"})
        .drop(columns=["age_group_id", "sex_id"])
    )
    aa_fhs_df = aa_fhs_df[
        aa_fhs_df["fhs_location_id"].isin(subnat_last_df["fhs_location_id"].unique())
    ].copy()

    subnat_last_df = subnat_last_df.merge(
        aa_fhs_df, on=["fhs_location_id", "year_id"], how="left"
    )
    subnat_last_df["fhs_fraction"] = subnat_last_df["population"] / subnat_last_df["fhs_population"]
    subnat_last_df = subnat_last_df[["location_id", "fhs_fraction"]]

    lsae_subnat_ids = (
        hierarchy_df[hierarchy_df["level"].isin([4, 5])]["location_id"].unique().tolist()
    )
    subnat_future_population_df = pd.DataFrame(
        list(itertools.product(lsae_subnat_ids, _FUTURE_YEARS)),
        columns=["location_id", "year_id"],
    ).merge(hierarchy_df[["location_id", "fhs_location_id"]], on="location_id", how="left")
    subnat_future_population_df = subnat_future_population_df.merge(
        aa_fhs_df, on=["fhs_location_id", "year_id"], how="left"
    ).merge(subnat_last_df, on=["location_id"], how="left")
    subnat_future_population_df["population"] = (
        subnat_future_population_df["fhs_population"] * subnat_future_population_df["fhs_fraction"]
    )
    subnat_future_population_df = subnat_future_population_df.drop(
        columns=["fhs_population", "fhs_fraction", "fhs_location_id"]
    )

    aa_full_population_df = pd.concat(
        [
            aa_fhs_level_0_3_population_df,
            aa_lsae_level_4_population_df,
            aa_lsae_level_5_population_df,
            subnat_future_population_df,
        ],
        ignore_index=True,
    )

    write_parquet(aa_full_population_df, aa_full_population_df_path)
    write_netcdf(
        convert_with_preset(aa_full_population_df, preset="aa_variables"),
        aa_full_population_ds_path,
    )

    # ── Age-sex combinations ───────────────────────────────────────────────────
    age_sex_df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )
    write_parquet(age_sex_df, age_sex_df_path)

    # ── Age-specific full population ───────────────────────────────────────────
    as_merge_variables = mbpc.as_merge_variables

    sub_aa_full_population_df = aa_full_population_df.merge(
        hierarchy_df[["location_id", "level"]], on="location_id", how="left"
    )
    sub_aa_full_population_df = sub_aa_full_population_df[
        sub_aa_full_population_df["level"] >= 3
    ].copy()

    sub_as_full_population_df = sub_aa_full_population_df.merge(
        age_sex_df, how="cross"
    ).rename(columns={"population": "aa_population"})
    sub_as_full_population_df = sub_as_full_population_df.merge(
        hierarchy_df[["location_id", "gbd_location_id", "fhs_location_id"]],
        on="location_id",
        how="left",
    )

    # Past: set by GBD directly where available; otherwise apply GBD age-sex fractions
    past_as_df = sub_as_full_population_df[sub_as_full_population_df["year_id"] <= 2023].copy()
    as_gbd_population_df["set_by_gbd"] = True
    past_as_df = past_as_df.merge(
        as_gbd_population_df[as_merge_variables + ["population", "set_by_gbd"]],
        on=as_merge_variables,
        how="left",
    )
    mask = past_as_df["set_by_gbd"].isna()
    past_as_df.loc[mask, "set_by_gbd"] = False
    past_as_df["set_by_gbd"] = past_as_df["set_by_gbd"].astype("boolean")

    as_gbd_population_df = as_gbd_population_df.rename(
        columns={"location_id": "gbd_location_id"}
    )
    past_as_df = past_as_df.merge(
        as_gbd_population_df[
            ["gbd_location_id", "year_id", "age_group_id", "sex_id", "as_population_fraction"]
        ],
        on=["gbd_location_id", "year_id", "age_group_id", "sex_id"],
        how="left",
    )
    mask = past_as_df["set_by_gbd"] == False  # noqa: E712
    past_as_df.loc[mask, "population"] = (
        past_as_df.loc[mask, "aa_population"] * past_as_df.loc[mask, "as_population_fraction"]
    )
    past_as_df = past_as_df.drop(
        columns=["gbd_location_id", "fhs_location_id", "set_by_gbd", "level"]
    )

    # Future: set by FHS directly where available; otherwise apply FHS age-sex fractions
    future_as_df = sub_as_full_population_df[sub_as_full_population_df["year_id"] >= 2024].copy()
    as_fhs_population_df["set_by_fhs"] = True
    future_as_df = future_as_df.merge(
        as_fhs_population_df[as_merge_variables + ["population", "set_by_fhs"]],
        on=as_merge_variables,
        how="left",
    )
    mask = future_as_df["set_by_fhs"].isna()
    future_as_df.loc[mask, "set_by_fhs"] = False
    future_as_df["set_by_fhs"] = future_as_df["set_by_fhs"].astype("boolean")

    as_fhs_population_df = as_fhs_population_df.rename(
        columns={"location_id": "fhs_location_id"}
    )
    future_as_df = future_as_df.merge(
        as_fhs_population_df[
            ["fhs_location_id", "year_id", "age_group_id", "sex_id", "as_population_fraction"]
        ],
        on=["fhs_location_id", "year_id", "age_group_id", "sex_id"],
        how="left",
    )
    mask = future_as_df["set_by_fhs"] == False  # noqa: E712
    future_as_df.loc[mask, "population"] = (
        future_as_df.loc[mask, "aa_population"] * future_as_df.loc[mask, "as_population_fraction"]
    )
    future_as_df = future_as_df.drop(
        columns=["gbd_location_id", "fhs_location_id", "set_by_fhs", "level"]
    )

    # Levels 0–2: directly from GBD (past) and FHS (future)
    low_level_past = (
        as_gbd_population_df.rename(columns={"gbd_location_id": "location_id"})
        .copy()
        .merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
    )
    low_level_past = low_level_past[low_level_past["level"] < 3].copy()
    low_level_past = low_level_past.drop(columns=["level", "set_by_gbd"])

    low_level_future = (
        as_fhs_population_df[as_fhs_population_df["year_id"] > 2023]
        .rename(columns={"fhs_location_id": "location_id"})
        .copy()
        .merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
    )
    low_level_future = low_level_future[low_level_future["level"] < 3].copy()
    low_level_future = low_level_future.drop(columns=["level", "set_by_fhs"])

    as_full_population_df = pd.concat(
        [low_level_past, past_as_df, low_level_future, future_as_df],
        ignore_index=True,
    )

    write_parquet(as_full_population_df, as_full_population_df_path)
    write_netcdf(
        convert_with_preset(as_full_population_df, preset="as_variables"),
        as_full_population_ds_path,
    )


if __name__ == "__main__":
    main()
