"""Pull the versioned GBD/FHS reference data for the malaria + dengue pipeline.

Canonical GBD getter. Writes one dated directory under ``constants.GBD_DATA_PATH``
(``01-raw_data/gbd/<GBD_DATA_DATE>/``) and repoints the ``current`` symlink after a
successful pull. Every downstream GBD reader resolves through
``constants.GBD_DATA_READ_PATH`` (= ``.../gbd/current``), so running this and then
re-running stage 02 refreshes the whole pipeline from a single source of truth.

Outputs (all parquet; the ``*_results`` files are GBD ``get_outputs`` long format):
    gbd_constants.json
    age_metadata.parquet                       (+ aggregate rows 1 / 22 / 27)
    {gbd,fhs}_2023_modeling_hierarchy.parquet
    {gbd,fhs}_2023_population.parquet
    aa_{malaria,dengue}_results.parquet         all-age (age 22), measures 1-6, count+rate
    as_{malaria,dengue}_results.parquet         age-specific, measures 1 & 6, count+rate

Location scope is GBD level <= 4 (national + subnational): the raking (stage-02
03/04) replaces LSAE values with GBD values wherever GBD has data (levels 4-5).
The only GBD level-5 units are the 9 regions of England (zero malaria/dengue), so
<= 4 is complete and numerically equivalent for both causes.

⚠ ENVIRONMENT: run in the IHME **my_gbd_sunset** conda env (provides ``db_queries``
/ the GBD shared functions), NOT ``idd-forecast-mbp``. This hits the live GBD
database, so it cannot be run or tested from the project env.

Supersedes the R getter (archived alongside as ``OLD_get_gbd_data.r``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from db_queries import get_age_metadata, get_location_metadata, get_outputs, get_population

# Make idd_forecast_mbp importable under my_gbd_sunset (which does not have the
# project installed). This file is src/idd_forecast_mbp/00_pull_raw_data/get_gbd_data.py
# so parents[2] is the repo's src/ directory. Relative — no committed absolute path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from idd_forecast_mbp import constants as mbpc  # noqa: E402

YEARS = list(range(2000, 2024))
SEX_IDS = [1, 2, 3]

# get_age_metadata returns only the most-detailed groups; append the aggregate rows
# the pipeline also needs (Under-5, All-age, Age-standardized).
_EXTRA_AGE_ROWS = pd.DataFrame({
    "age_group_id": [1, 22, 27],
    "age_group_years_start": [0, 0, 0],
    "age_group_years_end": [5, 125, 125],
    "age_group_name": ["Under 5", "All age", "Age standardized"],
})
_HIER_DROP = ["start_date", "end_date", "date_inserted",
              "last_updated", "last_updated_by", "last_updated_action"]


def main(
    data_date: str | None = None,
    make_current: bool = True,
    causes: tuple[str, ...] = ("malaria", "dengue"),
) -> None:
    release_id = mbpc.gbd_constants["release_2023_id"]
    compare_v = mbpc.gbd_constants["compare_2023_v"]
    data_date = data_date or mbpc.GBD_DATA_DATE
    data_path = mbpc.GBD_DATA_PATH / data_date
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"[get_gbd_data] writing to {data_path}")

    with open(data_path / "gbd_constants.json", "w") as f:
        json.dump(mbpc.gbd_constants, f, indent=2)

    # ── age metadata ───────────────────────────────────────────────────────────
    age_metadata = get_age_metadata(release_id=release_id)[
        ["age_group_id", "age_group_years_start", "age_group_years_end", "age_group_name"]
    ]
    age_metadata = pd.concat([age_metadata, _EXTRA_AGE_ROWS], ignore_index=True)
    age_metadata.to_parquet(data_path / "age_metadata.parquet", index=False)
    age_group_ids = age_metadata["age_group_id"].tolist()

    # ── hierarchies ────────────────────────────────────────────────────────────
    fhs_hierarchy = get_location_metadata(
        location_set_id=mbpc.gbd_constants["fhs_location_set_id"], release_id=release_id,
    ).drop(columns=_HIER_DROP)
    gbd_hierarchy = get_location_metadata(
        location_set_id=mbpc.gbd_constants["gbd_location_set_id"], release_id=release_id,
    ).drop(columns=_HIER_DROP)
    fhs_hierarchy.to_parquet(data_path / "fhs_2023_modeling_hierarchy.parquet", index=False)
    gbd_hierarchy.to_parquet(data_path / "gbd_2023_modeling_hierarchy.parquet", index=False)

    location_ids = gbd_hierarchy[gbd_hierarchy["level"] <= 4]["location_id"].tolist()

    # ── populations ────────────────────────────────────────────────────────────
    pop_cols = ["age_group_id", "location_id", "year_id", "sex_id", "population"]
    gbd_population = get_population(
        age_group_id=age_group_ids, release_id=release_id, year_id=YEARS,
        location_id=location_ids, sex_id=SEX_IDS,
    )[pop_cols]
    fhs_population = get_population(
        age_group_id=age_group_ids, release_id=release_id, year_id=YEARS,
        location_id=fhs_hierarchy["location_id"].tolist(), sex_id=SEX_IDS,
    )[pop_cols]
    gbd_population.to_parquet(data_path / "gbd_2023_population.parquet", index=False)
    fhs_population.to_parquet(data_path / "fhs_2023_population.parquet", index=False)

    # ── per-cause GBD outcomes (all-age + age-specific) ─────────────────────────
    # get_outputs already carries location_name/location_type, so drop them from the
    # hierarchy before merging — otherwise the merge yields location_name_x/_y
    # suffixes. Dropping keeps a single clean location_name/location_type.
    merge_keys = ["age_group_id", "location_id", "year_id", "sex_id"]
    hier_for_merge = gbd_hierarchy.drop(columns=["location_name", "location_type"])
    for cause_key in causes:
        info = mbpc.cause_map[cause_key]
        cause_id = info["cause_id"]

        print(f"[get_gbd_data] {info['cause_name']}: all-age")
        aa = get_outputs(
            "cause", cause_id=cause_id, measure_id=[1, 2, 3, 4, 5, 6], year_id=YEARS,
            location_id=location_ids, age_group_id=[22], release_id=release_id,
            metric_id=[1, 3], sex_id=[3], compare_version_id=compare_v,
        )
        aa = (aa.merge(hier_for_merge, on="location_id", how="left")
                .merge(gbd_population, on=merge_keys, how="left"))
        aa.to_parquet(data_path / f"aa_{cause_key}_results.parquet", index=False)

        print(f"[get_gbd_data] {info['cause_name']}: age-specific")
        as_df = get_outputs(
            "cause", cause_id=cause_id, measure_id=[1, 6], year_id=YEARS,
            location_id=location_ids, age_group_id=age_group_ids, release_id=release_id,
            metric_id=[1, 3], sex_id=[1, 2, 3], compare_version_id=compare_v,
        )
        as_df = (as_df.merge(hier_for_merge, on="location_id", how="left")
                      .merge(gbd_population, on=merge_keys, how="left"))
        as_df.to_parquet(data_path / f"as_{cause_key}_results.parquet", index=False)

    # ── finalize `current` only after every write succeeded ─────────────────────
    if make_current:
        current = mbpc.GBD_DATA_PATH / "current"
        if current.is_symlink() or current.exists():
            current.unlink()
        current.symlink_to(data_date)
        print(f"[get_gbd_data] current -> {data_date}")
    print("[get_gbd_data] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pull versioned GBD/FHS reference data. Run in the my_gbd_sunset env."
    )
    parser.add_argument("--data-date", default=None,
                        help=f"output dir name under gbd/ (default: constants.GBD_DATA_DATE={mbpc.GBD_DATA_DATE})")
    parser.add_argument("--no-make-current", action="store_true",
                        help="do not repoint the current symlink after the pull")
    parser.add_argument("--causes", nargs="+", default=["malaria", "dengue"])
    args = parser.parse_args()
    main(data_date=args.data_date, make_current=not args.no_make_current,
         causes=tuple(args.causes))
