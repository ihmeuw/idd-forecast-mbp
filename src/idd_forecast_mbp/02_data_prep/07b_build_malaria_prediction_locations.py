"""Build the malaria forecast prediction-location set with full covariate coverage.

DEPENDENCY ORDER: must run BEFORE 08_build_malaria_forecast_inputs.py.

Starts from `malaria_prediction_location_ids` (any level-5 location in an
endemic A0). Drop policy (2026-05-27):

1. Read the upstream gridded population (`mbpc.LSAE_POP_PATH`) for the
   forecast check window. Identify (loc, year) rows where pop == 0 — these
   carry no real per-capita signal and any covariate NaN there is expected.
2. For each covariate, count NaN ONLY in rows where pop > 0. That is the
   "real NaN" — a gap not explained by the polygon being uninhabited.
3. Report drops by reason ("real_nan" vs "pop_zero_all_years_in_window").
4. A location is dropped if EITHER (a) it has pop == 0 in every check-window
   year (uninhabited in the forecast era), or (b) any covariate has real
   NaN (NaN where pop > 0) anywhere in the window.

The pre-2026-05-27 policy dropped on ANY NaN in the check window, which
over-dropped locations like 93390 that are uninhabited early and populated
later — those have no real NaN once pop_zero rows are masked.

Output (versioned artifact `_A04_MAL_FORECAST_LOCATIONS`):
  prediction_location_ids.parquet — kept locs with A0_location_id metadata.
  dropped_locations.parquet       — audit log; long format. For real-NaN
                                    drops, one row per (location_id,
                                    covariate, ssp_scenario). For
                                    uninhabited drops, one row per
                                    location_id with synthetic covariate
                                    and ssp_scenario placeholders.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.array_builders import read_shared_covariates, wide_to_array
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.processing.helpers import level_filter
from idd_forecast_mbp.lib.processing.locations import malaria_prediction_location_ids
from idd_forecast_mbp.lib.versioning import finalize_artifact

# Forecast covariates checked for full coverage. Matches the default malaria
# forecast covariate set in lib/io/covariate_registry.py.
SHARED_VARS_TO_CHECK: tuple[str, ...] = (
    "gdppc_mean",
    "weighted_1km_urban_threshold_300.0_simple_mean",
    "people_flood_days_per_capita",
)
DRAW_VARS_TO_CHECK: tuple[str, ...] = ("malaria_suitability",)

# Past-year NaN is tolerated because past-year R model fits use a separate
# past-inputs parquet, not this artifact. With the 2026-05-27 pop_zero
# filter, this could in principle drop to 2000 — pre-2015 NaN on
# loc 93390-like locations is now correctly masked as pop_zero rather
# than counted as a real coverage gap. Leaving at 2023 for now; flag
# if you want to lower it.
FORECAST_CHECK_START_YEAR: int = 2023


def _resolve_flooding_path(lsae_hierarchy: str, ssp_scenario: str) -> str | None:
    base = Path(f"/mnt/team/rapidresponse/pub/flooding/results/output/{lsae_hierarchy}")
    for fname in [
        f"fldfrc_weightedmin_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
        f"fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
    ]:
        candidate = base / fname
        if candidate.exists():
            return str(candidate)
    return None


def main(
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    suitability_variant: str = mbpc.MALARIA_SUITABILITY_VARIANT,
    ssp_scenarios: list[str] | None = None,
    check_start_year: int = FORECAST_CHECK_START_YEAR,
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    mal_raked_aa_read_path: Path = mbpc.MAL_RAKED_AA_READ_PATH,
    gdppc_read_path: Path = mbpc.GDPPC_READ_PATH,
    ldipc_read_path: Path = mbpc.LDIPC_READ_PATH,
    urban_read_path: Path = mbpc.URBAN_READ_PATH,
    output_path: Path = mbpc.MAL_FORECAST_LOCATIONS_WRITE_PATH,
) -> None:
    ssp_scenarios = ssp_scenarios or list(mbpc.ssp_scenarios.keys())
    check_years = list(range(check_start_year, 2101))
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Hierarchy + raw prediction set ────────────────────────────────────
    print("Loading hierarchy...")
    hierarchy_df = read_parquet_with_integer_ids(
        Path(hierarchy_read_path) / f"full_hierarchy_2023_{lsae_hierarchy}.parquet"
    )
    print("Loading AA raked data...")
    aa_df = read_parquet_with_integer_ids(
        Path(mal_raked_aa_read_path) / "aa_full_malaria_df.parquet",
        filters=[level_filter(hierarchy_df, start_level=3, end_level=5)],
    )
    raw_loc_ids = malaria_prediction_location_ids(aa_df, hierarchy_df)
    print(f"  Raw prediction locations: {len(raw_loc_ids):,}")
    print(f"  Check window: {check_start_year}..2100 ({len(check_years)} years)")

    drop_log: list[dict] = []

    # ── 2. Read upstream pop, build pop_zero mask, log uninhabited locs ──────
    print("\nReading upstream gridded population...")
    pop_raw = pd.read_parquet(mbpc.LSAE_POP_PATH)
    pop_arr = (
        pop_raw[
            pop_raw["location_id"].isin(raw_loc_ids)
            & pop_raw["year_id"].isin(check_years)
        ]
        .pivot(index="location_id", columns="year_id", values="population")
        .reindex(index=raw_loc_ids, columns=check_years)
        .to_numpy()
    )
    # Missing (NaN) is treated the same as pop==0: no real data for that
    # (loc, year), so any covariate NaN there is expected, not actionable.
    pop_zero_mask = (pop_arr == 0) | np.isnan(pop_arr)  # (n_loc, n_year)
    print(f"  (loc, year) rows with pop==0 in window: "
          f"{int(pop_zero_mask.sum()):,} / {pop_zero_mask.size:,}")

    uninhabited_mask = pop_zero_mask.all(axis=1)
    uninhabited_loc_ids = [
        int(raw_loc_ids[i]) for i in range(len(raw_loc_ids)) if uninhabited_mask[i]
    ]
    print(f"  Uninhabited locs (pop==0 every year of window): "
          f"{len(uninhabited_loc_ids):,}")
    for loc in uninhabited_loc_ids:
        drop_log.append({
            "location_id": loc,
            "covariate": "<all_years_pop_zero>",
            "ssp_scenario": "<all>",
            "drop_reason": "pop_zero_all_years_in_window",
            "n_nan_years_in_check_window": len(check_years),
        })

    # ── 3. Per-SSP NaN check (real NaN only — pop==0 rows masked out) ────────
    for ssp_scenario in ssp_scenarios:
        print(f"\n=== Checking {ssp_scenario} ===")
        rcp_scenario = mbpc.ssp_scenarios[ssp_scenario]["rcp_scenario"]
        flooding_path = _resolve_flooding_path(lsae_hierarchy, ssp_scenario)

        print("  Reading shared scalars + flooding (forecast window only)...")
        shared_arrays = read_shared_covariates(
            location_ids=raw_loc_ids,
            years=check_years,
            gdppc_read_path=gdppc_read_path,
            ldipc_read_path=ldipc_read_path,
            urban_read_path=urban_read_path,
            flooding_path=flooding_path,
            rcp_scenario=rcp_scenario,
            med_consumppc_read_path=None,
        )
        for var in SHARED_VARS_TO_CHECK:
            if var not in shared_arrays:
                raise KeyError(
                    f"{var!r} not returned by read_shared_covariates; "
                    "check var name vs. source parquet column."
                )
            arr = shared_arrays[var]  # (n_loc, n_year)
            real_nan_mask = np.isnan(arr) & ~pop_zero_mask
            real_nan_per_loc = real_nan_mask.sum(axis=1)
            for i, loc in enumerate(raw_loc_ids):
                if real_nan_per_loc[i] > 0:
                    drop_log.append({
                        "location_id": int(loc),
                        "covariate": var,
                        "ssp_scenario": ssp_scenario,
                        "drop_reason": "real_nan",
                        "n_nan_years_in_check_window": int(real_nan_per_loc[i]),
                    })

        print("  Reading suitability...")
        suit_path = mbpc.get_malaria_suitability_path(
            suitability_variant, ssp_scenario, lsae_hierarchy
        )
        suit_arr = wide_to_array(suit_path, raw_loc_ids, check_years)  # (loc, year, draw)
        # Broadcast pop_zero_mask (loc, year) → (loc, year, 1) across draws.
        real_nan_mask = np.isnan(suit_arr) & ~pop_zero_mask[:, :, None]
        real_nan_per_loc = real_nan_mask.sum(axis=(1, 2))
        for i, loc in enumerate(raw_loc_ids):
            if real_nan_per_loc[i] > 0:
                drop_log.append({
                    "location_id": int(loc),
                    "covariate": "malaria_suitability",
                    "ssp_scenario": ssp_scenario,
                    "drop_reason": "real_nan",
                    "n_nan_years_in_check_window": int(real_nan_per_loc[i]),
                })

    # ── 4. Aggregate → kept / dropped lists ──────────────────────────────────
    dropped_loc_ids = sorted({entry["location_id"] for entry in drop_log})
    kept_loc_ids    = sorted(set(raw_loc_ids) - set(dropped_loc_ids))

    loc_to_a0 = hierarchy_df.set_index("location_id")["A0_location_id"].to_dict()

    kept_df = pd.DataFrame({
        "location_id":    kept_loc_ids,
        "A0_location_id": [int(loc_to_a0[loc]) for loc in kept_loc_ids],
    })

    dropped_df = pd.DataFrame(drop_log)
    if not dropped_df.empty:
        dropped_df["A0_location_id"] = dropped_df["location_id"].map(
            lambda loc: int(loc_to_a0[loc])
        )
        dropped_df = dropped_df[
            ["location_id", "A0_location_id", "covariate", "ssp_scenario",
             "drop_reason", "n_nan_years_in_check_window"]
        ].sort_values(["drop_reason", "location_id", "covariate", "ssp_scenario"])

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  Raw:     {len(raw_loc_ids):,}")
    print(f"  Dropped: {len(dropped_loc_ids):,} unique locations "
          f"({len(drop_log):,} audit rows)")
    print(f"  Kept:    {len(kept_loc_ids):,}")
    if not dropped_df.empty:
        by_reason = (
            dropped_df.drop_duplicates("location_id")
            .groupby("drop_reason").size().sort_values(ascending=False)
        )
        print("  Unique locs dropped by reason:")
        for reason, n in by_reason.items():
            print(f"    {reason}: {n}")
        real_nan_df = dropped_df[dropped_df["drop_reason"] == "real_nan"]
        if not real_nan_df.empty:
            by_cov = (
                real_nan_df.groupby("covariate")["location_id"]
                .nunique().sort_values(ascending=False)
            )
            print("  Real-NaN locs by covariate:")
            for cov, n in by_cov.items():
                print(f"    {cov}: {n}")

    # ── 6. Write outputs ─────────────────────────────────────────────────────
    write_parquet(kept_df,    output_path / "prediction_location_ids.parquet")
    write_parquet(dropped_df, output_path / "dropped_locations.parquet")
    print(f"\nWrote {output_path / 'prediction_location_ids.parquet'} ({len(kept_df):,} rows)")
    print(f"Wrote {output_path / 'dropped_locations.parquet'}        ({len(dropped_df):,} rows)")

    if Path(output_path) == Path(mbpc.MAL_FORECAST_LOCATIONS_WRITE_PATH):
        finalize_artifact(mbpc._A04_MAL_FORECAST_LOCATIONS)
    else:
        print(f"output_path overridden to {output_path}; skipping finalize_artifact.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build malaria forecast prediction-location set with full coverage"
    )
    parser.add_argument("--lsae_hierarchy", default=mbpc.LSAE_HIERARCHY)
    parser.add_argument("--suitability_variant", default=mbpc.MALARIA_SUITABILITY_VARIANT)
    parser.add_argument("--ssp_scenarios", nargs="+", default=None)
    parser.add_argument("--check_start_year", type=int, default=FORECAST_CHECK_START_YEAR)
    args = parser.parse_args()
    main(
        lsae_hierarchy=args.lsae_hierarchy,
        suitability_variant=args.suitability_variant,
        ssp_scenarios=args.ssp_scenarios,
        check_start_year=args.check_start_year,
    )
