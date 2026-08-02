"""Build the dengue forecast prediction-location set with full covariate coverage.

DEPENDENCY ORDER: run AFTER 06b_build_dengue_past_inputs.py (06b defines which
covariates must exist; this script coverage-checks them).

Dengue analog of 07b. Starts from `dengue_prediction_location_ids` (any level-5
location whose A0 cleared the prediction count thresholds in any year), then
applies the same drop policy as malaria's 07b (2026-05-27):

1. Read upstream gridded population (`mbpc.LSAE_POP_PATH`) for the forecast
   check window; mark (loc, year) rows where pop == 0 (or NaN).
2. For each covariate, count NaN ONLY where pop > 0 ("real NaN").
3. A location is dropped if EITHER (a) pop == 0 in every check-window year, or
   (b) any covariate has real NaN anywhere in the window.

Output (versioned artifact `_A04_DEN_FORECAST_LOCATIONS`):
  prediction_location_ids.parquet — kept locs + A0_location_id.
  dropped_locations.parquet       — audit log (long format).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.array_builders import read_shared_covariates, wide_to_array
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.processing.helpers import level_filter
from idd_forecast_mbp.lib.processing.locations import dengue_prediction_location_ids
from idd_forecast_mbp.lib.versioning import finalize_artifact

# Forecast covariates checked for full coverage. Mirror 07b's minimal set;
# extend (e.g. urban, more climate) as the dengue model's covariate set firms up.
SHARED_VARS_TO_CHECK: tuple[str, ...] = (
    "gdppc_mean",
    "people_flood_days_per_capita",
)
DRAW_VARS_TO_CHECK: tuple[str, ...] = ("dengue_suitability",)

FORECAST_CHECK_START_YEAR: int = 2023


def _resolve_flooding_path(lsae_hierarchy: str, ssp_scenario: str) -> str | None:
    base = Path(f"/mnt/team/rapidresponse/pub/flooding/results/output/{lsae_hierarchy}/{mbpc.FLOODING_RUN_DATE}")
    for fname in [
        f"fldfrc_weightedmin_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
        f"fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
    ]:
        candidate = base / fname
        if candidate.exists():
            return str(candidate)
    return None


def _dengue_suitability_path(lsae_hierarchy: str, ssp_scenario: str) -> Path:
    return mbpc.CLIMATE_AGGREGATES_PATH / lsae_hierarchy / f"dengue_suitability_{ssp_scenario}.parquet"


def main(
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    ssp_scenarios: list[str] | None = None,
    check_start_year: int = FORECAST_CHECK_START_YEAR,
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    den_raked_aa_read_path: Path = mbpc.DEN_RAKED_AA_READ_PATH,
    gdppc_read_path: Path = mbpc.GDPPC_READ_PATH,
    ldipc_read_path: Path = mbpc.LDIPC_READ_PATH,
    urban_read_path: Path = mbpc.URBAN_READ_PATH,
    output_path: Path = mbpc.DEN_FORECAST_LOCATIONS_WRITE_PATH,
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
    print("Loading AA raked dengue data...")
    aa_df = read_parquet_with_integer_ids(
        Path(den_raked_aa_read_path) / "aa_full_dengue_df.parquet",
        filters=[level_filter(hierarchy_df, start_level=3, end_level=5)],
    )
    raw_loc_ids = dengue_prediction_location_ids(aa_df, hierarchy_df)
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
    pop_zero_mask = (pop_arr == 0) | np.isnan(pop_arr)  # (n_loc, n_year)
    print(f"  (loc, year) rows with pop==0 in window: "
          f"{int(pop_zero_mask.sum()):,} / {pop_zero_mask.size:,}")

    uninhabited_mask = pop_zero_mask.all(axis=1)
    uninhabited_loc_ids = [
        int(raw_loc_ids[i]) for i in range(len(raw_loc_ids)) if uninhabited_mask[i]
    ]
    print(f"  Uninhabited locs (pop==0 every year of window): {len(uninhabited_loc_ids):,}")
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
            variables=SHARED_VARS_TO_CHECK,
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

        for var in DRAW_VARS_TO_CHECK:
            print(f"  Reading {var}...")
            suit_arr = wide_to_array(
                str(_dengue_suitability_path(lsae_hierarchy, ssp_scenario)),
                raw_loc_ids, check_years,
            )  # (loc, year, draw)
            real_nan_mask = np.isnan(suit_arr) & ~pop_zero_mask[:, :, None]
            real_nan_per_loc = real_nan_mask.sum(axis=(1, 2))
            for i, loc in enumerate(raw_loc_ids):
                if real_nan_per_loc[i] > 0:
                    drop_log.append({
                        "location_id": int(loc),
                        "covariate": var,
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
        dropped_df["A0_location_id"] = dropped_df["location_id"].map(lambda loc: int(loc_to_a0[loc]))
        dropped_df = dropped_df[
            ["location_id", "A0_location_id", "covariate", "ssp_scenario",
             "drop_reason", "n_nan_years_in_check_window"]
        ].sort_values(["drop_reason", "location_id", "covariate", "ssp_scenario"])

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  Raw:     {len(raw_loc_ids):,}")
    print(f"  Dropped: {len(dropped_loc_ids):,} unique locations ({len(drop_log):,} audit rows)")
    print(f"  Kept:    {len(kept_loc_ids):,}")

    # ── 6. Write outputs ─────────────────────────────────────────────────────
    write_parquet(kept_df,    output_path / "prediction_location_ids.parquet")
    write_parquet(dropped_df, output_path / "dropped_locations.parquet")
    print(f"\nWrote {output_path / 'prediction_location_ids.parquet'} ({len(kept_df):,} rows)")
    print(f"Wrote {output_path / 'dropped_locations.parquet'}        ({len(dropped_df):,} rows)")

    if Path(output_path) == Path(mbpc.DEN_FORECAST_LOCATIONS_WRITE_PATH):
        finalize_artifact(mbpc._A04_DEN_FORECAST_LOCATIONS)
    else:
        print(f"output_path overridden to {output_path}; skipping finalize_artifact.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build dengue forecast prediction-location set with full coverage"
    )
    parser.add_argument("--lsae_hierarchy", default=mbpc.LSAE_HIERARCHY)
    parser.add_argument("--ssp_scenarios", nargs="+", default=None)
    parser.add_argument("--check_start_year", type=int, default=FORECAST_CHECK_START_YEAR)
    args = parser.parse_args()
    main(
        lsae_hierarchy=args.lsae_hierarchy,
        ssp_scenarios=args.ssp_scenarios,
        check_start_year=args.check_start_year,
    )
