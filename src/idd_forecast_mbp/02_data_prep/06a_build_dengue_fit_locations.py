"""Build the dengue FIT location-id set (06a).

DEPENDENCY ORDER: must run BEFORE 06b_build_dengue_past_inputs.py.

Emits the most-detailed location IDs eligible for dengue model fitting, per
`dengue_fit_location_ids`: A2s whose A0's all-age record in the gate year
(last modeling year, 2023) had `dengue_mort_count > dengue_fit_mort_threshold`
AND `dengue_inc_count > dengue_fit_inc_threshold` (both 0.0 for now —
exploratory; we expect to tighten these later).

The per-(location, year) "non-zero all-age deaths or cases" row filter is NOT
applied here — that happens at past-inputs build time (06b). 06a only fixes
which locations are eligible at all.

Output (versioned artifact `_A03_DEN_FIT_LOCATIONS`):
  fit_location_ids.parquet — location_id + A0_location_id.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.processing.helpers import level_filter
from idd_forecast_mbp.lib.processing.locations import dengue_fit_locations
from idd_forecast_mbp.lib.versioning import finalize_artifact


def main(
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    den_raked_aa_read_path: Path = mbpc.DEN_RAKED_AA_READ_PATH,
    output_path: Path = mbpc.DEN_FIT_LOCATIONS_WRITE_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading hierarchy...")
    hierarchy_df = read_parquet_with_integer_ids(
        Path(hierarchy_read_path) / f"full_hierarchy_2023_{lsae_hierarchy}.parquet"
    )

    print("Loading AA raked dengue data...")
    aa_df = read_parquet_with_integer_ids(
        Path(den_raked_aa_read_path) / "aa_full_dengue_df.parquet",
        filters=[level_filter(hierarchy_df, start_level=3, end_level=5)],
    )

    # Every location at levels 3-5 (country / admin-1 / admin-2) within the
    # dengue-eligible A0s, tagged with most_detailed_lsae/fhs/gbd so 06b can build
    # one all-grain past-inputs table and downstream picks a grain by filtering.
    out = dengue_fit_locations(aa_df, hierarchy_df)
    n_by_level = out['level'].value_counts().sort_index().to_dict()
    print(f"  Dengue fit locations: {len(out):,} across levels 3-5 {n_by_level} "
          f"(A0 {mbpc.MODELING_YEARS[-1]}: mort_count > {mbpc.dengue_fit_mort_threshold} "
          f"AND inc_count > {mbpc.dengue_fit_inc_threshold})")

    out_file = output_path / "fit_location_ids.parquet"
    write_parquet(out, out_file)
    print(f"Wrote {out_file} ({len(out):,} rows)")

    finalize_artifact(mbpc._A03_DEN_FIT_LOCATIONS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dengue fit location-id set")
    parser.add_argument("--lsae_hierarchy", default=mbpc.LSAE_HIERARCHY)
    args = parser.parse_args()
    main(lsae_hierarchy=args.lsae_hierarchy)
