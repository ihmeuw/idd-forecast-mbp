"""Build dengue past input NetCDF for regression modeling.

Dengue differs from malaria in three ways:
  - R model predicts at age-sex (AS) level directly, so past outcomes are
    (location × year × age_group_id × sex_id)
  - No DAH covariate
  - No suitability variant dimension (dengue_suitability is one of the
    standard draw-varying climate variables)

Produces one NetCDF with all historical covariate and outcome data needed
to fit dengue models.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.io.array_builders import (
    scalar_to_array, read_shared_covariates, read_draw_climate,
)
from idd_forecast_mbp.lib.processing.helpers import level_filter
from idd_forecast_mbp.lib.versioning import finalize_artifact

PAST_YEARS = list(range(2000, 2023))
DRAW_IDS = list(range(100))


def _endemic_location_ids(
    as_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
) -> list[int]:
    """Return sorted most-detailed location IDs with any dengue burden."""
    df = as_df.merge(
        hierarchy_df[['location_id', 'most_detailed_lsae']],
        on='location_id', how='left',
    )
    df = df[
        (df['dengue_inc_count'] > 0) &
        (df['most_detailed_lsae'] == 1)
    ]
    return sorted(df['location_id'].unique().tolist())


def main(
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    ssp_scenario: str = "ssp245",
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    den_raked_as_read_path: Path = mbpc.DEN_RAKED_AS_READ_PATH,
    gdppc_read_path: Path = mbpc.GDPPC_READ_PATH,
    ldipc_read_path: Path = mbpc.LDIPC_READ_PATH,
    urban_read_path: Path = mbpc.URBAN_READ_PATH,
    output_path: Path = mbpc.DEN_PAST_INPUTS_WRITE_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Hierarchy and endemic locations ────────────────────────────────────
    print("Loading hierarchy...")
    hierarchy_df = read_parquet_with_integer_ids(
        Path(hierarchy_read_path) / f"full_hierarchy_2023_{lsae_hierarchy}.parquet"
    )

    print("Loading AS raked data...")
    as_df = read_parquet_with_integer_ids(
        Path(den_raked_as_read_path) / "as_full_dengue_df.parquet",
        filters=[
            ('year_id', 'in', PAST_YEARS),
            level_filter(hierarchy_df, start_level=3, end_level=5),
        ],
    )
    print(f"  AS columns: {list(as_df.columns)}")

    location_ids = _endemic_location_ids(as_df, hierarchy_df)
    age_group_ids = sorted(as_df['age_group_id'].unique().tolist())
    sex_ids = sorted(as_df['sex_id'].unique().tolist())
    n_loc = len(location_ids)
    n_year = len(PAST_YEARS)
    n_age = len(age_group_ids)
    n_sex = len(sex_ids)
    print(f"  Locations: {n_loc}, years: {n_year}, age groups: {n_age}, sexes: {n_sex}")

    # ── 2. AS outcomes (location × year × age × sex) ─────────────────────────
    print("Building AS outcome arrays...")
    as_outcome_cols = [
        c for c in as_df.columns
        if c not in ('location_id', 'year_id', 'age_group_id', 'sex_id')
        and as_df[c].dtype.kind == 'f'
    ]
    full_idx = pd.MultiIndex.from_product(
        [location_ids, PAST_YEARS, age_group_ids, sex_ids],
        names=['location_id', 'year_id', 'age_group_id', 'sex_id'],
    )
    as_pivot = (
        as_df[as_df['location_id'].isin(location_ids)]
        .set_index(['location_id', 'year_id', 'age_group_id', 'sex_id'])[as_outcome_cols]
        .reindex(full_idx)
    )
    as_arrays = {
        col: as_pivot[col].values.reshape(n_loc, n_year, n_age, n_sex).astype(np.float32)
        for col in as_outcome_cols
    }

    # ── 3. Non-draw scalar covariates (location × year) ──────────────────────
    print("Reading non-draw covariates...")

    # Flooding path — try weightedmin naming (lsae_1285) first
    flooding_path_str = None
    for fname in [
        f"fldfrc_weightedmin_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
        f"fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
    ]:
        candidate = Path(
            f"/mnt/team/rapidresponse/pub/flooding/results/output/{lsae_hierarchy}/{fname}"
        )
        if candidate.exists():
            flooding_path_str = str(candidate)
            break

    rcp_scenario = mbpc.ssp_scenarios[ssp_scenario]['rcp_scenario']
    shared_arrays = read_shared_covariates(
        location_ids=location_ids,
        years=PAST_YEARS,
        gdppc_read_path=gdppc_read_path,
        ldipc_read_path=ldipc_read_path,
        urban_read_path=urban_read_path,
        flooding_path=flooding_path_str,
        rcp_scenario=rcp_scenario,
    )

    # ── 4. Draw-varying climate (location × year × draw) ──────────────────────
    print("Reading draw-varying climate covariates...")
    CLIMATE = mbpc.CLIMATE_AGGREGATES_PATH / lsae_hierarchy
    climate_arrays = read_draw_climate(
        location_ids=location_ids,
        years=PAST_YEARS,
        ssp_scenario=ssp_scenario,
        lsae_hierarchy=lsae_hierarchy,
        extra_vars={
            'dengue_suitability': str(CLIMATE / f"dengue_suitability_{ssp_scenario}.parquet"),
        },
    )

    # ── 5. Assemble xarray Dataset ────────────────────────────────────────────
    print("Building xarray Dataset...")
    coords = {
        'location_id':  np.array(location_ids,  dtype=np.int32),
        'year_id':      np.array(PAST_YEARS,     dtype=np.int32),
        'age_group_id': np.array(age_group_ids,  dtype=np.int32),
        'sex_id':       np.array(sex_ids,        dtype=np.int32),
        'draw_id':      np.array(DRAW_IDS,       dtype=np.int32),
    }
    data_vars: dict[str, xr.DataArray] = {}

    # AS outcomes
    for col, arr in as_arrays.items():
        data_vars[col] = xr.DataArray(
            arr, dims=['location_id', 'year_id', 'age_group_id', 'sex_id']
        )

    # Shared non-draw covariates (gdppc, ldipc, urban, flooding)
    for var_name, arr in shared_arrays.items():
        data_vars[var_name] = xr.DataArray(arr, dims=['location_id', 'year_id'])

    # Draw-varying climate (includes dengue_suitability)
    for var_name, arr in climate_arrays.items():
        data_vars[var_name] = xr.DataArray(arr, dims=['location_id', 'year_id', 'draw_id'])

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update({
        'ssp_scenario':   ssp_scenario,
        'lsae_hierarchy': lsae_hierarchy,
        'created_date':   mbpc.RUN_DATE,
        'n_locations':    n_loc,
        'n_age_groups':   n_age,
    })

    # ── 6. Write NetCDF ───────────────────────────────────────────────────────
    out_file = output_path / "dengue_past_inputs.nc"
    print(f"Writing {out_file}  ({n_loc} loc × {n_year} yr × {n_age} age × {n_sex} sex × 100 draw)...")
    encoding = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} for v in data_vars}
    ds.to_netcdf(out_file, encoding=encoding)
    print(f"  Done. File size: {out_file.stat().st_size / 1e9:.2f} GB")

    finalize_artifact(mbpc._A03_DEN_PAST_INPUTS)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build dengue past input NetCDF")
    parser.add_argument("--ssp_scenario", default="ssp245")
    parser.add_argument("--lsae_hierarchy", default=mbpc.LSAE_HIERARCHY)
    args = parser.parse_args()
    main(ssp_scenario=args.ssp_scenario, lsae_hierarchy=args.lsae_hierarchy)
