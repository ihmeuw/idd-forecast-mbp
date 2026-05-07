"""Build malaria past input NetCDF for regression modeling.

Produces one NetCDF file with all historical covariate and outcome data
needed to fit malaria models. Variables carry only the dimensions they
actually vary over:

  location_id × year_id                            — AA outcomes, non-draw covariates
  location_id × year_id × draw_id                  — draw-varying climate
  location_id × year_id × draw_id × suit_variant   — malaria suitability

Locations are restricted to the endemic subset used in modeling (same burden
filters as the former 05_malaria_modeling_dataframe.py). Climate is read from
SSP245, which for the historical period (2000-2022) is equivalent to the
observed record.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.io.array_builders import (
    wide_to_array, scalar_to_array, read_shared_covariates, read_draw_climate,
)
from idd_forecast_mbp.lib.processing.helpers import level_filter
from idd_forecast_mbp.lib.versioning import finalize_artifact

PAST_YEARS = list(range(2000, 2023))
DRAW_IDS = list(range(100))


def _endemic_location_ids(
    aa_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    mort_threshold: float = 1.0,
) -> list[int]:
    """Return sorted endemic most-detailed location IDs (same filter as script 05)."""
    df = aa_df.merge(
        hierarchy_df[['location_id', 'A0_location_id', 'most_detailed_lsae']],
        on='location_id', how='left',
    )
    df = df[
        (df['malaria_pfpr'] > 0) &
        (df['malaria_mort_count'] > 0) &
        (df['malaria_inc_count'] >= 0)
    ]
    a0_2022 = df[
        (df['location_id'] == df['A0_location_id']) & (df['year_id'] == 2022)
    ]
    endemic_a0 = a0_2022[a0_2022['malaria_mort_count'] >= mort_threshold]['A0_location_id'].unique()
    md = df[df['A0_location_id'].isin(endemic_a0) & (df['most_detailed_lsae'] == 1)]
    return sorted(md['location_id'].unique().tolist())


def main(
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    ssp_scenario: str = "ssp245",
    add_base: bool = False,
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    mal_raked_aa_read_path: Path = mbpc.MAL_RAKED_AA_READ_PATH,
    mal_raked_as_read_path: Path = mbpc.MAL_RAKED_AS_READ_PATH,
    gdppc_read_path: Path = mbpc.GDPPC_READ_PATH,
    ldipc_read_path: Path = mbpc.LDIPC_READ_PATH,
    dah_read_path: Path = mbpc.DAH_READ_PATH,
    urban_read_path: Path = mbpc.URBAN_READ_PATH,
    output_path: Path = mbpc.MAL_PAST_INPUTS_WRITE_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    CLIMATE = mbpc.CLIMATE_AGGREGATES_PATH / lsae_hierarchy

    # ── 1. Hierarchy and endemic locations ────────────────────────────────────
    print("Loading hierarchy...")
    hierarchy_df = read_parquet_with_integer_ids(
        Path(hierarchy_read_path) / f"full_hierarchy_2023_{lsae_hierarchy}.parquet"
    )

    print("Loading AA raked data...")
    aa_df = read_parquet_with_integer_ids(
        Path(mal_raked_aa_read_path) / "aa_full_malaria_df.parquet",
        filters=[
            ('year_id', 'in', PAST_YEARS),
            level_filter(hierarchy_df, start_level=3, end_level=5),
        ],
    )
    print(f"  AA columns: {list(aa_df.columns)}")

    location_ids = _endemic_location_ids(aa_df, hierarchy_df)
    n_loc, n_year, n_draw = len(location_ids), len(PAST_YEARS), len(DRAW_IDS)
    print(f"  Endemic locations: {n_loc}, years: {n_year}, draws: {n_draw}")

    # ── 2. AA outcomes (location × year) ─────────────────────────────────────
    print("Building AA outcome arrays...")
    aa_sub = aa_df[aa_df['location_id'].isin(location_ids)].copy()
    aa_outcome_cols = [
        c for c in aa_sub.columns
        if c not in ('location_id', 'year_id') and not c.endswith('_count')
    ]
    aa_arrays = {
        col: scalar_to_array(aa_sub, col, location_ids, PAST_YEARS)
        for col in aa_outcome_cols
    }

    if add_base:
        reference_age_group_id = mbpc.cause_map['malaria']['reference_age_group_id']
        reference_sex_id = mbpc.cause_map['malaria']['reference_sex_id']
        as_ref_df = read_parquet_with_integer_ids(
            Path(mal_raked_as_read_path) / "as_full_malaria_df.parquet",
            filters=[
                ('year_id', 'in', PAST_YEARS),
                ('age_group_id', '==', reference_age_group_id),
                ('sex_id', '==', reference_sex_id),
                ('location_id', 'in', location_ids),
            ],
        )
        as_ref_outcome_cols = [
            c for c in as_ref_df.columns
            if c not in ('location_id', 'year_id', 'age_group_id', 'sex_id')
        ]
        for col in as_ref_outcome_cols:
            aa_arrays[f"base_{col}"] = scalar_to_array(as_ref_df, col, location_ids, PAST_YEARS)

    # A0_location_id per location (needed for country fixed effects in R)
    loc_to_a0 = hierarchy_df.set_index('location_id')['A0_location_id'].to_dict()

    # ── 3. Non-draw scalar covariates (location × year) ──────────────────────
    print("Reading non-draw covariates...")

    # DAH — at A0 level; broadcast to endemic locations via hierarchy
    dah_df = read_parquet_with_integer_ids(
        Path(dah_read_path) / "dah_df.parquet",
        filters=[('year_id', 'in', PAST_YEARS)],
    )
    dah_df_renamed = dah_df.rename(columns={'location_id': 'A0_location_id'})
    loc_year_df = pd.DataFrame(
        [(loc, yr) for loc in location_ids for yr in PAST_YEARS],
        columns=['location_id', 'year_id'],
    )
    loc_year_df['A0_location_id'] = loc_year_df['location_id'].map(loc_to_a0)
    dah_cols = [c for c in dah_df_renamed.columns
                if c not in ('A0_location_id', 'year_id', 'location_name', 'iso3', 'population')]
    dah_merged = loc_year_df.merge(
        dah_df_renamed[['A0_location_id', 'year_id'] + dah_cols],
        on=['A0_location_id', 'year_id'], how='left',
    )

    # Flooding path — try weightedmin naming first (lsae_1285), fall back to shifted
    flooding_path_str = None
    for fname in [
        f"fldfrc_weightedmin_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
        f"fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
    ]:
        candidate = Path(f"/mnt/team/rapidresponse/pub/flooding/results/output/{lsae_hierarchy}/{fname}")
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
    climate_arrays = read_draw_climate(
        location_ids=location_ids,
        years=PAST_YEARS,
        ssp_scenario=ssp_scenario,
        lsae_hierarchy=lsae_hierarchy,
    )

    # ── 5. Malaria suitability (location × year × draw × suit_variant) ────────
    print("Reading malaria suitability variants...")
    seen_paths: dict[str, str] = {}
    suit_arrays: dict[str, np.ndarray] = {}
    for variant in mbpc.MALARIA_SUITABILITY_VARIANTS:
        path = mbpc.get_malaria_suitability_path(variant, ssp_scenario, lsae_hierarchy)
        if path in seen_paths:
            print(f"  {variant} → same file as {seen_paths[path]}, skipping")
            continue
        seen_paths[path] = variant
        print(f"  {variant}...")
        suit_arrays[variant] = wide_to_array(path, location_ids, PAST_YEARS)

    # ── 6. Assemble xarray Dataset ────────────────────────────────────────────
    print("Building xarray Dataset...")
    a0_ids = np.array([loc_to_a0.get(loc, -1) for loc in location_ids], dtype=np.int32)
    coords = {
        'location_id':    np.array(location_ids, dtype=np.int32),
        'year_id':        np.array(PAST_YEARS,   dtype=np.int32),
        'draw_id':        np.array(DRAW_IDS,      dtype=np.int32),
        'A0_location_id': ('location_id', a0_ids),
    }
    data_vars: dict[str, xr.DataArray] = {}

    # AA outcomes
    for col, arr in aa_arrays.items():
        data_vars[col] = xr.DataArray(arr, dims=['location_id', 'year_id'])

    # DAH
    for col in dah_cols:
        data_vars[col] = xr.DataArray(
            scalar_to_array(dah_merged, col, location_ids, PAST_YEARS),
            dims=['location_id', 'year_id'],
        )

    # Shared non-draw covariates (gdppc, ldipc, urban, flooding)
    for var_name, arr in shared_arrays.items():
        data_vars[var_name] = xr.DataArray(arr, dims=['location_id', 'year_id'])

    # Draw-varying climate
    for var_name, arr in climate_arrays.items():
        data_vars[var_name] = xr.DataArray(arr, dims=['location_id', 'year_id', 'draw_id'])

    # Malaria suitability
    variant_names = list(suit_arrays.keys())
    suit_stack = np.stack(list(suit_arrays.values()), axis=-1)  # (loc, year, draw, variant)
    coords['suit_variant'] = variant_names
    data_vars['malaria_suitability'] = xr.DataArray(
        suit_stack, dims=['location_id', 'year_id', 'draw_id', 'suit_variant']
    )

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update({
        'ssp_scenario':    ssp_scenario,
        'lsae_hierarchy':  lsae_hierarchy,
        'created_date':    mbpc.RUN_DATE,
        'n_locations':     n_loc,
        'n_suit_variants': len(variant_names),
        'suit_variants':   ','.join(variant_names),
    })

    # ── 7. Write NetCDF ───────────────────────────────────────────────────────
    out_file = output_path / "malaria_past_inputs.nc"
    print(f"Writing {out_file}  ({n_loc} loc × {n_year} yr × {n_draw} draw)...")
    encoding = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} for v in data_vars}
    ds.to_netcdf(out_file, encoding=encoding)
    print(f"  Done. File size: {out_file.stat().st_size / 1e9:.2f} GB")

    finalize_artifact(mbpc._A03_MAL_PAST_INPUTS)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build malaria past input NetCDF")
    parser.add_argument("--ssp_scenario", default="ssp245")
    parser.add_argument("--lsae_hierarchy", default=mbpc.LSAE_HIERARCHY)
    parser.add_argument("--add_base", action="store_true", default=False)
    args = parser.parse_args()
    main(ssp_scenario=args.ssp_scenario, lsae_hierarchy=args.lsae_hierarchy, add_base=args.add_base)
