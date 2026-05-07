"""Build malaria past input parquet for regression modeling.

Produces one flat parquet with all historical covariate and outcome data
needed to fit malaria models. One row per (location_id, year_id) — valid
endemic observations only (malaria_pfpr > 0, mort_count > 0, inc_count >= 0).
Climate is read from SSP245 draw 000 (past draws are identical; variance opens at 2024).
DAH missingness in source implies $0; NaN values are filled with 0.
Suitability variants are stored as separate columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.io.array_builders import (
    wide_to_array, read_shared_covariates, read_draw_climate,
)
from idd_forecast_mbp.lib.processing.helpers import level_filter
from idd_forecast_mbp.lib.versioning import finalize_artifact

PAST_YEARS = list(range(2000, 2023))


def _endemic_location_ids(
    aa_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    mort_threshold: float = 1.0,
) -> list[int]:
    """Return sorted endemic most-detailed location IDs."""
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


def _arrays_to_df(arrays: dict, location_ids: list[int], years: list[int]) -> pd.DataFrame:
    """Convert dict of (n_loc, n_year) float32 arrays to a flat location×year DataFrame."""
    idx = pd.MultiIndex.from_product(
        [location_ids, years], names=['location_id', 'year_id']
    )
    return pd.DataFrame(
        {col: arr.flatten() for col, arr in arrays.items()},
        index=idx,
    ).reset_index()


def main(
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    ssp_scenario: str = "ssp245",
    add_base: bool = False,
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    mal_raked_aa_read_path: Path = mbpc.MAL_RAKED_AA_READ_PATH,
    mal_raked_as_read_path: Path = mbpc.MAL_RAKED_AS_READ_PATH,
    gdppc_read_path: Path = mbpc.GDPPC_READ_PATH,
    ldipc_read_path: Path = mbpc.LDIPC_READ_PATH,
    med_consumppc_read_path: Path = mbpc.MED_CONSUMPPC_READ_PATH,
    dah_read_path: Path = mbpc.DAH_READ_PATH,
    urban_read_path: Path = mbpc.URBAN_READ_PATH,
    output_path: Path = mbpc.MAL_PAST_INPUTS_WRITE_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

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

    location_ids = _endemic_location_ids(aa_df, hierarchy_df)
    print(f"  Endemic locations: {len(location_ids)}, years: {len(PAST_YEARS)}")

    # ── 2. Base dataframe: valid AA rows only ─────────────────────────────────
    print("Building base dataframe...")
    aa_sub = aa_df[
        aa_df['location_id'].isin(location_ids) &
        (aa_df['malaria_pfpr'] > 0) &
        (aa_df['malaria_mort_count'] > 0) &
        (aa_df['malaria_inc_count'] >= 0)
    ].copy()

    outcome_cols = [
        c for c in aa_sub.columns
        if c not in ('location_id', 'year_id') and not c.endswith('_count')
    ]
    df = aa_sub[['location_id', 'year_id'] + outcome_cols].copy()
    df['logit_malaria_pfpr'] = np.log(
        0.999 * df['malaria_pfpr'] / (1 - 0.999 * df['malaria_pfpr'])
    )

    loc_to_a0 = hierarchy_df.set_index('location_id')['A0_location_id'].to_dict()
    df['A0_location_id'] = df['location_id'].map(loc_to_a0).astype('int32')

    print(f"  Valid rows: {len(df):,}")

    # ── 3. Add base (reference age-sex) rates if requested ───────────────────
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
        as_ref_cols = [
            c for c in as_ref_df.columns
            if c not in ('location_id', 'year_id', 'age_group_id', 'sex_id')
        ]
        as_ref_df = as_ref_df[['location_id', 'year_id'] + as_ref_cols].rename(
            columns={c: f"base_{c}" for c in as_ref_cols}
        )
        df = df.merge(as_ref_df, on=['location_id', 'year_id'], how='left')

    # ── 4. DAH — broadcast from A0, fill missing years with $0 ───────────────
    print("Reading DAH...")
    dah_df = read_parquet_with_integer_ids(
        Path(dah_read_path) / "dah_df.parquet",
        filters=[('year_id', 'in', PAST_YEARS)],
    )
    dah_df = dah_df.rename(columns={'location_id': 'A0_location_id'})
    dah_cols = [c for c in dah_df.columns
                if c not in ('A0_location_id', 'year_id', 'location_name', 'iso3', 'population')]
    df = df.merge(
        dah_df[['A0_location_id', 'year_id'] + dah_cols],
        on=['A0_location_id', 'year_id'], how='left',
    )
    for col in dah_cols:
        df[col] = df[col].fillna(0.0)

    # ── 5. Shared scalar covariates (gdppc, ldipc, urban, flooding, med_consumppc) ──
    print("Reading shared covariates...")
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
        med_consumppc_read_path=med_consumppc_read_path,
    )
    shared_df = _arrays_to_df(shared_arrays, location_ids, PAST_YEARS)
    df = df.merge(shared_df, on=['location_id', 'year_id'], how='left')

    # ── 6. Climate draw 000 ───────────────────────────────────────────────────
    print("Reading climate covariates (draw 000)...")
    climate_arrays_draws = read_draw_climate(
        location_ids=location_ids,
        years=PAST_YEARS,
        ssp_scenario=ssp_scenario,
        lsae_hierarchy=lsae_hierarchy,
    )
    climate_arrays = {k: v[:, :, 0] for k, v in climate_arrays_draws.items()}
    climate_df = _arrays_to_df(climate_arrays, location_ids, PAST_YEARS)
    df = df.merge(climate_df, on=['location_id', 'year_id'], how='left')

    # ── 7. Malaria suitability variants as columns ────────────────────────────
    print("Reading malaria suitability variants...")
    seen_paths: dict[str, str] = {}
    for variant in mbpc.MALARIA_SUITABILITY_VARIANTS:
        path = mbpc.get_malaria_suitability_path(variant, ssp_scenario, lsae_hierarchy)
        if path in seen_paths:
            print(f"  {variant} → same file as {seen_paths[path]}, skipping")
            continue
        seen_paths[path] = variant
        print(f"  {variant}...")
        arr = wide_to_array(path, location_ids, PAST_YEARS)[:, :, 0]
        suit_df = _arrays_to_df({f"malaria_suitability_{variant}": arr}, location_ids, PAST_YEARS)
        df = df.merge(suit_df, on=['location_id', 'year_id'], how='left')

    # ── 8. Write parquet ──────────────────────────────────────────────────────
    out_file = output_path / "malaria_past_inputs.parquet"
    print(f"Writing {out_file}  ({len(df):,} rows × {len(df.columns)} cols)...")
    write_parquet(df, out_file)
    print(f"  Done. File size: {out_file.stat().st_size / 1e6:.1f} MB")

    finalize_artifact(mbpc._A03_MAL_PAST_INPUTS)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build malaria past input parquet")
    parser.add_argument("--ssp_scenario", default="ssp245")
    parser.add_argument("--lsae_hierarchy", default=mbpc.LSAE_HIERARCHY)
    parser.add_argument("--add_base", action="store_true", default=False)
    args = parser.parse_args()
    main(ssp_scenario=args.ssp_scenario, lsae_hierarchy=args.lsae_hierarchy, add_base=args.add_base)
