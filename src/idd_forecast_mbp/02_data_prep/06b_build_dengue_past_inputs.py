"""Build the dengue past-input parquet for regression modeling (06b).

DEPENDENCY ORDER: run AFTER 06a_build_dengue_fit_locations.py (consumes its
fit-location set) and BEFORE 07c (which coverage-checks the covariates defined
here).

Dengue differs from malaria (05):
  - Outcomes are age-sex (AS) level: one row per (location, year, age, sex).
  - No DAH covariate; no median-consumption covariate.
  - dengue_suitability is a single draw-varying climate variable (no variants).

Design (matching the malaria past-inputs rationale):
  - Location universe: exactly the FHS most-detailed set (`--grain fhs`,
    default) — 473 locations, selected on `most_detailed_fhs`, NOT on hierarchy
    level, because that set straddles levels 3 and 4. Covered whether or not
    dengue was ever observed there (`--location-set all`).
    06a's A0 gate is kept as the `fit_eligible` FLAG rather than applied as a
    filter, so the old gated set is recoverable by filtering.
  - Row inclusion: every (location, year) in the universe, dense over age/sex.
    A location with zero cases in every year is an OBSERVED ZERO, not missing
    data: the source (`as_full_dengue_df`) carries exact zeros with no NaN for
    those cells, and dropping them makes the absent-vs-zero question
    undecidable downstream and hides the zero half of covariate space.
    `--require-nonzero-incidence` restores the historical filter.
  - Climate collapsed to draw 000 (past draws are identical — the reason this
    is a flat parquet, not a draw-dimensioned netCDF). Only draw 000 is READ,
    which is what keeps a whole-hierarchy covariate read affordable.
  - Outcomes stored as RATES + population only; counts are recoverable as
    rate × population, so they are not stored.

A log-rate fit still drops the zero rows on its own finite-response filter --
that is the FIT's decision, made in `lib/data/dengue_inputs`, and is no longer
pre-empted here by the artifact simply not containing them.

Output (versioned artifact `_A03_DEN_PAST_INPUTS`):
  dengue_past_inputs.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.dengue_inputs import GRAIN_FLAG
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.io.array_builders import read_shared_covariates, read_draw_climate
from idd_forecast_mbp.lib.processing.aggregation import aggregate_aa_rate_lsae_to_gbd

from idd_forecast_mbp.lib.versioning import finalize_artifact

PAST_YEARS = list(mbpc.MODELING_YEARS)

#: Hierarchy levels the past-inputs table spans (country / admin-1 / admin-2).
PAST_LEVELS = (3, 4, 5)

#: Only draw 000 is read: past climate draws are identical, and reading all 100
#: over the whole hierarchy costs ~455 MB per variable against ~4.5 MB.
PAST_CLIMATE_DRAWS = ["000"]


def resolve_location_universe(
    hierarchy_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    *,
    location_set: str = "all",
    grain: str = "fhs",
) -> pd.DataFrame:
    """Locations the past-inputs table covers, tagged with ``fit_eligible``.

    ``grain`` selects on the hierarchy's most-detailed FLAG, never on ``level``.
    The FHS most-detailed set straddles levels 3 and 4 (national for countries FHS
    does not subnationalise, admin-1 for those it does), so a level cut is not the
    same set: ``level <= 4`` also admits the national rows of subnationalised
    countries and the admin-1 rows of ones that are not, which are aggregates or
    duplicates of the grain rather than members of it. ``grain="all"`` keeps the
    historical levels 3-5 span for callers that want every node.

    ``location_set="all"`` covers the grain whether or not dengue was observed
    there; ``"fit"`` intersects with 06a's A0-gated set. ``fit_eligible`` marks
    06a membership either way, so the gate stays recoverable by filtering.
    """
    if location_set not in ("all", "fit"):
        msg = f"location_set must be 'all' or 'fit'; got {location_set!r}"
        raise ValueError(msg)
    if grain != "all" and grain not in GRAIN_FLAG:
        msg = f"grain must be 'all' or one of {tuple(GRAIN_FLAG)}; got {grain!r}"
        raise ValueError(msg)

    keep = (hierarchy_df["level"].isin(list(PAST_LEVELS)) if grain == "all"
            else hierarchy_df[GRAIN_FLAG[grain]] == 1)

    fit_ids = set(fit_df["location_id"].astype(int))
    if location_set == "fit":
        keep = keep & hierarchy_df["location_id"].isin(fit_ids)

    out = hierarchy_df.loc[keep, ["location_id", "level", "A0_location_id"]].copy()
    out["fit_eligible"] = out["location_id"].isin(fit_ids)
    return out.sort_values("location_id").reset_index(drop=True)


def select_location_years(
    as_df: pd.DataFrame,
    *,
    require_nonzero_incidence: bool = False,
) -> pd.DataFrame:
    """The (location, year) pairs to make dense over age/sex.

    Default keeps every pair present in the source. ``require_nonzero_incidence``
    restores the historical rule (the location's OWN all-age cases > 0), which is
    what reduced the FHS-most-detailed coverage to 305 of 473 locations.
    """
    ly = as_df[["location_id", "year_id", "aa_dengue_inc_count"]].drop_duplicates(
        subset=["location_id", "year_id"])
    if require_nonzero_incidence:
        ly = ly[ly["aa_dengue_inc_count"] > 0]
    return ly[["location_id", "year_id"]].reset_index(drop=True)


def _arrays_to_df(arrays: dict, location_ids: list[int], years: list[int]) -> pd.DataFrame:
    """Convert dict of (n_loc, n_year) float32 arrays to a flat location×year DataFrame."""
    idx = pd.MultiIndex.from_product(
        [location_ids, years], names=['location_id', 'year_id']
    )
    return pd.DataFrame(
        {col: arr.flatten() for col, arr in arrays.items()},
        index=idx,
    ).reset_index()


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


def main(
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    ssp_scenario: str = "ssp245",
    location_set: str = "all",
    require_nonzero_incidence: bool = False,
    grain: str = "fhs",
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    den_raked_as_read_path: Path = mbpc.DEN_RAKED_AS_READ_PATH,
    fit_locations_read_path: Path = mbpc.DEN_FIT_LOCATIONS_READ_PATH,
    gdppc_read_path: Path = mbpc.GDPPC_READ_PATH,
    ldipc_read_path: Path = mbpc.LDIPC_READ_PATH,
    urban_read_path: Path = mbpc.URBAN_READ_PATH,
    output_path: Path = mbpc.DEN_PAST_INPUTS_WRITE_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Hierarchy + fit locations (from 06a) ───────────────────────────────
    print("Loading hierarchy...")
    hierarchy_df = read_parquet_with_integer_ids(
        Path(hierarchy_read_path) / f"full_hierarchy_2023_{lsae_hierarchy}.parquet"
    )

    print("Loading dengue fit locations (06a)...")
    fit_df = read_parquet_with_integer_ids(
        Path(fit_locations_read_path) / "fit_location_ids.parquet"
    )
    universe = resolve_location_universe(
        hierarchy_df, fit_df, location_set=location_set, grain=grain)
    location_ids = sorted(universe['location_id'].astype(int).unique().tolist())
    by_level = universe['level'].value_counts().sort_index().to_dict()
    print(f"  Universe: grain={grain!r}, location_set={location_set!r} -> "
          f"{len(location_ids):,} locations {by_level}, {len(PAST_YEARS)} years")
    print(f"  06a fit-eligible among those: "
          f"{int(universe['fit_eligible'].sum()):,} of {len(universe):,}")

    # ── 2. AS raked dengue outcomes (rates + AS population + all-age counts) ──
    print("Loading AS raked dengue data...")
    as_df = read_parquet_with_integer_ids(
        Path(den_raked_as_read_path) / "as_full_dengue_df.parquet",
        filters=[('year_id', 'in', PAST_YEARS), ('location_id', 'in', location_ids)],
    )

    # Row inclusion (see module docstring): every (location, year) by default.
    # Zero-incidence location-years are observed zeros and are KEPT.
    keep_ly = select_location_years(
        as_df, require_nonzero_incidence=require_nonzero_incidence)
    n_all_ly = as_df[['location_id', 'year_id']].drop_duplicates().shape[0]
    rule = ("all-age cases > 0" if require_nonzero_incidence
            else "all (zeros kept)")
    print(f"  Included (location, year) pairs: {len(keep_ly):,} "
          f"of {n_all_ly:,} ({rule})")

    # Dense over age/sex within the included years (NaN where an age/sex is absent).
    age_group_ids = sorted(as_df['age_group_id'].unique().tolist())
    sex_ids       = sorted(as_df['sex_id'].unique().tolist())
    agesex = pd.MultiIndex.from_product(
        [age_group_ids, sex_ids], names=['age_group_id', 'sex_id']
    ).to_frame(index=False)
    dense = keep_ly.merge(agesex, how='cross')

    keys = ['location_id', 'year_id', 'age_group_id', 'sex_id']
    df = dense.merge(
        as_df[keys + ['dengue_inc_rate', 'dengue_mort_rate', 'population']],
        on=keys, how='left',
    )
    print(f"  Dense AS rows: {len(df):,} "
          f"({len(keep_ly):,} loc-years × {len(age_group_ids)} age × {len(sex_ids)} sex)")

    # ── 3. Hierarchy ids: A0 / region / super_region (mapped by location_id) ──
    hier_ids = hierarchy_df.set_index('location_id')[
        ['A0_location_id', 'region_id', 'super_region_id']
    ]
    df['A0_location_id']           = df['location_id'].map(hier_ids['A0_location_id']).astype('int32')
    df['region_location_id']       = df['location_id'].map(hier_ids['region_id']).astype('int32')
    df['super_region_location_id'] = df['location_id'].map(hier_ids['super_region_id']).astype('int32')

    # ── 4. Covariates at LEVEL 5 (admin-2) — sources are LSAE admin-2 products ─
    # Read the covariates at their native level-5 grain, then (section 5) roll
    # them up to the FHS/country parents so every row (levels 3-5) has covariates.
    # Level-5 units under every A0 in the universe -- NOT just the written rows:
    # an FHS parent's covariates are the pop-weighted mean of its admin-2 children,
    # so the children must be read even when only levels 3-4 are written.
    universe_a0 = universe['A0_location_id'].unique()
    level5_ids = sorted(
        hierarchy_df.loc[
            hierarchy_df['A0_location_id'].isin(universe_a0)
            & (hierarchy_df['level'] == 5),
            'location_id'
        ].astype(int).unique().tolist()
    )
    print(f"Reading shared covariates (level 5: {len(level5_ids):,} admin-2)...")
    rcp_scenario = mbpc.ssp_scenarios[ssp_scenario]['rcp_scenario']
    shared_arrays = read_shared_covariates(
        location_ids=level5_ids,
        years=PAST_YEARS,
        gdppc_read_path=gdppc_read_path,
        ldipc_read_path=ldipc_read_path,
        urban_read_path=urban_read_path,
        flooding_path=_resolve_flooding_path(lsae_hierarchy, ssp_scenario),
        rcp_scenario=rcp_scenario,
        med_consumppc_read_path=None,   # dengue: skip med_consumppc (and DAH)
    )
    cov5 = _arrays_to_df(shared_arrays, level5_ids, PAST_YEARS)

    print("Reading climate covariates (level 5, draw 000)...")
    CLIMATE = mbpc.CLIMATE_AGGREGATES_PATH / lsae_hierarchy
    climate_arrays_draws = read_draw_climate(
        location_ids=level5_ids,
        years=PAST_YEARS,
        ssp_scenario=ssp_scenario,
        lsae_hierarchy=lsae_hierarchy,
        extra_vars={'dengue_suitability': str(CLIMATE / f"dengue_suitability_{ssp_scenario}.parquet")},
        draws=PAST_CLIMATE_DRAWS,
    )
    climate_arrays = {k: v[:, :, 0] for k, v in climate_arrays_draws.items()}
    cov5 = cov5.merge(
        _arrays_to_df(climate_arrays, level5_ids, PAST_YEARS),
        on=['location_id', 'year_id'], how='left',
    )

    # ── 5. Population-weighted roll-up of each covariate to levels 3-5 ─────────
    # aggregate_aa_rate_lsae_to_gbd does rate→count→sum-to-parent→count/pop, i.e.
    # a population-weighted mean up the hierarchy. Level-5 values pass through
    # unchanged; levels 3-4 get the pop-weighted aggregate of their admin-2s.
    print("Rolling covariates up to FHS/country levels (population-weighted)...")
    aa_full_population_df = read_parquet_with_integer_ids(
        mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
        columns=['location_id', 'year_id', 'population'],
    )
    hier_fit = hierarchy_df[hierarchy_df['A0_location_id'].isin(universe_a0)].copy()
    cov_cols = [c for c in cov5.columns if c not in ('location_id', 'year_id')]
    cov_all = None
    for c in cov_cols:
        # the rollup returns [location_id, year_id, population, c]; keep only the
        # covariate (df already carries the age-sex population from the raked data).
        full_c = aggregate_aa_rate_lsae_to_gbd(
            c, hier_fit, cov5[['location_id', 'year_id', c]],
            aa_full_population_df, return_full_df=True,
        )[['location_id', 'year_id', c]]
        cov_all = full_c if cov_all is None else cov_all.merge(
            full_c, on=['location_id', 'year_id'], how='outer')
    cov_all = cov_all[cov_all['location_id'].isin(location_ids)]
    df = df.merge(cov_all, on=['location_id', 'year_id'], how='left')

    # ── 5b. Grain flags + level for downstream filtering ──────────────────────
    flag_cols = ['level', 'most_detailed_lsae', 'most_detailed_fhs', 'most_detailed_gbd']
    df = df.merge(hierarchy_df[['location_id'] + flag_cols], on='location_id', how='left')
    # 06a's A0 gate travels as a flag, not a filter: `fit_eligible & inc_rate > 0`
    # reproduces the historical 305-location frame from this artifact.
    df = df.merge(universe[['location_id', 'fit_eligible']], on='location_id', how='left')

    # ── 6. Write parquet ──────────────────────────────────────────────────────
    df = df.sort_values(keys).reset_index(drop=True)
    out_file = output_path / "dengue_past_inputs.parquet"
    print(f"Writing {out_file}  ({len(df):,} rows × {len(df.columns)} cols)...")
    write_parquet(df, out_file)
    print(f"  Done. File size: {out_file.stat().st_size / 1e6:.1f} MB")

    finalize_artifact(mbpc._A03_DEN_PAST_INPUTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dengue past input parquet")
    parser.add_argument("--ssp_scenario", default="ssp245")
    parser.add_argument("--lsae_hierarchy", default=mbpc.LSAE_HIERARCHY)
    parser.add_argument(
        "--location-set", choices=["all", "fit"], default="all",
        help="'all' (default): every location in the chosen grain, observed or "
             "not, zeros kept. 'fit': intersect with 06a's A0-gated set "
             "(historical behaviour).")
    parser.add_argument(
        "--require-nonzero-incidence", action="store_true",
        help="Restore the historical row filter (keep only location-years whose "
             "own all-age case count is > 0). Off by default: zeros are data.")
    parser.add_argument(
        "--grain", choices=["fhs", "lsae", "all"], default="fhs",
        help="Which locations to WRITE, selected on the hierarchy's most-detailed "
             "flag: 'fhs' (default) = exactly the FHS most-detailed set the dengue "
             "models fit on; 'lsae' = admin-2; 'all' = every node at levels 3-5. "
             "The covariate roll-up reads level 5 regardless.")
    args = parser.parse_args()
    main(
        ssp_scenario=args.ssp_scenario,
        lsae_hierarchy=args.lsae_hierarchy,
        location_set=args.location_set,
        require_nonzero_incidence=args.require_nonzero_incidence,
        grain=args.grain,
    )
