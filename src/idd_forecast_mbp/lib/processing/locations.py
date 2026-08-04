"""Malaria + dengue location selection helpers.

malaria_fit_location_ids:        the endemic set used for malaria model fitting.
malaria_prediction_location_ids: superset used for prediction — all level-5
                                 locations in any A0 with malaria history,
                                 regardless of whether the location itself
                                 had nonzero historical pfpr.
dengue_fit_location_ids:         level-5 locations whose A0 cleared the dengue
                                 fit count thresholds in the gate year (2023).
dengue_prediction_location_ids:  level-5 locations whose A0 cleared the dengue
                                 prediction count thresholds in ANY year.
"""

from __future__ import annotations

import pandas as pd

from idd_forecast_mbp import constants as mbpc


def malaria_fit_location_ids(
    aa_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    mort_threshold: float = 1.0,
    endemic_year: int = mbpc.MODELING_YEARS[-1],
) -> list[int]:
    """Return sorted endemic most-detailed location IDs for model fitting.

    Endemic A0s are those with all-age malaria mort_count >= mort_threshold in
    `endemic_year`, which defaults to the last modeling year
    (mbpc.MODELING_YEARS[-1]) — NOT a hardcoded literal, so it tracks the
    modeling window. The returned location set is the level-5 (most-detailed)
    locations within those A0s that also have positive per-row history
    (pfpr > 0, mort_count > 0, inc_count > 0).
    """
    df = aa_df.merge(
        hierarchy_df[['location_id', 'A0_location_id', 'most_detailed_lsae']],
        on='location_id', how='left',
    )
    df = df[
        (df['malaria_pfpr'] > 0) &
        (df['malaria_mort_count'] > 0) &
        (df['malaria_inc_count'] > 0)
    ]
    a0_endemic_year = df[
        (df['location_id'] == df['A0_location_id']) & (df['year_id'] == endemic_year)
    ]
    endemic_a0 = a0_endemic_year[a0_endemic_year['malaria_mort_count'] >= mort_threshold]['A0_location_id'].unique()
    md = df[df['A0_location_id'].isin(endemic_a0) & (df['most_detailed_lsae'] == 1)]
    return sorted(md['location_id'].unique().tolist())


def malaria_prediction_location_ids(
    aa_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    mort_threshold: float = 0.0,
) -> list[int]:
    """Return sorted prediction-set most-detailed location IDs.

    Broader than the fit set: any level-5 location whose A0 had at least one
    record with malaria_mort_count >= mort_threshold (no per-row history
    requirement on the location itself). Includes locations with zero
    historical pfpr — the R model can still predict for them because their A0
    fixed effect was estimated. A0s not in the fit set must be handled
    separately at predict time.
    """
    df = aa_df.merge(
        hierarchy_df[['location_id', 'A0_location_id', 'most_detailed_lsae']],
        on='location_id', how='left',
    )
    df = df[
        (df['malaria_pfpr'] > 0) &
        (df['malaria_mort_count'] > 0) &
        (df['malaria_inc_count'] > 0)
    ]
    a0_df = df[df['location_id'] == df['A0_location_id']]
    endemic_a0 = a0_df[a0_df['malaria_mort_count'] >= mort_threshold]['A0_location_id'].unique()
    md = hierarchy_df[
        (hierarchy_df['most_detailed_lsae'] == 1) &
        (hierarchy_df['A0_location_id'].isin(endemic_a0))
    ]
    return sorted(md['location_id'].unique().tolist())


def dengue_eligible_a0_ids(
    aa_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    mort_threshold: float = mbpc.dengue_fit_mort_threshold,
    inc_threshold: float = mbpc.dengue_fit_inc_threshold,
    gate_year: int = mbpc.MODELING_YEARS[-1],
):
    """Return the dengue-eligible A0 (country) IDs.

    An A0 qualifies iff its own all-age `gate_year` record (default = last
    modeling year, 2023) had BOTH `dengue_mort_count > mort_threshold` AND
    `dengue_inc_count > inc_threshold`. Thresholds default to the (exploratory,
    0.0) constants in constants.py.

    Shared gate used by both dengue_fit_location_ids (level-5 only) and
    dengue_fit_locations (all levels 3-5). `aa_df` is the all-age raked dengue
    frame (must include A0/national rows).
    """
    df = aa_df.merge(
        hierarchy_df[['location_id', 'A0_location_id']],
        on='location_id', how='left',
    )
    a0_gate = df[(df['location_id'] == df['A0_location_id']) & (df['year_id'] == gate_year)]
    return a0_gate[
        (a0_gate['dengue_mort_count'] > mort_threshold) &
        (a0_gate['dengue_inc_count'] > inc_threshold)
    ]['A0_location_id'].unique()


def dengue_fit_location_ids(
    aa_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    mort_threshold: float = mbpc.dengue_fit_mort_threshold,
    inc_threshold: float = mbpc.dengue_fit_inc_threshold,
    gate_year: int = mbpc.MODELING_YEARS[-1],
) -> list[int]:
    """Return sorted most-detailed (level-5) location IDs for dengue model FITTING.

    A level-5 location qualifies iff its A0 cleared the dengue count thresholds
    in the gate year (see dengue_eligible_a0_ids). No per-location history
    requirement — the per-(location, year) "non-zero all-age deaths or cases"
    filter is applied at past-inputs build time (06b), not in this location set.

    `aa_df` is the all-age raked dengue frame (must include A0/national rows).
    """
    endemic_a0 = dengue_eligible_a0_ids(
        aa_df, hierarchy_df, mort_threshold, inc_threshold, gate_year
    )
    md = hierarchy_df[
        (hierarchy_df['most_detailed_lsae'] == 1) &
        (hierarchy_df['A0_location_id'].isin(endemic_a0))
    ]
    return sorted(md['location_id'].unique().tolist())


def dengue_fit_locations(
    aa_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    levels: tuple[int, ...] = (3, 4, 5),
    mort_threshold: float = mbpc.dengue_fit_mort_threshold,
    inc_threshold: float = mbpc.dengue_fit_inc_threshold,
    gate_year: int = mbpc.MODELING_YEARS[-1],
) -> pd.DataFrame:
    """Return every location at `levels` (default 3-5: country / admin-1 /
    admin-2) within the dengue-eligible A0s, tagged with the three grain flags.

    Lets 06b build one past-inputs table spanning all grains; downstream picks a
    grain by filtering most_detailed_lsae / most_detailed_fhs / most_detailed_gbd.

    Returns columns:
      location_id, A0_location_id, level,
      most_detailed_lsae, most_detailed_fhs, most_detailed_gbd
    """
    endemic_a0 = dengue_eligible_a0_ids(
        aa_df, hierarchy_df, mort_threshold, inc_threshold, gate_year
    )
    cols = ['location_id', 'A0_location_id', 'level',
            'most_detailed_lsae', 'most_detailed_fhs', 'most_detailed_gbd']
    out = hierarchy_df[
        hierarchy_df['A0_location_id'].isin(endemic_a0) &
        hierarchy_df['level'].isin(list(levels))
    ][cols].copy()
    return out.sort_values('location_id').reset_index(drop=True)


def dengue_prediction_location_ids(
    aa_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    mort_threshold: float = mbpc.dengue_pred_mort_threshold,
    inc_threshold: float = mbpc.dengue_pred_inc_threshold,
) -> list[int]:
    """Return sorted most-detailed location IDs for the dengue PREDICTION set.

    Broader than the fit set: a most-detailed location qualifies iff its A0 had
    ANY year whose all-age record has BOTH `dengue_mort_count > mort_threshold`
    AND `dengue_inc_count > inc_threshold` (same year). No per-location history
    requirement. Thresholds default to the (exploratory, 0.0) constants.

    `aa_df` is the all-age raked dengue frame (must include A0/national rows).
    """
    df = aa_df.merge(
        hierarchy_df[['location_id', 'A0_location_id', 'most_detailed_lsae']],
        on='location_id', how='left',
    )
    a0_df = df[df['location_id'] == df['A0_location_id']]
    endemic_a0 = a0_df[
        (a0_df['dengue_mort_count'] > mort_threshold) &
        (a0_df['dengue_inc_count'] > inc_threshold)
    ]['A0_location_id'].unique()
    md = hierarchy_df[
        (hierarchy_df['most_detailed_lsae'] == 1) &
        (hierarchy_df['A0_location_id'].isin(endemic_a0))
    ]
    return sorted(md['location_id'].unique().tolist())
