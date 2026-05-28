"""Malaria location selection helpers.

malaria_fit_location_ids:        the endemic set used for model fitting.
malaria_prediction_location_ids: superset used for prediction — all level-5
                                 locations in any A0 with malaria history,
                                 regardless of whether the location itself
                                 had nonzero historical pfpr.
"""

from __future__ import annotations

import pandas as pd


def malaria_fit_location_ids(
    aa_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    mort_threshold: float = 1.0,
) -> list[int]:
    """Return sorted endemic most-detailed location IDs for model fitting.

    Endemic A0s are those with 2022 all-age malaria mort_count >= mort_threshold.
    The returned location set is the level-5 (most-detailed) locations within
    those A0s that also have positive per-row history (pfpr > 0, mort_count > 0,
    inc_count > 0).
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
    a0_2022 = df[
        (df['location_id'] == df['A0_location_id']) & (df['year_id'] == 2022)
    ]
    endemic_a0 = a0_2022[a0_2022['malaria_mort_count'] >= mort_threshold]['A0_location_id'].unique()
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
