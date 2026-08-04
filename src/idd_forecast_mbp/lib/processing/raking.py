"""
Raking functions for the idd-forecast-mbp pipeline.

Three separate methods are preserved — they solve fundamentally different problems
and are not consolidatable (H4 finding from Phase 1 audit).

  rake_level / rake_aa_count_lsae_to_gbd:
    Count-based ratio raking for reconciling LSAE subnational estimates to GBD totals.
    Extracted from: rake_and_aggregate_functions.py

  logit_shift_rake:
    Additive logit-space shift for dengue CFR raking only.
    Extracted from: 04_forecasting/rake_dengue.py (inline script code refactored to function)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import write_parquet
from idd_forecast_mbp.lib.processing._helpers import make_aa_df_square, prep_df
from idd_forecast_mbp.lib.utils.transforms import logit


# ---------------------------------------------------------------------------
# Count-based ratio raking
# ---------------------------------------------------------------------------

def rake_level(
    count_variable: str,
    level_df: pd.DataFrame,
    level_m1_df: pd.DataFrame,
    problematic_rules: dict,
    hierarchy_df: pd.DataFrame,
    level: int,
    year_range: list[int] | None = None,
) -> pd.DataFrame:
    """Rake one hierarchy level to match its parent level using count-based ratio raking.

    Computes a raking factor = parent_count / sum_of_child_counts and applies it
    multiplicatively. Falls back to population-based raking for rows where the
    count-based rate would exceed problematic_rules thresholds.

    Parameters
    ----------
    count_variable:
        Name of the count column to rake (e.g. 'aa_malaria_mort_count').
    level_df:
        DataFrame for the current level (children).
    level_m1_df:
        DataFrame for the parent level (level - 1 aggregated targets).
    problematic_rules:
        Dict controlling when to switch from count-based to population-based raking:
          {'rate_max': {level: float},
           'count_raking_factor_max': float,
           'count_raking_factor_conditional': float,
           'rate_max_conditional': float}
    hierarchy_df:
        Full hierarchy DataFrame (needs 'location_id', 'parent_id', 'level').
    level:
        Current hierarchy level being raked.

    Note: level_df must contain a 'set_by_gbd' boolean column. Rows where
    set_by_gbd is True are left unchanged regardless of raking factors.

    year_range:
        Years to bound the output to. Defaults to mbpc.MODELING_YEARS. Pass an
        explicit list to use a different window (e.g. forecast raking).

    # Extracted from: rake_and_aggregate_functions.py:85
    """
    if year_range is None:
        year_range = mbpc.MODELING_YEARS
    level_df    = level_df[level_df["year_id"].isin(year_range)].copy()
    level_m1_df = level_m1_df[level_m1_df["year_id"].isin(year_range)].copy()
    level_m1_df = level_m1_df.rename(columns={
        count_variable: f'parent_{count_variable}',
        'location_id': 'parent_id',
    })

    level_df = level_df.merge(
        hierarchy_df[['location_id', 'parent_id']],
        on='location_id',
        how='left',
    )
    level_df['current_rate'] = level_df[count_variable] / level_df['population']
    level_df['effective_population'] = np.where(
        level_df[count_variable] > 0, level_df['population'], 0
    )

    level_m1_agg_df = level_df.groupby(['parent_id', 'year_id']).agg({
        count_variable: 'sum',
        'population': 'sum',
        'effective_population': 'sum',
    }).reset_index()
    level_m1_agg_df = level_m1_agg_df.rename(columns={
        'population': 'parent_population',
        'effective_population': 'parent_effective_population',
    })

    level_m1_agg_df = level_m1_agg_df.merge(
        level_m1_df[['year_id', 'parent_id', f'parent_{count_variable}']],
        on=['year_id', 'parent_id'],
        how='left',
    )

    level_m1_agg_df['count_raking_factor'] = (
        level_m1_agg_df[f'parent_{count_variable}'] / level_m1_agg_df[count_variable]
    )
    level_m1_agg_df['full_population_raking_factor'] = (
        level_m1_agg_df[f'parent_{count_variable}'] / level_m1_agg_df['parent_population']
    )
    level_m1_agg_df['population_raking_factor'] = (
        level_m1_agg_df[f'parent_{count_variable}'] / level_m1_agg_df['parent_effective_population']
    )

    for col in ['count_raking_factor', 'full_population_raking_factor', 'population_raking_factor']:
        level_m1_agg_df.loc[level_m1_agg_df[f'parent_{count_variable}'] == 0, col] = 0

    level_df = level_df.merge(
        level_m1_agg_df[[
            'year_id', 'parent_id', f'parent_{count_variable}',
            'count_raking_factor', 'full_population_raking_factor', 'population_raking_factor',
        ]],
        on=['year_id', 'parent_id'],
        how='left',
    )

    level_df['used_full_population_raking_factor'] = (
        level_df['population_raking_factor'] > problematic_rules['rate_max'][level]
    )
    level_df['population_raking_factor'] = np.where(
        level_df['population_raking_factor'] > problematic_rules['rate_max'][level],
        level_df['full_population_raking_factor'],
        level_df['population_raking_factor'],
    )
    level_df = level_df.drop(columns=['full_population_raking_factor'])

    level_df['population_to_use'] = level_df['population']
    eff_mask = level_df['used_full_population_raking_factor'] == False  # noqa: E712
    level_df.loc[eff_mask, 'population_to_use'] = level_df.loc[eff_mask, 'effective_population']

    level_df['count_based_count'] = level_df[count_variable] * level_df['count_raking_factor']
    level_df['population_based_count'] = (
        level_df['population_to_use'] * level_df['population_raking_factor']
    )
    level_df['count_based_rate'] = level_df['count_based_count'] / level_df['population']
    level_df['population_based_rate'] = level_df['population_based_count'] / level_df['population']
    level_df['parent_year_id'] = (
        level_df['parent_id'].astype(str).str.cat(level_df['year_id'].astype(str), sep='_')
    )

    problematic_rows = level_df[
        (level_df['count_raking_factor'] > problematic_rules['count_raking_factor_max']) |
        (level_df['count_based_rate'] > problematic_rules['rate_max'][level]) |
        (
            (level_df['count_raking_factor'] > problematic_rules['count_raking_factor_conditional']) &
            (level_df['count_based_rate'] > problematic_rules['rate_max_conditional'])
        )
    ]
    problematic_rows = problematic_rows[
        problematic_rows['count_based_rate'] > problematic_rows['population_based_rate']
    ]
    npinf_rows = level_df[level_df['count_raking_factor'] == np.inf]
    problematic_rows = pd.concat([problematic_rows, npinf_rows]).drop_duplicates()

    problematic_parent_years = (
        problematic_rows['parent_year_id']
        .drop_duplicates()
        .reset_index(drop=True)
        .to_frame(name='parent_year_id')
    )
    problematic_parent_years['use_population'] = True
    level_df = level_df.merge(problematic_parent_years, on='parent_year_id', how='left')
    level_df['use_population'] = level_df['use_population'].fillna(False).astype('boolean')

    population_mask = (level_df['use_population'] == True) & (level_df['set_by_gbd'] == False)  # noqa: E712
    count_mask = (level_df['use_population'] == False) & (level_df['set_by_gbd'] == False)  # noqa: E712

    level_df.loc[count_mask, count_variable] = (
        level_df.loc[count_mask, count_variable] * level_df.loc[count_mask, 'count_raking_factor']
    )
    level_df.loc[population_mask, count_variable] = (
        level_df.loc[population_mask, 'population_to_use'] *
        level_df.loc[population_mask, 'population_raking_factor']
    )

    drop_cols = (
        [c for c in level_df.columns if 'based' in c or 'raking' in c] +
        ['parent_id', f'parent_{count_variable}', 'use_population',
         'used_full_population_raking_factor', 'population_to_use',
         'effective_population', 'parent_year_id', 'current_rate']
    )
    level_df = level_df.drop(columns=drop_cols)

    return level_df


def rake_aa_count_lsae_to_gbd(
    count_variable: str,
    hierarchy_df: pd.DataFrame,
    aa_gbd_count_df: pd.DataFrame,
    aa_lsae_count_df: pd.DataFrame,
    problematic_rules: dict,
    aa_full_count_df_path: str | Path | None = None,
    return_full_df: bool = False,
    year_range: list[int] | None = None,
) -> pd.DataFrame | None:
    """Rake LSAE all-age counts to match GBD all-age counts at levels 4 and 5.

    GBD numbers are internally inconsistent across levels, so the strategy is:
      1. Replace LSAE values with GBD values wherever GBD has data (set_by_gbd=True).
         These rows are left unchanged by rake_level.
      2. Rake level 4 to GBD level 3 totals (level 4 children must sum to GBD level 3 parents).
      3. Rake level 5 to the already-raked level 4 (shoehorning LSAE to be consistent
         with what level 4 now reports after step 2).

    Parameters
    ----------
    count_variable:
        Name of the count column.
    hierarchy_df:
        Full hierarchy DataFrame.
    aa_gbd_count_df:
        GBD all-age counts (levels 0–4). Used both as replacement values and as
        level-3 raking targets.
    aa_lsae_count_df:
        LSAE all-age counts (levels 4–5, inputs to rake).
    problematic_rules:
        Passed through to rake_level(); controls fallback to population raking.
    aa_full_count_df_path:
        If provided, write the raked full DataFrame to this path.
    return_full_df:
        If True, return the raked DataFrame. If False (default), return None.

    year_range:
        Years to bound the output to. Defaults to mbpc.MODELING_YEARS. Pass an
        explicit list to use a different window.

    # Extracted from: rake_and_aggregate_functions.py:212
    """
    if year_range is None:
        year_range = mbpc.MODELING_YEARS
    aa_gbd_count_df  = aa_gbd_count_df[aa_gbd_count_df["year_id"].isin(year_range)].copy()
    aa_lsae_count_df = aa_lsae_count_df[aa_lsae_count_df["year_id"].isin(year_range)].copy()

    aa_gbd_count_df = prep_df(aa_gbd_count_df, hierarchy_df)
    aa_gbd_count_0_to_3_df = aa_gbd_count_df[aa_gbd_count_df['level'] <= 3].copy()
    aa_lsae_count_df = prep_df(aa_lsae_count_df, hierarchy_df)

    # Ensure all level-3/4/5 location-year combinations exist in LSAE (fill with 0)
    aa_lsae_count_df = make_aa_df_square(
        count_variable, aa_lsae_count_df, hierarchy_df, level_start=3, level_end=5
    )

    # Replace LSAE values with GBD values at GBD-defined locations and mark them
    aa_gbd_count_df[f'{count_variable}_gbd'] = aa_gbd_count_df[count_variable]
    aa_gbd_count_df['set_by_gbd'] = True
    aa_lsae_count_df = aa_lsae_count_df.merge(
        aa_gbd_count_df[['location_id', 'year_id', f'{count_variable}_gbd', 'set_by_gbd']],
        on=['location_id', 'year_id'],
        how='left',
    )
    mask = aa_lsae_count_df['set_by_gbd'].isna()
    aa_lsae_count_df.loc[mask, 'set_by_gbd'] = False
    aa_lsae_count_df['set_by_gbd'] = aa_lsae_count_df['set_by_gbd'].astype('boolean')
    aa_lsae_count_df[count_variable] = aa_lsae_count_df[f'{count_variable}_gbd'].fillna(
        aa_lsae_count_df[count_variable]
    )
    aa_lsae_count_df = aa_lsae_count_df.drop(columns=[f'{count_variable}_gbd'])

    # Rake level 4 to GBD level 3 (parent-level targets)
    aa_gbd_level_3_df = aa_gbd_count_0_to_3_df[aa_gbd_count_0_to_3_df['level'] == 3].copy()
    level_4_df = aa_lsae_count_df[aa_lsae_count_df['level'] == 4].copy()
    level_4_df = make_aa_df_square(count_variable, level_4_df, hierarchy_df, 4, 4)
    level_4_df = rake_level(
        count_variable, level_4_df, aa_gbd_level_3_df, problematic_rules, hierarchy_df,
        level=4, year_range=year_range,
    )

    # Rake level 5 to the raked level 4 (not original LSAE level 4)
    level_5_df = aa_lsae_count_df[aa_lsae_count_df['level'] == 5].copy()
    level_5_df = make_aa_df_square(count_variable, level_5_df, hierarchy_df, 5, 5)
    level_5_df = rake_level(
        count_variable, level_5_df, level_4_df, problematic_rules, hierarchy_df,
        level=5, year_range=year_range,
    )

    aa_full_count_df = pd.concat([
        aa_gbd_count_0_to_3_df,
        level_4_df,
        level_5_df,
    ], ignore_index=True)

    if 'level' in aa_full_count_df.columns:
        aa_full_count_df = aa_full_count_df.drop(columns=['level'])

    if aa_full_count_df_path is not None:
        write_parquet(aa_full_count_df, aa_full_count_df_path)

    if return_full_df:
        return aa_full_count_df
    return None


# ---------------------------------------------------------------------------
# Logit-shift raking (dengue CFR only)
# ---------------------------------------------------------------------------

def logit_shift_rake(
    forecast_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    rate_column: str,
    pred_column: str,
    rake_year: int = 2022,
    clip_upper: float = 0.99,
) -> pd.DataFrame:
    """Rake by computing and applying an additive logit-space shift.

    For each location/age/sex:
      shift = logit(observed_rate at rake_year) - logit(predicted_rate at rake_year)
    The shift is then added to all forecast years' predicted rates:
      raked_pred = raw_pred + shift

    Used for dengue CFR raking only. Preserves valid probability range via logit space.

    Parameters
    ----------
    forecast_df:
        Full forecast DataFrame (all years) with a raw predicted rate column
        named f'{pred_column}_raw'. Must contain the join keys used in observed_df.
    observed_df:
        DataFrame containing observed rates at rake_year.
        Must have the same join keys as forecast_df (typically location_id,
        age_group_id, sex_id).
    rate_column:
        Name of the observed rate column in observed_df (e.g. 'dengue_cfr').
    pred_column:
        Base name of the predicted column. The function expects
        f'{pred_column}_raw' in forecast_df and will produce f'{pred_column}'.
    rake_year:
        Year at which to compute the shift. Default 2022.
    clip_upper:
        Clip observed rates at this value before logit transform. Default 0.99.

    Returns
    -------
    forecast_df with f'{pred_column}' added and f'{pred_column}_raw' dropped.

    # Extracted from: 04_forecasting/rake_dengue.py:162-176 (inline script code)
    """
    join_keys = [c for c in observed_df.columns if c != rate_column]

    obs_rake_year = observed_df.copy()
    obs_rake_year[f'logit_{rate_column}'] = logit(obs_rake_year[rate_column], clip_upper=clip_upper)

    pred_raw_col = f'{pred_column}_raw'
    forecast_rake_year = forecast_df[forecast_df['year_id'] == rake_year].copy()

    forecast_rake_year = forecast_rake_year.merge(
        obs_rake_year[join_keys + [f'logit_{rate_column}']],
        on=join_keys,
        how='left',
    )
    forecast_rake_year['shift'] = (
        forecast_rake_year[f'logit_{rate_column}'] - forecast_rake_year[pred_raw_col]
    )

    forecast_df = forecast_df.merge(
        forecast_rake_year[join_keys + ['shift']],
        on=join_keys,
        how='left',
    )
    forecast_df[pred_column] = forecast_df[pred_raw_col] + forecast_df['shift']
    forecast_df = forecast_df.drop(columns=['shift', pred_raw_col], errors='ignore')

    return forecast_df
