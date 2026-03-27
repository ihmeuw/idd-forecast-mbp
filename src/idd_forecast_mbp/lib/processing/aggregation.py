"""
Hierarchy aggregation functions for the idd-forecast-mbp pipeline.

Two variants are preserved:
  AA (all-age): groups on [parent_id, year_id] — collapses age/sex dims
  AS (age-sex): groups on [parent_id, year_id, age_group_id, sex_id] — preserves them

Extracted from: rake_and_aggregate_functions.py
               05_aggregation/cause_as_aggregation_by_draw.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from idd_forecast_mbp.lib.io.parquet import write_parquet
from idd_forecast_mbp.lib.processing._helpers import make_aa_df_square, prep_df


# ---------------------------------------------------------------------------
# Core level aggregation
# ---------------------------------------------------------------------------

def aggregate_level(
    count_variable: str,
    level_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate count_variable from level_df to the next higher (parent) level.

    AA version: groups on [parent_id, year_id] only — does not preserve age/sex.
    For the AS variant, use aggregate_to_parent(preserve_age_sex=True).

    Parameters
    ----------
    count_variable:
        Name of the count column to sum.
    level_df:
        DataFrame for the current level (children).
    hierarchy_df:
        Full hierarchy DataFrame (needs 'location_id', 'parent_id').

    Returns
    -------
    DataFrame with aggregated counts at the parent level, keyed by
    (location_id=parent_id, year_id).

    # Extracted from: rake_and_aggregate_functions.py:284
    """
    level_df = level_df.merge(
        hierarchy_df[['location_id', 'parent_id']],
        on='location_id',
        how='left',
    )
    agg_df = (
        level_df.groupby(['parent_id', 'year_id'])[count_variable]
        .sum()
        .reset_index()
        .rename(columns={'parent_id': 'location_id'})
    )
    return agg_df


# ---------------------------------------------------------------------------
# Full hierarchy aggregation — counts
# ---------------------------------------------------------------------------

def aggregate_aa_count_lsae_to_gbd(
    count_variable: str,
    hierarchy_df: pd.DataFrame,
    aa_lsae_count_df: pd.DataFrame,
    aa_full_count_df_path: str | Path | None = None,
    return_full_df: bool = False,
) -> pd.DataFrame | None:
    """Aggregate LSAE all-age counts up through GBD hierarchy levels (5→0).

    Starts from level 5, calls aggregate_level iteratively to produce all levels.
    Fills missing location/year combinations with 0 before aggregating.

    Parameters
    ----------
    count_variable:
        Name of the count column.
    hierarchy_df:
        Full hierarchy DataFrame.
    aa_lsae_count_df:
        Level-5 LSAE count DataFrame to aggregate upward.
    aa_full_count_df_path:
        If provided, write the result to this path.
    return_full_df:
        If True, return the full DataFrame. If False (default), return None.

    # Extracted from: rake_and_aggregate_functions.py:301
    """
    aa_lsae_count_df = prep_df(aa_lsae_count_df, hierarchy_df)
    level_5_df = aa_lsae_count_df[aa_lsae_count_df['level'] == 5].copy()
    level_5_hierarchy_df = hierarchy_df[hierarchy_df['level'] == 5].copy()
    level_5_df = make_aa_df_square(count_variable, level_5_df, level_5_hierarchy_df, 5, 5)

    level_dfs = [level_5_df]
    level_df = level_5_df.copy()
    for _ in range(4, -1, -1):
        level_df = aggregate_level(count_variable, level_df, hierarchy_df)
        level_dfs.append(level_df)

    aa_full_count_df = pd.concat(level_dfs, ignore_index=True)

    if aa_full_count_df_path is not None:
        write_parquet(aa_full_count_df, aa_full_count_df_path)

    if return_full_df:
        return aa_full_count_df
    return None


# ---------------------------------------------------------------------------
# Full hierarchy aggregation — rates
# ---------------------------------------------------------------------------

def aggregate_aa_rate_lsae_to_gbd(
    rate_variable: str,
    hierarchy_df: pd.DataFrame,
    aa_lsae_rate_df: pd.DataFrame,
    aa_full_population_df: pd.DataFrame,
    aa_full_rate_df_path: str | Path | None = None,
    return_full_df: bool = False,
) -> pd.DataFrame | None:
    """Aggregate LSAE all-age rates by converting rate→count→aggregate→rate.

    Steps:
      1. Multiply level-5 rates by population to get counts.
      2. Call aggregate_aa_count_lsae_to_gbd to get full hierarchy counts.
      3. Divide counts by population to recover rates at each level.

    Parameters
    ----------
    rate_variable:
        Name of the rate column.
    hierarchy_df:
        Full hierarchy DataFrame.
    aa_lsae_rate_df:
        Level-5 LSAE rate DataFrame.
    aa_full_population_df:
        All-age population DataFrame (location_id, year_id, population).
    aa_full_rate_df_path:
        If provided, write the result to this path.
    return_full_df:
        If True, return the full rate DataFrame.

    # Extracted from: rake_and_aggregate_functions.py:342
    """
    if 'population' in aa_lsae_rate_df.columns:
        aa_lsae_rate_df = aa_lsae_rate_df.drop(columns=['population'])

    tmp_count_variable = 'tmp_count'
    aa_lsae_rate_df = prep_df(aa_lsae_rate_df, hierarchy_df)
    tmp_df = aa_lsae_rate_df[aa_lsae_rate_df['level'] == 5].copy()
    tmp_df = make_aa_df_square(tmp_count_variable, tmp_df, hierarchy_df, level_start=5, level_end=5)
    tmp_df = tmp_df.merge(aa_full_population_df, on=['location_id', 'year_id'], how='left')
    tmp_df[tmp_count_variable] = tmp_df[rate_variable] * tmp_df['population']
    tmp_df = tmp_df.drop(columns=[rate_variable, 'population'])

    aa_full_count_df = aggregate_aa_count_lsae_to_gbd(
        tmp_count_variable, hierarchy_df, tmp_df, return_full_df=True
    )
    aa_full_rate_df = make_rate_from_count(
        rate_variable, tmp_count_variable,
        aa_full_count_df, aa_full_population_df,
        aa_full_rate_df_path=aa_full_rate_df_path,
        return_full_df=True,
    )

    if 'level' in aa_full_rate_df.columns:
        aa_full_rate_df = aa_full_rate_df.drop(columns=['level'])

    if return_full_df:
        return aa_full_rate_df
    return None


def make_rate_from_count(
    rate_variable: str,
    count_variable: str,
    aa_full_count_df: pd.DataFrame,
    aa_full_population_df: pd.DataFrame,
    aa_full_rate_df_path: str | Path | None = None,
    return_full_df: bool = False,
) -> pd.DataFrame | None:
    """Divide count by population to produce a rate variable.

    Sets rate = 0 where population = 0 to avoid division by zero.
    Drops count_variable and level column from the result.

    Parameters
    ----------
    rate_variable:
        Name for the output rate column.
    count_variable:
        Name of the input count column (dropped from result).
    aa_full_count_df:
        Full-hierarchy count DataFrame.
    aa_full_population_df:
        Population DataFrame (location_id, year_id, population).
    aa_full_rate_df_path:
        If provided, write the result to this path.
    return_full_df:
        If True, return the rate DataFrame.

    # Extracted from: rake_and_aggregate_functions.py:325
    """
    if 'population' in aa_full_count_df.columns:
        aa_full_count_df = aa_full_count_df.drop(columns=['population'])

    df = aa_full_count_df.merge(
        aa_full_population_df[['location_id', 'year_id', 'population']],
        on=['location_id', 'year_id'],
        how='left',
    ).copy()
    df[rate_variable] = df[count_variable] / df['population']
    df.loc[df['population'] == 0, rate_variable] = 0
    df = df.drop(columns=[count_variable])
    if 'level' in df.columns:
        df = df.drop(columns=['level'])

    if aa_full_rate_df_path is not None:
        write_parquet(df, aa_full_rate_df_path)

    if return_full_df:
        return df
    return None


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------

def aggregate_to_parent(
    df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    count_variable: str,
    preserve_age_sex: bool = False,
) -> pd.DataFrame:
    """Aggregate count_variable to the parent level of each location.

    Parameters
    ----------
    df:
        Input DataFrame. Must contain 'location_id', 'year_id', count_variable,
        and 'age_group_id'/'sex_id' if preserve_age_sex=True.
    hierarchy_df:
        Full hierarchy DataFrame (needs 'location_id', 'parent_id').
    count_variable:
        Name of the count column to sum.
    preserve_age_sex:
        False (default): AA variant — groups on [parent_id, year_id] only.
        True: AS variant — groups on [parent_id, year_id, age_group_id, sex_id].

    # Extracted from: rake_and_aggregate_functions.py:284 (AA)
    #                 05_aggregation/cause_as_aggregation_by_draw.py:112 (AS)
    """
    df = df.merge(
        hierarchy_df[['location_id', 'parent_id']],
        on='location_id',
        how='left',
    )

    if preserve_age_sex:
        group_keys = ['parent_id', 'year_id', 'age_group_id', 'sex_id']
    else:
        group_keys = ['parent_id', 'year_id']

    agg_df = (
        df.groupby(group_keys)[count_variable]
        .sum()
        .reset_index()
        .rename(columns={'parent_id': 'location_id'})
    )
    return agg_df
