"""
Shared processing helpers used across lib/processing/ and by pipeline scripts.

make_aa_df_square: fills missing location/year rows with zeros.
prep_df:           adds hierarchy level column, drops parent_id.
level_filter:      builds a parquet location_id filter for given level range.
"""

from __future__ import annotations

import pandas as pd


def level_filter(
    hierarchy_df: pd.DataFrame,
    start_level: int,
    end_level: int | None = None,
    return_ids: bool = False,
) -> tuple | tuple[tuple, list]:
    """Build a parquet location_id filter for hierarchy levels [start_level, end_level].

    Parameters
    ----------
    hierarchy_df:
        Full hierarchy DataFrame with 'location_id' and 'level' columns.
    start_level:
        Lowest level to include (inclusive).
    end_level:
        Highest level to include (inclusive). Defaults to start_level.
    return_ids:
        If True, return (filter_tuple, location_ids_list).
        If False (default), return filter_tuple only.

    # Extracted from: helper_functions.py:92
    """
    if end_level is None:
        end_level = start_level
    levels_to_filter_on = list(range(start_level, end_level + 1))
    location_ids = (
        hierarchy_df[hierarchy_df['level'].isin(levels_to_filter_on)]
        ['location_id'].unique().tolist()
    )
    location_filter = ('location_id', 'in', location_ids)
    if return_ids:
        return location_filter, location_ids
    return location_filter


def prep_df(df: pd.DataFrame, hierarchy_df: pd.DataFrame) -> pd.DataFrame:
    """Add 'level' column from hierarchy; drop 'parent_id' if present.

    # Extracted from: rake_and_aggregate_functions.py:74
    """
    if 'level' not in df.columns:
        df = df.merge(
            hierarchy_df[['location_id', 'level']],
            on='location_id',
            how='left',
        ).copy()
    if 'parent_id' in df.columns:
        df = df.drop(columns=['parent_id'])
    return df


def make_aa_df_square(
    variable: str | list[str],
    df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    level_start: int,
    level_end: int,
) -> pd.DataFrame:
    """Fill missing location/year combinations with zeros for the given level range.

    Ensures every location in [level_start, level_end] has a row for every
    year present in df, with zero for any missing variable values.

    # Extracted from: rake_and_aggregate_functions.py:36
    """
    df = df.copy()
    years = df['year_id'].unique()
    level_hierarchy_df = hierarchy_df[
        (hierarchy_df['level'] >= level_start) &
        (hierarchy_df['level'] <= level_end)
    ].copy()

    variables = [variable] if isinstance(variable, str) else list(variable)
    missing_dfs = []

    for year in years:
        year_df = df[df['year_id'] == year]
        missing_rows = level_hierarchy_df[
            ~level_hierarchy_df['location_id'].isin(year_df['location_id'])
        ]
        if not missing_rows.empty:
            missing_dict: dict = {
                'location_id': missing_rows['location_id'].values,
                'year_id': year,
                'level': missing_rows['level'].values,
            }
            for var in variables:
                missing_dict[var] = 0
            missing_dfs.append(pd.DataFrame(missing_dict))

    if missing_dfs:
        missing_df = pd.concat(missing_dfs, ignore_index=True).drop(columns=['level'])
        df = pd.concat([df, missing_df], ignore_index=True)

    return df
