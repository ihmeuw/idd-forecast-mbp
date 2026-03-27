"""
Internal helpers shared by raking.py and aggregation.py.

Not part of the public lib API — import only within lib/processing/.
"""

from __future__ import annotations

import pandas as pd


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
