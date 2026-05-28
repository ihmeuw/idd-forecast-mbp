"""
Hierarchy loading and filtering utilities for the idd-forecast-mbp pipeline.

Extracted from: src/idd_forecast_mbp/helper_functions.py (level_filter — canonical)
                Inline patterns in 12+ pipeline scripts (load_hierarchy, make_location_filter)

level_filter() is the canonical implementation. Most pipeline scripts inlined the same
logic directly — Phase 4 will replace those with calls to this function.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids


def load_hierarchy(path: str | Path | None = None) -> pd.DataFrame:
    """Load the full LSAE hierarchy parquet.

    Parameters
    ----------
    path:
        Explicit path to the hierarchy parquet. If None, defaults to
        constants.HIERARCHY_READ_PATH/full_hierarchy_2023_{LSAE_HIERARCHY}.parquet.

    # Extracted from: inline in 12+ scripts, e.g. 02_data_prep/00_make_covariate_means.py:30
    """
    if path is None:
        from idd_forecast_mbp import constants as mbpc
        path = mbpc.HIERARCHY_READ_PATH / f"full_hierarchy_2023_{mbpc.LSAE_HIERARCHY}.parquet"
    return read_parquet_with_integer_ids(path)


def level_filter(
    hierarchy_df: pd.DataFrame,
    start_level: int,
    end_level: int | None = None,
    return_ids: bool = False,
) -> tuple | tuple[tuple, list[int]]:
    """Return a parquet filter tuple for locations at one or more hierarchy levels.

    Parameters
    ----------
    hierarchy_df:
        Full hierarchy DataFrame (must contain 'level' and 'location_id' columns).
    start_level:
        Lowest level to include. When end_level is None, only this level is selected.
    end_level:
        Highest level to include (inclusive). Defaults to start_level (single level).
    return_ids:
        If True, return (filter_tuple, location_ids) instead of just the filter_tuple.

    Returns
    -------
    filter_tuple : ('location_id', 'in', [...])
        Parquet filter tuple for use in read_parquet_with_integer_ids(filters=[...]).
    location_ids : list[int]
        Only returned when return_ids=True.

    Examples
    --------
    # Single level
    f = level_filter(hierarchy_df, 5)

    # Level range
    f = level_filter(hierarchy_df, 3, 5)

    # With IDs
    f, ids = level_filter(hierarchy_df, 5, return_ids=True)

    # Extracted from: helper_functions.py:92 (canonical implementation)
    """
    if end_level is None:
        end_level = start_level
    levels = list(range(start_level, end_level + 1))
    location_ids = (
        hierarchy_df[hierarchy_df['level'].isin(levels)]['location_id']
        .unique()
        .tolist()
    )
    location_filter = ('location_id', 'in', location_ids)
    if return_ids:
        return location_filter, location_ids
    return location_filter


def get_location_ids(
    hierarchy_df: pd.DataFrame,
    levels: int | list[int],
    extra_filter: pd.Series | None = None,
) -> list[int]:
    """Return location_ids for one or more hierarchy levels.

    Parameters
    ----------
    hierarchy_df:
        Full hierarchy DataFrame.
    levels:
        A single level or list of levels to include.
    extra_filter:
        Optional boolean Series (same index as hierarchy_df) applied after
        the level filter. Used for cases like filtering by gbd_location_id
        membership before extracting location_ids.
        Example: hierarchy_df['gbd_location_id'].isin(some_gbd_ids)

    # Extracted from: inline patterns across pipeline scripts;
    #   extra_filter handles the special case in 04_forecasting/as_malaria_fractions.py:204
    """
    if isinstance(levels, int):
        levels = [levels]
    mask = hierarchy_df['level'].isin(levels)
    if extra_filter is not None:
        mask = mask & extra_filter
    return hierarchy_df[mask]['location_id'].unique().tolist()


def make_location_filter(location_ids: list[int]) -> tuple:
    """Return a ('location_id', 'in', location_ids) parquet filter tuple.

    # Extracted from: inline pattern across all pipeline stages (L1 pattern)
    """
    return ('location_id', 'in', location_ids)
