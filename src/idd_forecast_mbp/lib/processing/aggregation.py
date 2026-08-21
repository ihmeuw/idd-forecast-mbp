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

import numpy as np
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

    if 'level' in aa_full_rate_df.columns:  # pragma: no cover
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
    join_cols: tuple[str, ...] | list[str] = ("location_id", "year_id"),
) -> pd.DataFrame | None:
    """Divide count by population to produce a rate variable.

    Sets rate = 0 where population = 0 to avoid division by zero.
    Drops count_variable and level column from the result.

    ``join_cols`` is the grain the population is matched at. It defaults to
    all-age ``(location_id, year_id)``. Pass
    ``("location_id", "year_id", "age_group_id", "sex_id")`` with an age/sex
    population frame to build age/sex rates -- leaving the default there would
    silently divide every age/sex count by the location's ALL-AGE population.

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

    join = list(join_cols)
    df = aa_full_count_df.merge(
        aa_full_population_df[[*join, 'population']],
        on=join,
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


def aggregate_outcomes_to_ancestors(
    leaf_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    count_cols: dict[str, str],
    population_df: pd.DataFrame,
    *,
    preserve_age_sex: bool = False,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Counts summed into every ancestor, with rates rebuilt from OWN population.

    The composition that is correct-by-construction, so no caller has to remember
    the rule: :func:`roll_up_to_ancestors` sums the counts, then
    :func:`make_rate_from_count` divides each ancestor's count by *that level's own*
    population row. Summing the leaves' populations instead inflates every aggregate
    rate, because a parent's population includes people no leaf contributed a count
    for. That is the single most common aggregation defect in this codebase and the
    thing ``lib/processing/products.validate_products`` exists to reject.

    ``preserve_age_sex=True`` keeps the age/sex grid and matches the population at
    age/sex grain, so an age-and-sex-specific rate for any ancestor is one call.
    ``population_df`` must be at the matching grain: all-age when
    ``preserve_age_sex`` is False, age/sex when it is True.

    Parameters
    ----------
    leaf_df:
        Leaf-level counts keyed by 'location_id', 'year_id' (plus age/sex when
        ``preserve_age_sex``). Leaves may sit at mixed levels.
    hierarchy_df:
        Needs 'location_id' and 'path_to_top_parent'.
    count_cols:
        ``{count_column: rate_column}`` — the counts to sum and the name each
        resulting rate gets.
    population_df:
        Population at every level, at the grain implied by ``preserve_age_sex``.
    preserve_age_sex:
        Keep the age/sex grid rather than collapsing to all-age.
    extra_group_cols:
        Additional dimensions to preserve (e.g. ``['draw']``).

    Returns
    -------
    One row per (ancestor location, year, [age, sex], [extra dims]) carrying every
    count, every rate, and the 'population' the rates were divided by.
    """
    counts = roll_up_to_ancestors(
        leaf_df, hierarchy_df, list(count_cols),
        extra_group_cols=extra_group_cols, preserve_age_sex=preserve_age_sex,
    )
    keys = ['location_id', 'year_id']
    if preserve_age_sex:
        keys += ['age_group_id', 'sex_id']
    keys += list(extra_group_cols or [])

    out = counts
    for count_col, rate_col in count_cols.items():
        rated = make_rate_from_count(
            rate_col, count_col, counts[[*keys, count_col]], population_df,
            return_full_df=True, join_cols=tuple(keys),
        )
        if 'population' in out.columns:
            rated = rated.drop(columns=['population'])
        out = out.merge(rated, on=keys, how='left')
    return out


def roll_up_covariates_to_ancestors(
    cov_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    covariate_cols: str | list[str],
    population_df: pd.DataFrame,
    *,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Population-weighted mean of location-year covariates into every ancestor.

    The ancestor-wise counterpart of :func:`roll_up_to_ancestors`, which sums
    counts. Covariates do not sum: a region's GDP per capita is the
    population-weighted mean of its children's, not their total. It is also the
    arbitrary-ancestor counterpart of :func:`aggregate_aa_rate_lsae_to_gbd`, which
    does the same weighting but only for the fixed lsae -> gbd step.

    Uses the same ``path_to_top_parent`` mechanism as
    :func:`roll_up_to_ancestors`, so leaves are returned unchanged alongside the
    aggregates and mixed-level leaf sets (the FHS 473 straddle levels 3 and 4) are
    handled without iterating level by level.

    Weights are each LEAF's own population. A covariate has no count to sum, so a
    weighted mean is the only available definition, and the weights must be the
    leaves' populations so they add up to the ancestor's aggregate.

    A NaN covariate at a leaf propagates to every ancestor above it rather than
    being skipped: dropping it would silently reweight the remaining children and
    hide the gap.

    Parameters
    ----------
    cov_df:
        Covariates at the leaf locations, keyed by 'location_id' and 'year_id'
        (plus any extra dims). Covariates are location-year attributes, so any
        age/sex duplication is collapsed before weighting.
    hierarchy_df:
        Needs 'location_id' and 'path_to_top_parent'.
    covariate_cols:
        Covariate column, or list of them, to aggregate.
    population_df:
        Leaf populations, columns 'location_id', 'year_id', 'population'.
    extra_group_cols:
        Additional dimensions to preserve through the aggregation.

    Returns
    -------
    One row per (ancestor location, year, [extra dims]) with the weighted means.
    """
    cols = [covariate_cols] if isinstance(covariate_cols, str) else list(covariate_cols)
    if 'path_to_top_parent' not in hierarchy_df.columns:
        msg = "hierarchy_df must carry 'path_to_top_parent' to roll up to ancestors"
        raise KeyError(msg)

    paths = (
        hierarchy_df.drop_duplicates('location_id')
        .set_index('location_id')['path_to_top_parent']
    )
    missing = set(cov_df['location_id'].unique()) - set(paths.index)
    if missing:
        msg = (f"{len(missing)} location_id(s) absent from the hierarchy, "
               f"e.g. {sorted(missing)[:5]}")
        raise KeyError(msg)

    keys = ['location_id', 'year_id', *(extra_group_cols or [])]
    out = cov_df[[*keys, *cols]].drop_duplicates(subset=keys)
    if 'population' in out.columns:
        out = out.drop(columns=['population'])
    out = out.merge(
        population_df[['location_id', 'year_id', 'population']],
        on=['location_id', 'year_id'], how='left',
    )
    if out['population'].isna().any():
        n = int(out['population'].isna().sum())
        msg = (f"{n} leaf (location, year) rows have no population to weight by; "
               "a weighted mean is undefined without them")
        raise ValueError(msg)

    out['_ancestor'] = [
        [int(x) for x in str(p).split(',')]
        for p in paths.loc[out['location_id']].to_numpy()
    ]
    out = out.explode('_ancestor')

    # A NaN leaf value must poison its ancestors, not vanish. pandas' groupby-sum
    # skips NaN, which would silently reweight the surviving children onto the full
    # population and hide the gap -- so count the NaNs per group and re-impose them.
    nan_flags = {c: f'_nan_{c}' for c in cols}
    for c, flag in nan_flags.items():
        out[flag] = out[c].isna().astype('int64')
        out[c] = out[c] * out['population']

    group_keys = ['_ancestor', 'year_id', *(extra_group_cols or [])]
    agg = (
        out.groupby(group_keys, sort=False)[[*cols, 'population', *nan_flags.values()]]
        .sum()
        .reset_index()
    )
    for c, flag in nan_flags.items():
        agg[c] = np.where(agg[flag] > 0, np.nan, agg[c] / agg['population'])
    return (
        agg.drop(columns=['population', *nan_flags.values()])
        .rename(columns={'_ancestor': 'location_id'})
        .astype({'location_id': 'int64'})
    )


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------

def aggregate_to_parent(
    df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    count_variable: str,
    preserve_age_sex: bool = False,
    extra_group_cols: list[str] | None = None,
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
    extra_group_cols:
        Additional columns to keep in the grouping (e.g. ['draw']). Appended to
        the group keys so a draw (or any other) dimension is preserved through
        the parent-sum instead of being collapsed. Sums stay within each distinct
        value of the extra columns.

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
    group_keys = group_keys + list(extra_group_cols or [])

    agg_df = (
        df.groupby(group_keys)[count_variable]
        .sum()
        .reset_index()
        .rename(columns={'parent_id': 'location_id'})
    )
    return agg_df


def roll_up_hierarchy(
    df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    count_variable: str,
    *,
    preserve_age_sex: bool = False,
    extra_group_cols: list[str] | None = None,
    start_level: int = 5,
) -> pd.DataFrame:
    """Roll counts up from ``start_level`` to level 0, returning every level stacked.

    Iteratively calls :func:`aggregate_to_parent`, so each higher level is the sum
    of its children. Draw- (and any-extra-dim-) aware via ``extra_group_cols``; the
    age/sex grid is preserved with ``preserve_age_sex=True``. The output contains the
    original ``start_level`` rows plus one row per ancestor location at every level
    above it — the count-space aggregation the finalize cores build on.

    Only the ``start_level`` rows present in ``df`` contribute; a parent with no
    children in ``df`` simply does not appear (callers that need the *complete*
    hierarchy zero-fill the result against the full location set afterward — a
    non-endemic location is a true zero, not a gap).

    Parameters
    ----------
    df:
        Counts at ``start_level`` (plus any extra/age-sex dims). A 'level' column
        is not required; it is looked up from ``hierarchy_df`` if absent.
    hierarchy_df:
        Full hierarchy DataFrame (needs 'location_id', 'parent_id', 'level').
    count_variable:
        Name of the count column to sum.
    preserve_age_sex / extra_group_cols:
        Passed through to :func:`aggregate_to_parent` at every level.
    start_level:
        Level of the input rows (default 5, LSAE admin-2).
    """
    extra = list(extra_group_cols or [])
    if 'level' not in df.columns:
        df = df.merge(hierarchy_df[['location_id', 'level']], on='location_id', how='left')
    current = df[df['level'] == start_level].drop(columns=['level'])

    levels = [current]
    child = current
    for _ in range(start_level - 1, -1, -1):
        child = aggregate_to_parent(
            child, hierarchy_df, count_variable,
            preserve_age_sex=preserve_age_sex, extra_group_cols=extra,
        )
        levels.append(child)

    return pd.concat(levels, ignore_index=True)


def roll_up_to_ancestors(
    df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    count_variables: str | list[str],
    *,
    extra_group_cols: list[str] | None = None,
    preserve_age_sex: bool = False,
) -> pd.DataFrame:
    """Sum leaf counts into every ancestor, via ``path_to_top_parent``.

    Use this instead of :func:`roll_up_hierarchy` whenever the input locations do
    not all sit at one level. The FHS most-detailed set is the motivating case:
    its 473 leaves span level 3 (193 countries with no subnational) and level 4
    (280 subnationals). Iterating parent-wards from the finest level silently
    drops the level-3 leaves, because they were never in the level-4 input.

    Each leaf's count is added to every location on its ``path_to_top_parent``,
    which includes the leaf itself, so the result carries the input rows and one
    row per ancestor at every level above them.

    Counts only. An aggregate *rate* is this count divided by that level's own
    population — see :func:`make_rate_from_count`. Never sum a population up the
    hierarchy to build a denominator: a location the forecast dropped contributes
    nothing to its parent's count while still belonging to its parent's
    population, and summing only the contributing children inflates the rate.

    Parameters
    ----------
    df:
        Counts at the leaf locations, keyed by 'location_id' and 'year_id' (plus
        age/sex and any extra dims). Leaves may sit at mixed levels.
    hierarchy_df:
        Needs 'location_id' and 'path_to_top_parent' (comma-separated ancestor
        ids, root first, self last).
    count_variables:
        Count column, or list of count columns, to sum.
    extra_group_cols:
        Additional dimensions to preserve through the sum — pass ``['draw']`` to
        aggregate within each draw. Aggregating per draw and collapsing to
        mean/lower/upper afterwards is required: summing children's quantiles
        would assume perfect cross-location correlation.
    preserve_age_sex:
        Keep the age/sex grid rather than collapsing it.

    Returns
    -------
    One row per (ancestor location, year, [age, sex], [extra dims]) with the
    summed counts.
    """
    if isinstance(count_variables, str):
        count_variables = [count_variables]
    if 'path_to_top_parent' not in hierarchy_df.columns:
        msg = "hierarchy_df must carry 'path_to_top_parent' to roll up to ancestors"
        raise KeyError(msg)

    paths = (
        hierarchy_df.drop_duplicates('location_id')
        .set_index('location_id')['path_to_top_parent']
    )
    missing = set(df['location_id'].unique()) - set(paths.index)
    if missing:
        msg = (f"{len(missing)} location_id(s) absent from the hierarchy, "
               f"e.g. {sorted(missing)[:5]}")
        raise KeyError(msg)

    out = df.copy()
    out['_ancestor'] = [
        [int(x) for x in str(p).split(',')]
        for p in paths.loc[out['location_id']].to_numpy()
    ]
    out = out.explode('_ancestor')

    group_keys = ['_ancestor', 'year_id']
    if preserve_age_sex:
        group_keys += ['age_group_id', 'sex_id']
    group_keys += list(extra_group_cols or [])

    return (
        out.groupby(group_keys, sort=False)[count_variables]
        .sum()
        .reset_index()
        .rename(columns={'_ancestor': 'location_id'})
        .astype({'location_id': 'int64'})
    )
