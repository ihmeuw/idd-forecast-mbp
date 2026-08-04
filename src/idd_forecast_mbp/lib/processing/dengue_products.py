"""Turn anchored dengue rates into hierarchy products.

The aggregation rule, which is the whole point of this module:

1. Rates become **counts** at the leaves, using each leaf's own population.
2. Counts roll up in **count space** — a parent is the sum of its children.
3. An aggregate **rate is that level's count divided by that level's own
   population**, read from the population artifact. Population is never summed
   from children.

Rule 3 is the one that bites. Summing only the modelled children's population
gives a partial denominator that inflates the rate — that is the bug behind the
2023 splice jump (``.claude/DECISIONS.md`` 2026-07-08). A location the forecast
dropped contributes nothing to its parent's count while still belonging to its
parent's population, so it is a true zero rather than a gap.

Where draws are present, aggregation happens **within** each draw and the
collapse to mean/lower/upper happens last. Summing children's quantiles would
assume perfect cross-location correlation and badly overstate the interval.
Because population carries no draw dimension, summarising counts and then
dividing is exactly equal to dividing and then summarising, so rate summaries
are derived from the summarised counts rather than recomputed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from idd_forecast_mbp.lib.processing.aggregation import roll_up_to_ancestors
from idd_forecast_mbp.lib.processing.summarize import summarize_draws

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

#: Outcome -> (rate column, count column) as they appear in the products.
MEASURES = {
    "inc": ("dengue_inc_rate", "dengue_inc_count"),
    "mort": ("dengue_mort_rate", "dengue_mort_count"),
}


def rates_to_counts(
    rates: pd.DataFrame,
    population: pd.DataFrame,
    measures: Sequence[str] = tuple(MEASURES),
) -> pd.DataFrame:
    """Multiply leaf rates by their own population to get counts.

    Rows whose location has no population are dropped: a rate with no
    denominator cannot become a count, and carrying it as zero would understate
    every ancestor.
    """
    keys = ["location_id", "year_id"]
    out = rates.merge(
        population[[*keys, "population"]].drop_duplicates(keys), on=keys, how="left",
    )
    out = out[out["population"].notna()]
    for measure in measures:
        rate_col, count_col = MEASURES[measure]
        out[count_col] = out[rate_col] * out["population"]
    return out


def age_sex_rates_to_all_age_counts(
    rates: pd.DataFrame,
    age_sex_population: pd.DataFrame,
    measures: Sequence[str] = tuple(MEASURES),
) -> pd.DataFrame:
    """Age/sex rates in, all-age leaf counts out.

    Anchoring per ``(location, age, sex)`` yields one rate per age/sex cell, and
    each cell must be multiplied by **its own** population before the cells are
    summed. Merging the all-age population onto every cell and summing instead
    multiplies the leaf count by roughly the number of cells — ~50x here — which
    is what made a first real run score a slope of 0.007 against observed.

    Cells with no population are dropped rather than zero-filled; a rate with no
    denominator cannot become a count.
    """
    keys = ["location_id", "year_id", "age_group_id", "sex_id"]
    out = rates.merge(
        age_sex_population[[*keys, "population"]].drop_duplicates(keys),
        on=keys, how="left",
    )
    out = out[out["population"].notna()]
    count_columns = []
    for measure in measures:
        rate_col, count_col = MEASURES[measure]
        out[count_col] = out[rate_col] * out["population"]
        count_columns.append(count_col)
    return (
        out.groupby(["location_id", "year_id"], as_index=False)[count_columns].sum()
    )


def aggregate_counts(
    counts: pd.DataFrame,
    hierarchy: pd.DataFrame,
    measures: Sequence[str] = tuple(MEASURES),
    *,
    draw_column: str | None = None,
) -> pd.DataFrame:
    """Roll leaf counts up to every ancestor, within each draw when present."""
    count_columns = [MEASURES[m][1] for m in measures]
    return roll_up_to_ancestors(
        counts, hierarchy, count_columns,
        extra_group_cols=[draw_column] if draw_column else None,
    )


def counts_to_rates(
    counts: pd.DataFrame,
    population: pd.DataFrame,
    measures: Sequence[str] = tuple(MEASURES),
) -> pd.DataFrame:
    """Divide each level's count by that level's own population.

    The population frame must carry a row for every location in ``counts``,
    including the aggregate levels. A missing denominator yields NaN rather than
    a silently wrong rate.
    """
    keys = ["location_id", "year_id"]
    out = counts.merge(
        population[[*keys, "population"]].drop_duplicates(keys), on=keys, how="left",
    )
    for measure in measures:
        rate_col, count_col = MEASURES[measure]
        out[rate_col] = out[count_col] / out["population"]
    return out


def summarize_products(
    rates: pd.DataFrame,
    measures: Sequence[str] = tuple(MEASURES),
    *,
    draw_column: str = "draw",
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> pd.DataFrame:
    """Collapse the draw axis to mean / lower / upper, per location and year."""
    keys = ["location_id", "year_id"]
    columns = [c for m in measures for c in MEASURES[m]]
    return summarize_draws(
        rates, value_cols=columns, group_cols=keys,
        draw_col=draw_column, quantiles=quantiles,
    )


def build_hierarchy_products(
    rates: pd.DataFrame,
    population: pd.DataFrame,
    hierarchy: pd.DataFrame,
    measures: Sequence[str] = tuple(MEASURES),
    *,
    draw_column: str | None = None,
    age_sex_population: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Leaf rates in, per-level rates and counts out.

    Pass ``age_sex_population`` when ``rates`` carries an age/sex grid — which it
    does whenever the anchor is per age/sex cell. Each cell is then costed at its
    own population and the cells are summed to an all-age leaf count before the
    roll-up. Omitting it silently multiplies every count by the number of cells.

    With ``draw_column`` set, the result carries mean / lower / upper per
    location-year; without it — the in-sample case, where the past covariates
    are single-realization — the result is a point estimate.
    """
    if age_sex_population is not None:
        counts = age_sex_rates_to_all_age_counts(rates, age_sex_population, measures)
    else:
        counts = rates_to_counts(rates, population, measures)
    aggregated = aggregate_counts(
        counts, hierarchy, measures, draw_column=draw_column,
    )
    with_rates = counts_to_rates(aggregated, population, measures)
    if draw_column is None:
        return with_rates
    return summarize_products(with_rates, measures, draw_column=draw_column)
