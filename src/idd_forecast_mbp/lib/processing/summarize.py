"""Collapse a draw axis into a point estimate and an uncertainty interval.

Every finished forecast product ends with this step. Before this module the only
implementation lived inline in ``05_aggregation/quick_gbd2023_timeseries.py``, so
each new script re-derived it.

One property worth knowing, because it saves real work upstream: when a value is
divided by a *deterministic* quantity (a rate is a count divided by a population
that carries no draw dimension), summarising and then dividing is exactly equal to
dividing and then summarising. Mean and quantiles both commute with scaling by a
positive constant. So a driver can aggregate counts, summarise once, and derive
rate mean/lower/upper by division — it never needs to build a draw-level rate frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def summarize_draws(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    group_cols: Sequence[str],
    *,
    quantiles: tuple[float, float] = (0.025, 0.975),
    draw_col: str = "draw",
) -> pd.DataFrame:
    """Collapse ``draw_col`` to ``{col}_mean`` / ``{col}_lower`` / ``{col}_upper``.

    Parameters
    ----------
    df:
        Draw-level frame. Must contain ``draw_col``, every column in ``group_cols``
        and every column in ``value_cols``.
    value_cols:
        Columns to summarise.
    group_cols:
        Columns identifying one summarised row (e.g. ``["location_id", "year_id"]``).
        ``draw_col`` must not appear here.
    quantiles:
        ``(lower, upper)`` probabilities for the interval. Defaults to a 95% interval.
    draw_col:
        Name of the draw axis.

    Returns
    -------
    One row per ``group_cols`` combination, with three columns per value column.
    Column order is stable: all statistics for the first value column, then the
    second, and so on.

    Notes
    -----
    NaN is skipped, matching the forecast contract where a masked outcome is written
    as NaN for every draw of a location. A group whose values are entirely NaN
    summarises to NaN rather than to zero.
    """
    lower_q, upper_q = quantiles
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError(
            f"quantiles must satisfy 0 <= lower < upper <= 1; got {quantiles}"
        )

    group_cols = list(group_cols)
    value_cols = list(value_cols)

    if draw_col not in df.columns:
        raise KeyError(f"draw column {draw_col!r} not in frame; have {list(df.columns)}")
    if draw_col in group_cols:
        raise ValueError(f"draw column {draw_col!r} must not be a group column")
    if not group_cols:
        raise ValueError("group_cols must be non-empty")

    missing = [c for c in (*group_cols, *value_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in frame: {missing}")

    grouped = df.groupby(group_cols, observed=True, sort=True)[value_cols]
    stats = {
        "mean": grouped.mean(),
        "lower": grouped.quantile(lower_q),
        "upper": grouped.quantile(upper_q),
    }

    out = pd.concat(stats, axis=1)
    # concat keys land as the OUTER level, so a column is (stat, value_col).
    out.columns = [f"{value_col}_{stat}" for stat, value_col in out.columns]
    out = out[[f"{c}_{s}" for c in value_cols for s in ("mean", "lower", "upper")]]
    return out.reset_index()
