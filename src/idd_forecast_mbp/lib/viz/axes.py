"""Axes that end on a labelled tick.

An axis should not run past its last labelled number: either the data's end is a tick (years end at
2100) or the axis extends to the next tick (a count axis ending past 900 goes on to the next
labelled value). Matplotlib's default margins do neither. Bobby, 2026-10-08: this is the default,
and running past the last number is something a figure has to ask for. The start of an axis keeps a
small gap before its first labelled value by default (``start="gap"``); ``start="at"`` begins the
axis exactly on it (an x axis starting AT 2000). Intended for idd-figures as its default; kept here
until it lands there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import ticker

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

YEAR_RANGE = (1900.0, 2200.0)  # an x axis whose data sit in here is a year axis
START_GAP = 0.02  # fraction of the axis span left before the first labelled value with start="gap"
Start = Literal["gap", "at"]


def _tick_bounds(lo: float, hi: float) -> tuple[float, float, list[float]] | None:
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    locator = ticker.MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10])
    ticks = np.asarray(locator.tick_values(lo, hi), dtype=float)
    # MaxNLocator's ticks always bracket [lo, hi], so both selections are non-empty
    t_lo = ticks[ticks <= lo].max()
    t_hi = ticks[ticks >= hi].min()
    return (
        float(t_lo),
        float(t_hi),
        [float(t) for t in ticks[(ticks >= t_lo) & (ticks <= t_hi)]],
    )


def end_on_ticks(
    ax: Axes, *, x: bool = True, y: bool = True, start: Start = "gap"
) -> None:
    """Each requested axis ends on the labelled tick at or above the data and starts on (``"at"``) or
    just before (``"gap"``) the labelled tick at or below it."""
    for on, interval, set_lim, axis in (
        (x, ax.dataLim.intervalx, ax.set_xlim, ax.xaxis),
        (y, ax.dataLim.intervaly, ax.set_ylim, ax.yaxis),
    ):
        if not on:
            continue
        bounds = _tick_bounds(float(interval[0]), float(interval[1]))
        if bounds is None:
            continue
        t_lo, t_hi, ticks = bounds
        pad = START_GAP * (t_hi - t_lo) if start == "gap" else 0.0
        set_lim(t_lo - pad, t_hi)
        axis.set_major_locator(ticker.FixedLocator(ticks))


def end_figure_axes_on_ticks(fig: Figure, *, start: Start = "gap") -> None:
    """Every data axes of ``fig``: y always; x only for a year axis (categorical x is left alone)."""
    for ax in fig.axes:
        lo, hi = ax.dataLim.intervalx
        is_year_axis = bool(
            np.isfinite([lo, hi]).all() and lo >= YEAR_RANGE[0] and hi <= YEAR_RANGE[1]
        )
        end_on_ticks(ax, x=is_year_axis, y=True, start=start)
