"""General time-series line painter + standalone layout.

Mark = lines over an ordered x (one line per series, optional [lo, hi] band). This
is the reusable atom for any "time-series of <something>" figure; a scatter painter
would be a sibling module reusing the same style helpers (idd_forecast_mbp.lib.viz.style).

  painter     timeseries_panel(ax, df, *, ...) -> ax
  standalone  plot_timeseries(df, *, ax=None, **opts) -> ax

Data contract: a tidy long DataFrame with an ordered x column, a value column, an
optional (lo, hi) band, and a `hue` column naming each series. Coupling is by
column name (each call site documents the columns it passes).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from idd_forecast_mbp.lib.viz.style import count_scale


def _resolve_scale(value_scale, values):
    """Return (scale, label_suffix).

    value_scale: 'auto' (compute via style.count_scale on this panel's max),
    None / 1 (no scaling), a bare float, or a precomputed (scale, suffix) tuple
    (e.g. a multiplier a layout computed once and shares across sharey panels).
    """
    if value_scale is None or value_scale == 1 or value_scale == 1.0:
        return 1.0, ""
    if isinstance(value_scale, tuple):
        return value_scale
    if value_scale == "auto":
        arr = np.asarray(values, dtype="float64")
        m = float(np.nanmax(arr)) if arr.size else 0.0
        if not np.isfinite(m):
            m = 0.0
        return count_scale(m)
    return float(value_scale), ""


def timeseries_panel(ax, df, *, x="year_id", value="mid", lo=None, hi=None,
                     hue="series", hue_order=None, colors=None, labels=None,
                     value_scale="auto", ylabel=None, xlabel=None,
                     anchor=None, show_ci=True):
    """Draw one time-series panel onto `ax` and return `ax`.

    df          : tidy rows for ONE panel.
    x, value    : column names for the ordered x and the central line.
    lo, hi      : optional column names for a [lo, hi] band (drawn when show_ci).
    hue         : column naming each series (one line per value).
    hue_order   : explicit draw order (else sorted unique).
    colors      : {hue_value: color};  labels: {hue_value: legend label}.
    value_scale : 'auto' (multiplier via style.count_scale, appended to ylabel),
                  None/1 (off), a float, or a (scale, suffix) tuple shared by a layout.

    Never makes a figure, never saves; sets only its own axis labels.
    """
    colors = colors or {}
    labels = labels or {}
    scale, suffix = _resolve_scale(value_scale, df[value])
    order = hue_order if hue_order is not None else sorted(df[hue].dropna().unique())
    for s in order:
        g = df[df[hue] == s].sort_values(x)
        if g.empty:
            continue
        c = colors.get(s)
        if show_ci and lo and hi and g[lo].notna().any():
            ax.fill_between(g[x], g[lo] * scale, g[hi] * scale, color=c, alpha=0.22, lw=0)
        ax.plot(g[x], g[value] * scale, color=c, lw=1.7, label=labels.get(s, s))
    if anchor is not None:
        ax.axvline(anchor, color="grey", ls=":", lw=0.8)
    ax.margins(x=0.01)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel + suffix)
    return ax


def plot_timeseries(df, *, ax=None, figsize=(8, 5), title=None, **opts):
    """Standalone one-painter layout: plot the time-series of `df` on its own axes.

    Returns the Axes (so it also works as a painter when an `ax` is passed in). The
    caller owns IO — no savefig/show here.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    timeseries_panel(ax, df, **opts)
    if title is not None:
        ax.set_title(title)
    return ax
