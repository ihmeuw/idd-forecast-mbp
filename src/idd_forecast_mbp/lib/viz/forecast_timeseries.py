"""Forecast time-series figures — forecast-specific layouts (gold-standard convention).

Forecast specializations of the general line painter
(idd_forecast_mbp.lib.viz.lines.timeseries_panel): SSP palette + black observed
line + 2023 anchor, and count multipliers via idd_forecast_mbp.lib.viz.style
(counts get a 'in Millions' / 'in 100,000s' multiplier; rates stay per-100k).

  painter   forecast_timeseries_panel(ax, panel_df, *, ...) -> ax   (thin wrapper)
  layouts   plot_forecast_metric_grid       (one group, 2x2 measure x metric)
            plot_forecast_superregion_grid   (small multiples, shared-y)
            plot_forecast_ssp_2x2            (all SSPs + each SSP alone, shared x/y)
              each owns the Figure, computes consistency knobs (one shared count
              multiplier across sharey panels) once, and RETURNS a Figure; the
              caller owns IO.

Prepared-data contract (tidy DataFrame): panel_data columns
  [group, measure, metric, series, year_id, mid, lo, hi]
  series ∈ {observed, ssp126, ssp245, ssp585}; observed has lo/hi = NaN.
  A "panel slice" = rows for one fixed (group, measure, metric).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.viz.lines import timeseries_panel
from idd_forecast_mbp.lib.viz.style import count_scale

# ── forecast palette / series styling (painter falls back to this) ─────────────
SSP_ORDER = list(mbpc.ssp_scenarios.keys())
SSP_COLOR = {s: mbpc.ssp_scenarios[s]["color"] for s in SSP_ORDER}
OBSERVED_COLOR = "black"
OBSERVED_LABEL = "observed (GBD2023)"
ANCHOR_YEAR = 2023

FORECAST_COLORS = {**SSP_COLOR, "observed": OBSERVED_COLOR}
FORECAST_HUE_ORDER = [*SSP_ORDER, "observed"]
FORECAST_LABELS = {s: mbpc.ssp_scenarios[s]["name"] for s in SSP_ORDER}
FORECAST_LABELS["observed"] = OBSERVED_LABEL

METRIC_TITLE = {
    ("inc", "rate"): "Incidence rate (per 1,000)",
    ("mort", "rate"): "Mortality rate (per 100,000)",
    ("inc", "count"): "Incidence count",
    ("mort", "count"): "Mortality count",
}


# ─────────────────────────────── painter ──────────────────────────────────────
def forecast_timeseries_panel(ax, panel_df, *, value_scale=None, show_ci=True,
                              ylabel=None, xlabel="year"):
    """Forecast specialization of lines.timeseries_panel: SSP palette + black
    observed line + 2023 anchor. panel_df columns: [series, year_id, mid, lo, hi].
    value_scale is set by the layout ('auto' / shared (scale, suffix) tuple for
    counts; None for rates). Returns `ax`."""
    return timeseries_panel(
        ax, panel_df, x="year_id", value="mid", lo="lo", hi="hi",
        hue="series", hue_order=FORECAST_HUE_ORDER, colors=FORECAST_COLORS,
        labels=FORECAST_LABELS, value_scale=value_scale, ylabel=ylabel,
        xlabel=xlabel, anchor=ANCHOR_YEAR, show_ci=show_ci)


# ─────────────────────────── shared layout helpers ────────────────────────────
def _panel_slice(panel_data, group, measure, metric, series=None):
    d = panel_data[(panel_data["group"] == group)
                   & (panel_data["measure"] == measure)
                   & (panel_data["metric"] == metric)]
    if series is not None:
        d = d[d["series"].isin(series)]
    return d


def _ordered_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    return list(seen.values()), list(seen.keys())


def forecast_groups(panel_data):
    """Super-region groups (excl. Global) that carry a forecast series."""
    fc = set(panel_data.loc[panel_data["series"] != "observed", "group"])
    return [g for g in sorted(fc) if g != "Global"]


def _shared_count_scale(panel_data, groups, measure, metric):
    """One (scale, suffix) for COUNT panels that will share a y-axis; None for rates
    (rate units are fixed in the label, not multiplier-scaled)."""
    if metric != "count":
        return None
    d = panel_data[(panel_data["measure"] == measure) & (panel_data["metric"] == metric)
                   & (panel_data["group"].isin(groups))]
    vals = d[["mid", "hi"]].to_numpy(dtype="float64")
    m = float(np.nanmax(vals)) if vals.size else 0.0
    if not np.isfinite(m):
        m = 0.0
    return count_scale(m)


# ─────────────────────────────── layouts ──────────────────────────────────────
def plot_forecast_metric_grid(panel_data, *, group="Global", show_ci=False):
    """2x2 (measure x metric) for one group; returns a Figure. Panels are independent
    (not shared-y), so each count panel auto-scales its own multiplier."""
    combos = [("inc", "rate"), ("mort", "rate"), ("inc", "count"), ("mort", "count")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (measure, metric) in zip(axes.ravel(), combos):
        title = METRIC_TITLE[(measure, metric)]
        value_scale = "auto" if metric == "count" else None  # rates: fixed unit, no multiplier
        forecast_timeseries_panel(ax, _panel_slice(panel_data, group, measure, metric),
                                  show_ci=show_ci, value_scale=value_scale, ylabel=title)
        ax.set_title(f"{group} — {title}")
    handles, labels = _ordered_legend(axes.ravel()[0])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle(f"Malaria GBD2023 — {group}, all-age (observed + forecast)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_forecast_superregion_grid(panel_data, *, measure, metric, show_ci=False):
    """Small multiples over forecast super-regions for one (measure, metric), shared y
    — so one count multiplier is computed once and shared across panels; returns a Figure."""
    groups = forecast_groups(panel_data)
    ncol = 3
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow),
                             squeeze=False, sharey=True)
    title = METRIC_TITLE[(measure, metric)]
    value_scale = _shared_count_scale(panel_data, groups, measure, metric)
    suffix = value_scale[1] if isinstance(value_scale, tuple) else ""
    flat = axes.ravel()
    for i, group in enumerate(groups):
        forecast_timeseries_panel(flat[i], _panel_slice(panel_data, group, measure, metric),
                                  show_ci=show_ci, value_scale=value_scale,
                                  xlabel=None, ylabel=None)
        flat[i].set_title(group, fontsize=9)
    for j in range(len(groups), nrow * ncol):
        flat[j].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(title + suffix)
    handles, labels = _ordered_legend(flat[0])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle(f"Malaria GBD2023 — super-region all-age — {title}")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_forecast_ssp_2x2(panel_data, *, group, measure, metric):
    """2x2 for one (group, measure, metric): top-left = all SSPs, the other three =
    each SSP alone with its 95% band. Shared x & y (one count multiplier across all
    four panels); returns a Figure."""
    base = _panel_slice(panel_data, group, measure, metric)
    title = METRIC_TITLE[(measure, metric)]
    value_scale = _shared_count_scale(panel_data, [group], measure, metric)
    suffix = value_scale[1] if isinstance(value_scale, tuple) else ""
    panels = ([("__all__", "All SSPs", False)]
              + [(s, f"{mbpc.ssp_scenarios[s]['name']} ({s})", True) for s in SSP_ORDER])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (which, ptitle, show_ci) in zip(axes.ravel(), panels):
        d = base if which == "__all__" else base[base["series"].isin([which, "observed"])]
        forecast_timeseries_panel(ax, d, show_ci=show_ci, value_scale=value_scale,
                                  xlabel=None, ylabel=None)
        ax.set_title(ptitle, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(title + suffix)
    for ax in axes[1, :]:
        ax.set_xlabel("year")
    handles, labels = _ordered_legend(axes.ravel()[0])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle(f"{group} — {title}")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig
