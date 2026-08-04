"""Shared low-level style helpers for idd-forecast-mbp visualizations.

Cross-painter helpers (used by the line painter, a future scatter painter, etc.),
mirroring the palettes.py / style.py split in the figure repo: figure-family-
agnostic helpers live here; family-specific drawing lives in the family modules.
"""
from __future__ import annotations

from idd_forecast_mbp.number_functions import get_multiplier


def count_scale(max_value, *, scale=2, override=None):
    """Display multiplier + label suffix for a magnitude.

    e.g. count_scale(2.6e8) -> (1e-6, " (in Millions)"); count_scale(9e5) ->
    (1e-5, " (in 100,000s)"). Thin wrapper over number_functions.get_multiplier so
    every painter/layout that puts a raw count onto a readable axis uses ONE rule:
    multiply the plotted values by the returned scale and append the suffix to the
    y-label. Counts of different magnitude therefore get different multipliers
    (DALYs in Millions, deaths in 100,000s) — that's intended, per-panel behavior.
    """
    return get_multiplier(max_value, scale=scale, override_multiplier=override)
