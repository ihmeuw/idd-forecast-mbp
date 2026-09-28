"""Synthetic selection summaries for the rank and gate tests.

A summary frame with random windowed metrics where one spec is ideal on every criterion, and
a spec subset that contains that spec's parsimony down-set, so the parsimonious pick has
somewhere simpler to go.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from idd_tools.model_selection import config_key, down_set

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idd_tools.model_selection.space import ModelUniverse

METRICS: dict[str, str] = {
    "oos_pfpr_r": "higher",
    "oos_pfpr_rmse": "lower",
    "oos_pfpr_mae": "lower",
}
ANCHOR_SPEC = (
    1585  # the fullest config in the 20260727_efs top tier; its down-set is large
)


def synthetic_summary(
    universe: ModelUniverse,  # kept for call-site symmetry with subset_with_downset
    spec_indices: list[int],
    best: int,
    windows: Sequence[str] = ("w1", "w2"),
    seed: int = 0,
) -> pd.DataFrame:
    """A summary frame with random windowed metrics where ``best`` is ideal on every criterion."""
    rng = np.random.default_rng(seed)
    n = len(spec_indices)
    frame = pd.DataFrame({"spec_index": spec_indices})
    for w in windows:
        for m, direction in METRICS.items():
            col = f"{w}__{m}"
            vals = (
                rng.uniform(0.5, 0.9, n)
                if direction == "higher"
                else rng.uniform(0.05, 0.2, n)
            )
            frame[col] = vals
            i = spec_indices.index(best)
            frame.loc[i, col] = (
                vals.max() + 0.05 if direction == "higher" else vals.min() - 0.01
            )
    return frame


def subset_with_downset(
    universe: ModelUniverse, anchor_spec: int, n_extra: int = 15
) -> list[int]:
    """``anchor_spec``, up to 20 of its down-set, and the first ``n_extra`` specs not already in."""
    cfg_of = {i + 1: c for i, c in enumerate(universe.configs)}
    key_to_spec = {config_key(c): i + 1 for i, c in enumerate(universe.configs)}
    ds = [
        key_to_spec[config_key(c)]
        for c in down_set(cfg_of[anchor_spec], universe.space, order="complexity")
    ]
    others = [s for s in range(1, n_extra + 1) if s not in ds and s != anchor_spec]
    return sorted({anchor_spec, *ds[:20], *others})
