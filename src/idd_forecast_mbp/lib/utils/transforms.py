"""
Mathematical transform utilities for the idd-forecast-mbp pipeline.

Small helpers used inside lib/processing/ functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def logit(
    p: np.ndarray | pd.Series,
    clip_upper: float = 0.99,
) -> np.ndarray | pd.Series:
    """Logit transform: log(p / (1 - p)).

    Clips values at clip_upper before transforming to prevent log(inf).

    Parameters
    ----------
    p:
        Probability values in (0, 1). Values above clip_upper are clipped.
    clip_upper:
        Upper bound applied before transform. Default 0.99.

    # Extracted from: inline in 04_forecasting/rake_dengue.py:166
    """
    p_clipped = np.clip(p, a_min=None, a_max=clip_upper)
    return np.log(p_clipped / (1.0 - p_clipped))


def expit(x: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Inverse logit: 1 / (1 + exp(-x)). Maps real line to (0, 1).

    Used when converting logit-shifted predictions back to probability space.
    """
    return 1.0 / (1.0 + np.exp(-x))
