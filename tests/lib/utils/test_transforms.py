"""
Tests for lib/utils/transforms.py
"""

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.utils.transforms import expit, logit


# ---------------------------------------------------------------------------
# logit
# ---------------------------------------------------------------------------

def test_logit_basic():
    result = logit(np.array([0.5]))
    assert np.isclose(result[0], 0.0)


def test_logit_clips_upper():
    # Value above clip_upper should be clipped to clip_upper before transform
    result_clipped = logit(np.array([1.0]), clip_upper=0.99)
    result_at_clip = logit(np.array([0.99]), clip_upper=0.99)
    assert np.isclose(result_clipped[0], result_at_clip[0])


def test_logit_custom_clip():
    result = logit(np.array([0.95]), clip_upper=0.9)
    expected = np.log(0.9 / 0.1)
    assert np.isclose(result[0], expected)


def test_logit_monotone():
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    out = logit(p)
    assert np.all(np.diff(out) > 0)


def test_logit_pandas_series():
    s = pd.Series([0.1, 0.5, 0.9])
    result = logit(s)
    assert len(result) == 3


def test_logit_no_clip_below():
    # No lower clip — very small positive values should give large negative logit
    result = logit(np.array([0.001]), clip_upper=0.99)
    assert result[0] < -5


# ---------------------------------------------------------------------------
# expit
# ---------------------------------------------------------------------------

def test_expit_zero():
    assert np.isclose(expit(np.array([0.0]))[0], 0.5)


def test_expit_large_positive():
    # expit(large) → ~1
    assert expit(np.array([100.0]))[0] > 0.999


def test_expit_large_negative():
    # expit(large negative) → ~0
    assert expit(np.array([-100.0]))[0] < 0.001


def test_expit_pandas_series():
    s = pd.Series([0.0, 1.0, -1.0])
    result = expit(s)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Round-trip: expit(logit(p)) ≈ p (for values below clip_upper)
# ---------------------------------------------------------------------------

def test_logit_expit_roundtrip():
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    # Use clip_upper=1.0 to avoid clipping these values
    recovered = expit(logit(p, clip_upper=1.0))
    assert np.allclose(recovered, p, atol=1e-10)
