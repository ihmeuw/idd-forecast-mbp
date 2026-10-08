"""Unit tests for 07_vetting/held_input_counts.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).parents[2] / "src/idd_forecast_mbp/07_vetting/held_input_counts.py"
)
_spec = importlib.util.spec_from_file_location("held_input_counts", SCRIPT_PATH)
hic = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hic
_spec.loader.exec_module(hic)


def _parts():
    return {
        "dah": np.array([[0.1]]),
        "gdp_pfpr": np.array([[-0.5]]),
        "log_gdp": np.array([[0.3]]),
        "suitability": np.array([[0.2, 0.4]]),
        "wind": np.array([[0.05, 0.0]]),
    }


def test_holding_gdp_zeroes_both_gdp_pieces_only():
    out = hic.held_parts(_parts(), "gdp")
    assert out["gdp_pfpr"].item() == 0.0
    assert out["log_gdp"].item() == 0.0
    np.testing.assert_array_equal(out["suitability"], [[0.2, 0.4]])
    assert out["dah"].item() == 0.1


def test_holding_suitability_keeps_the_draw_shape():
    out = hic.held_parts(_parts(), "suitability")
    np.testing.assert_array_equal(out["suitability"], [[0.0, 0.0]])
    assert out["gdp_pfpr"].item() == -0.5


def test_holding_wind_zeroes_wind_only():
    out = hic.held_parts(_parts(), "wind")
    np.testing.assert_array_equal(out["wind"], [[0.0, 0.0]])
    np.testing.assert_array_equal(out["suitability"], [[0.2, 0.4]])
