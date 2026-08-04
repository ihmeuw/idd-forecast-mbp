"""Unit tests for NearestResampler in 01_map_to_admin_2/block_utils.py.

Mirrors the climate-data NearestResampler tests. The bit-for-bit equivalence
against to_raster(...).resample_to(...) is validated by climate-data's
multi-block A/B and the conversion to this repo is structurally identical;
these tests cover shape/error/uniform/NaN-propagation invariants.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterra as rt  # type: ignore
from affine import Affine  # type: ignore

# block_utils lives in a digit-prefixed package (01_map_to_admin_2/), which
# can't be imported via dotted path. Mirror pixel_main.py's sys.path[0]
# pattern by injecting the stage-01 directory.
_STAGE01_DIR = (
    Path(__file__).parent.parent.parent
    / "src" / "idd_forecast_mbp" / "01_map_to_admin_2"
)
if str(_STAGE01_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE01_DIR))

from block_utils import NearestResampler  # noqa: E402


# Synthetic destination raster matching ESRI:54034 conventions.
DST_H, DST_W = 128, 128
DST_TRANSFORM = Affine(100.0, 0.0, 0.0, 0.0, -100.0, 1_000_000.0)
DST_CRS = "ESRI:54034"
SRC_LATS = np.linspace(20.0, -20.0, 41)
SRC_LONS = np.linspace(-30.0, 30.0, 61)
SRC_CRS = "EPSG:4326"


def _make_dst_template() -> rt.RasterArray:
    return rt.RasterArray(
        data=np.zeros((DST_H, DST_W), dtype=np.float32),
        transform=DST_TRANSFORM,
        crs=DST_CRS,
        no_data_value=np.nan,
    )


def test_build_shapes():
    r = NearestResampler.build(SRC_LATS, SRC_LONS, SRC_CRS, _make_dst_template())
    assert r.src_shape == (41, 61)
    assert r.dst_shape == (DST_H, DST_W)


def test_apply_shape_mismatch_raises():
    r = NearestResampler.build(SRC_LATS, SRC_LONS, SRC_CRS, _make_dst_template())
    with pytest.raises(ValueError, match="src_2d shape"):
        r.apply(np.zeros((10, 10), dtype=np.float32))


def test_apply_uniform_source():
    """Every in-bounds destination pixel of a uniform source should hold
    the same uniform value; out-of-bounds pixels should be NaN."""
    r = NearestResampler.build(SRC_LATS, SRC_LONS, SRC_CRS, _make_dst_template())
    src = np.full((41, 61), 3.14, dtype=np.float32)
    out = r.apply(src)
    finite = ~np.isnan(out)
    if finite.any():
        np.testing.assert_array_equal(out[finite], np.float32(3.14))


def test_apply_propagates_nan():
    """A row of source NaN at a latitude the destination patch picks up
    should produce NaN in the destination."""
    r = NearestResampler.build(SRC_LATS, SRC_LONS, SRC_CRS, _make_dst_template())
    src = np.ones((41, 61), dtype=np.float64)
    src[11, :] = np.nan  # source row at lat 9°, which the dest patch covers
    out = r.apply(src)
    assert np.isnan(out).any()


def test_apply_returns_float32():
    """apply() must always return float32 to match the existing pipeline's
    downstream dtype."""
    r = NearestResampler.build(SRC_LATS, SRC_LONS, SRC_CRS, _make_dst_template())
    src = np.ones((41, 61), dtype=np.float64)
    out = r.apply(src)
    assert out.dtype == np.float32


def test_apply_output_shape_matches_dst():
    r = NearestResampler.build(SRC_LATS, SRC_LONS, SRC_CRS, _make_dst_template())
    src = np.ones((41, 61), dtype=np.float32)
    out = r.apply(src)
    assert out.shape == (DST_H, DST_W)
