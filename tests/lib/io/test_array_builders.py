"""Tests for the draw-subset read in wide_to_array / read_draw_climate.

The default must stay all 100 draws; a caller that wants only draw 000 must get
a (n_loc, n_year, 1) array without the reader materialising the other 99, which
is what makes a whole-hierarchy past-covariate read affordable.
"""
import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.io.array_builders import DRAWS, wide_to_array


@pytest.fixture
def wide_parquet(tmp_path):
    """Wide climate parquet with location_id/year_id as the MultiIndex."""
    locs, years = [10, 20], [2000, 2001]
    idx = pd.MultiIndex.from_product([locs, years], names=["location_id", "year_id"])
    # Cell value encodes (location, year, draw) so a mis-slice is detectable.
    # The three components are spaced far enough apart to stay distinguishable
    # in float32 -- draws 1e-6 apart at magnitude 10 are not.
    data = {
        d: [i * 1_000_000 + loc * 1000 + (year - 2000) for loc, year in idx]
        for i, d in enumerate(DRAWS)
    }
    path = tmp_path / "wide.parquet"
    pd.DataFrame(data, index=idx).to_parquet(path)
    return path, locs, years


def test_default_reads_all_draws(wide_parquet):
    path, locs, years = wide_parquet
    arr = wide_to_array(str(path), locs, years)
    assert arr.shape == (len(locs), len(years), len(DRAWS))


def test_single_draw_subset_shape(wide_parquet):
    path, locs, years = wide_parquet
    arr = wide_to_array(str(path), locs, years, draws=["000"])
    assert arr.shape == (len(locs), len(years), 1)


def test_single_draw_matches_the_same_slice_of_the_full_read(wide_parquet):
    """The narrowed read must be the same numbers, not merely the same shape."""
    path, locs, years = wide_parquet
    full = wide_to_array(str(path), locs, years)
    one = wide_to_array(str(path), locs, years, draws=["000"])
    np.testing.assert_allclose(one[:, :, 0], full[:, :, 0])


def test_draw_subset_picks_the_requested_column(wide_parquet):
    path, locs, years = wide_parquet
    full = wide_to_array(str(path), locs, years)
    third = wide_to_array(str(path), locs, years, draws=["002"])
    np.testing.assert_allclose(third[:, :, 0], full[:, :, 2])
    # and is NOT draw 000 -- guards against a silently ignored argument
    assert not np.allclose(third[:, :, 0], full[:, :, 0])


def test_missing_location_year_is_nan(wide_parquet):
    path, locs, years = wide_parquet
    arr = wide_to_array(str(path), [*locs, 999], years, draws=["000"])
    assert np.isnan(arr[-1]).all()
