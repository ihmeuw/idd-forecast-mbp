"""
Tests for lib/processing/fhs_format.py

Uses small synthetic xarray Datasets — no real pipeline data.
"""

import numpy as np
import pytest
import xarray as xr

from idd_forecast_mbp.lib.processing.fhs_format import aggregate_locations, counts_to_rates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_count_ds():
    """Count dataset with 4 locations, 2 years, 1 draw.
    Shape: (draw_id=1, location_id=4, year_id=2)
    """
    data = np.array([[[10., 11.],   # loc 100
                      [20., 21.],   # loc 200
                      [30., 31.],   # loc 300
                      [40., 41.]]]) # loc 400
    return xr.Dataset(
        {"val": (["draw_id", "location_id", "year_id"], data)},
        coords={"draw_id": [0], "location_id": [100, 200, 300, 400], "year_id": [2030, 2031]},
    )


@pytest.fixture
def small_pop_ds():
    """Population dataset matching small_count_ds (no draw_id dim)."""
    return xr.Dataset(
        {"population": (["location_id", "year_id"], np.array([[100., 200.],
                                                               [200., 400.],
                                                               [300., 600.],
                                                               [400., 800.]]))},
        coords={"location_id": [100, 200, 300, 400], "year_id": [2030, 2031]},
    )


# ---------------------------------------------------------------------------
# aggregate_locations
# ---------------------------------------------------------------------------

def test_aggregate_locations_sums_counts(small_count_ds, small_pop_ds):
    """Source locations are summed into the target location."""
    count_out, _ = aggregate_locations(small_count_ds, small_pop_ds,
                                       source_locs=[100, 200], target_loc=999)
    # target location val for draw=0, year=2030 should be 10+20=30
    assert float(count_out["val"].sel(location_id=999, draw_id=0, year_id=2030)) == pytest.approx(30.0)


def test_aggregate_locations_sums_population(small_count_ds, small_pop_ds):
    """Population is also summed correctly."""
    _, pop_out = aggregate_locations(small_count_ds, small_pop_ds,
                                     source_locs=[100, 200], target_loc=999)
    assert float(pop_out["population"].sel(location_id=999, year_id=2030)) == pytest.approx(300.0)


def test_aggregate_locations_removes_sources(small_count_ds, small_pop_ds):
    """Source locations are removed from the output."""
    count_out, pop_out = aggregate_locations(small_count_ds, small_pop_ds,
                                              source_locs=[100, 200], target_loc=999)
    assert 100 not in count_out.location_id.values
    assert 200 not in count_out.location_id.values
    assert 100 not in pop_out.location_id.values


def test_aggregate_locations_keeps_others(small_count_ds, small_pop_ds):
    """Non-source locations are untouched."""
    count_out, _ = aggregate_locations(small_count_ds, small_pop_ds,
                                        source_locs=[100, 200], target_loc=999)
    assert 300 in count_out.location_id.values
    assert 400 in count_out.location_id.values
    assert float(count_out["val"].sel(location_id=300, draw_id=0, year_id=2030)) == pytest.approx(30.0)


def test_aggregate_locations_no_sources_present(small_count_ds, small_pop_ds):
    """If none of the source locations are present, datasets are returned unchanged."""
    count_out, pop_out = aggregate_locations(small_count_ds, small_pop_ds,
                                              source_locs=[999, 888], target_loc=777)
    assert set(count_out.location_id.values) == {100, 200, 300, 400}
    assert set(pop_out.location_id.values) == {100, 200, 300, 400}


def test_aggregate_locations_partial_present(small_count_ds, small_pop_ds):
    """Source locations not in the dataset are silently skipped."""
    count_out, _ = aggregate_locations(small_count_ds, small_pop_ds,
                                        source_locs=[100, 999], target_loc=777)
    # Only 100 was present, so target should equal 100's value
    assert float(count_out["val"].sel(location_id=777, draw_id=0, year_id=2030)) == pytest.approx(10.0)
    assert 100 not in count_out.location_id.values


def test_aggregate_locations_ethiopian_pattern(small_count_ds, small_pop_ds):
    """Full Ethiopian pattern: 3 sub-nationals → 1 national."""
    # Rename to match Ethiopian location IDs for clarity
    count_ds = small_count_ds.assign_coords(location_id=[60908, 95069, 94364, 500])
    pop_ds = small_pop_ds.assign_coords(location_id=[60908, 95069, 94364, 500])

    count_out, pop_out = aggregate_locations(
        count_ds, pop_ds,
        source_locs=[60908, 95069, 94364],
        target_loc=44858,
    )
    assert 44858 in count_out.location_id.values
    assert 500 in count_out.location_id.values
    assert 60908 not in count_out.location_id.values
    # Sum of first 3 locations for draw=0, year=2030: 10+20+30 = 60
    assert float(count_out["val"].sel(location_id=44858, draw_id=0, year_id=2030)) == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# counts_to_rates
# ---------------------------------------------------------------------------

@pytest.fixture
def count_da():
    return xr.DataArray([10.0, 20.0, 0.0, 5.0], dims=["location_id"],
                        coords={"location_id": [1, 2, 3, 4]})


@pytest.fixture
def pop_da():
    return xr.DataArray([100.0, 200.0, 0.0, 50.0], dims=["location_id"],
                        coords={"location_id": [1, 2, 3, 4]})


def test_counts_to_rates_basic(count_da, pop_da):
    """Standard division: count / population."""
    result = counts_to_rates(count_da, pop_da)
    assert float(result.sel(location_id=1)) == pytest.approx(0.1)
    assert float(result.sel(location_id=2)) == pytest.approx(0.1)


def test_counts_to_rates_zero_population(count_da, pop_da):
    """Zero population produces 0.0, not NaN or inf."""
    result = counts_to_rates(count_da, pop_da)
    assert float(result.sel(location_id=3)) == pytest.approx(0.0)
    assert not np.isnan(float(result.sel(location_id=3)))


def test_counts_to_rates_zero_count(count_da, pop_da):
    """Zero count with nonzero population produces 0.0."""
    result = counts_to_rates(count_da, pop_da)
    assert float(result.sel(location_id=3)) == pytest.approx(0.0)


def test_counts_to_rates_broadcasts_over_draws():
    """Population (no draw dim) broadcasts correctly over counts (with draw dim)."""
    count = xr.DataArray(
        [[10.0, 20.0], [30.0, 40.0]],
        dims=["draw_id", "location_id"],
        coords={"draw_id": [0, 1], "location_id": [1, 2]},
    )
    pop = xr.DataArray(
        [100.0, 200.0],
        dims=["location_id"],
        coords={"location_id": [1, 2]},
    )
    result = counts_to_rates(count, pop)
    assert result.dims == ("draw_id", "location_id")
    assert float(result.sel(draw_id=0, location_id=1)) == pytest.approx(0.1)
    assert float(result.sel(draw_id=1, location_id=2)) == pytest.approx(0.2)


def test_counts_to_rates_no_nan_in_output(count_da, pop_da):
    """Output should never contain NaN."""
    result = counts_to_rates(count_da, pop_da)
    assert not bool(np.any(np.isnan(result.values)))
