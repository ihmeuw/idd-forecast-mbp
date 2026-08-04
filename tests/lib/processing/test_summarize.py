"""Tests for lib/processing/summarize.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.summarize import summarize_draws


@pytest.fixture
def two_group_draws() -> pd.DataFrame:
    """Two locations x one year x 100 draws, with hand-checkable values.

    Location 1 draws are 0..99, location 2 draws are 100..199.
    """
    return pd.DataFrame(
        {
            "location_id": np.repeat([1, 2], 100),
            "year_id": 2030,
            "draw": np.tile(np.arange(100), 2),
            "inc_count": np.concatenate([np.arange(100.0), np.arange(100.0) + 100.0]),
        }
    )


def test_mean_matches_hand_computed(two_group_draws):
    out = summarize_draws(two_group_draws, ["inc_count"], ["location_id", "year_id"])
    means = out.set_index("location_id")["inc_count_mean"]
    assert means[1] == pytest.approx(49.5)
    assert means[2] == pytest.approx(149.5)


def test_quantiles_match_pandas_directly(two_group_draws):
    out = summarize_draws(two_group_draws, ["inc_count"], ["location_id", "year_id"])
    row = out[out.location_id == 1].iloc[0]
    expected = two_group_draws.query("location_id == 1")["inc_count"]
    assert row["inc_count_lower"] == pytest.approx(expected.quantile(0.025))
    assert row["inc_count_upper"] == pytest.approx(expected.quantile(0.975))


def test_lower_le_mean_le_upper(two_group_draws):
    out = summarize_draws(two_group_draws, ["inc_count"], ["location_id", "year_id"])
    assert (out.inc_count_lower <= out.inc_count_mean).all()
    assert (out.inc_count_mean <= out.inc_count_upper).all()


def test_one_row_per_group(two_group_draws):
    out = summarize_draws(two_group_draws, ["inc_count"], ["location_id", "year_id"])
    assert len(out) == 2
    assert out.columns.tolist() == [
        "location_id",
        "year_id",
        "inc_count_mean",
        "inc_count_lower",
        "inc_count_upper",
    ]


def test_column_order_groups_by_value_col(two_group_draws):
    df = two_group_draws.assign(mort_count=two_group_draws.inc_count / 10.0)
    out = summarize_draws(df, ["inc_count", "mort_count"], ["location_id", "year_id"])
    assert out.columns.tolist()[2:] == [
        "inc_count_mean",
        "inc_count_lower",
        "inc_count_upper",
        "mort_count_mean",
        "mort_count_lower",
        "mort_count_upper",
    ]


def test_summarize_then_divide_equals_divide_then_summarize(two_group_draws):
    """The property the driver relies on to avoid building draw-level rate frames."""
    pop = pd.Series({1: 1000.0, 2: 4000.0}, name="population")

    after = summarize_draws(two_group_draws, ["inc_count"], ["location_id", "year_id"])
    after = after.set_index("location_id")
    rate_after = after["inc_count_mean"] / pop
    lower_after = after["inc_count_lower"] / pop

    before = two_group_draws.assign(
        inc_rate=lambda d: d.inc_count / d.location_id.map(pop)
    )
    rate_before = summarize_draws(
        before, ["inc_rate"], ["location_id", "year_id"]
    ).set_index("location_id")

    pd.testing.assert_series_equal(
        rate_after, rate_before["inc_rate_mean"], check_names=False
    )
    pd.testing.assert_series_equal(
        lower_after, rate_before["inc_rate_lower"], check_names=False
    )


def test_all_nan_group_stays_nan_not_zero():
    """A masked outcome is NaN for every draw; it must not summarise to zero."""
    df = pd.DataFrame(
        {
            "location_id": np.repeat([1, 2], 10),
            "year_id": 2030,
            "draw": np.tile(np.arange(10), 2),
            "mort_count": np.concatenate([np.full(10, np.nan), np.arange(10.0)]),
        }
    )
    out = summarize_draws(df, ["mort_count"], ["location_id", "year_id"]).set_index(
        "location_id"
    )
    assert np.isnan(out.loc[1, "mort_count_mean"])
    assert out.loc[2, "mort_count_mean"] == pytest.approx(4.5)


def test_rejects_draw_in_group_cols(two_group_draws):
    with pytest.raises(ValueError, match="must not be a group column"):
        summarize_draws(two_group_draws, ["inc_count"], ["location_id", "draw"])


def test_rejects_missing_draw_column(two_group_draws):
    with pytest.raises(KeyError, match="draw column"):
        summarize_draws(
            two_group_draws.drop(columns="draw"), ["inc_count"], ["location_id"]
        )


def test_rejects_missing_value_column(two_group_draws):
    with pytest.raises(KeyError, match="columns not in frame"):
        summarize_draws(two_group_draws, ["nope"], ["location_id"])


def test_rejects_empty_group_cols(two_group_draws):
    with pytest.raises(ValueError, match="non-empty"):
        summarize_draws(two_group_draws, ["inc_count"], [])


@pytest.mark.parametrize("bad", [(0.9, 0.1), (-0.1, 0.5), (0.5, 1.5), (0.5, 0.5)])
def test_rejects_bad_quantiles(two_group_draws, bad):
    with pytest.raises(ValueError, match="quantiles"):
        summarize_draws(
            two_group_draws, ["inc_count"], ["location_id"], quantiles=bad
        )


def test_custom_quantiles_widen_interval(two_group_draws):
    narrow = summarize_draws(
        two_group_draws, ["inc_count"], ["location_id"], quantiles=(0.25, 0.75)
    )
    wide = summarize_draws(
        two_group_draws, ["inc_count"], ["location_id"], quantiles=(0.025, 0.975)
    )
    assert (wide.inc_count_upper >= narrow.inc_count_upper).all()
    assert (wide.inc_count_lower <= narrow.inc_count_lower).all()
