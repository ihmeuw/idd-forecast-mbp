"""Tests for the shared aggregation weights.

These pin the behaviour that produced a real analytical error: a covariate change measured
population-weighted was 14x smaller than the burden-weighted truth, because population
weighting spreads mass over the ~75% of admin-2 units with no malaria
(``.claude/GDP_LEVERAGE_INVESTIGATION.md``). The distinguishing properties below --
zero-burden locations carry exactly zero weight, and ``total`` is not a rescaled ``mean`` --
are the ones that make the two schemes answer different questions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.weights import (
    WEIGHT_SCHEMES,
    weighted_rollup_to_levels,
)


@pytest.fixture
def hierarchy() -> pd.DataFrame:
    """Four admin-2 units in two super-regions."""
    return pd.DataFrame(
        {
            "location_id": [101, 102, 201, 202],
            "super_region_id": [10, 10, 20, 20],
        }
    )


@pytest.fixture
def values() -> pd.DataFrame:
    """One year, a per-capita covariate that differs across locations."""
    return pd.DataFrame(
        {
            "location_id": [101, 102, 201, 202],
            "year_id": [2023] * 4,
            "value": [1.0, 3.0, 10.0, 20.0],
        }
    )


def _weights(pairs: list[tuple[int, float]], year: int = 2023) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": [p[0] for p in pairs],
            "year_id": [year] * len(pairs),
            "weight": [p[1] for p in pairs],
        }
    )


def test_scheme_registry_documents_every_scheme():
    assert set(WEIGHT_SCHEMES) == {"population", "mort2023", "inc2023"}
    assert all(isinstance(v, str) and v for v in WEIGHT_SCHEMES.values())


def test_weighted_mean_of_constant_field_returns_that_constant(hierarchy):
    """A weighted mean must be weight-invariant when the field is flat."""
    flat = pd.DataFrame(
        {
            "location_id": [101, 102, 201, 202],
            "year_id": [2023] * 4,
            "value": [7.5] * 4,
        }
    )
    w = _weights([(101, 1.0), (102, 999.0), (201, 3.0), (202, 0.5)])
    out = weighted_rollup_to_levels(flat, w, hierarchy)
    assert out.value.round(10).eq(7.5).all()


def test_global_weighted_mean_is_the_weighted_average(values, hierarchy):
    w = _weights([(101, 1.0), (102, 1.0), (201, 2.0), (202, 6.0)])
    out = weighted_rollup_to_levels(values, w, hierarchy)
    g = out.loc[out.location_id == 1, "value"].iloc[0]
    # (1*1 + 3*1 + 10*2 + 20*6) / 10
    assert g == pytest.approx((1 + 3 + 20 + 120) / 10)


def test_super_region_means_use_only_their_own_children(values, hierarchy):
    w = _weights([(101, 1.0), (102, 3.0), (201, 1.0), (202, 1.0)])
    out = weighted_rollup_to_levels(values, w, hierarchy)
    sr10 = out.loc[out.location_id == 10, "value"].iloc[0]
    sr20 = out.loc[out.location_id == 20, "value"].iloc[0]
    assert sr10 == pytest.approx((1 * 1 + 3 * 3) / 4)
    assert sr20 == pytest.approx((10 + 20) / 2)


def test_zero_weight_locations_contribute_nothing(values, hierarchy):
    """The burden schemes drop zero-burden locations; a zero weight must not shift a mean."""
    with_zero = _weights([(101, 0.0), (102, 1.0), (201, 0.0), (202, 1.0)])
    without = _weights([(102, 1.0), (202, 1.0)])
    a = weighted_rollup_to_levels(values, with_zero, hierarchy)
    b = weighted_rollup_to_levels(values, without, hierarchy)
    ga = a.loc[a.location_id == 1, "value"].iloc[0]
    gb = b.loc[b.location_id == 1, "value"].iloc[0]
    assert ga == pytest.approx(gb)


def test_absent_locations_restrict_the_denominator(values, hierarchy):
    """A burden scheme silently restricts to endemic locations -- that is the point.

    The two schemes therefore have different denominators and their results are not
    directly subtractable.
    """
    endemic_only = _weights([(201, 1.0), (202, 1.0)])
    out = weighted_rollup_to_levels(values, endemic_only, hierarchy)
    assert 10 not in set(out.location_id)          # super-region with no endemic children
    assert out.loc[out.location_id == 1, "value"].iloc[0] == pytest.approx(15.0)


def test_total_is_not_a_rescaled_mean(values, hierarchy):
    """``total=True`` sums value*weight; it is a different quantity, not a scaled mean."""
    w = _weights([(101, 1.0), (102, 1.0), (201, 2.0), (202, 6.0)])
    mean = weighted_rollup_to_levels(values, w, hierarchy)
    total = weighted_rollup_to_levels(values, w, hierarchy, total=True)
    gm = mean.loc[mean.location_id == 1, "value"].iloc[0]
    gt = total.loc[total.location_id == 1, "value"].iloc[0]
    assert gt == pytest.approx(1 + 3 + 20 + 120)
    assert gt == pytest.approx(gm * 10)            # equals mean x total weight, by definition
    assert gt != pytest.approx(gm)


def test_totals_sum_up_the_hierarchy(values, hierarchy):
    """Extensive totals must be additive: super-regions sum to global."""
    w = _weights([(101, 1.0), (102, 1.0), (201, 2.0), (202, 6.0)])
    out = weighted_rollup_to_levels(values, w, hierarchy, total=True)
    g = out.loc[out.location_id == 1, "value"].iloc[0]
    srs = out.loc[out.location_id.isin([10, 20]), "value"].sum()
    assert g == pytest.approx(srs)


def test_year_axis_is_preserved_independently(hierarchy):
    """Each year aggregates on its own weights; years must not bleed together."""
    vals = pd.DataFrame(
        {
            "location_id": [101, 102, 101, 102],
            "year_id": [2023, 2023, 2024, 2024],
            "value": [1.0, 3.0, 10.0, 30.0],
        }
    )
    w = pd.DataFrame(
        {
            "location_id": [101, 102, 101, 102],
            "year_id": [2023, 2023, 2024, 2024],
            "weight": [1.0, 1.0, 1.0, 3.0],
        }
    )
    out = weighted_rollup_to_levels(vals, w, hierarchy)
    g = out[out.location_id == 1].set_index("year_id").value
    assert g[2023] == pytest.approx(2.0)
    assert g[2024] == pytest.approx((10 + 90) / 4)


def test_no_overlap_between_values_and_weights_returns_empty(values, hierarchy):
    w = _weights([(999, 1.0)])
    out = weighted_rollup_to_levels(values, w, hierarchy)
    assert out.empty
    assert list(out.columns) == ["location_id", "year_id", "value"]


def test_locations_missing_from_hierarchy_are_dropped(values, hierarchy):
    """An inner join on the hierarchy means unmapped locations cannot silently inflate."""
    w = _weights([(101, 1.0), (102, 1.0), (201, 1.0), (202, 1.0), (777, 1000.0)])
    vals = pd.concat(
        [values, pd.DataFrame({"location_id": [777], "year_id": [2023], "value": [1e6]})],
        ignore_index=True,
    )
    out = weighted_rollup_to_levels(vals, w, hierarchy)
    assert out.loc[out.location_id == 1, "value"].iloc[0] == pytest.approx(
        (1 + 3 + 10 + 20) / 4
    )
