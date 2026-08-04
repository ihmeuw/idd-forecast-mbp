"""Tests for :mod:`idd_forecast_mbp.lib.modeling.year_path`."""

from __future__ import annotations

import numpy as np
import pytest

from idd_forecast_mbp.lib.modeling.year_path import (
    effective_year,
    growth_factor,
    identity,
    linear_decay,
    logistic_decay,
)

ANCHOR = 2023
END = 2100
YEARS = np.arange(2000, 2101)


def _increments(path):
    """Per-year advance in effective year — the marginal time effect."""
    return np.diff(path)


class TestIdentity:
    def test_effective_year_is_the_calendar_year(self):
        np.testing.assert_allclose(identity(YEARS, ANCHOR), YEARS)


class TestPastIsUntouched:
    """The fit saw calendar years, so the anchor window must too."""

    @pytest.mark.parametrize("fn", [linear_decay, logistic_decay])
    def test_years_up_to_the_anchor_are_the_calendar_year(self, fn):
        path = fn(YEARS, ANCHOR, END)
        past = YEARS <= ANCHOR
        np.testing.assert_allclose(path[past], YEARS[past])


class TestLinearDecay:
    def test_marginal_effect_reaches_zero_at_the_end_year(self):
        path = linear_decay(YEARS, ANCHOR, END)
        assert _increments(path)[-1] == pytest.approx(0.0, abs=1e-9)

    def test_marginal_effect_never_increases(self):
        inc = _increments(linear_decay(YEARS, ANCHOR, END))
        future = inc[YEARS[:-1] >= ANCHOR]
        assert np.all(np.diff(future) <= 1e-12)

    def test_cumulative_effect_does_not_return_to_the_anchor(self):
        """Growth plateaus; it is not undone. That would be a different claim."""
        path = linear_decay(YEARS, ANCHOR, END)
        assert path[-1] > ANCHOR

    def test_effective_advance_is_about_half_the_span(self):
        """Weights fall linearly 1 -> 0, so they average one half."""
        path = linear_decay(YEARS, ANCHOR, END)
        advance = path[-1] - ANCHOR
        assert advance == pytest.approx((END - ANCHOR) / 2, rel=0.03)

    def test_advances_less_than_no_decay(self):
        decayed = linear_decay(YEARS, ANCHOR, END)[-1] - ANCHOR
        assert decayed < (END - ANCHOR)

    def test_rejects_end_year_at_or_before_the_anchor(self):
        with pytest.raises(ValueError, match="must be after anchor_year"):
            linear_decay(YEARS, ANCHOR, ANCHOR)


class TestLogisticDecay:
    def test_marginal_effect_reaches_zero_at_the_end_year(self):
        path = logistic_decay(YEARS, ANCHOR, END)
        assert _increments(path)[-1] == pytest.approx(0.0, abs=1e-9)

    def test_marginal_effect_never_increases(self):
        inc = _increments(logistic_decay(YEARS, ANCHOR, END))
        future = inc[YEARS[:-1] >= ANCHOR]
        assert np.all(np.diff(future) <= 1e-9)

    def test_holds_near_the_fitted_trend_before_rolling_off(self):
        """The point of the logistic shape: near-term looks like the fit."""
        path = logistic_decay(YEARS, ANCHOR, END)
        inc = _increments(path)
        early = inc[(YEARS[:-1] > ANCHOR) & (YEARS[:-1] <= ANCHOR + 10)]
        assert early.mean() > 0.85

    def test_rolls_off_later_than_linear(self):
        lin = linear_decay(YEARS, ANCHOR, END)
        log = logistic_decay(YEARS, ANCHOR, END)
        mid = np.argmax(YEARS >= ANCHOR + 20)
        assert log[mid] > lin[mid]

    def test_steeper_gives_a_sharper_roll_off(self):
        gentle = logistic_decay(YEARS, ANCHOR, END, steepness=4.0)
        sharp = logistic_decay(YEARS, ANCHOR, END, steepness=16.0)
        early = np.argmax(YEARS >= ANCHOR + 15)
        assert sharp[early] > gentle[early]

    def test_cumulative_effect_does_not_return_to_the_anchor(self):
        assert logistic_decay(YEARS, ANCHOR, END)[-1] > ANCHOR

    @pytest.mark.parametrize("bad", [2023, 2100, 1990, 2200])
    def test_rejects_a_midpoint_outside_the_window(self, bad):
        with pytest.raises(ValueError, match="midpoint_year"):
            logistic_decay(YEARS, ANCHOR, END, midpoint_year=bad)

    def test_default_midpoint_is_2060(self):
        from idd_forecast_mbp.lib.modeling.year_path import DEFAULT_MIDPOINT_YEAR
        assert DEFAULT_MIDPOINT_YEAR == 2060
        np.testing.assert_allclose(
            logistic_decay(YEARS, ANCHOR, END),
            logistic_decay(YEARS, ANCHOR, END, midpoint_year=2060))

    def test_rejects_end_year_at_or_before_the_anchor(self):
        with pytest.raises(ValueError, match="must be after anchor_year"):
            logistic_decay(YEARS, ANCHOR, ANCHOR)


class TestOrderIndependence:
    def test_result_does_not_depend_on_input_ordering(self):
        shuffled = YEARS.copy()
        rng = np.random.default_rng(0)
        rng.shuffle(shuffled)
        a = dict(zip(YEARS, linear_decay(YEARS, ANCHOR, END), strict=True))
        b = dict(zip(shuffled, linear_decay(shuffled, ANCHOR, END), strict=True))
        for year in YEARS:
            assert a[year] == pytest.approx(b[year])


class TestSamplingIndependence:
    """The path must not depend on which years you ask about."""

    @pytest.mark.parametrize("fn", [linear_decay, logistic_decay])
    def test_sparse_request_matches_the_dense_path(self, fn):
        dense = fn(YEARS, ANCHOR, END)
        dense_lookup = dict(zip(YEARS, dense, strict=True))
        sparse_years = np.array([2023, 2050, 2100])
        sparse = fn(sparse_years, ANCHOR, END)
        for year, got in zip(sparse_years, sparse, strict=True):
            assert got == pytest.approx(dense_lookup[year], rel=1e-9)

    @pytest.mark.parametrize("fn", [linear_decay, logistic_decay])
    def test_a_single_far_year_still_accumulates_the_whole_span(self, fn):
        only_2100 = fn(np.array([2100]), ANCHOR, END)[0]
        full = fn(YEARS, ANCHOR, END)[-1]
        assert only_2100 == pytest.approx(full, rel=1e-9)
        assert only_2100 > ANCHOR + 20      # not just one year's worth


class TestEffectiveYear:
    def test_dispatches_by_name(self):
        np.testing.assert_allclose(
            effective_year(YEARS, ANCHOR, "linear_decay", end_year=END),
            linear_decay(YEARS, ANCHOR, END))

    def test_default_is_identity(self):
        np.testing.assert_allclose(effective_year(YEARS, ANCHOR), YEARS)

    def test_accepts_a_user_supplied_callable(self):
        """The escape hatch for specifying the math directly."""
        def freeze(years, anchor_year):
            return np.minimum(np.asarray(years, float), anchor_year)
        got = effective_year(YEARS, ANCHOR, freeze)
        assert got[-1] == ANCHOR      # time stops dead at the anchor


class TestGrowthFactor:
    def test_undecayed_growth_matches_the_closed_form(self):
        beta = 0.043
        got = growth_factor(beta, ANCHOR, END, "identity")
        assert got == pytest.approx(np.exp(beta * (END - ANCHOR)), rel=1e-9)

    def test_undecayed_dengue_year_term_is_about_27x(self):
        """The number that makes a decay non-optional."""
        assert growth_factor(0.043, ANCHOR, END, "identity") == pytest.approx(27, rel=0.1)

    def test_linear_decay_roughly_halves_the_exponent(self):
        assert growth_factor(0.043, ANCHOR, END, "linear_decay") == pytest.approx(
            5.3, rel=0.15)

    def test_a_symmetric_logistic_totals_the_same_as_linear(self):
        """Surprising but exact: symmetric weights average 0.5, so do linear ones.

        The two shapes differ in WHEN the effect is spent, not in how much is
        spent. Making the logistic meaningfully different requires moving
        `midpoint_fraction` off centre.
        """
        beta = 0.043
        lin = growth_factor(beta, ANCHOR, END, "linear_decay")
        log = growth_factor(beta, ANCHOR, END, "logistic_decay",
                            midpoint_year=(ANCHOR + END) / 2)
        assert log == pytest.approx(lin, rel=0.02)

    def test_any_decay_is_less_than_none(self):
        beta = 0.043
        none = growth_factor(beta, ANCHOR, END, "identity")
        for path in ("linear_decay", "logistic_decay"):
            assert growth_factor(beta, ANCHOR, END, path) < none

    def test_midpoint_moves_the_total_in_the_expected_direction(self):
        """Later roll-off holds the fitted trend longer, so permits more growth."""
        beta = 0.043
        early = growth_factor(beta, ANCHOR, END, "logistic_decay",
                              midpoint_year=2042)
        late = growth_factor(beta, ANCHOR, END, "logistic_decay",
                             midpoint_year=2080)
        assert early < late

    def test_a_negative_coefficient_gives_shrinkage(self):
        assert growth_factor(-0.043, ANCHOR, END, "identity") < 1.0
