"""Tests for :mod:`idd_forecast_mbp.lib.modeling.anchor`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.modeling.anchor import (
    AnchorSpec,
    Baseline,
    Eligibility,
    apply_shift,
    baseline_value,
    compute_shift,
    outlier_years,
)

YEARS = (2019, 2020, 2021, 2022, 2023)


def _frame(location_ids, years, values, col):
    """Tidy (location_id, year_id, <col>) frame from a nested value list."""
    rows = [
        {"location_id": loc, "year_id": yr, col: val}
        for loc, series in zip(location_ids, values, strict=True)
        for yr, val in zip(years, series, strict=True)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Eligibility / outlier detection
# ---------------------------------------------------------------------------

class TestOutlierYears:
    def test_no_method_keeps_everything(self):
        vals = [1.0, 1.0, 1.0, 50.0, 1.0]
        assert outlier_years(YEARS, vals, Eligibility(method=None)) == []

    def test_loo_mean_flags_a_spike(self):
        vals = [1.0, 1.1, 0.9, 1.0, 12.0]
        assert outlier_years(YEARS, vals, Eligibility(method="mean")) == [2023]

    def test_loo_mean_false_flags_a_trend_endpoint(self):
        """A clean linear trend has no outliers, but the LOO *mean* flags an end."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        flagged = outlier_years(YEARS, vals, Eligibility(method="mean", threshold_sd=1.5))
        assert flagged, "expected the LOO-mean method to mis-flag a pure trend"

    def test_loo_trend_does_not_flag_a_clean_trend(self):
        """The reason 'trend' exists: a perfectly linear series has no outliers."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert outlier_years(YEARS, vals, Eligibility(method="trend")) == []

    def test_loo_trend_flags_a_departure_from_trend(self):
        vals = [1.0, 2.0, 3.0, 4.0, 40.0]
        assert outlier_years(YEARS, vals, Eligibility(method="trend")) == [2023]

    @pytest.mark.parametrize(
        ("direction", "expected"),
        [("both", [2023]), ("above", [2023]), ("below", [])],
    )
    def test_direction_filters_the_side(self, direction, expected):
        vals = [1.0, 1.1, 0.9, 1.0, 12.0]
        got = outlier_years(YEARS, vals, Eligibility(method="mean", direction=direction))
        assert got == expected

    def test_direction_below_catches_a_dip(self):
        vals = [10.0, 10.1, 9.9, 10.0, 0.5]
        got = outlier_years(YEARS, vals, Eligibility(method="mean", direction="below"))
        assert got == [2023]

    def test_draw_uncertainty_suppresses_a_flag(self):
        """A year that is uncertain enough to explain its own deviation survives."""
        vals = [1.0, 1.1, 0.9, 1.0, 12.0]
        ses = [0.01, 0.01, 0.01, 0.01, 500.0]
        assert outlier_years(YEARS, vals, Eligibility(method="mean"), ses) == []
        # ...and is flagged again once the standard error is ignored.
        no_se = Eligibility(method="mean", use_draw_se=False)
        assert outlier_years(YEARS, vals, no_se, ses) == [2023]

    @pytest.mark.parametrize(("method", "n_years"), [("mean", 2), ("trend", 3)])
    def test_too_few_years_flags_nothing(self, method, n_years):
        yrs = YEARS[:n_years]
        vals = [1.0, 99.0, 1.0][:n_years]
        assert outlier_years(yrs, vals, Eligibility(method=method)) == []

    def test_zero_spread_flags_nothing(self):
        vals = [2.0, 2.0, 2.0, 2.0, 2.0]
        assert outlier_years(YEARS, vals, Eligibility(method="mean")) == []

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method must be"):
            Eligibility(method="magic")

    def test_rejects_nonpositive_threshold(self):
        with pytest.raises(ValueError, match="threshold_sd must be positive"):
            Eligibility(method="mean", threshold_sd=0.0)

    def test_rejects_unknown_direction(self):
        with pytest.raises(ValueError, match="direction must be"):
            Eligibility(method="mean", direction="upwards")


# ---------------------------------------------------------------------------
# Baseline statistics
# ---------------------------------------------------------------------------

class TestBaselineValue:
    @pytest.mark.parametrize(
        ("statistic", "expected"),
        [("mean", 4.0), ("median", 3.0), ("min", 1.0), ("max", 10.0)],
    )
    def test_simple_statistics(self, statistic, expected):
        vals = [1.0, 2.0, 3.0, 4.0, 10.0]
        assert baseline_value(YEARS, vals, Baseline(statistic=statistic)) == expected

    def test_single_takes_the_only_year(self):
        assert baseline_value((2023,), (7.5,), Baseline(statistic="single")) == 7.5

    def test_trend_reads_the_fitted_line(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        got = baseline_value(YEARS, vals, Baseline(statistic="trend", effective_year=2023))
        assert got == pytest.approx(5.0)

    def test_trend_defaults_to_the_window_midpoint(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        got = baseline_value(YEARS, vals, Baseline(statistic="trend"))
        assert got == pytest.approx(3.0)

    def test_trend_is_not_the_median_on_a_skewed_series(self):
        """Trend and median genuinely differ — this is why both are offered."""
        vals = [1.0, 2.0, 3.0, 4.0, 40.0]
        trend = baseline_value(YEARS, vals, Baseline(statistic="trend", effective_year=2023))
        median = baseline_value(YEARS, vals, Baseline(statistic="median"))
        assert trend != pytest.approx(median)

    def test_trim_min_max_drops_the_extremes(self):
        vals = [1.0, 2.0, 3.0, 4.0, 100.0]
        got = baseline_value(YEARS, vals, Baseline(statistic="mean", trim="min_max"))
        assert got == pytest.approx(3.0)

    def test_trim_extremes_drops_beyond_threshold(self):
        vals = [1.0, 1.0, 1.0, 1.0, 100.0]
        got = baseline_value(YEARS, vals, Baseline("mean", trim="extremes", trim_threshold=1.0))
        assert got == pytest.approx(1.0)

    def test_trim_is_a_noop_below_three_values(self):
        got = baseline_value((2022, 2023), (1.0, 100.0), Baseline("mean", trim="min_max"))
        assert got == pytest.approx(50.5)

    def test_non_finite_values_are_ignored(self):
        vals = [1.0, np.nan, 3.0, np.inf, 5.0]
        assert baseline_value(YEARS, vals, Baseline(statistic="median")) == 3.0

    def test_all_missing_gives_nan(self):
        vals = [np.nan] * 5
        assert np.isnan(baseline_value(YEARS, vals, Baseline(statistic="mean")))

    def test_rejects_unknown_statistic(self):
        with pytest.raises(ValueError, match="statistic must be"):
            Baseline(statistic="mode")

    def test_rejects_unknown_trim(self):
        with pytest.raises(ValueError, match="trim must be"):
            Baseline(trim="everything")

    def test_trim_extremes_is_a_noop_on_zero_spread(self):
        vals = [3.0, 3.0, 3.0, 3.0, 3.0]
        got = baseline_value(YEARS, vals, Baseline("mean", trim="extremes"))
        assert got == pytest.approx(3.0)

    def test_trend_falls_back_to_the_only_finite_value(self):
        vals = [np.nan, np.nan, np.nan, np.nan, 6.0]
        assert baseline_value(YEARS, vals, Baseline(statistic="trend")) == 6.0

    def test_trim_removing_everything_gives_nan(self):
        """`extremes` with a zero threshold keeps nothing but the exact mean."""
        vals = [1.0, 2.0, 3.0, 4.0, 100.0]
        got = baseline_value(YEARS, vals, Baseline("mean", trim="extremes", trim_threshold=0.0))
        assert np.isnan(got)


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------

class TestAnchorSpec:
    def test_single_requires_exactly_one_year(self):
        with pytest.raises(ValueError, match="needs exactly one year"):
            AnchorSpec(years=YEARS, baseline=Baseline(statistic="single"))

    def test_rejects_empty_window(self):
        with pytest.raises(ValueError, match="years must be non-empty"):
            AnchorSpec(years=())

    def test_rejects_unknown_applied_to(self):
        with pytest.raises(ValueError, match="applied_to must be"):
            AnchorSpec(years=YEARS, applied_to="sideways")

    def test_constructors_match_their_legacy_semantics(self):
        assert AnchorSpec.point(2023).years == (2023,)
        assert AnchorSpec.point(2023).baseline.statistic == "single"
        assert AnchorSpec.median_diff(YEARS).applied_to == "separate"
        assert AnchorSpec.median_resid(YEARS).applied_to == "residual"


# ---------------------------------------------------------------------------
# compute_shift / apply_shift
# ---------------------------------------------------------------------------

class TestComputeShift:
    def test_point_recovers_obs_minus_pred_at_the_year(self):
        obs = _frame([1], (2023,), [[10.0]], "observed")
        pred = _frame([1], (2023,), [[4.0]], "predicted")
        shift = compute_shift(obs, pred, AnchorSpec.point(2023))
        assert shift.loc[1] == pytest.approx(6.0)

    def test_median_diff_and_median_resid_disagree(self):
        """The median is not linear, so the two definitions are genuinely distinct."""
        obs = _frame([1], YEARS, [[1.0, 2.0, 3.0, 4.0, 100.0]], "observed")
        pred = _frame([1], YEARS, [[0.0, 0.0, 0.0, 90.0, 0.0]], "predicted")
        diff = compute_shift(obs, pred, AnchorSpec.median_diff(YEARS)).loc[1]
        resid = compute_shift(obs, pred, AnchorSpec.median_resid(YEARS)).loc[1]
        assert diff != pytest.approx(resid)

    def test_median_diff_equals_median_resid_when_pred_is_constant(self):
        """With a flat prediction the two coincide — a sanity check on both paths."""
        obs = _frame([1], YEARS, [[1.0, 5.0, 3.0, 9.0, 2.0]], "observed")
        pred = _frame([1], YEARS, [[2.0] * 5], "predicted")
        diff = compute_shift(obs, pred, AnchorSpec.median_diff(YEARS)).loc[1]
        resid = compute_shift(obs, pred, AnchorSpec.median_resid(YEARS)).loc[1]
        assert diff == pytest.approx(resid)

    def test_only_window_years_contribute(self):
        obs = _frame([1], (2017, 2018, 2023), [[999.0, 999.0, 10.0]], "observed")
        pred = _frame([1], (2017, 2018, 2023), [[0.0, 0.0, 4.0]], "predicted")
        shift = compute_shift(obs, pred, AnchorSpec.point(2023))
        assert shift.loc[1] == pytest.approx(6.0)

    def test_outlier_year_is_excluded_from_the_anchor(self):
        obs_vals = [1.0, 1.0, 1.0, 1.0, 50.0]
        obs = _frame([1], YEARS, [obs_vals], "observed")
        pred = _frame([1], YEARS, [[0.0] * 5], "predicted")
        plain = compute_shift(obs, pred, AnchorSpec(years=YEARS, baseline=Baseline("mean")))
        filtered = compute_shift(
            obs, pred,
            AnchorSpec(years=YEARS, baseline=Baseline("mean"),
                       eligibility=Eligibility(method="mean")),
        )
        assert plain.loc[1] == pytest.approx(10.8)
        assert filtered.loc[1] == pytest.approx(1.0)

    def test_residual_path_carries_the_standard_error(self):
        """A high-SE year survives outlier filtering on the residual path too."""
        obs = _frame([1], YEARS, [[1.0, 1.1, 0.9, 1.0, 12.0]], "observed")
        obs["obs_se"] = [0.01, 0.01, 0.01, 0.01, 500.0]
        pred = _frame([1], YEARS, [[0.0] * 5], "predicted")
        spec = AnchorSpec(years=YEARS, baseline=Baseline("mean"),
                          eligibility=Eligibility(method="mean"), applied_to="residual")
        with_se = compute_shift(obs, pred, spec, se_col="obs_se").loc[1]
        without_se = compute_shift(obs.drop(columns="obs_se"), pred, spec).loc[1]
        # With the SE the spike is kept (mean of all five); without it, dropped.
        assert with_se == pytest.approx(3.2)
        assert without_se == pytest.approx(1.0)

    def test_groups_are_independent(self):
        obs = _frame([1, 2], (2023,), [[10.0], [20.0]], "observed")
        pred = _frame([1, 2], (2023,), [[4.0], [5.0]], "predicted")
        shift = compute_shift(obs, pred, AnchorSpec.point(2023))
        assert shift.loc[1] == pytest.approx(6.0)
        assert shift.loc[2] == pytest.approx(15.0)

    def test_group_without_an_anchor_is_dropped(self):
        obs = _frame([1], (2023,), [[10.0]], "observed")
        pred = _frame([1, 2], (2023,), [[4.0], [4.0]], "predicted")
        shift = compute_shift(obs, pred, AnchorSpec.point(2023))
        assert 2 not in shift.index

    def test_age_sex_granularity(self):
        """Anchoring per (location, age, sex) — the granularity that makes as_id cancel."""
        rows = []
        for age in (3, 4):
            for sex in (1, 2):
                rows.append({"location_id": 1, "age_group_id": age, "sex_id": sex,
                             "year_id": 2023, "observed": 10.0 * age + sex,
                             "predicted": 1.0})
        frame = pd.DataFrame(rows)
        keys = ("location_id", "age_group_id", "sex_id")
        shift = compute_shift(frame, frame, AnchorSpec.point(2023), group_cols=keys,
                              obs_col="observed", pred_col="predicted")
        assert shift.loc[(1, 3, 1)] == pytest.approx(30.0)
        assert shift.loc[(1, 4, 2)] == pytest.approx(41.0)


class TestCancellation:
    """The key concept: a time-constant additive term cancels under the anchor.

    See ``.claude/DECISIONS.md`` 2026-08-03 — this is why the prediction frame can
    be decoupled from the fit design matrix, and why `as_id` never needs to enter
    the prediction. If this test fails the whole pipeline shape is wrong.
    """

    @pytest.mark.parametrize("offset", [0.0, 2.5, -7.0, 100.0])
    def test_constant_offset_leaves_the_anchored_result_unchanged(self, offset):
        years = (2020, 2021, 2022, 2023, 2024, 2025)
        obs = _frame([1], YEARS, [[1.0, 1.5, 2.0, 2.5, 3.0]], "observed")

        base = pd.DataFrame({"location_id": 1, "year_id": years,
                             "predicted": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]})
        offset_frame = base.assign(predicted=base["predicted"] + offset)

        spec = AnchorSpec.median_diff(YEARS)
        plain = apply_shift(base, compute_shift(obs, base, spec))
        shifted = apply_shift(offset_frame, compute_shift(obs, offset_frame, spec))

        np.testing.assert_allclose(plain["anchored"], shifted["anchored"], rtol=1e-12)

    def test_cancellation_holds_per_age_sex_cell(self):
        keys = ("location_id", "age_group_id", "sex_id")
        cells = [(1, 3, 1), (1, 4, 2)]
        years = (2022, 2023, 2024)

        obs = pd.DataFrame([
            {"location_id": c[0], "age_group_id": c[1], "sex_id": c[2],
             "year_id": 2023, "observed": 5.0 + i}
            for i, c in enumerate(cells)
        ])
        base = pd.DataFrame([
            {"location_id": c[0], "age_group_id": c[1], "sex_id": c[2],
             "year_id": y, "predicted": 0.1 * k}
            for c in cells for k, y in enumerate(years)
        ])
        # A per-cell, time-constant offset — exactly what an `as_id` term is.
        offsets = {(1, 3, 1): 3.0, (1, 4, 2): -2.0}
        with_as_id = base.copy()
        with_as_id["predicted"] += [
            offsets[(r.location_id, r.age_group_id, r.sex_id)] for r in base.itertuples()
        ]

        spec = AnchorSpec.point(2023)
        a = apply_shift(base, compute_shift(obs, base, spec, group_cols=keys),
                        group_cols=keys)
        b = apply_shift(with_as_id, compute_shift(obs, with_as_id, spec, group_cols=keys),
                        group_cols=keys)
        np.testing.assert_allclose(a["anchored"], b["anchored"], rtol=1e-12)


class TestApplyShift:
    def test_shift_is_added_to_every_year(self):
        pred = pd.DataFrame({"location_id": 1, "year_id": [2023, 2050, 2100],
                             "predicted": [1.0, 2.0, 3.0]})
        out = apply_shift(pred, pd.Series({1: 10.0}, name="shift"))
        np.testing.assert_allclose(out["anchored"], [11.0, 12.0, 13.0])

    def test_unanchored_group_dropped_by_default(self):
        pred = pd.DataFrame({"location_id": [1, 2], "year_id": [2023, 2023],
                             "predicted": [1.0, 2.0]})
        out = apply_shift(pred, pd.Series({1: 10.0}, name="shift"))
        assert out["location_id"].tolist() == [1]

    def test_unanchored_group_kept_unshifted_when_asked(self):
        """Matches the malaria rocket, which treats a missing anchor as shift zero."""
        pred = pd.DataFrame({"location_id": [1, 2], "year_id": [2023, 2023],
                             "predicted": [1.0, 2.0]})
        out = apply_shift(pred, pd.Series({1: 10.0}, name="shift"), drop_unanchored=False)
        np.testing.assert_allclose(out["anchored"], [11.0, 2.0])
