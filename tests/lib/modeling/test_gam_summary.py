"""Tests for :mod:`idd_forecast_mbp.lib.modeling.gam_summary`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.modeling import fit as fit_mod
from idd_forecast_mbp.lib.modeling import specs as specs_mod
from idd_forecast_mbp.lib.modeling.gam_summary import linear_effects, summarize_gam

SPEC = (
    specs_mod.Term("x_smooth", "mpi", 5),
    specs_mod.Term("x_linear", "linear"),
    specs_mod.Term("x_other", "linear"),
    specs_mod.Term("A0_af", "factor"),
)


@pytest.fixture
def fitted():
    """A model with a known linear effect, so the coefficient is checkable."""
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame({
        "x_smooth": rng.uniform(0, 10, n),
        "x_linear": rng.normal(0, 1, n),
        "x_other": rng.normal(0, 1, n),
        "A0_af": rng.integers(0, 3, n),
        "location_id": rng.integers(1, 20, n),
        "year_id": rng.integers(2000, 2024, n),
    })
    frame["y"] = (0.30 * frame["x_smooth"] + 2.0 * frame["x_linear"]
                  - 1.0 * frame["x_other"] + rng.normal(0, 0.1, n))
    frame["natural"] = np.exp(frame["y"])
    outcome = fit_mod.Outcome("y", "natural", "log")
    return fit_mod.fit_gam(frame, SPEC, outcome), frame


class TestSummarizeGam:
    def test_terms_carry_real_column_names(self, fitted):
        gam, _ = fitted
        got = summarize_gam(gam, SPEC)
        assert list(got["term"])[:4] == ["x_smooth", "x_linear", "x_other", "A0_af"]

    def test_intercept_is_labelled(self, fitted):
        gam, _ = fitted
        assert "intercept" in set(summarize_gam(gam, SPEC)["term"])

    def test_forms_are_named_not_positional(self, fitted):
        gam, _ = fitted
        got = summarize_gam(gam, SPEC).set_index("term")
        assert got.loc["x_linear", "form"] == "linear"
        assert got.loc["x_smooth", "form"] == "spline"
        assert got.loc["A0_af", "form"] == "factor"

    def test_coefficient_recovers_a_known_linear_effect(self, fitted):
        """The whole point: a linear term must report an interpretable number."""
        gam, _ = fitted
        got = summarize_gam(gam, SPEC).set_index("term")
        assert got.loc["x_linear", "coef"] == pytest.approx(2.0, abs=0.15)
        assert got.loc["x_other", "coef"] == pytest.approx(-1.0, abs=0.15)

    def test_standard_errors_are_reported_for_linear_terms(self, fitted):
        gam, _ = fitted
        got = summarize_gam(gam, SPEC).set_index("term")
        assert got.loc["x_linear", "se"] > 0
        assert np.isfinite(got.loc["x_linear", "z"])
        assert 0.0 <= got.loc["x_linear", "p_coef"] <= 1.0

    def test_no_scalar_coefficient_for_spline_or_factor_terms(self, fitted):
        """A basis coefficient or a per-level effect is not one interpretable number."""
        gam, _ = fitted
        got = summarize_gam(gam, SPEC).set_index("term")
        assert np.isnan(got.loc["x_smooth", "coef"])
        assert np.isnan(got.loc["A0_af", "coef"])

    def test_coefficient_counts_match_the_model(self, fitted):
        gam, _ = fitted
        got = summarize_gam(gam, SPEC)
        assert got["n_coef"].sum() == len(gam.coef_)

    def test_multi_coefficient_terms_report_their_size(self, fitted):
        gam, _ = fitted
        got = summarize_gam(gam, SPEC).set_index("term")
        assert got.loc["x_smooth", "n_coef"] > 1
        assert got.loc["A0_af", "n_coef"] == 3      # three factor levels

    def test_effective_dof_is_populated(self, fitted):
        gam, _ = fitted
        got = summarize_gam(gam, SPEC)
        assert (got["edf"].dropna() >= 0).all()
        assert got["edf"].notna().any()

    def test_term_p_values_are_present(self, fitted):
        gam, _ = fitted
        assert summarize_gam(gam, SPEC)["p_term"].notna().any()

    def test_tolerates_a_model_with_no_statistics(self, fitted):
        """Never crash on a model whose statistics were not computed."""
        gam, _ = fitted
        gam.statistics_ = {}
        got = summarize_gam(gam, SPEC)
        assert len(got) == len(gam.terms)
        assert got["se"].isna().all()


class TestLinearEffects:
    def test_only_linear_terms_survive(self, fitted):
        gam, _ = fitted
        got = linear_effects(gam, SPEC)
        assert set(got["term"]) == {"x_linear", "x_other"}

    def test_interval_brackets_the_estimate(self, fitted):
        gam, _ = fitted
        got = linear_effects(gam, SPEC).set_index("term")
        for term in ("x_linear", "x_other"):
            assert got.loc[term, "ci_low"] < got.loc[term, "coef"]
            assert got.loc[term, "coef"] < got.loc[term, "ci_high"]

    def test_interval_covers_the_truth(self, fitted):
        gam, _ = fitted
        got = linear_effects(gam, SPEC).set_index("term")
        assert got.loc["x_linear", "ci_low"] <= 2.0 <= got.loc["x_linear", "ci_high"]
