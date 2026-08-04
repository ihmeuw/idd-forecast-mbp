"""Tests for :mod:`idd_forecast_mbp.lib.modeling.dengue_pipeline`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.data.dengue_inputs import DengueInputs
from idd_forecast_mbp.lib.modeling import specs as specs_mod
from idd_forecast_mbp.lib.modeling.anchor import AnchorSpec
from idd_forecast_mbp.lib.modeling.dengue_formulations import (
    CFR,
    INCIDENCE,
    MORTALITY,
    Formulation,
)
from idd_forecast_mbp.lib.modeling.dengue_pipeline import (
    run_formulations,
    NATURAL_COLUMN,
    model_frame,
    observed_anchor_frame,
    run_formulation,
)

ANCHOR = 2023
YEARS = range(2015, 2024)
CELLS = ((3, 1), (4, 2))


@pytest.fixture
def inputs():
    """Two locations x two age/sex cells x nine years, with a usable signal.

    Incidence rises with suitability so a fit is well posed; CFR sits strictly
    inside (0, 1) so its logit is finite.
    """
    rng = np.random.default_rng(0)
    rows = []
    for loc in (10, 20):
        for year in YEARS:
            suitability = 0.3 + 0.02 * (year - 2015) + 0.1 * (loc == 20)
            for age, sex in CELLS:
                cell_scale = 3.0 if age == 4 else 1.0
                inc = 0.01 * cell_scale * np.exp(2.0 * suitability)
                cfr = 0.01 if age == 3 else 0.02
                rows.append({
                    "location_id": loc, "fhs_location_id": loc, "year_id": year,
                    "age_group_id": age, "sex_id": sex,
                    "A0_location_id": 163, "A0_af": 0,
                    "dengue_suitability": suitability,
                    "log_gdppc_mean": 7.0 + 0.01 * (year - 2015),
                    "population": 1000.0,
                    "rr_inc_as": cell_scale,
                    "dengue_inc_rate": inc * (1 + 0.001 * rng.standard_normal()),
                    "dengue_mort_rate": inc * cfr,
                })
    frame = pd.DataFrame(rows)
    frame["log_dengue_inc_rate"] = np.log(frame["dengue_inc_rate"])
    frame["log_dengue_mort_rate"] = np.log(frame["dengue_mort_rate"])
    frame["dengue_cfr"] = frame["dengue_mort_rate"] / frame["dengue_inc_rate"]
    frame["as_id"] = (frame["age_group_id"].astype(str) + "_"
                      + frame["sex_id"].astype(str)).astype("category").cat.codes
    frame["logit_dengue_cfr"] = np.log(
        frame["dengue_cfr"] / (1 - frame["dengue_cfr"]))

    hierarchy = pd.DataFrame([
        {"location_id": 1, "parent_id": 0, "level": 0, "path_to_top_parent": "1"},
        {"location_id": 10, "parent_id": 1, "level": 3, "path_to_top_parent": "1,10"},
        {"location_id": 20, "parent_id": 1, "level": 3, "path_to_top_parent": "1,20"},
    ])
    return DengueInputs(
        fit_frame=frame,
        observed_all_age=pd.DataFrame(),
        population=pd.DataFrame(),
        hierarchy=hierarchy,
        grain="fhs",
        anchor_year=ANCHOR,
        reference_age_group_id=3,
        reference_sex_id=1,
        base_location_ids=np.array([10, 20]),
    )


def _formulation(**overrides) -> Formulation:
    base = {
        "id": "t1",
        "description": "test",
        "structure": "inc_cfr",
        "terms": {
            INCIDENCE: (specs_mod.Term("dengue_suitability", "linear"),),
            CFR: (specs_mod.Term("log_gdppc_mean", "linear"),),
        },
        "anchor": AnchorSpec.point(ANCHOR),
    }
    return Formulation(**{**base, **overrides})


class TestFrames:
    def test_base_cell_frame_is_one_row_per_location_year(self, inputs):
        frame = model_frame(inputs, _formulation(), INCIDENCE)
        assert len(frame) == 2 * len(list(YEARS))
        assert set(frame["age_group_id"]) == {3}

    def test_observed_anchor_frame_keeps_age_sex_when_anchoring_per_cell(self, inputs):
        got = observed_anchor_frame(inputs, _formulation(), INCIDENCE)
        assert set(zip(got["age_group_id"], got["sex_id"], strict=True)) == set(CELLS)

    def test_observed_anchor_frame_collapses_when_anchoring_per_location(self, inputs):
        f = _formulation(anchor_group=("location_id",))
        got = observed_anchor_frame(inputs, f, INCIDENCE)
        assert "age_group_id" not in got.columns
        assert len(got) == 2 * len(list(YEARS))

    def test_observed_anchor_frame_drops_non_finite_responses(self, inputs):
        inputs.fit_frame.loc[0, "log_dengue_inc_rate"] = -np.inf
        got = observed_anchor_frame(inputs, _formulation(), INCIDENCE)
        assert np.isfinite(got["observed"]).all()


class TestRunFormulation:
    def test_fits_the_outcomes_the_structure_declares(self, inputs):
        run = run_formulation(inputs, _formulation())
        assert set(run.fits) == {INCIDENCE, CFR}

    def test_derives_mortality_from_incidence_and_cfr(self, inputs):
        run = run_formulation(inputs, _formulation())
        rates = run.rates
        np.testing.assert_allclose(
            rates[NATURAL_COLUMN[MORTALITY]],
            rates[NATURAL_COLUMN[INCIDENCE]] * rates[NATURAL_COLUMN[CFR]],
            rtol=1e-12,
        )

    def test_mort_cfr_derives_incidence(self, inputs):
        f = _formulation(
            structure="mort_cfr",
            terms={MORTALITY: (specs_mod.Term("dengue_suitability", "linear"),),
                   CFR: (specs_mod.Term("log_gdppc_mean", "linear"),)},
        )
        run = run_formulation(inputs, f)
        rates = run.rates
        np.testing.assert_allclose(
            rates[NATURAL_COLUMN[INCIDENCE]],
            rates[NATURAL_COLUMN[MORTALITY]] / rates[NATURAL_COLUMN[CFR]],
            rtol=1e-12,
        )

    def test_inc_mort_fits_both_and_derives_nothing(self, inputs):
        f = _formulation(
            structure="inc_mort",
            terms={INCIDENCE: (specs_mod.Term("dengue_suitability", "linear"),),
                   MORTALITY: (specs_mod.Term("dengue_suitability", "linear"),)},
        )
        run = run_formulation(inputs, f)
        assert set(run.fits) == {INCIDENCE, MORTALITY}
        assert f.derived_outcome is None

    def test_mort_then_inc_feeds_mortality_into_the_incidence_model(self, inputs):
        """The coupled structure: incidence consumes predicted mortality."""
        f = _formulation(
            structure="mort_then_inc",
            terms={
                MORTALITY: (specs_mod.Term("dengue_suitability", "linear"),),
                INCIDENCE: (specs_mod.Term("dengue_suitability", "linear"),
                            specs_mod.Term("log_dengue_mort_rate", "linear")),
            },
        )
        run = run_formulation(inputs, f)
        assert set(run.fits) == {MORTALITY, INCIDENCE}
        assert np.isfinite(run.rates[NATURAL_COLUMN[INCIDENCE]]).any()

    def test_anchored_prediction_hits_the_observed_anchor(self, inputs):
        """A point anchor must reproduce observed at the anchor year, per cell."""
        run = run_formulation(inputs, _formulation())
        pred = run.fits[INCIDENCE].predictions
        at_anchor = pred[pred["year_id"] == ANCHOR]
        observed = inputs.fit_frame.merge(
            at_anchor[["location_id", "age_group_id", "sex_id", "anchored"]],
            on=["location_id", "age_group_id", "sex_id"], how="inner",
        )
        observed = observed[observed["year_id"] == ANCHOR]
        np.testing.assert_allclose(
            observed["anchored"], observed["log_dengue_inc_rate"], rtol=1e-10,
        )

    def test_each_age_sex_cell_lands_on_its_own_anchor(self, inputs):
        """Cells differ threefold in the fixture; the anchored output must too.

        Tolerance is set by the 0.1% multiplicative noise the fixture injects per
        row — the ratio of two noisy observations carries it — not by anything
        about the anchor, which is exact.
        """
        run = run_formulation(inputs, _formulation())
        pred = run.fits[INCIDENCE].predictions
        at_anchor = pred[(pred["year_id"] == ANCHOR) & (pred["location_id"] == 10)]
        by_cell = at_anchor.set_index("age_group_id")[NATURAL_COLUMN[INCIDENCE]]
        assert by_cell.loc[4] / by_cell.loc[3] == pytest.approx(3.0, rel=1e-2)

    def test_natural_space_column_inverts_the_link(self, inputs):
        run = run_formulation(inputs, _formulation())
        pred = run.fits[INCIDENCE].predictions
        np.testing.assert_allclose(
            pred[NATURAL_COLUMN[INCIDENCE]], np.exp(pred["anchored"]), rtol=1e-12,
        )
        cfr = run.fits[CFR].predictions
        expected = 1 / (1 + np.exp(-cfr["anchored"].to_numpy(dtype=float)))
        np.testing.assert_allclose(cfr[NATURAL_COLUMN[CFR]], expected, rtol=1e-12)

    def test_fit_row_count_is_reported(self, inputs):
        run = run_formulation(inputs, _formulation())
        assert run.fits[INCIDENCE].n_fit_rows == 2 * len(list(YEARS))

    def test_rejects_a_grain_mismatch(self, inputs):
        with pytest.raises(ValueError, match="grain"):
            run_formulation(inputs, _formulation(grain="lsae"))

    def test_as_id_is_allowed_and_predicts_on_the_age_sex_grid(self, inputs):
        """as_id belongs in the regression; it must not be refused.

        An earlier version raised on any as_id term, conflating "should not widen
        the prediction frame" with "may not be a term". The cancellation that
        motivated it applies to the RAKED result only — unraked, as_id changes the
        answer, which is why both are reported.
        """
        f = _formulation(
            terms={INCIDENCE: (specs_mod.Term("dengue_suitability", "linear"),),
                   CFR: (specs_mod.Term("log_gdppc_mean", "linear"),
                         specs_mod.Term("as_id", "factor"))},
            response_grain={CFR: "age_sex"},
        )
        run = run_formulation(inputs, f)
        cfr_pred = run.fits[CFR].predictions
        assert {"age_group_id", "sex_id"} <= set(cfr_pred.columns)
        assert len(cfr_pred) > len(run.fits[INCIDENCE].predictions.drop_duplicates(
            ["location_id", "year_id"]))

    def test_unraked_keeps_the_models_own_level(self, inputs):
        """No anchor means no shift: the anchored column is the raw prediction."""
        f = _formulation(anchor=None)
        run = run_formulation(inputs, f)
        pred = run.fits[INCIDENCE].predictions
        assert run.formulation.is_raked is False
        assert pred["anchored"].notna().all()


class TestPredictionFrameStaysNarrow:
    def test_prediction_is_made_on_the_base_cell_only(self, inputs, monkeypatch):
        """The model must be evaluated once per (location, year), not per age/sex.

        This is the cancellation shortcut made observable: if a future change
        widened the prediction frame, this count would jump by the number of
        age/sex cells.
        """
        from idd_forecast_mbp.lib.modeling import dengue_pipeline as pipeline

        calls: list[int] = []
        real_fit_one = pipeline._fit_one

        def spy(fit_frame, terms, outcome, predict_frame=None, **kw):
            calls.append(len(fit_frame))
            return real_fit_one(fit_frame, terms, outcome, predict_frame, **kw)

        monkeypatch.setattr(pipeline, "_fit_one", spy)
        run_formulation(inputs, _formulation())

        n_location_years = 2 * len(list(YEARS))
        assert calls == [n_location_years, n_location_years]


class TestRunFormulations:
    def test_serial_runs_every_formulation(self, inputs):
        fs = [_formulation(id="a"), _formulation(id="b")]
        got = run_formulations(inputs, fs)
        assert list(got) == ["a", "b"]

    def test_parallel_matches_serial_exactly(self, inputs):
        """Formulations share no state, so parallelism must not change a number."""
        fs = [_formulation(id="a"), _formulation(id="b")]
        serial = run_formulations(inputs, fs, n_jobs=1)
        parallel = run_formulations(inputs, fs, n_jobs=2, backend="threading")
        for name in ("a", "b"):
            np.testing.assert_allclose(
                serial[name].rates[NATURAL_COLUMN[INCIDENCE]].to_numpy(),
                parallel[name].rates[NATURAL_COLUMN[INCIDENCE]].to_numpy(),
                rtol=1e-12,
            )

    def test_result_order_follows_the_input_list(self, inputs):
        fs = [_formulation(id="z"), _formulation(id="a")]
        assert list(run_formulations(inputs, fs, n_jobs=2, backend="threading")) == ["z", "a"]
