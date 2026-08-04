"""Tests for :mod:`idd_forecast_mbp.lib.modeling.dengue_formulations`."""

from __future__ import annotations

import pytest

from idd_forecast_mbp.lib.modeling import specs as specs_mod
from idd_forecast_mbp.lib.modeling.anchor import AnchorSpec, Baseline, Eligibility
from idd_forecast_mbp.lib.modeling.dengue_formulations import (
    CFR,
    INCIDENCE,
    LINK,
    MORTALITY,
    RESPONSE_COLUMN,
    STRUCTURES,
    Formulation,
    validate_unique_ids,
)


def _terms(*columns: str) -> tuple:
    """Real spec terms — a fake would not catch the field-name mismatch."""
    return tuple(specs_mod.Term(c, "linear") for c in columns)


def _formulation(**overrides) -> Formulation:
    base = {
        "id": "d1",
        "description": "test formulation",
        "structure": "inc_cfr",
        "terms": {
            INCIDENCE: _terms("dengue_suitability", "relative_humidity"),
            CFR: _terms("log_gdppc_mean"),
        },
    }
    return Formulation(**{**base, **overrides})


class TestStructures:
    def test_all_four_structures_are_registered(self):
        assert set(STRUCTURES) == {"inc_cfr", "inc_mort", "mort_then_inc", "mort_cfr"}

    @pytest.mark.parametrize(
        ("key", "fits", "derives"),
        [
            ("inc_cfr", (INCIDENCE, CFR), MORTALITY),
            ("inc_mort", (INCIDENCE, MORTALITY), ""),
            ("mort_then_inc", (MORTALITY, INCIDENCE), ""),
            ("mort_cfr", (MORTALITY, CFR), INCIDENCE),
        ],
    )
    def test_each_structure_declares_what_it_fits_and_derives(self, key, fits, derives):
        s = STRUCTURES[key]
        assert s.fits == fits
        assert s.derives == derives

    def test_mort_then_inc_is_the_only_coupled_structure(self):
        coupled = {k for k, s in STRUCTURES.items() if s.feeds_forward}
        assert coupled == {"mort_then_inc"}
        assert STRUCTURES["mort_then_inc"].feeds_forward == MORTALITY

    def test_mortality_is_fitted_before_incidence_when_it_feeds_forward(self):
        """Fit order matters: incidence consumes predicted mortality."""
        fits = STRUCTURES["mort_then_inc"].fits
        assert fits.index(MORTALITY) < fits.index(INCIDENCE)

    def test_cfr_is_the_only_logit_outcome(self):
        assert LINK[CFR] == "logit"
        assert LINK[INCIDENCE] == LINK[MORTALITY] == "log"

    def test_every_outcome_has_a_response_column(self):
        for outcome in (INCIDENCE, MORTALITY, CFR):
            assert RESPONSE_COLUMN[outcome]


class TestFormulationValidation:
    def test_rejects_unknown_structure(self):
        with pytest.raises(ValueError, match="structure must be one of"):
            _formulation(structure="vibes")

    def test_rejects_unknown_grain(self):
        with pytest.raises(ValueError, match="grain must be one of"):
            _formulation(grain="admin_7")

    def test_rejects_unknown_engine(self):
        with pytest.raises(ValueError, match="engine must be one of"):
            _formulation(engine="fortran")

    def test_rejects_unknown_year_term(self):
        with pytest.raises(ValueError, match="year_term must be"):
            _formulation(year_term="quadratic")

    def test_rejects_missing_terms_for_a_fitted_outcome(self):
        """A structure that fits mortality needs mortality terms."""
        with pytest.raises(ValueError, match="has no terms for"):
            _formulation(structure="inc_mort")

    def test_accepts_terms_for_every_fitted_outcome(self):
        f = _formulation(
            structure="mort_then_inc",
            terms={MORTALITY: _terms("log_gdppc_mean"),
                   INCIDENCE: _terms("dengue_suitability")},
        )
        assert f.fitted_outcomes == (MORTALITY, INCIDENCE)


class TestFormulationProperties:
    def test_fitted_and_derived_outcomes(self):
        f = _formulation()
        assert f.fitted_outcomes == (INCIDENCE, CFR)
        assert f.derived_outcome == MORTALITY

    def test_structures_with_no_derived_outcome_report_none(self):
        f = _formulation(
            structure="inc_mort",
            terms={INCIDENCE: _terms("a"), MORTALITY: _terms("b")},
        )
        assert f.derived_outcome is None

    def test_covariates_per_outcome(self):
        f = _formulation()
        assert f.covariates(INCIDENCE) == ("dengue_suitability", "relative_humidity")
        assert f.covariates(CFR) == ("log_gdppc_mean",)

    def test_covariates_of_an_unfitted_outcome_is_empty(self):
        assert _formulation().covariates(MORTALITY) == ()

    def test_hold_covariates_span_every_fitted_model(self):
        f = _formulation()
        assert f.hold_covariates == (
            "dengue_suitability", "relative_humidity", "log_gdppc_mean",
        )

    def test_hold_covariates_are_deduplicated(self):
        """A covariate in two models yields one hold arm, not two."""
        f = _formulation(
            terms={INCIDENCE: _terms("log_gdppc_mean", "dengue_suitability"),
                   CFR: _terms("log_gdppc_mean")},
        )
        assert f.hold_covariates == ("log_gdppc_mean", "dengue_suitability")

    def test_a_covariate_outside_the_model_gets_no_hold(self):
        """Holds follow the final regressions — no urban term, no urban hold."""
        assert "urban_fraction" not in _formulation().hold_covariates

    def test_default_anchor_is_per_age_sex_cell(self):
        """The default must satisfy the cancellation precondition."""
        assert _formulation().anchors_by_age_sex is True

    def test_all_age_anchor_is_flagged_as_not_by_age_sex(self):
        f = _formulation(anchor_group=("location_id",))
        assert f.anchors_by_age_sex is False


class TestRegistryRecord:
    def test_key_convention_matches_malaria(self):
        rec = _formulation().registry_record("2026_08_03")
        assert rec["run_date"] == "2026_08_03_d1"

    def test_best_defaults_false(self):
        assert _formulation().registry_record("2026_08_03")["best"] is False

    def test_best_can_be_flagged(self):
        assert _formulation().registry_record("2026_08_03", best=True)["best"] is True

    def test_records_the_structural_choices_the_forecaster_reads_back(self):
        f = _formulation(with_as_id=True, year_term="by_super_region", engine="R",
                         grain="lsae", reference_age_group_id=4, reference_sex_id=2)
        rec = f.registry_record("2026_08_03")
        assert rec["structure"] == "inc_cfr"
        assert rec["fits"] == [INCIDENCE, CFR]
        assert rec["derives"] == MORTALITY
        assert rec["grain"] == "lsae"
        assert rec["engine"] == "R"
        assert rec["with_as_id"] is True
        assert rec["year_term"] == "by_super_region"
        assert rec["reference_age_group_id"] == 4
        assert rec["reference_sex_id"] == 2

    def test_records_the_anchor_configuration(self):
        f = _formulation(anchor=AnchorSpec(
            years=(2019, 2020, 2021, 2022, 2023),
            baseline=Baseline(statistic="median"),
            eligibility=Eligibility(method="trend"),
            applied_to="residual",
        ))
        rec = f.registry_record("2026_08_03")
        assert rec["anchor_years"] == [2019, 2020, 2021, 2022, 2023]
        assert rec["anchor_statistic"] == "median"
        assert rec["anchor_applied_to"] == "residual"
        assert rec["anchor_eligibility"] == "trend"

    def test_records_the_hold_set(self):
        rec = _formulation().registry_record("2026_08_03")
        assert rec["hold_covariates"] == [
            "dengue_suitability", "relative_humidity", "log_gdppc_mean",
        ]

    def test_record_is_json_serialisable(self):
        import json
        json.dumps(_formulation().registry_record("2026_08_03"))

    def test_cause_is_recorded(self):
        assert _formulation().registry_record("2026_08_03")["cause"] == "dengue"


class TestVariant:
    def test_variant_suffixes_the_id_and_overrides_a_field(self):
        v = _formulation().variant("lsae", grain="lsae")
        assert v.id == "d1_lsae"
        assert v.grain == "lsae"

    def test_variant_leaves_the_original_untouched(self):
        f = _formulation()
        f.variant("lsae", grain="lsae")
        assert f.grain == "fhs"
        assert f.id == "d1"

    def test_variant_can_sweep_the_as_id_axis(self):
        f = _formulation()
        assert f.with_as_id is False
        assert f.variant("asid", with_as_id=True).with_as_id is True


class TestValidateUniqueIds:
    def test_accepts_distinct_ids(self):
        validate_unique_ids([_formulation(id="a"), _formulation(id="b")])

    def test_rejects_a_collision(self):
        with pytest.raises(ValueError, match="duplicate formulation id"):
            validate_unique_ids([_formulation(id="a"), _formulation(id="a")])
