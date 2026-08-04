"""The dengue formulations we are currently trying.

This is the editable list — the axes live in
:mod:`idd_forecast_mbp.lib.modeling.dengue_formulations`, the choices live here.

Nothing here is settled. There is no final dengue model and no settled model
*structure*, so the point of this file is to make several structurally different
candidates runnable end to end and comparable, not to encode a winner.

The covariate specs come from ``03_modeling/final_models_dengue.r`` and the
``spec_A``..``spec_D`` block in
``reports/03_modeling/pygam_dengue_models_explore.ipynb``. Note the R file — which
the planning prompt calls the settled specification — carries **no year term**,
while the better-performing notebook formulations do. That disagreement is real
and unresolved; both are represented below so the comparison can settle it.
"""

from __future__ import annotations

from idd_forecast_mbp.lib.modeling import specs as specs_mod
from idd_forecast_mbp.lib.modeling.anchor import AnchorSpec, Baseline, Eligibility
from idd_forecast_mbp.lib.modeling.dengue_formulations import (
    CFR,
    INCIDENCE,
    MORTALITY,
    Formulation,
    validate_unique_ids,
)

ANCHOR_YEAR = 2023
ANCHOR_WINDOW = (2019, 2020, 2021, 2022, 2023)

#: Pin to the single anchor year — the first-submission behaviour.
ANCHOR_POINT = AnchorSpec.point(ANCHOR_YEAR)

#: Median over the window. More robust to a spiky single year, but the dengue
#: window spans COVID, which is exactly what the outlier filter below is for.
ANCHOR_MEDIAN = AnchorSpec.median_diff(ANCHOR_WINDOW)

#: Median over the window, dropping years more than 2 SD from a leave-one-out
#: linear expectation. `trend` rather than `mean` because dengue series trend,
#: and a leave-one-out mean would flag the window's endpoints on trend alone.
ANCHOR_ROBUST = AnchorSpec(
    years=ANCHOR_WINDOW,
    baseline=Baseline(statistic="median"),
    eligibility=Eligibility(method="trend", threshold_sd=2.0),
)

# --- covariate blocks -------------------------------------------------------

_CLIMATE = (
    specs_mod.Term("dengue_suitability", "mpi", 6),
    specs_mod.Term("urban_fraction", "linear"),
    specs_mod.Term("relative_humidity", "linear"),
)
_COUNTRY = (specs_mod.Term("A0_af", "factor"),)
_YEAR = (specs_mod.Term("year_id", "linear"),)

#: CFR as specified in final_models_dengue.r::mod_cfr_all, minus the as_id term —
#: as_id belongs in the fit (with_as_id), never the prediction spec.
_CFR_TERMS = (specs_mod.Term("log_gdppc_mean", "linear"), *_COUNTRY)

#: Mortality regressed directly, for the structures that do not route through CFR.
_MORT_TERMS = (*_CLIMATE, specs_mod.Term("log_gdppc_mean", "linear"), *_COUNTRY)


def build_formulations() -> list[Formulation]:
    """The current candidate set, one per structural question we want answered."""
    formulations = [
        Formulation(
            id="d1_inc_cfr",
            description=(
                "First-submission structure: regress incidence and CFR, derive "
                "mortality. No year term. Point anchor at 2023."
            ),
            structure="inc_cfr",
            terms={INCIDENCE: (*_CLIMATE, *_COUNTRY), CFR: _CFR_TERMS},
            anchor=ANCHOR_POINT,
        ),
        Formulation(
            id="d2_inc_cfr_year",
            description=(
                "As d1 plus a linear year term. Tests how much of the trajectory "
                "the year term drives — it does NOT cancel under the anchor."
            ),
            structure="inc_cfr",
            terms={INCIDENCE: (*_CLIMATE, *_YEAR, *_COUNTRY), CFR: _CFR_TERMS},
            anchor=ANCHOR_POINT,
            year_term="linear",
        ),
        Formulation(
            id="d3_inc_cfr_robust_anchor",
            description=(
                "As d1 but anchored on the median of 2019-2023 with leave-one-out "
                "trend outlier exclusion. Tests anchor sensitivity across COVID."
            ),
            structure="inc_cfr",
            terms={INCIDENCE: (*_CLIMATE, *_COUNTRY), CFR: _CFR_TERMS},
            anchor=ANCHOR_ROBUST,
        ),
        Formulation(
            id="d4_inc_mort",
            description="Regress incidence and mortality independently; no CFR.",
            structure="inc_mort",
            terms={INCIDENCE: (*_CLIMATE, *_COUNTRY), MORTALITY: _MORT_TERMS},
            anchor=ANCHOR_POINT,
        ),
        Formulation(
            id="d5_mort_then_inc",
            description=(
                "Regress mortality, then incidence using predicted mortality as a "
                "covariate. The coupled structure."
            ),
            structure="mort_then_inc",
            terms={
                MORTALITY: _MORT_TERMS,
                INCIDENCE: (*_CLIMATE, *_COUNTRY,
                            specs_mod.Term("log_dengue_mort_rate_pred", "linear")),
            },
            anchor=ANCHOR_POINT,
        ),
        Formulation(
            id="d6_mort_cfr",
            description=(
                "Regress mortality and CFR, derive incidence. Mirror image of d1."
            ),
            structure="mort_cfr",
            terms={MORTALITY: _MORT_TERMS, CFR: _CFR_TERMS},
            anchor=ANCHOR_POINT,
        ),
    ]
    validate_unique_ids(formulations)
    return formulations


#: Same specs fitted with `as_id` in the design matrix. It does not change the
#: age/sex pattern — that always comes from the observed anchor — but it de-biases
#: the coefficients on the time-varying covariates, which does move the forecast.
def build_as_id_variants() -> list[Formulation]:
    """The `with_as_id=True` counterpart of every formulation above."""
    variants = [f.variant("asid", with_as_id=True) for f in build_formulations()]
    validate_unique_ids(variants)
    return variants
