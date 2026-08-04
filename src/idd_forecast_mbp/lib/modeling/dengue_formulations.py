"""What a dengue formulation *is*, and what it promises the forecaster.

There is no final dengue model and no settled model *structure*. The pipeline
therefore has to carry several structurally different formulations all the way
through to full predictions, so the structural choices are recorded per
formulation and read back downstream — never hardcoded. This module is that
contract.

The outcome structures
----------------------
Four ways to get incidence and mortality, which are genuinely different models
rather than options on one model:

``inc_cfr``
    Regress incidence and CFR; derive mortality as ``incidence x CFR``. The
    first-submission structure (``04_forecasting/OLD_rake_dengue.py``).
``inc_mort``
    Regress incidence and mortality independently.
``mort_then_inc``
    Regress mortality, then regress incidence *using predicted mortality as a
    covariate*. Ordered and coupled, not independent.
``mort_cfr``
    Regress mortality and CFR; derive incidence as ``mortality / CFR``. The
    mirror image of ``inc_cfr``.

So "mortality = incidence x CFR" is a property of *one* structure, not an
invariant of dengue.

What is deliberately NOT an axis
--------------------------------
The age/sex *mechanism* is not a structural choice, because under a
per-``(location, age, sex)`` anchor the fitted ``as_id`` term cancels out of the
forecast exactly — see ``.claude/DECISIONS.md`` 2026-08-03. Whether ``as_id`` sat
in the design matrix changes the *fitted covariate coefficients* (it de-biases
them), so it is recorded as :attr:`Formulation.with_as_id`, a fit-quality flag.
It does not change the age/sex pattern, which always comes from the observed
anchor, and it must never widen the prediction frame.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from idd_forecast_mbp.lib.modeling import specs as specs_mod
from idd_forecast_mbp.lib.modeling.anchor import AnchorSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Outcomes a formulation can regress or derive.
INCIDENCE = "incidence"
MORTALITY = "mortality"
CFR = "cfr"


@dataclass(frozen=True)
class Structure:
    """One outcome structure: what gets regressed, what gets derived, in what order.

    ``fits`` is ordered — ``mort_then_inc`` needs mortality before incidence
    because the incidence model consumes it.
    """

    key: str
    fits: tuple[str, ...]
    derives: str
    description: str
    #: Outcome fed into a later model as a covariate, if any.
    feeds_forward: str | None = None


STRUCTURES: dict[str, Structure] = {
    s.key: s
    for s in (
        Structure(
            key="inc_cfr",
            fits=(INCIDENCE, CFR),
            derives=MORTALITY,
            description="Regress incidence and CFR; mortality = incidence x CFR.",
        ),
        Structure(
            key="inc_mort",
            fits=(INCIDENCE, MORTALITY),
            derives="",
            description="Regress incidence and mortality independently.",
        ),
        Structure(
            key="mort_then_inc",
            fits=(MORTALITY, INCIDENCE),
            derives="",
            feeds_forward=MORTALITY,
            description=(
                "Regress mortality, then incidence using predicted mortality as a "
                "covariate."
            ),
        ),
        Structure(
            key="mort_cfr",
            fits=(MORTALITY, CFR),
            derives=INCIDENCE,
            description="Regress mortality and CFR; incidence = mortality / CFR.",
        ),
    )
}

#: Model-space response per outcome. Sets the space the anchor is applied in:
#: log for rates, logit for CFR.
RESPONSE_COLUMN = {
    INCIDENCE: "log_dengue_inc_rate",
    MORTALITY: "log_dengue_mort_rate",
    CFR: "logit_dengue_cfr",
}

LINK = {INCIDENCE: "log", MORTALITY: "log", CFR: "logit"}

_GRAINS = ("fhs", "lsae")
_RESPONSE_GRAINS = ("base_cell", "all_age", "age_sex")

#: Where the age/sex pattern of the result comes from.
AGE_SEX_FROM_ANCHOR = "anchor"            # each cell lands on its own observed anchor
AGE_SEX_FROM_RELATIVE_RISK = "relative_risk"   # base rate x observed rr, unrescaled
AGE_SEX_FROM_REDISTRIBUTION = "redistribute"   # all-age total split by rr shares
_ENGINES = ("pygam", "R")


@dataclass(frozen=True)
class Formulation:
    """A complete, runnable specification — fit through to prediction.

    Parameters
    ----------
    terms:
        Per-outcome model terms, keyed by outcome. Only the outcomes the
        structure fits need an entry.
    anchor_group:
        Granularity of the anchor. Leaving this at
        ``("location_id", "age_group_id", "sex_id")`` is what makes a
        time-constant term cancel; an all-age-only anchor breaks that and
        reinstates a genuine age/sex prediction.
    with_as_id:
        Whether ``as_id`` was in the design matrix at fit time. Affects the
        fitted covariate coefficients only — never the prediction frame.
    year_term:
        Whether the model carries a time term, and in what form. ``None``, a
        global ``"linear"`` year, or ``"by_super_region"`` slopes. Recorded
        because a year term does NOT cancel under the anchor — it is
        time-varying by construction and so drives the forecast trajectory
        directly, and because it is the covariate a future decay would reshape.
    engine:
        Which language fitted this. Both are supported and coexist in the
        registry.
    """

    id: str
    description: str
    structure: str
    terms: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    grain: str = "fhs"
    response_grain: dict[str, str] = field(default_factory=dict)
    anchor: AnchorSpec | None = field(default_factory=lambda: AnchorSpec.point(2023))
    anchor_group: tuple[str, ...] = ("location_id", "age_group_id", "sex_id")
    reference_age_group_id: int = 3
    reference_sex_id: int = 1
    with_as_id: bool = False
    year_term: str | None = None
    #: Fit one model per super-region instead of one pooled model.
    #:
    #: The reason this exists: a per-super-region NONLINEAR time effect cannot be
    #: had from a pooled pyGAM fit. Masking year to 0 outside a super-region works
    #: for a linear term (out-of-group rows contribute beta*0 regardless of beta,
    #: so they carry no information about it) but NOT for a spline, because s(0) is
    #: not zero -- 71-98% of rows then pile onto a single x value in the middle of
    #: the basis carrying other super-regions' responses. Fitting separately is the
    #: only way to get a genuine per-group smooth here. The cost is no pooling:
    #: High-income has 168 base-cell rows, so its fit is thin.
    by_super_region: bool = False
    engine: str = "pygam"

    def __post_init__(self) -> None:
        if self.structure not in STRUCTURES:
            msg = f"structure must be one of {tuple(STRUCTURES)}; got {self.structure!r}"
            raise ValueError(msg)
        if self.grain not in _GRAINS:
            msg = f"grain must be one of {_GRAINS}; got {self.grain!r}"
            raise ValueError(msg)
        if self.engine not in _ENGINES:
            msg = f"engine must be one of {_ENGINES}; got {self.engine!r}"
            raise ValueError(msg)
        if self.year_term not in (None, "linear", "by_super_region"):
            msg = (f"year_term must be None, 'linear' or 'by_super_region'; "
                   f"got {self.year_term!r}")
            raise ValueError(msg)
        bad = {o: g for o, g in self.response_grain.items()
               if g not in _RESPONSE_GRAINS}
        if bad:
            msg = f"response_grain values must be in {_RESPONSE_GRAINS}; got {bad}"
            raise ValueError(msg)
        unknown = set(self.response_grain) - set(self.spec.fits)
        if unknown:
            msg = (f"formulation {self.id!r} sets response_grain for {sorted(unknown)}, "
                   f"which structure {self.structure!r} does not fit")
            raise ValueError(msg)
        missing = set(self.spec.fits) - set(self.terms)
        if missing:
            msg = (f"formulation {self.id!r} uses structure {self.structure!r}, "
                   f"which fits {self.spec.fits}, but has no terms for "
                   f"{sorted(missing)}")
            raise ValueError(msg)

    @property
    def spec(self) -> Structure:
        """The outcome structure this formulation uses."""
        return STRUCTURES[self.structure]

    @property
    def fitted_outcomes(self) -> tuple[str, ...]:
        """Outcomes that get their own regression, in fit order."""
        return self.spec.fits

    @property
    def derived_outcome(self) -> str | None:
        """The outcome computed from the others, if any."""
        return self.spec.derives or None

    def grain_for(self, outcome: str) -> str:
        """Response grain for one outcome, defaulting to the base age/sex cell.

        Per OUTCOME, not per formulation: a structure can legitimately regress
        incidence on the base cell while regressing CFR across the full age/sex
        grid, which is exactly what the first-submission shape does.
        """
        return self.response_grain.get(outcome, "base_cell")

    @property
    def is_raked(self) -> bool:
        """Whether predictions are shifted onto observed at all.

        ``anchor=None`` is the no-rake variant: the model's own level stands, which
        is the only configuration that shows how much of the fit the anchor was
        doing versus the covariates.
        """
        return self.anchor is not None

    @property
    def age_sex_source(self) -> str:
        """Where the age/sex pattern comes from — derived, not configured.

        The three cases are forced by the other choices rather than chosen freely:

        - An **all-age** response has no per-cell prediction, so age/sex can only
          come from redistributing the all-age total by observed shares.
        - A **base-cell** response with **no anchor** has nothing observed to land
          on, so age/sex comes from the observed relative risks directly.
        - A **base-cell** response **with** a per-cell anchor gets age/sex from the
          anchor itself, which is why a time-constant ``as_id`` term cancels.

        Note redistribution and the relative-risk broadcast produce identical age/sex
        *shares* — ``pop * rr / sum(pop * rr)`` either way. They differ only in what
        sets the all-age total: the all-age model, or ``exp(base) * sum(pop * rr)``.
        """
        grains = {self.grain_for(o) for o in self.fitted_outcomes}
        if grains == {"all_age"}:
            return AGE_SEX_FROM_REDISTRIBUTION
        if not self.is_raked:
            return AGE_SEX_FROM_RELATIVE_RISK
        return AGE_SEX_FROM_ANCHOR

    @property
    def anchors_by_age_sex(self) -> bool:
        """Whether the anchor is per age/sex cell — the cancellation precondition.

        False when there is no anchor at all: nothing cancels if nothing shifts.
        """
        if self.anchor is None:
            return False
        return {"age_group_id", "sex_id"} <= set(self.anchor_group)

    def covariates(self, outcome: str) -> tuple[str, ...]:
        """Column names the model for ``outcome`` reads.

        This is what determines the hold set: a covariate that is not in any
        fitted model does not need a hold arm. Resolved through
        :func:`specs.spec_columns`, which is the canonical accessor and also
        handles tensor and group terms — reaching into a term's attributes by
        hand gets the field name wrong and silently yields term objects.
        """
        terms = self.terms.get(outcome, ())
        if not terms:
            return ()
        return tuple(specs_mod.spec_columns(terms))

    @property
    def hold_covariates(self) -> tuple[str, ...]:
        """Every covariate across the fitted models, deduplicated, in order.

        Holds are derived from the formulation rather than fixed in advance — if
        urban never enters the final regressions there is no urban hold to run.
        """
        seen: dict[str, None] = {}
        for outcome in self.fitted_outcomes:
            for column in self.covariates(outcome):
                seen.setdefault(column, None)
        return tuple(seen)

    def registry_record(self, run_date: str, *, best: bool = False) -> dict[str, Any]:
        """Metadata the forecaster needs to rebuild the prediction frame.

        Key convention and the ``best`` flag match the malaria registry
        (``lib/model_registry.R``) so the two stay readable by the same tooling.
        Unifying them into one cause-parameterised registry is a separate,
        coordinated change — malaria's is live and must not be disturbed here.
        """
        return {
            "run_date": f"{run_date}_{self.id}",
            "best": best,
            "description": self.description,
            "cause": "dengue",
            "structure": self.structure,
            "fits": list(self.fitted_outcomes),
            "derives": self.derived_outcome,
            "grain": self.grain,
            "engine": self.engine,
            "with_as_id": self.with_as_id,
            "year_term": self.year_term,
            "reference_age_group_id": self.reference_age_group_id,
            "reference_sex_id": self.reference_sex_id,
            "response_grain": dict(self.response_grain),
            "age_sex_source": self.age_sex_source,
            "raked": self.is_raked,
            "anchor_group": list(self.anchor_group) if self.is_raked else None,
            "anchor_years": list(self.anchor.years) if self.anchor else None,
            "anchor_statistic": self.anchor.baseline.statistic if self.anchor else None,
            "anchor_applied_to": self.anchor.applied_to if self.anchor else None,
            "anchor_eligibility": (self.anchor.eligibility.method
                                   if self.anchor else None),
            "hold_covariates": list(self.hold_covariates),
        }

    def variant(self, id_suffix: str, **overrides: Any) -> Formulation:
        """A copy with fields replaced — for sweeping one axis at a time."""
        return replace(self, id=f"{self.id}_{id_suffix}", **overrides)


def validate_unique_ids(formulations: Sequence[Formulation]) -> None:
    """Raise if two formulations share an id — they would collide in the registry."""
    counts = Counter(f.id for f in formulations)
    duplicates = sorted(name for name, n in counts.items() if n > 1)
    if duplicates:
        msg = f"duplicate formulation id(s): {duplicates}"
        raise ValueError(msg)
