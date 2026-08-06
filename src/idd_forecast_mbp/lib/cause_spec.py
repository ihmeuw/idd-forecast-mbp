"""What differs between causes, so the pipeline code itself does not.

Stages 03 onward — fit, forecast, products, figures — are the same operations for malaria and
dengue. The differences are *parameters*: which grain the model is fitted at, how the anchor is
defined, which locations are eligible, which outcomes exist and in what order. Historically those
parameters were embedded in cause-specific modules, which is why ``lib/`` accumulated six forked
``dengue_*`` files: there was no cause-blind object to hook into, so forking was the only way to
proceed.

This module is that object. A stage script takes a :class:`CauseSpec` and stops mentioning
malaria or dengue.

Design rules learned the hard way (see ``.claude/DENGUE_CONSULT_ROUND3.md``):

**Alternative futures are NOT in here.** A covariate may have several supplied future
trajectories — malaria's DAH scenarios, dengue's time-decay functions, any covariate held at its
2023 value — and a run may be executed across them. That is a property of a *run*, so it lives in
the run manifest. Three successive attempts to model it here (``has_dah: bool``, then an
``AxisSpec`` secondary axis, then an ``EligibilityRule`` taxonomy) were each a real observation
promoted into a structural type; each time the honest answer was "that is a parameter". So
:attr:`CauseSpec.covariates` holds covariate *names* and nothing about their futures.

**Nothing here names one cause's choice as the default.** ``persist_as_draws_admin2: bool`` was
rejected for exactly this: dengue does need age/sex draws persisted, just at FHS grain rather than
admin-2, so the honest boolean was ``False`` — which reads as "dengue needs no age/sex draws", the
opposite of true. It is a grain, and deliberately independent of :attr:`CauseSpec.fit_grain`,
because a cause may be fitted at one grain and persisted at another.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from idd_forecast_mbp import constants as mbpc


class Grain(str, Enum):
    """The location grain a cause is fitted or persisted at.

    ``ADMIN2`` is a uniform level-5 leaf set. ``FHS_MOST_DETAILED`` is *mixed level* — its 473
    leaves span level 3 (countries with no subnational detail) and level 4 — which is why any
    roll-up over it must use ``roll_up_to_ancestors`` rather than a fixed ``start_level``.
    """

    ADMIN2 = "admin2"
    FHS_MOST_DETAILED = "fhs_most_detailed"

    @property
    def is_mixed_level(self) -> bool:
        return self is Grain.FHS_MOST_DETAILED

    @property
    def uniform_level(self) -> int | None:
        """The single hierarchy level, or None when the grain spans several."""
        return 5 if self is Grain.ADMIN2 else None


class AnchorKind(str, Enum):
    """How a forecast is tied to observed history."""

    #: The anchor year's prediction is forced to equal that year's observed value.
    POINT = "point"
    #: The prediction is tied to the mean of observed values over a window of years.
    WINDOW_MEAN = "window_mean"


@dataclass(frozen=True)
class AnchorSpec:
    """How predictions are tied to observed data, and what "correct" therefore means.

    The reason this is not just an integer year: the two kinds imply *different correctness
    checks*, and using the wrong one is actively misleading. For a point anchor, the prediction
    must equal observed at that year and any deviation is a bug. For a window mean it must equal
    the window's mean, and equality with any single year — including the last one — is not
    expected. Dengue's F4 lands within 4% of its 2014–2023 observed mean while sitting 45% below
    observed 2023, because 2023 was an epidemic spike the decade mean deliberately does not chase.
    Checked against observed-2023 that correct run looks catastrophic; checked against its actual
    target it passes.

    Use :meth:`target` to get the thing a prediction should be compared against, and never
    hand-roll "observed at ``years[-1]``".
    """

    kind: AnchorKind
    years: tuple[int, ...]
    #: Leave-one-out outlier eligibility in SD, when the window screens outliers. Dengue: 10.0.
    outlier_sd: float | None = None

    def __post_init__(self) -> None:
        if not self.years:
            raise ValueError("AnchorSpec.years must not be empty")
        if len(set(self.years)) != len(self.years):
            raise ValueError(f"AnchorSpec.years has duplicates: {self.years}")
        if self.kind is AnchorKind.POINT and len(self.years) != 1:
            raise ValueError(
                f"a point anchor needs exactly one year, got {self.years}; use WINDOW_MEAN "
                "for a multi-year baseline"
            )
        if self.kind is AnchorKind.WINDOW_MEAN and len(self.years) < 2:
            raise ValueError(
                f"a window-mean anchor needs at least two years, got {self.years}; use POINT "
                "for a single-year anchor"
            )
        if self.outlier_sd is not None and self.outlier_sd <= 0:
            raise ValueError(f"outlier_sd must be positive, got {self.outlier_sd}")

    @property
    def reproduces_observed(self) -> bool:
        """True when a correct run reproduces an observed *year* exactly.

        Only point anchors do. Gate the hard anchor check on this; when it is False the right
        artifact is a diagnostic against :meth:`target`, not a pass/fail on observed equality.
        """
        return self.kind is AnchorKind.POINT

    @property
    def label(self) -> str:
        """Human-readable description, for figure annotation."""
        if self.kind is AnchorKind.POINT:
            return f"anchored to observed {self.years[0]}"
        return f"anchored to the {min(self.years)}–{max(self.years)} observed mean"

    def target(
        self,
        observed: pd.DataFrame,
        value_col: str,
        *,
        group_cols: Sequence[str] = ("location_id",),
        year_col: str = "year_col_placeholder",
    ) -> pd.DataFrame:
        """The value a prediction should be compared against, per group.

        For ``POINT`` this is the observed value at the anchor year. For ``WINDOW_MEAN`` it is
        the mean over the window's years, computed only from years actually present so a partial
        window degrades rather than silently returning NaN.

        Returns ``group_cols + ["anchor_target", "n_years"]``. ``n_years`` is how many of the
        window's years contributed, which is what tells you a target is thin.
        """
        year_col = "year_id" if year_col == "year_col_placeholder" else year_col
        cols = list(group_cols)
        want = {int(y) for y in self.years}
        sub = observed[observed[year_col].isin(want)]
        if sub.empty:
            raise ValueError(
                f"no observed rows in anchor years {sorted(want)}; cannot form an anchor target"
            )
        # For a point anchor the mean of one year IS that year, so one code path serves both.
        return (
            sub.groupby(cols, observed=True)[value_col]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "anchor_target", "count": "n_years"})
        )


@dataclass(frozen=True)
class BurdenFilter:
    """Which locations are eligible to be fitted: those with some observed burden.

    Both causes do this and it is one rule with one parameter. Malaria keeps
    ``malaria_inc_count >= 0``, which is inert on a non-negative count and therefore keeps
    everything; dengue keeps ``dengue_inc_count > 0``, which drops the 168 of 473 FHS locations
    with no observed dengue. The only thing distinguishing them is whether the comparison is
    strict, hence one bool rather than a taxonomy of filter kinds.
    """

    column: str
    threshold: float = 0.0
    #: True -> ``> threshold`` (drops rows exactly at it). False -> ``>= threshold``.
    strict: bool = False

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """Boolean mask of eligible rows. Raises if the column is absent."""
        if self.column not in df.columns:
            raise KeyError(
                f"burden filter column {self.column!r} not in frame; have {list(df.columns)[:12]}"
            )
        col = df[self.column]
        return col > self.threshold if self.strict else col >= self.threshold

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.mask(df)]

    @property
    def is_inert(self) -> bool:
        """True when the filter cannot exclude any non-negative row."""
        return not self.strict and self.threshold <= 0

    @property
    def description(self) -> str:
        return f"{self.column} {'>' if self.strict else '>='} {self.threshold:g}"


@dataclass(frozen=True)
class MeasureSpec:
    """One modelled outcome, and what it is derived from.

    ``depends_on`` carries feed-forward: dengue predicts mortality and then uses it as a
    covariate for incidence, or combines incidence with a case-fatality ratio to get mortality.
    Malaria's two measures are independent, so their ``depends_on`` is empty.
    """

    name: str
    count_col: str | None
    rate_col: str | None
    #: Names of measures this one consumes. Must refer to measures in the same structure.
    depends_on: tuple[str, ...] = ()
    #: True for quantities that are neither counts nor rates and are not delivered (e.g. CFR).
    intermediate: bool = False

    def __post_init__(self) -> None:
        if not self.intermediate and not (self.count_col and self.rate_col):
            raise ValueError(
                f"measure {self.name!r} is deliverable so it needs both count_col and rate_col; "
                "set intermediate=True for a derived quantity like CFR"
            )
        if self.name in self.depends_on:
            raise ValueError(f"measure {self.name!r} depends on itself")


@dataclass(frozen=True)
class MeasureStructure:
    """The cause's outcomes as a dependency structure, not a flat list.

    A flat list cannot express that one outcome feeds another, and evaluating them in the wrong
    order silently uses an unpopulated covariate. :meth:`order` returns a topological order and
    refuses to guess when the dependencies are cyclic.
    """

    measures: tuple[MeasureSpec, ...]

    def __post_init__(self) -> None:
        if not self.measures:
            raise ValueError("MeasureStructure needs at least one measure")
        names = [m.name for m in self.measures]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate measure names: {names}")
        known = set(names)
        for m in self.measures:
            missing = set(m.depends_on) - known
            if missing:
                raise ValueError(
                    f"measure {m.name!r} depends on unknown measure(s) {sorted(missing)}"
                )
        self.order()  # fail at construction, not at first use, if cyclic

    def __iter__(self):
        return iter(self.measures)

    def __len__(self) -> int:
        return len(self.measures)

    def __getitem__(self, name: str) -> MeasureSpec:
        for m in self.measures:
            if m.name == name:
                return m
        raise KeyError(f"no measure named {name!r}; have {[m.name for m in self.measures]}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.measures)

    @property
    def deliverable(self) -> tuple[MeasureSpec, ...]:
        """Measures that become products; excludes intermediates such as CFR."""
        return tuple(m for m in self.measures if not m.intermediate)

    def order(self) -> tuple[str, ...]:
        """Names in an order where every dependency precedes its consumer."""
        remaining = {m.name: set(m.depends_on) for m in self.measures}
        out: list[str] = []
        while remaining:
            ready = sorted(n for n, deps in remaining.items() if not deps - set(out))
            if not ready:
                raise ValueError(
                    f"cyclic measure dependencies among {sorted(remaining)}; cannot order"
                )
            out.extend(ready)
            for n in ready:
                remaining.pop(n)
        return tuple(out)


@dataclass(frozen=True)
class CauseSpec:
    """What is true of a cause *regardless of which run you are doing*.

    The admission test for a field here is: **would it be the same for every run of this
    cause?** If a run could legitimately change it, it belongs to the run, not the cause.

    That test excludes more than it first appears, and excluding it is what makes this
    extensible. Anything on the *refit* list — formulation, engine, anchor spec, reference
    cell, suitability variant, burden threshold — varies run to run by definition, since
    changing it is what forces a refit. So none of them are here, and adding a covariate or a
    variant never means editing this class.

    Deliberately absent, with reasons:

    ``covariates``
        Which covariates a run uses is set by its *formulation*, and malaria's model selection
        exists precisely to vary that. Covariate metadata already has a canonical owner in
        ``lib/io/covariate_registry.COVARIATE_REGISTRY``; duplicating names here would be a
        second source of truth that goes stale the moment a formulation changes.
    ``anchor`` / ``measures`` / ``reference_*`` / burden threshold
        All on the refit list — see above. They are real, they are just per-run. The types
        (:class:`AnchorSpec`, :class:`MeasureStructure`, :class:`BurdenFilter`) live in this
        module and a run spec instantiates them.
    alternative covariate futures
        DAH scenarios, decay functions, holds. Per-run by construction; see module docstring.
    """

    name: str
    cause_id: int

    #: Permanent data property, not a run choice: dengue has no genuine admin-2 data (it was
    #: distributed down from national using suitability), so it is fitted at FHS most-detailed.
    fit_grain: Grain

    #: Which column defines "has burden" for this cause. The *threshold* applied to it is a run
    #: choice and lives in the run spec; only the column name is cause-invariant.
    burden_column: str

    #: True when a location absent from the output means zero, so consumers fill 0 rather than
    #: inheriting NaN from a hierarchy join. Dengue's 168 excluded FHS locations are the case
    #: this exists for.
    absent_means_zero: bool

    raked_aa_read_path: Path
    past_inputs_path: Path
    forecast_inputs_path: Path

    #: ``run_key -> products dir``; the products root is per-run, so this is a callable.
    products_read_path: object

    #: The one genuinely cause-specific behaviour: at which grain age/sex draw-level output must
    #: be persisted, or None if it need not be. Independent of :attr:`fit_grain` by design.
    as_draw_persist_grain: Grain | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CauseSpec.name must not be empty")
        if not self.burden_column:
            raise ValueError(f"{self.name}: burden_column must not be empty")

    def default_burden_filter(self, threshold: float = 0.0, *, strict: bool = False):
        """A :class:`BurdenFilter` on this cause's burden column.

        Convenience only. The threshold and strictness are run choices, so a run spec should
        pass its own values rather than relying on these defaults.
        """
        return BurdenFilter(self.burden_column, threshold=threshold, strict=strict)

    def product_filename(self, ssp_scenario: str, trajectory: str) -> str:
        """Product filename for one ``(ssp, trajectory)`` arm.

        ``trajectory`` is the selected covariate-trajectory value for this arm, named by the run
        manifest — ``Baseline``/``Constant`` for malaria's DAH, ``no_decay``/``logistic_k8`` for
        dengue's decay. Putting it in the path rather than in a column follows the house rule
        against storing a redundant constant column.
        """
        if not ssp_scenario or not trajectory:
            raise ValueError("both ssp_scenario and trajectory are required")
        return f"all_age_summary_{ssp_scenario}_{trajectory}.parquet"

    def anchor_filename(
        self, ssp_scenario: str, trajectory: str, anchor: AnchorSpec
    ) -> str:
        """Anchor-diagnostic filename for one arm.

        The anchor is passed in rather than stored: it is on the refit list, so two runs of the
        same cause can legitimately use different anchors.
        """
        if not ssp_scenario or not trajectory:
            raise ValueError("both ssp_scenario and trajectory are required")
        return f"anchor_{min(anchor.years)}_{ssp_scenario}_{trajectory}.parquet"

    @property
    def uses_mixed_level_leaves(self) -> bool:
        """True when a roll-up must use ancestors rather than a fixed start level."""
        return self.fit_grain.is_mixed_level


def _malaria() -> CauseSpec:
    return CauseSpec(
        name="malaria",
        cause_id=mbpc.malaria_id,
        fit_grain=Grain.ADMIN2,
        burden_column="malaria_inc_count",
        absent_means_zero=False,
        raked_aa_read_path=mbpc.MAL_RAKED_AA_READ_PATH,
        past_inputs_path=mbpc.MAL_PAST_INPUTS_READ_PATH,
        forecast_inputs_path=mbpc.MAL_FORECAST_INPUTS_READ_PATH,
        products_read_path=mbpc.mal_products_read_path,
        as_draw_persist_grain=Grain.ADMIN2,
    )


def _dengue() -> CauseSpec:
    return CauseSpec(
        name="dengue",
        cause_id=mbpc.dengue_id,
        fit_grain=Grain.FHS_MOST_DETAILED,
        burden_column="dengue_inc_count",
        absent_means_zero=True,
        raked_aa_read_path=mbpc.DEN_RAKED_AA_READ_PATH,
        past_inputs_path=mbpc.DEN_PAST_INPUTS_READ_PATH,
        forecast_inputs_path=mbpc.DEN_FORECAST_INPUTS_READ_PATH,
        products_read_path=mbpc.den_products_read_path,
        as_draw_persist_grain=Grain.FHS_MOST_DETAILED,
    )


#: Registry. Values are built lazily by :func:`get_cause_spec` so importing this module does not
#: touch the filesystem via the artifact ``current`` symlinks.
_BUILDERS = {"malaria": _malaria, "dengue": _dengue}


def get_cause_spec(name: str) -> CauseSpec:
    """The :class:`CauseSpec` for ``malaria`` or ``dengue``."""
    key = name.strip().lower()
    if key not in _BUILDERS:
        raise ValueError(f"unknown cause {name!r}; known: {sorted(_BUILDERS)}")
    return _BUILDERS[key]()


def known_causes() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))
