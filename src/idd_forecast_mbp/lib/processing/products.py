"""The product contract: what a finished all-age product must look like, and how to check it.

Two causes produce these files with two different producers — malaria via
``05_aggregation/finish_run.py``, dengue via its own F4 driver — and one consumer reads both
(``plot_run_comparison.py``). Sharing the *producer* was rejected as needlessly expensive; what
has to be shared is the **contract**. A contract with no enforcement drifts, so the schema and
its validator live together here and both producers call :func:`validate_products`.

The invariant that matters most, and the one a summary file can actually be checked against:

    An aggregate rate is a count divided by **that level's own population row** — never a sum of
    the children's populations, and never a population-weighted average of children's rates.

:func:`validate_products` re-derives every rate from its count and population and fails if they
disagree, which catches the whole family of denominator errors mechanically instead of by review.

The anchor check is deliberately *not* "does the run reproduce observed at the anchor year". That
question is only meaningful for a point anchor. For a window-mean anchor the correct target is the
window's mean, and a correct run can sit far from any single year: dengue's F4 lands within 4% of
its 2014–2023 observed mean while sitting 45% below observed 2023, because 2023 was an epidemic
spike the decade mean deliberately does not chase. Checked the wrong way that correct run looks
catastrophic — and, worse, a genuinely broken one can look fine. See
:func:`anchor_diagnostic`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from idd_forecast_mbp.lib.cause_spec import AnchorSpec, CauseSpec, MeasureStructure

#: Columns every product carries, whatever the cause.
BASE_COLUMNS: tuple[str, ...] = ("location_id", "year_id", "level", "population")

#: ID columns that must be integer-typed. A float ID in a parquet file is a bug at the source.
ID_COLUMNS: tuple[str, ...] = ("location_id", "year_id", "level")

#: Summary statistics carried for every measure, in both count and rate space.
STATS: tuple[str, ...] = ("mean", "lower", "upper")

#: Dimension names that must live in the PATH rather than as a constant column, per the house
#: rule against storing a redundant constant column.
PATH_DIMENSIONS: frozenset[str] = frozenset(
    {"ssp_scenario", "ssp", "dah_scenario", "dah", "trajectory", "decay", "cause", "cause_id"}
)

#: Relative tolerance when re-deriving a rate from its count and population.
RATE_RTOL = 1e-6

#: A point anchor should reproduce its target essentially exactly; anything above this is a bug.
POINT_ANCHOR_RTOL = 1e-3

#: A window-mean anchor has no exactness guarantee. Beyond this the run is worth investigating,
#: but it is a diagnostic threshold, not a correctness criterion.
WINDOW_ANCHOR_WARN_RTOL = 0.10


def measure_columns(measures: MeasureStructure) -> tuple[str, ...]:
    """Every summary column implied by a measure structure, in a stable order.

    Intermediates (CFR and similar) are excluded: they are not delivered, so they are not part
    of the product contract even though the model computes them.
    """
    out: list[str] = []
    for m in measures.deliverable:
        for base in (m.count_col, m.rate_col):
            out.extend(f"{base}_{stat}" for stat in STATS)
    return tuple(out)


def expected_columns(measures: MeasureStructure) -> tuple[str, ...]:
    """Full required column set: base columns plus every measure summary column."""
    return BASE_COLUMNS + measure_columns(measures)


@dataclass
class ValidationReport:
    """Everything wrong with one product, collected rather than raised one at a time.

    A producer wants the whole list in one run, not to fix-and-rerun once per violation.
    """

    label: str
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def fail(self, msg: str) -> None:
        self.problems.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def raise_if_failed(self) -> None:
        if not self.ok:
            joined = "\n  - ".join(self.problems)
            raise ValueError(f"{self.label} violates the product contract:\n  - {joined}")

    def summary(self) -> str:
        head = f"{self.label}: {'OK' if self.ok else f'{len(self.problems)} problem(s)'}"
        lines = [head]
        lines.extend(f"  PROBLEM {p}" for p in self.problems)
        lines.extend(f"  note    {n}" for n in self.notes)
        return "\n".join(lines)


def validate_products(
    df: pd.DataFrame,
    cause: CauseSpec,
    measures: MeasureStructure,
    *,
    label: str = "product",
    hierarchy: pd.DataFrame | None = None,
    expected_levels: Sequence[int] | None = None,
) -> ValidationReport:
    """Check one all-age product against the contract.

    Passing ``hierarchy`` additionally checks count additivity — that each parent's count equals
    the sum of its present children. Without it that check is skipped and noted, because it
    cannot be done from the file alone.
    """
    rep = ValidationReport(label=label)

    missing = [c for c in expected_columns(measures) if c not in df.columns]
    if missing:
        rep.fail(f"missing required columns: {missing}")
        return rep  # everything below assumes the columns exist

    if df.empty:
        rep.fail("product is empty")
        return rep

    _check_ids(df, rep)
    _check_no_draw_axis(df, rep)
    _check_redundant_constant_columns(df, rep)
    _check_levels(df, rep, expected_levels)
    _check_population(df, rep)
    _check_intervals(df, measures, rep)
    _check_rates_match_counts(df, measures, rep)
    _check_nonneg_counts(df, measures, rep)

    if hierarchy is None:
        rep.note("count additivity not checked (no hierarchy supplied)")
    else:
        _check_count_additivity(df, measures, hierarchy, rep)

    if cause.absent_means_zero:
        rep.note(
            "absent_means_zero=True: locations missing from this product mean ZERO, not unknown; "
            "consumers joining the full hierarchy must fill 0 rather than inherit NaN"
        )
    return rep


def _check_ids(df: pd.DataFrame, rep: ValidationReport) -> None:
    for col in ID_COLUMNS:
        if not pd.api.types.is_integer_dtype(df[col]):
            rep.fail(f"{col} must be an integer dtype, got {df[col].dtype}")
        if df[col].isna().any():
            rep.fail(f"{col} contains nulls")
    dupe = df.duplicated(subset=["location_id", "year_id"]).sum()
    if dupe:
        rep.fail(f"{dupe} duplicated (location_id, year_id) row(s)")


def _check_no_draw_axis(df: pd.DataFrame, rep: ValidationReport) -> None:
    for col in ("draw", "draw_id"):
        if col in df.columns:
            rep.fail(f"{col!r} present: an all-age product is summarised, draws must be collapsed")


def _check_redundant_constant_columns(df: pd.DataFrame, rep: ValidationReport) -> None:
    for col in df.columns:
        if col.lower() in PATH_DIMENSIONS and df[col].nunique(dropna=False) <= 1:
            rep.fail(
                f"{col!r} is a single-valued dimension column; it belongs in the filename, "
                "not as a redundant constant column"
            )


def _check_levels(
    df: pd.DataFrame, rep: ValidationReport, expected_levels: Sequence[int] | None
) -> None:
    present = sorted(int(x) for x in df.level.unique())
    if expected_levels is not None:
        want = sorted(int(x) for x in expected_levels)
        if present != want:
            rep.fail(f"levels present {present} != expected {want}")
    if any(lv < 0 for lv in present):
        rep.fail(f"negative hierarchy level(s): {present}")


def _check_population(df: pd.DataFrame, rep: ValidationReport) -> None:
    if (df.population < 0).any():
        rep.fail("negative population")
    if df.population.isna().any():
        rep.fail("null population")
    zero = int((df.population == 0).sum())
    if zero:
        # Real and expected at admin-2 (162 units in 2023). Rates are undefined there, so the
        # rate check must skip them rather than divide by zero.
        rep.note(f"{zero} row(s) with population == 0; rate checks skip them")


def _check_intervals(df: pd.DataFrame, measures: MeasureStructure, rep: ValidationReport) -> None:
    for m in measures.deliverable:
        for base in (m.count_col, m.rate_col):
            # Name these explicitly: STATS is ordered (mean, lower, upper), so positional
            # unpacking silently compares the wrong pair.
            mid = df[f"{base}_mean"]
            lo = df[f"{base}_lower"]
            hi = df[f"{base}_upper"]
            bad_lo = int((lo > mid + abs(mid) * 1e-9).sum())
            bad_hi = int((hi < mid - abs(mid) * 1e-9).sum())
            if bad_lo:
                rep.fail(f"{base}: lower exceeds mean in {bad_lo} row(s)")
            if bad_hi:
                rep.fail(f"{base}: upper below mean in {bad_hi} row(s)")


def _check_rates_match_counts(
    df: pd.DataFrame, measures: MeasureStructure, rep: ValidationReport
) -> None:
    """The load-bearing check: rate == count / that level's OWN population row."""
    usable = df.population > 0
    if not usable.any():
        rep.note("no rows with positive population; rate/count consistency not checked")
        return
    sub = df[usable]
    for m in measures.deliverable:
        for stat in STATS:
            count = sub[f"{m.count_col}_{stat}"].to_numpy(dtype="float64")
            rate = sub[f"{m.rate_col}_{stat}"].to_numpy(dtype="float64")
            implied = count / sub.population.to_numpy(dtype="float64")
            close = np.isclose(rate, implied, rtol=RATE_RTOL, atol=0.0, equal_nan=True)
            n_bad = int((~close).sum())
            if n_bad:
                idx = np.flatnonzero(~close)[:3]
                examples = ", ".join(
                    f"loc {int(sub.location_id.iloc[i])} yr {int(sub.year_id.iloc[i])}: "
                    f"rate {rate[i]:.6g} vs count/pop {implied[i]:.6g}"
                    for i in idx
                )
                rep.fail(
                    f"{m.name} {stat}: rate != count / own population in {n_bad} row(s) "
                    f"[{examples}] -- an aggregate rate must divide by that level's own "
                    "population row, never a sum of children's populations"
                )


def _check_nonneg_counts(
    df: pd.DataFrame, measures: MeasureStructure, rep: ValidationReport
) -> None:
    for m in measures.deliverable:
        for stat in STATS:
            col = f"{m.count_col}_{stat}"
            n = int((df[col] < 0).sum())
            if n:
                rep.fail(f"{col}: {n} negative value(s)")


def _check_count_additivity(
    df: pd.DataFrame,
    measures: MeasureStructure,
    hierarchy: pd.DataFrame,
    rep: ValidationReport,
    *,
    rtol: float = 1e-6,
) -> None:
    """Each parent's count equals the sum of its PRESENT children.

    Only parents whose children are all present are checked; a partially-present parent would
    legitimately exceed its visible children, and flagging that would be a false alarm.
    """
    if "parent_id" not in hierarchy.columns:
        rep.note("hierarchy has no parent_id; additivity not checked")
        return
    h = hierarchy[["location_id", "parent_id"]]
    have = set(df.location_id.unique())
    kids = h[h.location_id.isin(have) & h.parent_id.isin(have)]
    if kids.empty:
        rep.note("no parent/child pairs both present; additivity not checked")
        return
    child_counts = h[h.parent_id.isin(kids.parent_id.unique())].groupby("parent_id").size()
    present_counts = kids.groupby("parent_id").size()
    complete = sorted(set(present_counts[present_counts == child_counts.reindex(
        present_counts.index)].index))
    if not complete:
        rep.note("no parent has all children present; additivity not checked")
        return

    for m in measures.deliverable:
        col = f"{m.count_col}_mean"
        merged = (
            df[df.location_id.isin(kids[kids.parent_id.isin(complete)].location_id)]
            .merge(h, on="location_id", how="left")
            .groupby(["parent_id", "year_id"], observed=True)[col]
            .sum()
            .reset_index()
            .rename(columns={"parent_id": "location_id", col: "child_sum"})
        )
        parents = df[df.location_id.isin(complete)][["location_id", "year_id", col]]
        cmp_ = parents.merge(merged, on=["location_id", "year_id"], how="inner")
        if cmp_.empty:
            continue
        close = np.isclose(
            cmp_[col].to_numpy(dtype="float64"),
            cmp_.child_sum.to_numpy(dtype="float64"),
            rtol=rtol,
            atol=0.0,
        )
        n_bad = int((~close).sum())
        if n_bad:
            rep.fail(
                f"{m.name}: parent count != sum of children in {n_bad} parent-year(s); counts "
                "must aggregate in count space"
            )


@dataclass
class AnchorReport:
    """Predicted level against the anchor's own target, per the anchor's semantics."""

    kind: str
    label: str
    #: Per-group comparison: group cols + anchor_target, predicted, ratio, rel_diff, n_years.
    table: pd.DataFrame
    #: True only when the anchor guarantees exactness and it held.
    passed: bool
    #: True when the deviation exceeds the applicable threshold.
    flagged: bool
    threshold: float
    message: str

    def summary(self) -> str:
        return f"[{self.kind}] {self.message}"


def anchor_diagnostic(
    predicted: pd.DataFrame,
    observed: pd.DataFrame,
    anchor: AnchorSpec,
    *,
    value_col: str,
    group_cols: Sequence[str] = ("location_id",),
    year_col: str = "year_id",
    point_rtol: float = POINT_ANCHOR_RTOL,
    window_warn_rtol: float = WINDOW_ANCHOR_WARN_RTOL,
) -> AnchorReport:
    """Compare the predicted level against the anchor's target — never against a bare year.

    The comparison year is ``max(anchor.years)``: the last year the anchor is informed by, which
    is where a forecast departs from history. For a point anchor that is the anchor year itself.

    For :attr:`AnchorKind.POINT` the target is that year's observed value and equality is a
    correctness criterion — ``passed`` is meaningful. For :attr:`AnchorKind.WINDOW_MEAN` the
    target is the window mean, exactness is not guaranteed, and ``passed`` is left False with
    ``flagged`` carrying whether the deviation is large enough to investigate.
    """
    target = anchor.target(observed, value_col, group_cols=group_cols, year_col=year_col)
    cmp_year = max(anchor.years)
    pred = predicted[predicted[year_col] == cmp_year]
    if pred.empty:
        raise ValueError(
            f"no predicted rows at year {cmp_year}; cannot compare against the anchor target"
        )
    pred = pred[[*group_cols, value_col]].rename(columns={value_col: "predicted"})

    table = target.merge(pred, on=list(group_cols), how="inner")
    if table.empty:
        raise ValueError("no groups shared between the anchor target and the prediction")
    tgt = table.anchor_target.to_numpy(dtype="float64")
    got = table.predicted.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        table["ratio"] = np.where(tgt != 0, got / tgt, np.nan)
        table["rel_diff"] = np.where(tgt != 0, (got - tgt) / tgt, np.nan)

    worst = float(np.nanmax(np.abs(table.rel_diff))) if len(table) else 0.0
    tot_t, tot_g = float(np.nansum(tgt)), float(np.nansum(got))
    agg_rel = (tot_g - tot_t) / tot_t if tot_t else float("nan")

    if anchor.reproduces_observed:
        passed = worst <= point_rtol
        flagged = not passed
        msg = (
            f"point anchor at {cmp_year}: worst per-group deviation {worst:.2%} "
            f"(tolerance {point_rtol:.2%}); aggregate {agg_rel:+.2%}. "
            + ("reproduces observed as required" if passed else "DOES NOT reproduce observed -- bug")
        )
        threshold = point_rtol
    else:
        passed = False
        flagged = abs(agg_rel) > window_warn_rtol
        msg = (
            f"{anchor.label}: predicted {cmp_year} vs the anchor target -- aggregate "
            f"{agg_rel:+.2%}, worst per-group {worst:.2%}. Exactness is NOT expected for a "
            "window-mean anchor; comparing against observed at a single year would be wrong. "
            + (
                f"Aggregate deviation exceeds {window_warn_rtol:.0%} -- worth investigating."
                if flagged
                else f"Within the {window_warn_rtol:.0%} diagnostic threshold."
            )
        )
        threshold = window_warn_rtol

    return AnchorReport(
        kind=anchor.kind.value,
        label=anchor.label,
        table=table,
        passed=passed,
        flagged=flagged,
        threshold=threshold,
        message=msg,
    )
