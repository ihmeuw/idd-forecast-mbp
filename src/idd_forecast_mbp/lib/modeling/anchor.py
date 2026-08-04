"""Anchor a prediction to observed data over a window.

The anchor adds a constant, per group, so that predictions line up with observed
values over an anchor window. It is applied in *model space* — log for rates,
logit for CFR — so the same code serves both; nothing here knows which.

Why this module exists in this shape
------------------------------------
Three separate bodies of work solved overlapping pieces of this problem and each
hardcoded its own combination:

- ``04_forecasting/forecast_malaria_admin_2s_rocket.r`` and
  :mod:`~idd_forecast_mbp.lib.modeling.shift` — single-year point anchor.
- ``04_forecasting/OLD_rake_dengue.py`` — point anchor, but per
  ``(location, age, sex)`` rather than per location.
- ``reports/03_modeling/pygam_dengue_models_explore.ipynb`` — ``point`` /
  ``median_diff`` / ``median_resid`` over a window, per location.
- ``idd-manuscripts`` ``GBD2023_DENV/gbd2023_denv_aroc_functions.py`` — leave-one-out
  outlier exclusion and a much richer set of baseline statistics.

Those are not four alternatives; they are points in a space of four *independent*
choices, which this module makes explicit:

1. **Eligibility** (:class:`Eligibility`) — which years in the window are allowed
   to count. Optionally drop years whose value sits more than ``threshold_sd``
   from a leave-one-out expectation.
2. **Statistic** (:class:`Baseline`) — how the surviving years collapse to one
   number.
3. **Applied to** (:attr:`AnchorSpec.applied_to`) — collapse observed and
   predicted separately then difference, or collapse the per-year residual.
   These differ whenever the statistic is non-linear, which the median is.
4. **Granularity** — the ``group_cols`` argument. ``("location_id",)`` anchors per
   location; ``("location_id", "age_group_id", "sex_id")`` anchors per age/sex
   cell.

Granularity is not a cosmetic choice. Under a per-``(location, age, sex)`` anchor,
any term that is additive in link space and constant in time cancels out of the
forecast entirely — see ``.claude/DECISIONS.md`` 2026-08-03 "The prediction frame
is decoupled from the fit design matrix". Changing granularity changes whether
that holds.

The legacy modes are recoverable as special cases::

    point        = Baseline("single"), years=(anchor_year,), applied_to="separate"
    median_diff  = Baseline("median"), applied_to="separate"
    median_resid = Baseline("median"), applied_to="residual"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

# Leave-one-out needs enough remaining years to say anything. A regression needs
# one more than a mean does: with three years, dropping one leaves two, which fit
# a line perfectly and give a zero-residual expectation for every year.
_MIN_YEARS = {"mean": 3, "trend": 4}

# A trim needs at least this many values to have anything to drop.
_MIN_TRIM_VALUES = 3
# A line needs at least this many points.
_MIN_TREND_POINTS = 2

# Collapsing functions for the non-degenerate statistics. "single" and "trend"
# are handled separately: they need the years, not just the values.
_STATISTIC_FN: dict[str, Callable[[Any], Any]] = {
    "mean": np.mean, "median": np.median, "min": np.min, "max": np.max,
}

_STATISTICS = ("single", "mean", "median", "min", "max", "trend")
_TRIMS = (None, "min_max", "extremes")
_DIRECTIONS = ("both", "above", "below")
_APPLIED_TO = ("separate", "residual")


@dataclass(frozen=True)
class Eligibility:
    """Which years in the anchor window are allowed to contribute.

    Parameters
    ----------
    method:
        ``None`` keeps every year. ``"mean"`` compares each year to the mean of
        the *other* years; ``"trend"`` compares it to a line fitted on the other
        years, which is what you want when the series trends and the endpoints
        would otherwise look anomalous.
    threshold_sd:
        Drop a year when ``|z| > threshold_sd``.
    direction:
        ``"above"`` drops only spikes, ``"below"`` only dips, ``"both"`` either.
    use_draw_se:
        When a per-year standard error is supplied, fold it into the denominator
        as ``sqrt(loo_std**2 + se**2)`` so a year is not flagged merely for being
        uncertain. Ignored when no standard error is available.

    Leave-one-out matters: the expectation for a year is built from the other
    years only, so an outlier cannot inflate the expectation it is judged against.
    """

    method: str | None = None
    threshold_sd: float = 2.0
    direction: str = "both"
    use_draw_se: bool = True

    def __post_init__(self) -> None:
        if self.method not in (None, "mean", "trend"):
            msg = f"method must be None, 'mean' or 'trend'; got {self.method!r}"
            raise ValueError(msg)
        if self.direction not in _DIRECTIONS:
            msg = f"direction must be one of {_DIRECTIONS}; got {self.direction!r}"
            raise ValueError(msg)
        if self.threshold_sd <= 0:
            msg = f"threshold_sd must be positive; got {self.threshold_sd}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Baseline:
    """How the eligible years collapse to a single anchor value.

    Parameters
    ----------
    statistic:
        ``"single"`` takes the lone year in the window (and requires exactly one).
        ``"trend"`` fits a line across the window and reads it at
        ``effective_year``, so the anchor is a de-noised level on a fitted line
        rather than any realised observation.
    trim:
        Applied before the statistic. ``"min_max"`` drops the extremes;
        ``"extremes"`` drops values beyond ``trim_threshold`` sample SDs of the
        mean. Both are no-ops on fewer than three values.
    effective_year:
        Year at which to read the fitted line; defaults to the window midpoint.
        Only used by ``statistic="trend"``.

    ``trim`` overlaps :class:`Eligibility` on purpose — the two come from
    different codebases. Prefer one or the other rather than stacking them, or
    the effective threshold becomes hard to reason about.
    """

    statistic: str = "median"
    trim: str | None = None
    trim_threshold: float = 5.0
    effective_year: int | None = None

    def __post_init__(self) -> None:
        if self.statistic not in _STATISTICS:
            msg = f"statistic must be one of {_STATISTICS}; got {self.statistic!r}"
            raise ValueError(msg)
        if self.trim not in _TRIMS:
            msg = f"trim must be one of {_TRIMS}; got {self.trim!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class AnchorSpec:
    """A complete anchor configuration.

    ``applied_to="separate"`` collapses observed and predicted independently and
    differences the results; ``"residual"`` collapses the per-year difference.
    They disagree whenever the statistic is non-linear — the median is, the mean
    is not — so both are kept rather than one being derived from the other.
    """

    years: tuple[int, ...]
    baseline: Baseline = field(default_factory=Baseline)
    eligibility: Eligibility = field(default_factory=Eligibility)
    applied_to: str = "separate"

    def __post_init__(self) -> None:
        if not self.years:
            msg = "years must be non-empty"
            raise ValueError(msg)
        if self.applied_to not in _APPLIED_TO:
            msg = f"applied_to must be one of {_APPLIED_TO}; got {self.applied_to!r}"
            raise ValueError(msg)
        if self.baseline.statistic == "single" and len(self.years) != 1:
            msg = f"statistic='single' needs exactly one year; got {self.years}"
            raise ValueError(msg)

    @classmethod
    def point(cls, year: int) -> AnchorSpec:
        """The legacy single-year anchor: ``shift = obs(year) - pred(year)``."""
        return cls(years=(year,), baseline=Baseline(statistic="single"))

    @classmethod
    def median_diff(
        cls, years: Sequence[int], eligibility: Eligibility | None = None,
    ) -> AnchorSpec:
        """``median(obs) - median(pred)``, each over its own window years."""
        return cls(years=tuple(years), baseline=Baseline(statistic="median"),
                   eligibility=eligibility or Eligibility(), applied_to="separate")

    @classmethod
    def median_resid(
        cls, years: Sequence[int], eligibility: Eligibility | None = None,
    ) -> AnchorSpec:
        """``median(obs_y - pred_y)`` over the years present in both."""
        return cls(years=tuple(years), baseline=Baseline(statistic="median"),
                   eligibility=eligibility or Eligibility(), applied_to="residual")


# ---------------------------------------------------------------------------
# Year eligibility
# ---------------------------------------------------------------------------

def outlier_years(
    years: Sequence[int],
    values: Sequence[float],
    eligibility: Eligibility,
    standard_errors: Sequence[float] | None = None,
) -> list[int]:
    """Years whose value is more than ``threshold_sd`` from a leave-one-out expectation.

    Returns an empty list when ``eligibility.method`` is ``None`` or when there
    are too few years for the chosen method to mean anything (three for
    ``"mean"``, four for ``"trend"``) — silently keeping every year rather than
    flagging on noise.
    """
    if eligibility.method is None:
        return []

    yrs = np.asarray(years, dtype=float)
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(yrs) & np.isfinite(vals)
    yrs, vals = yrs[finite], vals[finite]
    ses = (np.asarray(standard_errors, dtype=float)[finite]
           if standard_errors is not None else None)

    if yrs.size < _MIN_YEARS[eligibility.method]:
        return []

    # Scale-relative tolerance for "the leave-one-out fit is exact". A perfectly
    # linear series leaves float-dust residuals; without this the z-score is
    # dust/dust and flags at random.
    scale = max(float(np.max(np.abs(vals))), 1.0)
    tol = 1e-9 * scale

    flagged: list[int] = []
    for i in range(yrs.size):
        z = _leave_one_out_z(i, yrs, vals, ses, eligibility, tol)
        if _is_outlier(z, eligibility):
            flagged.append(int(yrs[i]))

    return flagged


def _leave_one_out_z(  # noqa: PLR0913
    i: int,
    yrs: np.ndarray[Any, Any],
    vals: np.ndarray[Any, Any],
    ses: np.ndarray[Any, Any] | None,
    eligibility: Eligibility,
    tol: float,
) -> float:
    """Standardised departure of year ``i`` from an expectation built without it."""
    others = np.arange(yrs.size) != i
    other_years, other_values = yrs[others], vals[others]

    if eligibility.method == "mean":
        expected = float(other_values.mean())
        residuals = other_values - expected
        dof = 1  # one parameter estimated: the mean
    else:
        slope, intercept = np.polyfit(other_years, other_values, 1)
        expected = float(slope * yrs[i] + intercept)
        residuals = other_values - (slope * other_years + intercept)
        dof = 2  # two parameters estimated: slope and intercept

    spread = float(np.std(residuals, ddof=dof)) if residuals.size > dof else 0.0
    if eligibility.use_draw_se and ses is not None and np.isfinite(ses[i]):
        spread = float(np.hypot(spread, ses[i]))

    deviation = float(vals[i] - expected)

    if spread > tol:
        return deviation / spread
    # The other years pin the expectation exactly, so there is no scale to
    # standardise against. A year matching it is not an outlier; one departing
    # from it is, however small the departure looks against the series.
    if abs(deviation) <= tol:
        return 0.0
    return float(np.inf) if deviation > 0 else float(-np.inf)


def _is_outlier(z: float, eligibility: Eligibility) -> bool:
    """Apply the threshold on the side(s) ``direction`` asks for."""
    if eligibility.direction == "above":
        return z > eligibility.threshold_sd
    if eligibility.direction == "below":
        return z < -eligibility.threshold_sd
    return abs(z) > eligibility.threshold_sd


# ---------------------------------------------------------------------------
# Baseline value
# ---------------------------------------------------------------------------

def _trim(values: np.ndarray[Any, Any], baseline: Baseline) -> np.ndarray[Any, Any]:
    """Drop extremes ahead of the statistic. No-op on fewer than three values."""
    if baseline.trim is None or values.size < _MIN_TRIM_VALUES:
        return values
    if baseline.trim == "min_max":
        return np.sort(values)[1:-1]
    spread = float(np.std(values))
    if spread <= 0:
        return values
    centre = float(np.mean(values))
    keep = np.abs(values - centre) <= baseline.trim_threshold * spread
    return values[keep]


def baseline_value(
    years: Sequence[int],
    values: Sequence[float],
    baseline: Baseline,
) -> float:
    """Collapse ``values`` to a single anchor level. NaN when nothing is usable."""
    yrs = np.asarray(years, dtype=float)
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(yrs) & np.isfinite(vals)
    yrs, vals = yrs[finite], vals[finite]
    if vals.size == 0:
        return float("nan")

    if baseline.statistic == "single":
        return float(vals[0])

    if baseline.statistic == "trend":
        if vals.size < _MIN_TREND_POINTS:
            return float(vals[0])
        effective = (baseline.effective_year if baseline.effective_year is not None
                     else float(np.mean(yrs)))
        slope, intercept = np.polyfit(yrs, vals, 1)
        return float(slope * effective + intercept)

    kept = _trim(vals, baseline)
    if kept.size == 0:
        return float("nan")
    return float(_STATISTIC_FN[baseline.statistic](kept))


# ---------------------------------------------------------------------------
# The shift
# ---------------------------------------------------------------------------

def _collapse(
    frame: pd.DataFrame, value_col: str, group_cols: list[str], spec: AnchorSpec,
    se_col: str | None,
) -> pd.Series:
    """Per-group baseline of ``value_col``, after dropping ineligible years."""

    def one(g: pd.DataFrame) -> float:
        yrs = g["year_id"].to_numpy()
        vals = g[value_col].to_numpy(dtype=float)
        ses = g[se_col].to_numpy(dtype=float) if se_col and se_col in g else None
        dropped = outlier_years(yrs, vals, spec.eligibility, ses)
        if dropped:
            keep = ~np.isin(yrs, dropped)
            yrs, vals = yrs[keep], vals[keep]
        return baseline_value(yrs, vals, spec.baseline)

    return frame.groupby(group_cols, sort=False).apply(one, include_groups=False)


def compute_shift(  # noqa: PLR0913
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    spec: AnchorSpec,
    *,
    group_cols: Sequence[str] = ("location_id",),
    obs_col: str = "observed",
    pred_col: str = "predicted",
    se_col: str | None = None,
) -> pd.Series:
    """Per-group additive shift, in whatever space ``obs_col``/``pred_col`` live in.

    ``observed`` and ``predicted`` are tidy frames carrying ``group_cols``,
    ``year_id`` and their value column. Only ``spec.years`` are used. Groups with
    no usable anchor get NaN — the caller decides whether that means "leave
    unshifted" or "drop", and those are different decisions (see the
    no-invented-deaths rule for CFR).

    Returns a Series indexed by ``group_cols``, to be added to the prediction.
    """
    keys = list(group_cols)
    obs = observed[observed["year_id"].isin(spec.years)]
    pred = predicted[predicted["year_id"].isin(spec.years)]

    if spec.applied_to == "residual":
        cols = [*keys, "year_id"]
        paired = obs[[*cols, obs_col]].merge(pred[[*cols, pred_col]], on=cols, how="inner")
        if se_col and se_col in observed.columns:
            paired = paired.merge(observed[[*cols, se_col]], on=cols, how="left")
        paired["_resid"] = paired[obs_col] - paired[pred_col]
        shift = _collapse(paired, "_resid", keys, spec, se_col)
    else:
        obs_base = _collapse(obs, obs_col, keys, spec, se_col)
        pred_base = _collapse(pred, pred_col, keys, spec, None)
        shift = obs_base - pred_base

    return shift.rename("shift").dropna()


def apply_shift(  # noqa: PLR0913
    predictions: pd.DataFrame,
    shift: pd.Series,
    *,
    group_cols: Sequence[str] = ("location_id",),
    pred_col: str = "predicted",
    out_col: str = "anchored",
    drop_unanchored: bool = True,
) -> pd.DataFrame:
    """Add ``shift`` to every row of its group.

    ``drop_unanchored=True`` removes rows whose group has no shift; ``False``
    leaves them at their unshifted value, matching the malaria rocket, which
    treats a missing anchor as shift zero.
    """
    keys = list(group_cols)
    out = predictions.copy()
    idx = pd.MultiIndex.from_frame(out[keys]) if len(keys) > 1 else pd.Index(out[keys[0]])
    mapped = pd.Series(idx.map(shift), index=out.index, dtype=float)
    if drop_unanchored:
        out = out[mapped.notna()].copy()
        mapped = mapped[mapped.notna()]
    else:
        mapped = mapped.fillna(0.0)
    out[out_col] = out[pred_col].to_numpy(dtype=float) + mapped.to_numpy(dtype=float)
    return out
