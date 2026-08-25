"""
vaccine_efficacy.py — malaria vaccine-efficacy curve construction and loading.

Scope (deliberately narrow, per repo inspection):
  * Build monthly VE curves for each product/channel/regimen from VE_ANCHORS.yaml.
  * Provide a VECurve accessor with continuous-age lookup.
  * Load + validate a VE curve frame against the forecast's loader contract.

Explicitly NOT in scope (lives elsewhere, already built and tested):
  * Birth-cohort C3/C4 resolution, trigger gating, dose-4 lag, bin averaging
    -> lib/processing/vaccine_cohort_fractions.py (fraction_for_bin, 111 tests).
  * age_group_id binning / age_metadata.parquet  -> the stage script.
  * Application to counts (burden-weighted collapse)  -> already implemented.

The VE module never imports age groups; its only age contract is
    VECurve.ve(vacc_name, column, age_years) -> float
continuous age in years, linear interpolation between whole months, 0 at/beyond
the last month.

Acceptance test: regenerate the four cells from VE_ANCHORS.yaml and diff against
the delivered CSVs -> byte-identical.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd

VE_COLUMNS = ["ve_case_d3", "ve_case_d34", "ve_death_d3", "ve_death_d34"]
REQUIRED_CSV_COLUMNS = ["vaccine", "age_months", *VE_COLUMNS]


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------
def _interp_piecewise(anchors, ages, logspace, eps):
    """Piecewise interpolation through anchors (each segment uses only its two
    bracketing anchors — NO regression). Before first anchor: hold peak.
    Beyond last anchor: NaN (filled by _extrapolate_tail)."""
    xs = np.array([a for a, _ in anchors], float)
    ys = np.array([v for _, v in anchors], float)
    if logspace:
        ly = np.log(np.clip(ys, eps, 1.0))
        out = np.interp(ages, xs, ly, left=ly[0], right=np.nan)
        return np.exp(out)
    return np.interp(ages, xs, ys, left=ys[0], right=np.nan)


def _extrapolate_tail(anchors, ages, logspace, out, eps):
    """Fill values beyond the last anchor using the slope of the last two anchors.
    linear -> straight to 0; loglinear -> log-line, cut to 0 below eps."""
    xs = np.array([a for a, _ in anchors], float)
    ys = np.array([v for _, v in anchors], float)
    x1, x2 = xs[-2], xs[-1]
    y1, y2 = ys[-2], ys[-1]
    beyond = ages > xs[-1]
    if logspace:
        ly1, ly2 = np.log(max(y1, eps)), np.log(max(y2, eps))
        slope = (ly2 - ly1) / (x2 - x1)
        vals = np.exp(ly2 + slope * (ages - xs[-1]))
        vals[vals < eps] = 0.0
    else:
        slope = (y2 - y1) / (x2 - x1)
        vals = y2 + slope * (ages - xs[-1])
        vals[vals < 0] = 0.0
    out[beyond] = vals[beyond]
    return out


def _back_extrap_reset(d34_anchors, booster_age, logspace):
    """Booster reset height = the two post-booster anchors extrapolated BACK to
    the booster age, using the cell's form. (label R — an assumption.)"""
    (xa, ya), (xb, yb) = d34_anchors[0], d34_anchors[1]
    if logspace:
        la, lb = np.log(ya), np.log(yb)
        slope = (lb - la) / (xb - xa)
        return float(np.exp(la + slope * (booster_age - xa)))
    slope = (yb - ya) / (xb - xa)
    return float(ya + slope * (booster_age - xa))


def _build_disease(pre, d3p, d34p, booster_age, dose3_age, ages, logspace, eps):
    """dose-3-only and dose-3+4 disease curves for one product."""
    peak = pre[0][1]
    first = pre[0][0]
    d3_anchors = pre + d3p
    d3 = _interp_piecewise(d3_anchors, ages, logspace, eps)
    d3 = _extrapolate_tail(d3_anchors, ages, logspace, d3, eps)
    d3 = np.minimum(d3, peak)
    d3[ages < first] = peak
    d3[ages < dose3_age] = 0.0

    reset = _back_extrap_reset(d34p, booster_age, logspace)
    d34 = d3.copy()
    d34_anchors = [(booster_age, reset)] + d34p
    post = ages >= booster_age
    d34_vals = _interp_piecewise(d34_anchors, ages, logspace, eps)
    d34_vals = _extrapolate_tail(d34_anchors, ages, logspace, d34_vals, eps)
    d34[post] = d34_vals[post]

    for arr in (d3, d34):
        arr[arr < (eps if logspace else 0.0)] = 0.0
        np.clip(arr, 0, 1, out=arr)
    d34[ages < booster_age] = d3[ages < booster_age]
    return d3, d34


def _build_severe(ages, d3, d34, booster_age, dose3_age, logspace,
                  severe_mode, r_pre, r_d34, eps):
    """severe/death channel = ratio x disease. dose-3+4 follows disease per form.
    dose-3-only: 'zero' -> 0 at booster; 'smooth' -> continues (loglinear borrows
    the dose-3+4 severe slope; linear scales disease)."""
    pre = ages < booster_age
    post = ~pre
    death_d34 = np.where(pre, d34 * r_pre, d34 * r_d34)
    death_d34[ages < dose3_age] = 0.0

    if severe_mode == "zero":
        death_d3 = np.where(pre, d3 * r_pre, 0.0)
    else:  # smooth
        death_d3 = d3 * r_pre
        if logspace:
            i_b = int(np.where(ages == booster_age)[0][0])
            base = death_d3[i_b]
            ds30 = d34[int(np.where(ages == 30)[0][0])] * r_d34
            ds44 = d34[int(np.where(ages == 44)[0][0])] * r_d34
            slope = (np.log(max(ds44, eps)) - np.log(max(ds30, eps))) / (44 - 30)
            vals = np.exp(np.log(max(base, eps)) + slope * (ages - booster_age))
            vals[vals < eps] = 0.0
            death_d3[post] = vals[post]
        # linear smooth: death_d3 = d3 * r_pre already covers all ages
    death_d3[ages < dose3_age] = 0.0
    for a in (death_d3, death_d34):
        np.clip(a, 0, 1, out=a)
    return death_d3, death_d34


def build_cell(anchors: dict, interpolation: str, severe_post_booster: str) -> pd.DataFrame:
    """Build one factorial cell -> frame satisfying the loader contract.

    anchors : parsed VE_ANCHORS.yaml (dict).
    interpolation : 'linear' | 'loglinear'.
    severe_post_booster : 'zero' | 'smooth'.
    """
    logspace = interpolation == "loglinear"
    sched = anchors["schedule"]
    dose3_age = sched["dose3_age_months"]
    booster_age = sched["booster_age_months"]
    C = anchors["constants"]
    max_age, eps, ndec = C["max_age_months"], C["eps_threshold"], C["round_decimals"]
    ratios = anchors["severe_disease_ratio"]
    r_pre, r_d34 = ratios["pre_booster"], ratios["boosted"]

    ages = np.arange(0, max_age + 1)
    frames = []
    for vacc in anchors["disease"]:
        dv = anchors["disease"][vacc]
        pre = [(a["age"], a["ve"]) for a in dv["pre"]]
        d3p = [(a["age"], a["ve"]) for a in dv["d3"]]
        d34p = [(a["age"], a["ve"]) for a in dv["d34"]]
        d3, d34 = _build_disease(pre, d3p, d34p, booster_age, dose3_age, ages, logspace, eps)
        sd3, sd34 = _build_severe(ages, d3, d34, booster_age, dose3_age, logspace,
                                  severe_post_booster, r_pre, r_d34, eps)
        frames.append(pd.DataFrame({
            "vaccine": vacc, "age_months": ages,
            "ve_case_d3": np.round(d3, ndec), "ve_case_d34": np.round(d34, ndec),
            "ve_death_d3": np.round(sd3, ndec), "ve_death_d34": np.round(sd34, ndec),
        }))
    return pd.concat(frames, ignore_index=True)


def build_all_cells(anchors: dict) -> dict[str, pd.DataFrame]:
    """Build every cell listed under anchors['build']['cells']."""
    out = {}
    for cell in anchors["build"]["cells"]:
        out[cell["name"]] = build_cell(anchors, cell["interpolation"], cell["severe_post_booster"])
    return out


# ---------------------------------------------------------------------------
# VECurve accessor  (the module's only age contract)
# ---------------------------------------------------------------------------
@dataclass
class VECurve:
    """Continuous-age VE lookup over a validated monthly frame.

    curves[vacc][column] is a monthly numpy array indexed by whole-month age.
    ve(vacc, column, age_years) linearly interpolates between whole months and
    returns 0 at or beyond the last month. Knows nothing about age groups.
    """
    curves: dict[str, dict[str, Sequence[float]]]
    max_month: int | None = None

    def __post_init__(self) -> None:
        if self.max_month is None:
            any_product = next(iter(self.curves.values()))
            object.__setattr__(self, "max_month",
                               len(any_product[VE_COLUMNS[0]]) - 1)

    def products(self) -> tuple[str, ...]:
        return tuple(sorted(self.curves))

    def n_months(self, vacc_name: str) -> int:
        return len(self.curves[vacc_name][VE_COLUMNS[0]])

    def nonzero_tails(self) -> dict[tuple[str, str], float]:
        """{(product, column): final value} for curves that end above zero.

        Lookups clip to 0 past the last month, so a curve truncated while still
        positive would introduce a silent discontinuity. This surfaces it.
        """
        return {
            (vacc, col): float(series[-1])
            for vacc, cols in self.curves.items()
            for col, series in cols.items()
            if len(series) and series[-1] > 0.0
        }

    def implied_trigger_months(self, vacc_name: str) -> tuple[int | None, int | None]:
        """(first month with non-zero dose-3 VE, first month where boosted VE
        departs from dose-3-only VE) -- the schedule the curve itself implies,
        for cross-checking against the cohort model's trigger ages. Keyed on the
        CASE channel; the severe columns are never consulted."""
        d3 = self.curves[vacc_name]["ve_case_d3"]
        d34 = self.curves[vacc_name]["ve_case_d34"]
        first_dose3 = next((m for m, v in enumerate(d3) if v > 0.0), None)
        first_boost = next((m for m, (a, b) in enumerate(zip(d3, d34)) if a != b), None)
        return first_dose3, first_boost

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "VECurve":
        validate_ve_frame(df)
        curves: dict[str, dict[str, np.ndarray]] = {}
        max_month = int(df["age_months"].max())
        for vacc, sub in df.groupby("vaccine"):
            sub = sub.sort_values("age_months")
            curves[str(vacc)] = {c: sub[c].to_numpy() for c in VE_COLUMNS}
        return cls(curves=curves, max_month=max_month)

    def ve(self, vacc_name: str, column: str, age_years: float) -> float:
        try:
            arr = self.curves[vacc_name][column]
        except KeyError as exc:
            raise KeyError(
                f"no VE curve for product {vacc_name!r} column {column!r}; "
                f"curve has products {self.products()} and columns {VE_COLUMNS}"
            ) from exc
        m = age_years * 12.0
        if m <= 0:
            return float(arr[0])
        if m >= self.max_month:
            return 0.0
        lo = int(np.floor(m))
        frac = m - lo
        return float(arr[lo] * (1.0 - frac) + arr[lo + 1] * frac)


# ---------------------------------------------------------------------------
# Loader / validation  (the forecast's enforced contract)
# ---------------------------------------------------------------------------
def validate_ve_frame(df: pd.DataFrame, expected_products: Sequence[str] | None = None,
                      dose3_age: int = 6, booster_age: int = 24) -> None:
    """Raise ValueError unless df satisfies the loader contract:
      * exact required columns
      * age_months a complete gapless run from 0 per product
      * values in [0,1], no nulls
      * case channel independently encodes dose 3 (first month case_d3>0) and
        booster (first month case_d3 != case_d34) at the expected ages
      * every expected product present
      * each curve reaches 0 by the last month (else flagged)
    """
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"VE frame missing columns: {missing}")
    if df[VE_COLUMNS].isnull().any().any() or df["age_months"].isnull().any():
        raise ValueError("VE frame contains nulls")
    vmin, vmax = df[VE_COLUMNS].min().min(), df[VE_COLUMNS].max().max()
    if vmin < 0.0 or vmax > 1.0:
        raise ValueError(f"VE values out of [0,1]: min={vmin}, max={vmax}")

    products = list(df["vaccine"].unique())
    if expected_products is not None:
        for p in expected_products:
            if p not in products:
                raise ValueError(f"VE frame missing product: {p}")

    for vacc, sub in df.groupby("vaccine"):
        sub = sub.sort_values("age_months")
        am = sub["age_months"].to_numpy()
        if am[0] != 0 or not np.array_equal(am, np.arange(am[0], am[-1] + 1)):
            raise ValueError(f"{vacc}: age_months not a gapless run from 0")
        d3 = sub["ve_case_d3"].to_numpy()
        d34 = sub["ve_case_d34"].to_numpy()
        first_dose3 = next((m for m, v in enumerate(d3) if v > 0.0), None)
        first_boost = next((m for m, (a, b) in enumerate(zip(d3, d34)) if a != b), None)
        if first_dose3 != dose3_age:
            raise ValueError(f"{vacc}: dose-3 trigger at month {first_dose3}, expected {dose3_age}")
        if first_boost != booster_age:
            raise ValueError(f"{vacc}: booster trigger at month {first_boost}, expected {booster_age}")
        last = sub.iloc[-1][VE_COLUMNS].to_numpy()
        if np.any(last > 0.0):
            raise ValueError(f"{vacc}: curve does not reach 0 at last month (values {last})")


def load_ve_curve(path: str, expected_products: Sequence[str] | None = None,
                  dose3_age: int = 6, booster_age: int = 24) -> VECurve:
    """Read a VE curve CSV, validate against the loader contract, return a VECurve.
    (Ported from the stage script — kept as the module's public loader.)"""
    df = pd.read_csv(path)
    validate_ve_frame(df, expected_products=expected_products,
                      dose3_age=dose3_age, booster_age=booster_age)
    return VECurve.from_frame(df)
