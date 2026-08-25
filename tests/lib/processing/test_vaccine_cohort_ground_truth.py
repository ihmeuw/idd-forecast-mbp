"""Permanent regression test for the vaccine cohort fraction logic.

Builds a synthetic ground-truth population with realistic, non-trivial,
NON-DIFFERENTIAL mortality (explicit headcount compartments: unvaccinated /
D3-only / D3+D4), aggregates it up to the real age_group_id bins, then runs
the ACTUAL production logic against only the resulting coarse population and
asserts the two agree within tolerance.

This is the automated replacement for the manual notebook audit that
originally found the dose-3 year-boundary bug: if the bin-width handling or
the 50/50 blend regresses, this fails rather than requiring another hand
audit.

Ported from .claude/vaccination_scripts/validate_against_synthetic_ground_truth.py
with the logic unchanged; it now imports the repo module rather than a local
copy. `_protection` is imported deliberately -- the ground truth has to
replicate the same protection curve to be a meaningful comparison.

Self-contained by design: no real data files, so it runs on every test pass
rather than only under -m slow.
"""
import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.vaccine_efficacy import VECurve, VE_COLUMNS
from idd_forecast_mbp.lib.processing.vaccine_cohort_fractions import (
    D3_TRIGGER_AGE,
    D4_TRIGGER_AGE,
    CoverageSeries,
    _protection,
    fraction_for_bin,
)

# Synthetic VE curve with the delivered files' structure (zero before month 6,
# boosted column identical to dose-3-only until month 24, then a step up), but
# piecewise-constant so the ground truth stays hand-checkable. Self-contained:
# this harness must not depend on a data file.
D3_CASE, D34_CASE, D3_DEATH, D34_DEATH = 0.60, 0.90, 0.45, 0.70
VE_LAST_MONTH = 300


def _build_ve() -> VECurve:
    cols = {c: [] for c in VE_COLUMNS}
    for month in range(VE_LAST_MONTH + 1):
        if month < 6:
            vals = (0.0, 0.0, 0.0, 0.0)
        elif month < 24:
            vals = (D3_CASE, D3_CASE, D3_DEATH, D3_DEATH)
        elif month < VE_LAST_MONTH:
            vals = (D3_CASE, D34_CASE, 0.0, D34_DEATH)
        else:
            vals = (0.0, 0.0, 0.0, 0.0)
        for col, v in zip(VE_COLUMNS, vals):
            cols[col].append(v)
    return VECurve(curves={"r21": cols})


VE = _build_ve()

# ---------------------------------------------------------------------------
# Synthetic setup -- a plausible ramp to a plateau, then flat.
# ---------------------------------------------------------------------------
PLATEAU_YEAR, PLATEAU_VAL = 2040, 0.40
RAMP = {2024: 0.03, 2025: 0.13, 2026: 0.26, 2027: 0.29, 2028: 0.30, 2029: 0.31,
        2030: 0.32, 2031: 0.33, 2032: 0.34, 2033: 0.35, 2034: 0.36, 2035: 0.37,
        2036: 0.38, 2037: 0.38, 2038: 0.39, 2039: 0.39, 2040: 0.40}
DROPOUT = 0.367

# Mortality hazard -- identical for every compartment (non-differential).
BG, PEAK, K = 0.004, 0.15, 6.0

REPORT_YEARS = list(range(2024, 2061))

AGE_GROUPS = pd.DataFrame([
    dict(age_group_id=2,   name="Early Neonatal", start=0.0,      end=0.019178),
    dict(age_group_id=3,   name="Late Neonatal",  start=0.019178, end=0.076712),
    dict(age_group_id=388, name="1-5 months",     start=0.076712, end=0.5),
    dict(age_group_id=389, name="6-11 months",    start=0.5,      end=1.0),
    dict(age_group_id=238, name="12-23 months",   start=1.0,      end=2.0),
    dict(age_group_id=34,  name="2 to 4",         start=2.0,      end=5.0),
    dict(age_group_id=6,   name="5 to 9",         start=5.0,      end=10.0),
    dict(age_group_id=7,   name="10 to 14",       start=10.0,     end=15.0),
    dict(age_group_id=8,   name="15 to 19",       start=15.0,     end=20.0),
])
NARROW_BINS = ["Early Neonatal", "Late Neonatal", "1-5 months", "6-11 months", "12-23 months"]
WIDE_BINS = ["2 to 4", "5 to 9", "10 to 14", "15 to 19"]


def build_coverage() -> CoverageSeries:
    dose3 = {}
    for y in range(2004, 2063):
        if y < 2024:
            dose3[y] = 0.0
        elif y <= PLATEAU_YEAR:
            dose3[y] = RAMP[y]
        else:
            dose3[y] = PLATEAU_VAL
    # dose_4 is UNCONDITIONAL: (1 - dropout) applied to the dose_3 of the
    # cohort that triggers 2 years earlier. Matches the real data's structure.
    dose4 = {
        y: (round((1 - DROPOUT) * dose3[y - 2], 4) if (y - 2) in dose3 and y >= 2026 else 0.0)
        for y in range(2004, 2063)
    }
    return CoverageSeries(dose3=dose3, dose4=dose4, vacc_name="r21", ve=VE)


def survival(age):
    age = np.maximum(age, 0.0)
    cum_hazard = BG * age + (PEAK / K) * (1 - np.exp(-K * age))
    return np.exp(-cum_hazard)


def build_ground_truth_cohorts(coverage: CoverageSeries) -> pd.DataFrame:
    """Monthly birth cohorts, each carrying its own immutable d3/d4 fractions."""
    base_births, growth = 10_000, 1.02
    rows = []
    for by in range(2004, 2061):
        births = base_births * growth ** (by - 2004)
        for bm in range(1, 13):
            birth_time = by + (bm - 0.5) / 12
            d3_year = int(birth_time + D3_TRIGGER_AGE)
            d4_year = int(birth_time + D4_TRIGGER_AGE)
            rows.append(dict(
                birth_time=birth_time,
                births=births,
                d3_ever=coverage.d3(d3_year) if d3_year >= 2024 else 0.0,
                d4_ever=coverage.d4(d4_year) if d4_year >= 2024 else 0.0,
            ))
    return pd.DataFrame(rows)


def ground_truth_state(row, age, coverage):
    """Explicit headcount compartments -- the ground truth being checked."""
    total = row.births * survival(age)
    d3_ever, d4_ever = row.d3_ever, row.d4_ever
    if age < D3_TRIGGER_AGE:
        n_3 = n_4 = 0.0
    elif age < D4_TRIGGER_AGE:
        n_3, n_4 = total * d3_ever, 0.0
    else:
        n_4 = total * d4_ever
        n_3 = total * (d3_ever - d4_ever)
    case, death = _protection(VE, coverage.vacc_name, age, d3_ever, d4_ever)
    return total, n_3 + n_4, n_4, total * case, total * death


def build_ground_truth_and_coarse_population(cohorts, coverage):
    gt_rows = []
    for y in REPORT_YEARS:
        ages = float(y) - cohorts.birth_time
        valid = ages >= 0
        sub, sub_ages = cohorts[valid], ages[valid]
        states = [ground_truth_state(r, a, coverage) for r, a in zip(sub.itertuples(), sub_ages)]
        tmp = sub.assign(
            age=sub_ages.values,
            total=[s[0] for s in states],
            n_ever3=[s[1] for s in states],
            n_ever4=[s[2] for s in states],
            n_prot=[s[3] for s in states],
            n_prot_death=[s[4] for s in states],
        )
        for ag in AGE_GROUPS.itertuples():
            piece = tmp[(tmp.age >= ag.start) & (tmp.age < ag.end)]
            if piece.total.sum() == 0:
                continue
            gt_rows.append(dict(
                year=y, age_group_id=ag.age_group_id, age_group_name=ag.name,
                population=piece.total.sum(),
                frac_ever_dose3=piece.n_ever3.sum() / piece.total.sum(),
                frac_ever_dose4=piece.n_ever4.sum() / piece.total.sum(),
                effective_protection_case=piece.n_prot.sum() / piece.total.sum(),
                effective_protection_death=piece.n_prot_death.sum() / piece.total.sum(),
            ))
    gt = pd.DataFrame(gt_rows)
    return gt, gt[["year", "age_group_id", "age_group_name", "population"]].copy()


def run_production_logic(coverage, coarse_pop) -> pd.DataFrame:
    """The real fraction_for_bin, given only the coarse population -- exactly
    what the orchestration script has access to (no mortality, no birth-month
    detail, no single-year-of-age split)."""
    rows = []
    for r in coarse_pop.itertuples():
        ag = AGE_GROUPS[AGE_GROUPS.age_group_id == r.age_group_id].iloc[0]
        d3, d4, case, death = fraction_for_bin(coverage, ag.start, ag.end, r.year)
        rows.append(dict(year=r.year, age_group_id=r.age_group_id,
                         age_group_name=r.age_group_name, population=r.population,
                         frac_ever_dose3=d3, frac_ever_dose4=d4,
                         effective_protection_case=case,
                         effective_protection_death=death))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def comparison() -> pd.DataFrame:
    coverage = build_coverage()
    cohorts = build_ground_truth_cohorts(coverage)
    truth, coarse = build_ground_truth_and_coarse_population(cohorts, coverage)
    production = run_production_logic(coverage, coarse)
    return truth.merge(
        production, on=["year", "age_group_id", "age_group_name"], suffixes=("_truth", "_prod")
    )


def test_comparison_is_non_trivial(comparison):
    """Guard the harness itself: if the merge or the synthetic build silently
    produced nothing, the tolerance assertions below would pass vacuously."""
    assert len(comparison) > 200
    assert comparison.frac_ever_dose3_truth.max() > 0.2
    assert comparison.frac_ever_dose4_truth.max() > 0.1


def test_narrow_bins_match_ground_truth(comparison):
    """Bins <=1 year wide.

    Tolerance is 0.01 (1 percentage point), not tighter, because "12-23
    months" carries a small INHERENT approximation error even at this
    resolution: protection decays exponentially in exact age, and a single
    age_int+0.5 representative point cannot capture that curvature across a
    full year of width. Measured at ~0.0065 during the original validation --
    expected, not a regression. Reintroducing the gating or year-boundary bug
    moves this well past 0.01.
    """
    narrow = comparison[comparison.age_group_name.isin(NARROW_BINS)]
    cols = ["frac_ever_dose3", "frac_ever_dose4",
            "effective_protection_case", "effective_protection_death"]
    err = (narrow[[f"{c}_prod" for c in cols]].values
           - narrow[[f"{c}_truth" for c in cols]].values)
    max_err = np.abs(err).max()
    assert max_err < 0.01, (
        f"narrow-bin fraction error {max_err} exceeds tolerance -- check for "
        "regressions in cohort_fraction_at_age / the trigger-age gating."
    )


def test_wide_bins_match_ground_truth(comparison):
    """Bins spanning multiple single years of age, where the uniform
    per-year average (no single-year population available) applies."""
    wide = comparison[comparison.age_group_name.isin(WIDE_BINS)]
    rel = ((wide.frac_ever_dose3_prod - wide.frac_ever_dose3_truth)
           / wide.frac_ever_dose3_truth.replace(0, np.nan)).abs().max() * 100
    assert rel < 5.0, (
        f"wide-bin relative dose-3 error {rel}% exceeds tolerance -- check the "
        "50/50 blend in single_year_of_age_fraction or the uniform-average "
        "fallback in fraction_for_bin."
    )


def test_dose3_boundary_bug_would_be_caught(comparison):
    """The specific historical bug: sampling one point that lands on the year
    boundary resolves to the LATER, higher coverage year and overestimates
    dose-3-ever during ramp-up. Confirm the production values are not
    systematically biased upward against ground truth.
    """
    ramp_years = comparison[comparison.year.between(2025, 2035)]
    bias = (ramp_years.frac_ever_dose3_prod - ramp_years.frac_ever_dose3_truth).mean()
    assert abs(bias) < 0.005, f"systematic dose-3 bias during ramp-up: {bias}"


def test_dose4_never_exceeds_dose3_in_output(comparison):
    assert (comparison.frac_ever_dose4_prod <= comparison.frac_ever_dose3_prod + 1e-12).all()
