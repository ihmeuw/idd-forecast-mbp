"""
Pure logic for tracking malaria vaccine (dose 3 / dose 4) coverage and
efficacy-adjusted protection by birth cohort.

No I/O, no population, no mortality. This module answers one question:
"for someone born at a given time (or currently a given single-year-of-age),
what fraction of their birth cohort ever received dose 3, ever received
dose 4, and what is their current efficacy-adjusted protection against
clinical cases and against severe/fatal outcomes?"

Design rationale (see .claude/vaccination_scripts/
claude_code_prompt_vaccine_cohort_tracking.md for the full write-up):

- Dose 3 is given at exactly 6 months of age (0.5 years); dose 4 at exactly
  2 years. Because both are fixed-age events, "time since dose" falls out
  of current age for free -- no extra state is needed.
- `dose_4` coverage as reported in the real data is ALREADY the
  unconditional fraction of the birth cohort that completed the booster
  (dropout already applied). Do not multiply it by dose_3 again. Verified
  against the real Kebbi, Nigeria series: dose_4(t)/dose_3(t-2) settles to
  a constant by 2029-2033.

  PRECISION NOTE: that constant computes to ~0.640-0.642 from the real
  rows, not exactly 0.633 (= 1 - 0.367). Do not hardcode 0.633 in any
  tolerance -- see tests/lib/processing/test_vaccine_cohort_fractions.py.
- Protection comes from a DELIVERED MONTHLY VE CURVE (`VECurve`, defined in
  `vaccine_efficacy.py` -- that module owns curve construction and the curve
  type; this one owns the birth-cohort logic), not from a parametric decay. The curve supplies, per vaccine product and per month of
  age, VE for the dose-3-only and the boosted (dose 3 + 4) states, for both
  clinical ("case") and severe/fatal ("death") outcomes. An earlier version
  of this module used placeholder initial-efficacy/half-life parameters; the
  real curves are piecewise interpolations through trial anchors and are not
  exponential, so that form is gone.
- Fraction-vaccinated is invariant to non-differential mortality (mortality
  that doesn't depend on vaccination status), so this module deliberately
  has no population/mortality logic at all -- multiply its outputs onto
  real population numbers at the call site.
- This implementation assumes vaccination does not affect all-cause or
  malaria-specific mortality. If a future requirement needs the vaccine's
  mortality-reduction effect reflected in the age structure (i.e.
  vaccinated children preferentially surviving and therefore becoming an
  over-represented share of older age groups over time), that requires a
  different, hazard-based multi-state model and is out of scope here --
  flagged as a known follow-up, deliberately not attempted. Note this is
  unchanged by the arrival of `ve_death_*`: those columns describe efficacy
  against death, they do not feed back into the cohort's age structure.
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from idd_forecast_mbp.lib.processing.vaccine_efficacy import VECurve, VE_COLUMNS

D3_TRIGGER_AGE = 0.5   # years
D4_TRIGGER_AGE = 2.0   # years

# The program started in 2024, so nobody older than this can carry a dose.
# Used by callers to skip age groups that are always all-zero.
VACCINE_RELEVANT_MAX_AGE = 20.0

# Where within the reporting year age is evaluated. This is a real modelling
# choice, not a detail: a child with `age_int` completed years at time t+phi was
# born in (t+phi-age_int-1, t+phi-age_int], so phi decides which calendar years
# each dose trigger falls in.
#
#   start_of_year : dose 3 blends two years 50/50, dose 4 is single-valued
#   mid_year      : dose 3 is single-valued, dose 4 blends 50/50
#   end_of_year   : as start_of_year, shifted one year later
#
# Population estimates are commonly mid-year, which would argue for `mid_year`;
# the pipeline has used `start_of_year` since 2026-08-24 and the ground-truth
# harness shares that convention, so it cannot falsify it. Measured difference on
# real coverage: up to 6.7 percentage points during ramp-up years, converging to
# ~0.4pp at plateau. See DECISIONS.md 2026-08-24 -- the choice is PARKED, and
# this parameter exists so revisiting it is a flag rather than surgery.
AGE_REFERENCE_PHI = {"start_of_year": 0.0, "mid_year": 0.5, "end_of_year": 1.0}
DEFAULT_AGE_REFERENCE = "start_of_year"


def year_weights(age_reference: str, trigger_age: float) -> Dict[int, float]:
    """Weights over calendar years for a dose trigger `trigger_age` after birth.

    Keys are offsets relative to `by + 1` (where `by = year - age_int - 1`);
    values sum to 1. Derived from the birth window a completed-age band implies:
    the trigger falls in (phi + trigger_age - 1, phi + trigger_age] measured from
    that reference year.
    """
    try:
        phi = AGE_REFERENCE_PHI[age_reference]
    except KeyError as exc:
        raise ValueError(
            f"unknown age_reference {age_reference!r}; expected one of "
            f"{sorted(AGE_REFERENCE_PHI)}"
        ) from exc
    lo, hi = phi + trigger_age - 1.0, phi + trigger_age
    weights, k = {}, math.floor(lo)
    while k < hi:
        overlap = min(hi, k + 1.0) - max(lo, float(k))
        if overlap > 1e-12:
            weights[k] = overlap
        k += 1
    return weights

@dataclass
class CoverageSeries:
    """Coverage lookup for a single location + vaccine product, plus the VE
    curves used to turn coverage into protection.

    dose3/dose4 are {calendar_year: coverage_fraction}. Any year not present
    (including anything before the program existed) is treated as 0
    coverage -- callers should confirm the input data's year range actually
    covers birth_year+2 for every cohort of interest; this class does not
    warn about out-of-range lookups.

    `vacc_name` and `ve` ride on the series rather than being passed
    separately: a coverage series is already scoped to one location and one
    product, so the VE lookup stays consistent by construction.
    """
    dose3: Dict[int, float]
    dose4: Dict[int, float]
    vacc_name: str
    ve: VECurve

    def d3(self, year: int) -> float:
        return self.dose3.get(int(year), 0.0)

    def d4(self, year: int) -> float:
        return self.dose4.get(int(year), 0.0)

    def protection(self, age: float, d3_ever: float, d4_ever: float) -> Tuple[float, float]:
        return _protection(self.ve, self.vacc_name, age, d3_ever, d4_ever)


def _protection(ve: VECurve, vacc_name: str, age: float,
                d3_ever: float, d4_ever: float) -> Tuple[float, float]:
    """(case, death) protection at `age`, given the cohort's eventual
    (un-gated) d3_ever/d4_ever fractions.

    Same three-branch structure as the parametric version it replaces, with
    curve lookups in place of the decay terms:

    - below the dose-3 trigger nobody is protected;
    - between the triggers the whole dose-3 share sits on the dose-3-only
      curve (nobody has been boosted yet);
    - at or above the dose-4 trigger the boosted share moves to the boosted
      curve and the remainder (d3_ever - d4_ever) stays on the dose-3-only
      curve.

    Deliberately branches rather than interpolating the boosted curve across
    the booster age: `ve_*_d34` steps discontinuously at month 24, and
    interpolating across that step would be meaningless. Below the trigger
    the boosted curve is never consulted.

    Still linear in d3_ever and d4_ever at fixed age, which is what makes
    averaging fractions across a bin equal to averaging protection.
    """
    if age < D3_TRIGGER_AGE:
        return 0.0, 0.0
    if age < D4_TRIGGER_AGE:
        return (d3_ever * ve.ve(vacc_name, "ve_case_d3", age),
                d3_ever * ve.ve(vacc_name, "ve_death_d3", age))
    d3_only = d3_ever - d4_ever
    case = (d4_ever * ve.ve(vacc_name, "ve_case_d34", age)
            + d3_only * ve.ve(vacc_name, "ve_case_d3", age))
    death = (d4_ever * ve.ve(vacc_name, "ve_death_d34", age)
             + d3_only * ve.ve(vacc_name, "ve_death_d3", age))
    return case, death


def cohort_fraction_at_age(coverage: CoverageSeries, birth_time: float,
                           age: float) -> Tuple[float, float, float, float]:
    """Fractions for a birth cohort identified by continuous `birth_time`
    (e.g. 2027.46 for a cohort born in June 2027), evaluated at continuous
    `age`. Use this for sub-annual reporting bins (Early Neonatal ...
    6-11 months), where a single representative age/birth_time point is
    accurate (validated: exact to numerical noise for bins <=1 year wide
    and entirely below age 2).

    Returns (d3_ever, d4_ever, protection_case, protection_death), where
    d3_ever/d4_ever are gated to 0 until `age` has actually reached the
    corresponding trigger age.
    """
    d3_trig_time = birth_time + D3_TRIGGER_AGE
    d4_trig_time = birth_time + D4_TRIGGER_AGE
    d3_year, d4_year = int(d3_trig_time // 1), int(d4_trig_time // 1)
    d3_ever_eventual = coverage.d3(d3_year)
    d4_ever_eventual = coverage.d4(d4_year)
    d3_ever = d3_ever_eventual if age >= D3_TRIGGER_AGE else 0.0
    d4_ever = d4_ever_eventual if age >= D4_TRIGGER_AGE else 0.0
    prot_case, prot_death = coverage.protection(age, d3_ever_eventual, d4_ever_eventual)
    return d3_ever, d4_ever, prot_case, prot_death


def single_year_of_age_fraction(coverage: CoverageSeries, age_int: int, year: int,
                                 age_reference: str = DEFAULT_AGE_REFERENCE
                                 ) -> Tuple[float, float, float, float]:
    """Fractions for someone who is exactly `age_int` completed years old
    as of calendar year `year`, using ONLY annual-resolution coverage data
    (no birth-month detail).

    Without birth-month detail, this person's birth calendar year is
    `year - age_int - 1` for virtually their entire age-year. From that:

    - Dose 3's 6-month trigger is a *half*-integer offset from birth, so it
      straddles TWO calendar years 50/50 -- average the two.
    - Dose 4's 2-year trigger is a *whole*-integer offset from birth, so it
      does NOT straddle any boundary -- it's deterministic.

    This function was the source of a real bug during development: an
    earlier version sampled a single point that happened to land exactly
    on the year boundary, which silently resolves to the LATER (higher)
    coverage year and systematically overestimates dose-3-ever fractions
    by several percentage points during ramp-up years. The 50/50 blend
    below is the fix -- see
    tests/lib/processing/test_vaccine_cohort_ground_truth.py.

    Use this for "12-23 months" and any wider bin (2 to 4, 5 to 9, ...).

    Requires age_int >= 1. The 50/50 dose-3 blend assumes the ENTIRE
    age-year in question is already past the 6-month trigger (true for
    anyone who has had at least 1 full completed year of life). Calling
    this with age_int=0 would be wrong: part of that age-year (ages
    0-0.5) hasn't reached the dose-3 trigger at all, and this function has
    no way to represent "partially triggered within the year" -- it would
    silently return a non-zero, biased-high d3_ever for the whole age-0
    slice. If you ever need a bin that includes age 0-1 as part of a wider
    bin, that age-0 slice must go through `cohort_fraction_at_age` at
    sub-annual resolution instead, not through this function.
    """
    if age_int < 1:
        raise ValueError(
            f"single_year_of_age_fraction requires age_int >= 1 (got {age_int}); "
            "age 0-1 straddles the dose-3 trigger itself and must be handled at "
            "sub-annual resolution via cohort_fraction_at_age, not averaged here."
        )
    by = year - age_int - 1  # birth calendar year for the reference convention
    d3_weights = year_weights(age_reference, D3_TRIGGER_AGE)
    d4_weights = year_weights(age_reference, D4_TRIGGER_AGE)
    d3_ever = sum(w * coverage.d3(by + 1 + k) for k, w in d3_weights.items())
    d4_ever = (sum(w * coverage.d4(by + 1 + k) for k, w in d4_weights.items())
               if age_int >= 2 else 0.0)
    age_exact = age_int + 0.5  # representative point in time for the VE lookup
    prot_case, prot_death = coverage.protection(age_exact, d3_ever, d4_ever)
    # NOTE: protection uses the blended d3_ever directly. This is exact
    # (not an approximation) because `_protection` is linear in d3_ever and
    # d4_ever for fixed age, so averaging inputs equals averaging outputs.
    return d3_ever, d4_ever, prot_case, prot_death


def fraction_for_bin(coverage: CoverageSeries, age_start: float, age_end: float, year: int,
                      single_year_weights: Optional[Dict[int, float]] = None,
                      age_reference: str = DEFAULT_AGE_REFERENCE
                      ) -> Tuple[float, float, float, float]:
    """Convenience wrapper: given a reporting bin [age_start, age_end) and
    calendar year, return (d3_ever, d4_ever, protection_case,
    protection_death) for the WHOLE bin.

    - Bins <=1 year wide, entirely below age 2: single representative point
      via cohort_fraction_at_age.
    - Wider bins (or "12-23 months" specifically, which is exactly 1 year
      wide starting at age 1): averaged across each single year of age in
      the bin via single_year_of_age_fraction.

    `single_year_weights`, if provided, maps {age_int: population_share}
    for a population-weighted average across the single years of age in
    the bin. If not provided, falls back to a uniform (equal-weight-per-
    year) average -- validated to introduce a peak error of well under 1%
    of the bin's population during the fastest coverage-ramp years,
    vanishing once coverage plateaus.

    For this repo that fallback is always what runs: the population source
    (02-processed_data/population/<hierarchy>/current/as_*_population_df.parquet)
    carries the standard 25 GBD age groups, whose finest resolution over
    ages 2-4 is the lumped "2 to 4" bin (age_group_id 34). Checked against
    age_specific_fhs/age_metadata.parquet: the only group at or above age 1
    that is <=1 year wide is 238 (12-23 months). No single-year-of-age
    population exists anywhere in the pipeline to weight with.

    A bin that is wider than 1 year but starts below age 1 would route to
    the averaging path and raise from single_year_of_age_fraction(age_int=0)
    -- deliberately loud, since that case needs sub-annual handling. No
    such bin exists in the GBD schema this repo uses.
    """
    if age_end - age_start <= 1.0 and age_end <= 2.0 and not (age_start == 1.0 and age_end == 2.0):
        mid_age = (age_start + age_end) / 2
        birth_time = year - mid_age
        return cohort_fraction_at_age(coverage, birth_time, mid_age)

    int_ages = list(range(int(age_start), int(age_end)))
    fracs = [single_year_of_age_fraction(coverage, a, year, age_reference)
             for a in int_ages]
    if single_year_weights is None:
        weights = [1.0 / len(int_ages)] * len(int_ages)
    else:
        total = sum(single_year_weights.get(a, 0.0) for a in int_ages)
        if total == 0:
            weights = [1.0 / len(int_ages)] * len(int_ages)
        else:
            weights = [single_year_weights.get(a, 0.0) / total for a in int_ages]
    return tuple(
        sum(w * f[i] for w, f in zip(weights, fracs)) for i in range(4)
    )
