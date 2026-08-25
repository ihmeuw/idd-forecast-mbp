"""Unit tests for malaria vaccine cohort fraction + VE-curve logic.

Covers the checks called for in the task spec:
  * d4_ever <= d3_ever always holds
  * d3_ever/d4_ever are exactly 0 below their trigger ages
  * the real Kebbi series' dose_4(t)/dose_3(t-2) ratio settles to a constant
    (guarding against reintroducing the "multiply by d3 again" bug)
  * the dose-3 50/50 boundary blend averages rather than picking an endpoint
  * single_year_of_age_fraction refuses age_int < 1
plus the VE-curve lookup that replaced the old parametric waning form.
"""
import pytest

from idd_forecast_mbp.lib.processing.vaccine_efficacy import VECurve, VE_COLUMNS
from idd_forecast_mbp.lib.processing.vaccine_cohort_fractions import (
    AGE_REFERENCE_PHI,
    D3_TRIGGER_AGE,
    D4_TRIGGER_AGE,
    CoverageSeries,
    cohort_fraction_at_age,
    fraction_for_bin,
    single_year_of_age_fraction,
    year_weights,
)

# Real Kebbi, Nigeria rows from the coverage handoff (see the task doc).
KEBBI_DOSE3 = {
    2024: 0.028659685, 2025: 0.1283919621597444, 2026: 0.2623181975453931,
    2027: 0.2865266480815513, 2028: 0.2944779002586327, 2029: 0.3019250129756307,
    2030: 0.3095975746099548, 2031: 0.3170679158042784, 2032: 0.324268412,
    2033: 0.3311618338175892,
}
KEBBI_DOSE4 = {
    2024: 0.0, 2025: 0.0, 2026: 0.032671702, 2027: 0.13042123958470406,
    2028: 0.17789014390935642, 2029: 0.18393324967818903, 2030: 0.18876193499024355,
    2031: 0.19354690272736397, 2032: 0.19833962775025285, 2033: 0.2029829478967828,
}
RAMP_DOSE3 = {2024: 0.03, 2025: 0.13, 2026: 0.26, 2027: 0.29, 2028: 0.30}
RAMP_DOSE4 = {2026: 0.019, 2027: 0.0823, 2028: 0.1646, 2029: 0.1836, 2030: 0.1899}

# Synthetic VE curve with the same STRUCTURE as the delivered files -- zero before
# month 6, boosted column identical to dose-3-only until month 24, then a step up --
# but piecewise-constant so expected values are computable by hand.
D3_CASE, D34_CASE, D3_DEATH, D34_DEATH = 0.60, 0.90, 0.45, 0.70
LAST_MONTH = 60


def _make_curve(products=("rtss", "r21")) -> VECurve:
    curves = {}
    for product in products:
        cols = {c: [] for c in VE_COLUMNS}
        for month in range(LAST_MONTH + 1):
            if month < 6:
                vals = (0.0, 0.0, 0.0, 0.0)
            elif month < 24:
                vals = (D3_CASE, D3_CASE, D3_DEATH, D3_DEATH)
            elif month < LAST_MONTH:
                # severe0 style: unboosted severe VE drops to 0 at the booster
                vals = (D3_CASE, D34_CASE, 0.0, D34_DEATH)
            else:
                vals = (0.0, 0.0, 0.0, 0.0)
            for col, v in zip(VE_COLUMNS, vals):
                cols[col].append(v)
        curves[product] = cols
    return VECurve(curves=curves)


VE = _make_curve()


@pytest.fixture
def kebbi() -> CoverageSeries:
    return CoverageSeries(dose3=KEBBI_DOSE3, dose4=KEBBI_DOSE4, vacc_name="r21", ve=VE)


@pytest.fixture
def ramp() -> CoverageSeries:
    return CoverageSeries(dose3=RAMP_DOSE3, dose4=RAMP_DOSE4, vacc_name="r21", ve=VE)


# ---------------------------------------------------------------------------
# VECurve
# ---------------------------------------------------------------------------
def test_curve_products_and_length():
    assert VE.products() == ("r21", "rtss")
    assert VE.n_months("r21") == LAST_MONTH + 1


@pytest.mark.parametrize("age_years, expected", [
    (0.0, 0.0),
    (-1.0, 0.0),
    (5 / 12, 0.0),           # month 5, before dose 3
    (0.5, D3_CASE),          # month 6 exactly
    (1.0, D3_CASE),
    (LAST_MONTH / 12, 0.0),  # at/after the final month
    (100.0, 0.0),            # far beyond the curve
])
def test_ve_lookup_at_and_beyond_bounds(age_years, expected):
    assert VE.ve("r21", "ve_case_d3", age_years) == pytest.approx(expected)


def test_ve_interpolates_between_whole_months():
    """Month 5 is 0 and month 6 is D3_CASE, so halfway between is half."""
    assert VE.ve("r21", "ve_case_d3", 5.5 / 12) == pytest.approx(D3_CASE / 2)


def test_ve_unknown_product_or_column_raises():
    with pytest.raises(KeyError, match="no VE curve"):
        VE.ve("nonexistent", "ve_case_d3", 1.0)
    with pytest.raises(KeyError, match="no VE curve"):
        VE.ve("r21", "ve_nonexistent", 1.0)


def test_nonzero_tails_empty_when_curve_decays_to_zero():
    assert _make_curve().nonzero_tails() == {}


def test_nonzero_tails_flags_a_truncated_curve():
    """A curve cut off while still positive must be reported, since lookups
    clip to 0 beyond the end and would introduce a silent discontinuity."""
    truncated = VECurve(curves={"r21": {c: [0.0] * 6 + [0.5] * 6 for c in VE_COLUMNS}})
    tails = truncated.nonzero_tails()
    assert set(tails) == {("r21", c) for c in VE_COLUMNS}
    assert all(v == 0.5 for v in tails.values())


def test_implied_trigger_months_matches_the_model():
    """The VE file independently encodes the dose schedule; it must agree with
    D3/D4_TRIGGER_AGE or every protection number would be wrong."""
    assert VE.implied_trigger_months("r21") == (round(D3_TRIGGER_AGE * 12),
                                                round(D4_TRIGGER_AGE * 12))


def test_implied_trigger_months_none_when_curve_is_flat():
    flat = VECurve(curves={"r21": {c: [0.0] * 12 for c in VE_COLUMNS}})
    assert flat.implied_trigger_months("r21") == (None, None)


# ---------------------------------------------------------------------------
# CoverageSeries
# ---------------------------------------------------------------------------
def test_absent_years_read_as_zero_coverage(ramp):
    assert ramp.d3(1999) == 0.0
    assert ramp.d4(2099) == 0.0
    assert ramp.d3(2025) == pytest.approx(0.13)


# ---------------------------------------------------------------------------
# Spec check: d4_ever <= d3_ever always
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("age", [0.0, 0.25, 0.5, 0.99, 1.5, 2.0, 2.5, 4.0, 4.9])
@pytest.mark.parametrize("birth_time", [2023.1, 2024.5, 2026.46, 2028.9, 2031.0])
def test_d4_never_exceeds_d3_continuous(kebbi, birth_time, age):
    d3, d4, case, death = cohort_fraction_at_age(kebbi, birth_time=birth_time, age=age)
    assert d4 <= d3 + 1e-12, "dose4 exceeded dose3 for the same cohort"
    assert 0.0 <= case <= 1.0
    assert 0.0 <= death <= 1.0


@pytest.mark.parametrize("age_int", [1, 2, 3, 4])
@pytest.mark.parametrize("year", [2024, 2028, 2033, 2040])
def test_d4_never_exceeds_d3_single_year(kebbi, age_int, year):
    d3, d4, _, _ = single_year_of_age_fraction(kebbi, age_int=age_int, year=year)
    assert d4 <= d3 + 1e-12


# ---------------------------------------------------------------------------
# Spec check: exact zeros below the trigger ages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("age", [0.0, 0.1, 0.4999])
def test_nothing_before_dose3_trigger(ramp, age):
    assert cohort_fraction_at_age(ramp, birth_time=2026.46, age=age) == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("age", [0.5, 1.0, 1.9999])
def test_dose4_still_zero_between_triggers(ramp, age):
    d3, d4, case, death = cohort_fraction_at_age(ramp, birth_time=2024.46, age=age)
    assert d3 > 0.0
    assert d4 == 0.0
    assert case > 0.0 and death > 0.0


def test_dose4_active_once_trigger_reached(ramp):
    _, before, _, _ = cohort_fraction_at_age(ramp, birth_time=2024.46, age=1.999)
    _, after, _, _ = cohort_fraction_at_age(ramp, birth_time=2024.46, age=2.0)
    assert before == 0.0
    assert after > 0.0


def test_single_year_dose4_gated_below_age_2(ramp):
    _, d4_age1, _, _ = single_year_of_age_fraction(ramp, age_int=1, year=2028)
    _, d4_age2, _, _ = single_year_of_age_fraction(ramp, age_int=2, year=2028)
    assert d4_age1 == 0.0
    assert d4_age2 > 0.0


def test_cohorts_triggering_before_program_start_stay_zero(kebbi):
    assert cohort_fraction_at_age(kebbi, birth_time=2019.6, age=4.0) == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Spec check: the real Kebbi dropout ratio, and no double-multiply
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("year", [2029, 2030, 2031, 2032, 2033])
def test_kebbi_dropout_ratio_settles_to_a_constant(year):
    """Confirms dose_4 is already unconditional. The band is deliberately loose:
    computed exactly from the delivered rows the ratio is ~0.640-0.642, NOT 0.633
    (= 1 - 0.367) as the handoff states. Asserting 0.633 tightly would fail on the
    real data; the loose band still catches the bug this guards, which moves the
    ratio by a factor of ~0.3."""
    assert 0.60 <= KEBBI_DOSE4[year] / KEBBI_DOSE3[year - 2] <= 0.66


def test_d4_ever_is_the_raw_dose4_value_not_multiplied_by_d3(kebbi):
    birth_year = 2029
    d3, d4, _, _ = cohort_fraction_at_age(kebbi, birth_time=birth_year + 0.46, age=3.0)
    expected = KEBBI_DOSE4[birth_year + 2]
    assert d4 == pytest.approx(expected)
    assert d4 != pytest.approx(expected * d3)


# ---------------------------------------------------------------------------
# Spec check: the dose-3 50/50 boundary blend
# ---------------------------------------------------------------------------
def test_dose3_blend_is_the_average_not_either_endpoint():
    cov = CoverageSeries(dose3={2026: 0.10, 2027: 0.90}, dose4={}, vacc_name="r21", ve=VE)
    d3, _, _, _ = single_year_of_age_fraction(cov, age_int=1, year=2028)  # born 2026
    assert d3 == pytest.approx(0.50)
    assert d3 != pytest.approx(0.10)
    assert d3 != pytest.approx(0.90)


def test_dose4_is_deterministic_not_blended():
    cov = CoverageSeries(dose3={}, dose4={2027: 0.10, 2028: 0.90}, vacc_name="r21", ve=VE)
    _, d4, _, _ = single_year_of_age_fraction(cov, age_int=2, year=2028)
    assert d4 == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Spec check: the age_int >= 1 guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("age_int", [0, -1])
def test_single_year_of_age_rejects_age_below_one(ramp, age_int):
    with pytest.raises(ValueError, match="requires age_int >= 1"):
        single_year_of_age_fraction(ramp, age_int=age_int, year=2028)


def test_wide_bin_starting_below_age_one_raises(ramp):
    """A bin wider than 1 year starting below age 1 routes to the averaging path,
    which must fail loudly rather than return a biased-high age-0 slice."""
    with pytest.raises(ValueError, match="requires age_int >= 1"):
        fraction_for_bin(ramp, age_start=0.0, age_end=2.0, year=2028)


# ---------------------------------------------------------------------------
# Protection: the two outcomes and the three branches
# ---------------------------------------------------------------------------
def test_protection_between_triggers_uses_dose3_only_curve(ramp):
    """No one is boosted yet, so the whole dose-3 share sits on the d3 curve
    for both outcomes."""
    d3, d4, case, death = cohort_fraction_at_age(ramp, birth_time=2024.46, age=1.0)
    assert d4 == 0.0
    assert case == pytest.approx(d3 * D3_CASE)
    assert death == pytest.approx(d3 * D3_DEATH)


def test_protection_above_booster_splits_boosted_and_dose3_only(ramp):
    """Boosted share on the d34 curve, remainder on the d3 curve. Under this
    synthetic curve's severe0 shape the unboosted severe VE is 0, so the death
    protection comes only from the boosted share."""
    d3, d4, case, death = cohort_fraction_at_age(ramp, birth_time=2024.46, age=2.5)
    assert d4 > 0.0
    assert case == pytest.approx(d4 * D34_CASE + (d3 - d4) * D3_CASE)
    assert death == pytest.approx(d4 * D34_DEATH)


def test_protection_is_linear_in_the_fractions(ramp):
    """The property that makes averaging fractions across a bin equal to
    averaging protection -- which the 50/50 blend and bin averaging rely on."""
    a = ramp.protection(2.5, 0.4, 0.2)
    b = ramp.protection(2.5, 0.8, 0.4)
    assert b[0] == pytest.approx(2 * a[0])
    assert b[1] == pytest.approx(2 * a[1])


def test_zero_coverage_gives_zero_protection():
    cov = CoverageSeries(dose3={}, dose4={}, vacc_name="rtss", ve=VE)
    for age in (0.6, 1.5, 3.0, 4.5):
        assert cohort_fraction_at_age(cov, birth_time=2026.0, age=age) == (0.0, 0.0, 0.0, 0.0)


def test_protection_zero_once_ve_curve_is_exhausted(ramp):
    """Beyond the curve's last month VE is 0, so protection is 0 even though the
    cohort still counts as ever-vaccinated."""
    d3, d4, case, death = cohort_fraction_at_age(ramp, birth_time=2026.46, age=LAST_MONTH / 12)
    assert d3 > 0.0
    assert (case, death) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# fraction_for_bin routing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("age_start, age_end", [
    (0.0, 0.019178), (0.019178, 0.076712), (0.076712, 0.5), (0.5, 1.0),
])
def test_narrow_sub_age_two_bins_use_the_midpoint_path(ramp, age_start, age_end):
    year = 2028
    mid = (age_start + age_end) / 2
    assert fraction_for_bin(ramp, age_start, age_end, year) == cohort_fraction_at_age(
        ramp, birth_time=year - mid, age=mid
    )


def test_twelve_to_23_months_uses_the_single_year_path(ramp):
    """238 is exactly 1 year wide but starts at age 1, so it must route to the
    single-year logic (the 50/50 blend), not the midpoint shortcut."""
    year = 2028
    assert fraction_for_bin(ramp, 1.0, 2.0, year) == single_year_of_age_fraction(
        ramp, age_int=1, year=year
    )


def test_multi_year_bin_uniform_average(ramp):
    year = 2030
    per_age = [single_year_of_age_fraction(ramp, a, year) for a in (2, 3, 4)]
    got = fraction_for_bin(ramp, 2.0, 5.0, year)
    for i in range(4):
        assert got[i] == pytest.approx(sum(f[i] for f in per_age) / 3)


def test_multi_year_bin_population_weighted(ramp):
    year = 2030
    weights = {2: 6.0, 3: 3.0, 4: 1.0}
    per_age = {a: single_year_of_age_fraction(ramp, a, year) for a in (2, 3, 4)}
    got = fraction_for_bin(ramp, 2.0, 5.0, year, single_year_weights=weights)
    total = sum(weights.values())
    for i in range(4):
        assert got[i] == pytest.approx(
            sum(weights[a] / total * per_age[a][i] for a in (2, 3, 4))
        )
    assert got != fraction_for_bin(ramp, 2.0, 5.0, year)


def test_all_zero_weights_falls_back_to_uniform(ramp):
    year = 2030
    got = fraction_for_bin(ramp, 2.0, 5.0, year, single_year_weights={2: 0.0, 3: 0.0, 4: 0.0})
    assert got == fraction_for_bin(ramp, 2.0, 5.0, year)


def test_weights_missing_an_age_are_treated_as_zero(ramp):
    year = 2030
    got = fraction_for_bin(ramp, 2.0, 5.0, year, single_year_weights={3: 1.0})
    assert got == pytest.approx(single_year_of_age_fraction(ramp, 3, year))


def test_older_bins_are_all_zero_during_early_program_years(kebbi):
    """Nobody aged 15-19 in 2030 could have been vaccinated -- the program started
    in 2024, when they were already past both triggers."""
    assert fraction_for_bin(kebbi, 15.0, 20.0, 2030) == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# age-reference convention (the parked decision, made a parameter)
# ---------------------------------------------------------------------------
def test_year_weights_sum_to_one_for_every_convention():
    for ref in AGE_REFERENCE_PHI:
        for trigger in (D3_TRIGGER_AGE, D4_TRIGGER_AGE):
            assert sum(year_weights(ref, trigger).values()) == pytest.approx(1.0)


def test_the_blend_moves_between_the_doses_with_the_convention():
    """Under start-of-year dose 3 straddles two calendar years and dose 4 does
    not; under mid-year it is the other way round. This is the whole substance of
    the convention choice."""
    assert year_weights("start_of_year", D3_TRIGGER_AGE) == {-1: 0.5, 0: 0.5}
    assert year_weights("start_of_year", D4_TRIGGER_AGE) == {1: 1.0}
    assert year_weights("mid_year", D3_TRIGGER_AGE) == {0: 1.0}
    assert year_weights("mid_year", D4_TRIGGER_AGE) == {1: 0.5, 2: 0.5}


def test_end_of_year_is_start_of_year_shifted_by_one():
    start = year_weights("start_of_year", D3_TRIGGER_AGE)
    end = year_weights("end_of_year", D3_TRIGGER_AGE)
    assert {k + 1: v for k, v in start.items()} == end


def test_unknown_age_reference_raises():
    with pytest.raises(ValueError, match="unknown age_reference"):
        year_weights("whenever", D3_TRIGGER_AGE)


def test_default_reference_reproduces_the_historical_blend(ramp):
    """The default must not change any number: start_of_year is what the pipeline
    has always done."""
    by = 2028 - 1 - 1
    expected = 0.5 * ramp.d3(by) + 0.5 * ramp.d3(by + 1)
    assert single_year_of_age_fraction(ramp, 1, 2028)[0] == pytest.approx(expected)


def test_mid_year_reads_a_single_coverage_year_for_dose3(ramp):
    by = 2028 - 1 - 1
    got = single_year_of_age_fraction(ramp, 1, 2028, age_reference="mid_year")[0]
    assert got == pytest.approx(ramp.d3(by + 1))


def test_conventions_differ_most_during_a_ramp(ramp):
    """The measured gap on real data was up to 6.7pp in ramp years; on a steep
    synthetic ramp the two conventions must visibly disagree."""
    start = single_year_of_age_fraction(ramp, 1, 2027)[0]
    mid = single_year_of_age_fraction(ramp, 1, 2027, age_reference="mid_year")[0]
    assert abs(mid - start) > 0.05


def test_conventions_agree_once_coverage_is_flat():
    flat = CoverageSeries(dose3={y: 0.4 for y in range(2020, 2040)},
                          dose4={y: 0.25 for y in range(2020, 2040)},
                          vacc_name="r21", ve=VE)
    vals = [single_year_of_age_fraction(flat, 3, 2035, age_reference=r)[:2]
            for r in AGE_REFERENCE_PHI]
    assert all(v == pytest.approx(vals[0]) for v in vals)


def test_fraction_for_bin_passes_the_convention_through(ramp):
    a = fraction_for_bin(ramp, 2.0, 5.0, 2030)
    b = fraction_for_bin(ramp, 2.0, 5.0, 2030, age_reference="mid_year")
    assert a != pytest.approx(b)
