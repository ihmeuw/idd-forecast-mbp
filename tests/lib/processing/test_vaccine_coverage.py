"""Tests for the coverage-series transforms.

These functions used to live in the stage script with no tests at all. The
contract they protect is that `dose_4` as delivered is UNCONDITIONAL -- never
multiplied by dose_3 -- and that the two pre-series-dose_4 treatments are
explicit rather than silent.
"""
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.vaccine_cohort_fractions import CoverageSeries
from idd_forecast_mbp.lib.processing.vaccine_coverage import (
    COVERAGE_MEASURES,
    DOSE4_EXCESS_TOLERANCE,
    INFANT_AGE_GROUP_IDS,
    NEVER_COVERABLE_AGE_GROUP_IDS,
    dose_counts,
    weighted_coverage,
    ID_DTYPES,
    VALUE_DTYPES,
    apply_product_scenario,
    backcast_prelag_dose3,
    build_coverage_series,
    build_protection_table,
    check_dose4_not_exceeding_dose3,
    compute_fractions,
    implied_dropout_ratios,
    validate_coverage_frame,
    zero_prelag_dose4,
)
from idd_forecast_mbp.lib.processing.vaccine_efficacy import VE_COLUMNS, VECurve

RATIO = 0.633
D3 = {2024: 0.10, 2025: 0.20, 2026: 0.30, 2027: 0.40, 2028: 0.50, 2029: 0.60, 2030: 0.70}


def _rows(subnat_id, vacc_name, prelag_dose4):
    """One location's series. `prelag_dose4` puts dose_4 in 2024/2025, whose
    dose-3 antecedent (2022/2023) predates the series -- the real-data defect."""
    out = []
    for year, d3 in D3.items():
        if year >= 2026:
            d4 = round(RATIO * D3[year - 2], 6)
        else:
            d4 = round(RATIO * D3[2024], 6) if prelag_dose4 else 0.0
        out.append(dict(subnat_id=subnat_id, country_id=1, country="X", subnat=f"S{subnat_id}",
                        year_id=year, dose_3=d3, dose_4=d4,
                        vacc_id=1 if vacc_name == "rtss" else 2, vacc_name=vacc_name))
    return out


@pytest.fixture
def coverage() -> pd.DataFrame:
    """Location 10 is clean; location 20 has the pre-lag dose_4 defect and rtss."""
    return pd.DataFrame(_rows(10, "r21", False) + _rows(20, "rtss", True))


@pytest.fixture
def ve() -> VECurve:
    curves = {}
    for product in ("rtss", "r21"):
        cols = {c: [] for c in VE_COLUMNS}
        for month in range(61):
            vals = (0.0, 0.0, 0.0, 0.0) if month < 6 else (
                (0.6, 0.6, 0.45, 0.45) if month < 24 else (0.6, 0.9, 0.0, 0.7))
            for col, v in zip(VE_COLUMNS, vals):
                cols[col].append(v)
        curves[product] = cols
    return VECurve(curves=curves)


# ---------------------------------------------------------------------------
# validate_coverage_frame
# ---------------------------------------------------------------------------
def test_valid_frame_passes(coverage, ve):
    validate_coverage_frame(coverage, ve)


def test_missing_column_raises(coverage, ve):
    with pytest.raises(ValueError, match="missing required column"):
        validate_coverage_frame(coverage.drop(columns=["dose_4"]), ve, source="f.csv")


def test_product_without_a_ve_curve_raises(coverage, ve):
    coverage.loc[coverage.index[0], "vacc_name"] = "brand_new"
    with pytest.raises(ValueError, match="have no VE curve"):
        validate_coverage_frame(coverage, ve)


@pytest.mark.parametrize("col, value", [("dose_3", 1.5), ("dose_4", -0.1)])
def test_dose_outside_unit_interval_raises(coverage, ve, col, value):
    coverage.loc[coverage.index[0], col] = value
    with pytest.raises(ValueError, match=rf"{col} outside \[0, 1\]"):
        validate_coverage_frame(coverage, ve)


def test_location_switching_product_raises(coverage, ve):
    mask = (coverage.subnat_id == 10) & (coverage.year_id == 2030)
    coverage.loc[mask, "vacc_name"] = "rtss"
    with pytest.raises(ValueError, match="more than one vacc_name"):
        validate_coverage_frame(coverage, ve)


# ---------------------------------------------------------------------------
# implied dropout ratios
# ---------------------------------------------------------------------------
def test_implied_ratio_recovers_the_construction_ratio(coverage):
    ratios = implied_dropout_ratios(coverage)
    assert set(ratios.index) == {10, 20}
    for loc in (10, 20):
        assert ratios[loc] == pytest.approx(RATIO, abs=1e-6)


def test_location_with_no_usable_denominator_falls_back_to_the_global_median(coverage):
    """A location whose dose_3 never clears the noise floor still gets a ratio."""
    tiny = coverage[coverage.subnat_id == 10].copy()
    tiny["subnat_id"] = 30
    tiny["dose_3"] = 0.001
    ratios = implied_dropout_ratios(pd.concat([coverage, tiny], ignore_index=True))
    assert ratios[30] == pytest.approx(ratios[[10, 20]].median())


# ---------------------------------------------------------------------------
# the two pre-series dose_4 treatments
# ---------------------------------------------------------------------------
def test_backcast_adds_the_recoverable_years_only(coverage):
    out, n_added = backcast_prelag_dose3(coverage)
    assert n_added == 2                                  # loc 20's 2024 and 2025
    added = out[(out.subnat_id == 20) & (out.year_id < 2024)]
    assert sorted(added.year_id) == [2022, 2023]
    assert (added.dose_4 == 0.0).all()                   # no booster evidence for those years
    assert added.dose_3.between(0, 1).all()
    # inverting the ratio recovers the dose_3 that the early dose_4 implies
    assert added.dose_3.iloc[0] == pytest.approx(RATIO * D3[2024] / RATIO, rel=1e-3)


def test_backcast_is_a_noop_when_nothing_predates_the_series(coverage):
    clean = coverage[coverage.subnat_id == 10]
    out, n_added = backcast_prelag_dose3(clean)
    assert n_added == 0
    assert out.equals(clean)


def test_backcast_refuses_to_imply_impossible_coverage(coverage):
    """dose_4 so high that dividing by the dropout ratio exceeds 1."""
    mask = (coverage.subnat_id == 20) & (coverage.year_id == 2024)
    coverage.loc[mask, "dose_4"] = 0.99
    with pytest.raises(ValueError, match="implies dose_3 > 1"):
        backcast_prelag_dose3(coverage)


def test_backcast_refuses_to_overwrite_delivered_rows(coverage):
    """If the series already carries the year the back-cast would write, that is a
    conflict, not something to silently overwrite.

    The placeholder row must have dose_3 == 0: a non-zero value would move
    `first_dose3_year` back to 2022 and there would be no pre-lag rows left to
    back-cast at all.
    """
    extra = coverage[(coverage.subnat_id == 20) & (coverage.year_id == 2024)].copy()
    extra["year_id"] = 2022
    extra["dose_3"] = 0.0
    extra["dose_4"] = 0.0
    with pytest.raises(ValueError, match="would overwrite delivered rows"):
        backcast_prelag_dose3(pd.concat([coverage, extra], ignore_index=True))


def test_zeroing_targets_exactly_the_prelag_rows(coverage):
    out, n_zeroed = zero_prelag_dose4(coverage)
    assert n_zeroed == 2
    early = out[out.year_id.isin([2024, 2025])]
    assert (early.dose_4 == 0.0).all()
    # everything from 2026 on is untouched
    later = out[out.year_id >= 2026].reset_index(drop=True)
    assert later.dose_4.equals(coverage[coverage.year_id >= 2026].reset_index(drop=True).dose_4)


# ---------------------------------------------------------------------------
# product scenario
# ---------------------------------------------------------------------------
def test_projected_scenario_is_a_passthrough(coverage):
    out, rows, locs = apply_product_scenario(coverage, "projected")
    assert (rows, locs) == (0, 0)
    assert out is coverage


def test_all_r21_switches_only_the_non_r21_locations(coverage):
    out, rows, locs = apply_product_scenario(coverage, "all_r21")
    assert locs == 1                       # only location 20 was rtss
    assert rows == len(D3)
    assert set(out.vacc_name.unique()) == {"r21"}
    assert set(out.vacc_id.unique()) == {2}
    assert set(coverage.vacc_name.unique()) == {"r21", "rtss"}   # input not mutated


def test_unknown_product_scenario_raises(coverage):
    with pytest.raises(ValueError, match="unknown product scenario"):
        apply_product_scenario(coverage, "all_rtss")


# ---------------------------------------------------------------------------
# CoverageSeries construction and the fraction grid
# ---------------------------------------------------------------------------
def test_build_coverage_series_carries_product_and_curve(coverage, ve):
    s = build_coverage_series(coverage, 20, ve)
    assert isinstance(s, CoverageSeries)
    assert s.vacc_name == "rtss"
    assert s.ve is ve
    assert s.d3(2026) == pytest.approx(D3[2026])
    assert s.d4(2026) == pytest.approx(RATIO * D3[2024])


def _age_groups() -> pd.DataFrame:
    return pd.DataFrame([
        dict(age_group_id=389, age_group_years_start=0.5, age_group_years_end=1.0),
        dict(age_group_id=238, age_group_years_start=1.0, age_group_years_end=2.0),
        dict(age_group_id=34, age_group_years_start=2.0, age_group_years_end=5.0),
    ])


def test_compute_fractions_covers_the_whole_grid(coverage, ve):
    years = [2028, 2029]
    out = compute_fractions(coverage, _age_groups(), years, ve)
    assert len(out) == 2 * len(years) * 3          # locations x years x age groups
    assert set(out.columns) == {
        "location_id", "year_id", "age_group_id",
        "frac_ever_dose3", "frac_ever_dose4",
        "effective_protection_case", "effective_protection_death"}
    assert (out.frac_ever_dose4 <= out.frac_ever_dose3 + 1e-12).all()
    # the 12-23 month bin cannot have had a booster
    assert (out.loc[out.age_group_id == 238, "frac_ever_dose4"] == 0).all()


# ---------------------------------------------------------------------------
# output assembly
# ---------------------------------------------------------------------------
def _pop(locations=(10, 20), years=(2028,), ages=(389, 238, 34)) -> pd.DataFrame:
    return pd.DataFrame([
        dict(location_id=l, year_id=y, age_group_id=a, sex_id=s, population=1000.0 + a)
        for l in locations for y in years for a in ages for s in (1, 2)
    ])


def test_build_protection_table_broadcasts_over_sex_and_adds_headcounts(coverage, ve):
    fractions = compute_fractions(coverage, _age_groups(), [2028], ve)
    pop = _pop()
    out = build_protection_table(pop, fractions)

    assert len(out) == len(pop)
    assert list(out.columns) == list(ID_DTYPES) + list(VALUE_DTYPES)
    for col, dtype in {**ID_DTYPES, **VALUE_DTYPES}.items():
        assert out[col].dtype == dtype
    assert out.n_ever_dose3.equals(out.population * out.frac_ever_dose3)
    assert out.n_protected_death_equiv.equals(out.population * out.effective_protection_death)
    # both sexes share the fraction, since coverage is not sex-specific
    by_sex = out.pivot_table(index=["location_id", "age_group_id"], columns="sex_id",
                             values="frac_ever_dose3")
    assert (by_sex[1] == by_sex[2]).all()


def test_build_protection_table_refuses_to_silently_drop_rows(coverage, ve):
    """A fraction grid that misses part of the population slice is an error, not
    a quietly shorter table."""
    fractions = compute_fractions(coverage, _age_groups(), [2028], ve)
    with pytest.raises(ValueError, match="merge dropped rows"):
        build_protection_table(_pop(years=(2028, 2029)), fractions)


# ---------------------------------------------------------------------------
# the dose4 <= dose3 contract check
# ---------------------------------------------------------------------------
def _result(d3_vals, d4_vals) -> pd.DataFrame:
    n = len(d3_vals)
    return pd.DataFrame(dict(
        location_id=[10] * n, year_id=range(2024, 2024 + n),
        age_group_id=[34] * n, sex_id=[1] * n, population=[1000.0] * n,
        frac_ever_dose3=d3_vals, frac_ever_dose4=d4_vals))


def test_contract_check_clean_when_dose4_never_exceeds(coverage):
    fatal, info = check_dose4_not_exceeding_dose3(_result([0.5, 0.5], [0.3, 0.3]), coverage)
    assert fatal == "" and info == ""


def test_contract_check_reports_sub_tolerance_excess_as_info(coverage):
    tiny = DOSE4_EXCESS_TOLERANCE / 2
    fatal, info = check_dose4_not_exceeding_dose3(_result([0.5], [0.5 + tiny]), coverage)
    assert fatal == ""
    assert "within tolerance" in info


def test_contract_check_is_fatal_above_tolerance(coverage):
    fatal, info = check_dose4_not_exceeding_dose3(_result([0.2], [0.4]), coverage)
    assert "dose_4 exceeds dose_3" in fatal
    assert "--zero-prelag-dose4" in fatal        # points at the remedies


# ---------------------------------------------------------------------------
# weighted coverage -- the population vs death-weighted framings
# ---------------------------------------------------------------------------
def _weightable() -> pd.DataFrame:
    """Two products, two cells each, with deliberately different weights so the
    population- and death-weighted answers must differ."""
    return pd.DataFrame({
        "year_id": [2030] * 4,
        "vacc_name": ["r21", "r21", "rtss", "rtss"],
        "pop": [1.0, 3.0, 1.0, 1.0],
        "deaths": [3.0, 1.0, 1.0, 1.0],
        "frac_ever_dose3": [0.4, 0.8, 0.2, 0.6],
        "frac_ever_dose4": [0.1, 0.3, 0.0, 0.2],
    })


def test_weighted_coverage_emits_nine_series():
    out = weighted_coverage(_weightable(), "pop")
    assert len(out) == 9
    assert set(out["product"]) == {"r21", "rtss", "either"}
    assert set(out["measure"]) == set(COVERAGE_MEASURES)


def test_weighted_coverage_is_a_weighted_mean():
    out = weighted_coverage(_weightable(), "pop")
    got = out.query("product == 'r21' and measure == 'ever dose 3'").coverage.iloc[0]
    assert got == pytest.approx((1 * 0.4 + 3 * 0.8) / 4)


def test_dose3_only_is_ever_minus_boosted():
    out = weighted_coverage(_weightable(), "pop").set_index(["product", "measure"]).coverage
    for product in ("r21", "rtss", "either"):
        assert out[(product, "dose 3 only")] == pytest.approx(
            out[(product, "ever dose 3")] - out[(product, "dose 3+4")])


def test_changing_the_weight_changes_the_answer():
    """The whole point of the death-weighted framing: same fractions, different
    weights, different headline coverage."""
    d = _weightable()
    by_pop = weighted_coverage(d, "pop").set_index(["product", "measure"]).coverage
    by_death = weighted_coverage(d, "deaths").set_index(["product", "measure"]).coverage
    assert by_pop[("r21", "ever dose 3")] == pytest.approx(0.70)
    assert by_death[("r21", "ever dose 3")] == pytest.approx((3 * 0.4 + 1 * 0.8) / 4)
    assert by_pop[("r21", "ever dose 3")] != pytest.approx(by_death[("r21", "ever dose 3")])


def test_zero_total_weight_yields_nan_not_a_divide_by_zero():
    d = _weightable()
    d["deaths"] = 0.0
    assert weighted_coverage(d, "deaths").coverage.isna().all()


def test_never_coverable_ages_are_the_pre_trigger_bins():
    """These carry death weight but can never be covered, which is why the
    death-weighted series has a ceiling below 1."""
    assert NEVER_COVERABLE_AGE_GROUP_IDS == (2, 3, 388)


# ---------------------------------------------------------------------------
# dose counts -- coverage x cohort, nothing more
# ---------------------------------------------------------------------------
def _dose_fixture():
    """Known cohort sizes and coverage so every dose count is hand-computable."""
    cov = pd.DataFrame({
        "subnat_id": [1, 1, 1], "year_id": [2024, 2025, 2026],
        "dose_3": [0.5, 0.6, 0.7], "dose_4": [0.0, 0.0, 0.4],
        "vacc_name": ["r21"] * 3,
    })
    infants = pd.DataFrame({
        "location_id": [1] * 5, "year_id": [2022, 2023, 2024, 2025, 2026],
        "infants": [100.0, 200.0, 300.0, 400.0, 500.0],
    })
    return cov, infants


def test_dose3_is_coverage_times_the_same_year_cohort():
    cov, infants = _dose_fixture()
    d = dose_counts(cov, infants).set_index("year_id")
    assert d.loc[2024, "doses_dose3"] == pytest.approx(0.5 * 300.0)
    assert d.loc[2026, "doses_dose3"] == pytest.approx(0.7 * 500.0)


def test_dose4_uses_the_cohort_from_two_years_earlier():
    """Children turning 24 months in year t were the infants of t-2."""
    cov, infants = _dose_fixture()
    d = dose_counts(cov, infants).set_index("year_id")
    assert d.loc[2026, "infants_t_minus_lag"] == pytest.approx(300.0)   # the 2024 cohort
    assert d.loc[2026, "doses_dose4"] == pytest.approx(0.4 * 300.0)


def test_dose4_lag_is_configurable():
    cov, infants = _dose_fixture()
    d = dose_counts(cov, infants, dose4_lag_years=1).set_index("year_id")
    assert d.loc[2026, "infants_t_minus_lag"] == pytest.approx(400.0)   # the 2025 cohort


def test_total_is_the_sum_of_both_doses():
    cov, infants = _dose_fixture()
    d = dose_counts(cov, infants)
    assert d.doses_total.equals(d.doses_dose3.fillna(0) + d.doses_dose4.fillna(0))


def test_missing_lagged_cohort_is_nan_not_a_silent_zero():
    """If population does not reach back far enough the dose-4 count must be
    absent, not quietly zero -- a zero would understate doses without warning."""
    cov, infants = _dose_fixture()
    # drop 2022/2023, so 2024's lagged cohort (2022) is gone but 2026's (2024) survives
    d = dose_counts(cov, infants[infants.year_id >= 2024]).set_index("year_id")
    assert pd.isna(d.loc[2024, "infants_t_minus_lag"])
    assert pd.isna(d.loc[2024, "doses_dose4"])
    assert d.loc[2026, "infants_t_minus_lag"] == pytest.approx(300.0)


def test_dose_counts_do_not_route_through_cohort_tracking():
    """A dose count is coverage x cohort size. If the ratio of doses to the
    cohort is not exactly the coverage, something has crept in."""
    cov, infants = _dose_fixture()
    d = dose_counts(cov, infants)
    assert (d.doses_dose3 / d.infants_t).round(10).tolist() == cov.dose_3.round(10).tolist()


def test_infant_age_groups_span_exactly_one_year():
    """The proxy is only valid because these four bins sum to age 0-1."""
    assert INFANT_AGE_GROUP_IDS == (2, 3, 388, 389)
