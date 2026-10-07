"""Arm weights: one definition serves the all-age products and the admin-2 age/sex recreation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.arm_factors import (
    admin2_protection,
    all_age_factors,
    arm_age_sex_counts,
    cell_weights,
    hold_population,
    structure_multiplier,
)
from idd_forecast_mbp.lib.processing.disaggregation import malaria_as_fractions

AGES = [2, 3, 6]  # 2 is the zeroed neonatal group
SEXES = [1, 2]
YEARS = [2023, 2030]
LOCS = [11, 12]


@pytest.fixture
def as_pop() -> pd.DataFrame:
    rows = []
    for loc in LOCS:
        for y in YEARS:
            for a in AGES:
                for s in SEXES:
                    # the population ages: the oldest group grows by 2030
                    pop = 100.0 * loc + 10 * a + s + (50.0 * a if y == 2030 else 0.0)
                    rows.append((loc, y, a, s, pop))
    return pd.DataFrame(
        rows, columns=["location_id", "year_id", "age_group_id", "sex_id", "population"]
    )


@pytest.fixture
def rr() -> pd.DataFrame:
    rows = [
        (loc, a, s, 1.0 + 0.5 * (a == 3), 2.0 - 0.5 * (s == 2))
        for loc in LOCS
        for a in AGES
        for s in SEXES
    ]
    return pd.DataFrame(
        rows,
        columns=["location_id", "age_group_id", "sex_id", "rr_inc_as", "rr_mort_as"],
    )


@pytest.fixture
def protection(as_pop: pd.DataFrame) -> pd.DataFrame:
    p = as_pop[["location_id", "year_id", "age_group_id", "sex_id"]].copy()
    p["effective_protection_case"] = np.where(p.age_group_id == 3, 0.4, 0.0)
    p["effective_protection_death"] = np.where(p.age_group_id == 3, 0.3, 0.0)
    return p


@pytest.fixture
def aa_counts() -> pd.DataFrame:
    rows = [
        (loc, y, d, 1000.0 * loc + y + d, 10.0 * loc + d)
        for loc in LOCS
        for y in YEARS
        for d in range(3)
    ]
    return pd.DataFrame(
        rows, columns=["location_id", "year_id", "draw", "inc_count", "mort_count"]
    )


def test_no_change_weights_are_the_fractions_and_sum_to_one(
    as_pop: pd.DataFrame, rr: pd.DataFrame
) -> None:
    f = malaria_as_fractions(rr, as_pop)
    w = cell_weights(f)
    np.testing.assert_allclose(w.inc_weight, f.inc_fraction)
    fac = all_age_factors(w)
    np.testing.assert_allclose(fac.inc_factor, 1.0)
    np.testing.assert_allclose(fac.mort_factor, 1.0)


def test_age_sex_counts_sum_to_all_age_times_factor(
    as_pop: pd.DataFrame,
    rr: pd.DataFrame,
    protection: pd.DataFrame,
    aa_counts: pd.DataFrame,
) -> None:
    """The recreation and the all-age product agree exactly (the module's contract)."""
    f = malaria_as_fractions(rr, as_pop)
    w = cell_weights(
        f, protection=protection, multiplier=structure_multiplier(as_pop, 2023)
    )
    as_counts = arm_age_sex_counts(aa_counts, w)
    summed = as_counts.groupby(["location_id", "year_id", "draw"], as_index=False)[
        ["inc_count", "mort_count"]
    ].sum()
    expect = aa_counts.merge(all_age_factors(w), on=["location_id", "year_id"])
    expect["inc_count"] *= expect.inc_factor
    expect["mort_count"] *= expect.mort_factor
    merged = summed.merge(
        expect, on=["location_id", "year_id", "draw"], suffixes=("", "_exp")
    )
    np.testing.assert_allclose(merged.inc_count, merged.inc_count_exp, rtol=1e-12)
    np.testing.assert_allclose(merged.mort_count, merged.mort_count_exp, rtol=1e-12)


def test_protection_reduces_only_protected_cells(
    as_pop: pd.DataFrame,
    rr: pd.DataFrame,
    protection: pd.DataFrame,
    aa_counts: pd.DataFrame,
) -> None:
    f = malaria_as_fractions(rr, as_pop)
    base = arm_age_sex_counts(aa_counts, cell_weights(f))
    vacc = arm_age_sex_counts(aa_counts, cell_weights(f, protection=protection))
    m = base.merge(
        vacc,
        on=["location_id", "year_id", "age_group_id", "sex_id", "draw"],
        suffixes=("_b", "_v"),
    )
    protected = m.age_group_id == 3
    np.testing.assert_allclose(m.inc_count_v[protected], 0.6 * m.inc_count_b[protected])
    np.testing.assert_allclose(
        m.mort_count_v[protected], 0.7 * m.mort_count_b[protected]
    )
    np.testing.assert_allclose(m.inc_count_v[~protected], m.inc_count_b[~protected])


def test_structure_multiplier_is_one_at_the_hold_year_and_matches_the_definition(
    as_pop: pd.DataFrame,
) -> None:
    m = structure_multiplier(as_pop, 2023)
    np.testing.assert_allclose(m.loc[m.year_id == 2023, "multiplier"], 1.0)
    # the DECISIONS 2026-07-31 definition, computed directly from the populations
    p = as_pop.copy()
    aa = (
        p.groupby(["location_id", "year_id"], as_index=False)["population"]
        .sum()
        .rename(columns={"population": "aa"})
    )
    p = p.merge(aa, on=["location_id", "year_id"])
    ref = p[p.year_id == 2023][
        ["location_id", "age_group_id", "sex_id", "population", "aa"]
    ]
    p = p.merge(
        ref, on=["location_id", "age_group_id", "sex_id"], suffixes=("", "_ref")
    )
    p["expect"] = (p.population_ref / p.population) / (p.aa_ref / p.aa)
    got = p.merge(m, on=["location_id", "year_id", "age_group_id", "sex_id"])
    np.testing.assert_allclose(got.multiplier, got.expect)


def test_structure_multiplier_refuses_a_missing_hold_year(as_pop: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no rows for hold year 1999"):
        structure_multiplier(as_pop, 1999)


def test_hold_population_freezes_every_year_at_the_hold_value(
    as_pop: pd.DataFrame,
) -> None:
    held = hold_population(
        as_pop, 2023, keys=["location_id", "year_id", "age_group_id", "sex_id"]
    )
    ref = (
        as_pop[as_pop.year_id == 2023]
        .set_index(["location_id", "age_group_id", "sex_id"])
        .population
    )
    for y in YEARS:
        got = (
            held[held.year_id == y]
            .set_index(["location_id", "age_group_id", "sex_id"])
            .population
        )
        pd.testing.assert_series_equal(
            got.sort_index(), ref.sort_index(), check_names=False
        )


def test_hold_population_refuses_missing_rows(as_pop: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no rows for hold year 1999"):
        hold_population(
            as_pop, 1999, keys=["location_id", "year_id", "age_group_id", "sex_id"]
        )
    partial = as_pop[~((as_pop.year_id == 2023) & (as_pop.location_id == 12))]
    with pytest.raises(ValueError, match="have no 2023 population"):
        hold_population(
            partial, 2023, keys=["location_id", "year_id", "age_group_id", "sex_id"]
        )


def test_admin2_protection_broadcasts_admin1_to_children(
    protection: pd.DataFrame,
) -> None:
    a1 = protection[protection.location_id == 11].copy()
    mapping = pd.DataFrame(
        {"location_id": [101, 102], "coverage_location_id": [11, 11]}
    )
    out = admin2_protection(a1, mapping)
    assert set(out.location_id) == {101, 102}
    assert len(out) == 2 * len(a1)
    np.testing.assert_allclose(
        out[out.location_id == 101].effective_protection_case.to_numpy(),
        a1.effective_protection_case.to_numpy(),
    )


def test_protection_outside_unit_interval_refuses(
    as_pop: pd.DataFrame, rr: pd.DataFrame, protection: pd.DataFrame
) -> None:
    bad = protection.copy()
    bad.loc[bad.index[0], "effective_protection_case"] = 1.5
    with pytest.raises(ValueError, match="protection outside"):
        cell_weights(malaria_as_fractions(rr, as_pop), protection=bad)
