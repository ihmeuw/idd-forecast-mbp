"""Unit tests for 06b's location-universe and row-inclusion rules.

These two pure helpers are what decide whether the dengue past frame covers all
473 FHS-most-detailed locations or only the 305 endemic ones, so they are tested
without touching the cluster artifacts.

The behaviour under test, stated as intent: a location-year with zero observed
cases is an OBSERVED ZERO and must survive into the artifact. The historical
rule (drop it) is still reachable, but is no longer the default.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "idd_forecast_mbp" / "02_data_prep"
    / "06b_build_dengue_past_inputs.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("build_06b_dengue_past_inputs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hierarchy():
    """Two countries plus a region, spanning the case a level cut gets wrong.

    Country 10 is subnationalised in FHS: its national row (level 3) is an
    AGGREGATE and its admin-1 row 11 is the FHS most-detailed unit. Country 20 is
    not subnationalised: its national row IS the FHS most-detailed unit, and its
    admin-1 row 22 is below the grain. So `most_detailed_fhs` picks {11, 20} while
    `level <= 4` would wrongly pick {10, 11, 20, 22}.
    """
    return pd.DataFrame({
        "location_id":        [1, 10, 11, 12, 13, 20, 22, 21],
        "level":              [2,  3,  4,  5,  5,  3,  4,  5],
        "A0_location_id":     [1, 10, 10, 10, 10, 20, 20, 20],
        "most_detailed_fhs":  [0,  0,  1,  0,  0,  1,  0,  0],
        "most_detailed_lsae": [0,  0,  0,  1,  1,  0,  0,  1],
    })


@pytest.fixture
def fit_locations():
    """06a's gated set: country 10's branch only — country 20 failed the A0 gate."""
    return pd.DataFrame({"location_id": [10, 11, 12, 13]})


# ── grain selection ─────────────────────────────────────────────────────────

def test_fhs_grain_is_exactly_the_most_detailed_flag(mod, hierarchy, fit_locations):
    """Nothing more, nothing less: the flag, not a level cut."""
    out = mod.resolve_location_universe(hierarchy, fit_locations, grain="fhs")
    assert out["location_id"].tolist() == [11, 20]


def test_fhs_grain_excludes_the_aggregate_parent_and_sub_grain_child(mod, hierarchy, fit_locations):
    out = mod.resolve_location_universe(hierarchy, fit_locations, grain="fhs")
    got = set(out["location_id"])
    assert 10 not in got, "national row of a subnationalised country is an aggregate"
    assert 22 not in got, "admin-1 row of a non-subnationalised country is below the grain"


def test_fhs_grain_straddles_two_levels(mod, hierarchy, fit_locations):
    """Guards the level-cut mistake: the grain is not confined to one level."""
    out = mod.resolve_location_universe(hierarchy, fit_locations, grain="fhs")
    assert set(out["level"]) == {3, 4}


def test_lsae_grain_is_the_admin2_flag(mod, hierarchy, fit_locations):
    out = mod.resolve_location_universe(hierarchy, fit_locations, grain="lsae")
    assert out["location_id"].tolist() == [12, 13, 21]


def test_unknown_grain_raises(mod, hierarchy, fit_locations):
    with pytest.raises(ValueError, match="grain"):
        mod.resolve_location_universe(hierarchy, fit_locations, grain="fhs_most_detailed")


# ── resolve_location_universe ────────────────────────────────────────────────

def test_grain_all_spans_levels_3_to_5(mod, hierarchy, fit_locations):
    out = mod.resolve_location_universe(hierarchy, fit_locations, grain="all")
    assert out["location_id"].tolist() == [10, 11, 12, 13, 20, 21, 22]
    assert 1 not in out["location_id"].values  # level 2 excluded


def test_location_set_all_includes_never_observed_locations(mod, hierarchy, fit_locations):
    """The point of the change: country 20, which 06a's gate excluded, is present."""
    out = mod.resolve_location_universe(hierarchy, fit_locations, location_set="all")
    assert 20 in out["location_id"].values


def test_location_set_fit_intersects_the_gated_set(mod, hierarchy, fit_locations):
    out = mod.resolve_location_universe(hierarchy, fit_locations, location_set="fit")
    assert out["location_id"].tolist() == [11]


def test_fit_eligible_flags_06a_membership_not_the_universe(mod, hierarchy, fit_locations):
    """The gate must be recoverable by filtering — the old set is a subset."""
    out = mod.resolve_location_universe(hierarchy, fit_locations, grain="all")
    eligible = set(out.loc[out["fit_eligible"], "location_id"])
    assert eligible == {10, 11, 12, 13}
    assert set(out.loc[~out["fit_eligible"], "location_id"]) == {20, 21, 22}


def test_fit_eligible_is_set_under_the_fit_universe_too(mod, hierarchy, fit_locations):
    out = mod.resolve_location_universe(hierarchy, fit_locations, location_set="fit")
    assert out["fit_eligible"].all()


def test_unknown_location_set_raises(mod, hierarchy, fit_locations):
    with pytest.raises(ValueError, match="location_set"):
        mod.resolve_location_universe(hierarchy, fit_locations, location_set="endemic")


# ── select_location_years ────────────────────────────────────────────────────

@pytest.fixture
def as_df():
    """Two locations × two years, broadcast over two age/sex rows each.

    Location 10 has cases in 2001 only; location 20 never has cases.
    """
    rows = []
    for loc, counts in [(10, {2000: 0.0, 2001: 5.0}), (20, {2000: 0.0, 2001: 0.0})]:
        for year, count in counts.items():
            for age in (2, 3):
                rows.append({"location_id": loc, "year_id": year,
                             "age_group_id": age, "sex_id": 1,
                             "aa_dengue_inc_count": count})
    return pd.DataFrame(rows)


def test_default_keeps_zero_incidence_location_years(mod, as_df):
    out = mod.select_location_years(as_df)
    assert len(out) == 4
    assert set(map(tuple, out.values)) == {(10, 2000), (10, 2001), (20, 2000), (20, 2001)}


def test_default_deduplicates_across_age_sex_rows(mod, as_df):
    """One row per (location, year), not one per age/sex cell."""
    out = mod.select_location_years(as_df)
    assert not out.duplicated().any()
    assert out.columns.tolist() == ["location_id", "year_id"]


def test_require_nonzero_restores_the_historical_filter(mod, as_df):
    out = mod.select_location_years(as_df, require_nonzero_incidence=True)
    assert set(map(tuple, out.values)) == {(10, 2001)}


def test_zero_and_positive_split_is_the_only_difference(mod, as_df):
    """Default output is a strict superset of the filtered one."""
    kept_all = set(map(tuple, mod.select_location_years(as_df).values))
    kept_nz = set(map(tuple, mod.select_location_years(
        as_df, require_nonzero_incidence=True).values))
    assert kept_nz < kept_all
