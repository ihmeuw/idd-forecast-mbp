"""Tests for lib/processing/finalize_forecast.py (all-age core + zero-fill)."""

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.aggregation import roll_up_hierarchy
from idd_forecast_mbp.lib.processing.disaggregation import disaggregate_malaria_draws
from idd_forecast_mbp.lib.processing.finalize_forecast import (
    finalize_age_sex_draws,
    finalize_age_sex_summary,
    finalize_all_age,
    square_to_locations,
)

_KEYS_AS = ["location_id", "year_id", "age_group_id", "sex_id"]


# ---------------------------------------------------------------------------
# Fixtures — a 6-level binary hierarchy (32 admin-2 leaves).
# Endemic set = the 16 admin-2 under super-region 2 (locs 32..47); super-region
# 3 (locs 48..63) is entirely non-endemic. That asymmetry is what distinguishes
# a full-population denominator from a summed-endemic one.
# ---------------------------------------------------------------------------

@pytest.fixture
def hier():
    rows = [{"location_id": 1, "parent_id": 0, "level": 0}]
    for i in range(2, 4):
        rows.append({"location_id": i, "parent_id": 1, "level": 1})
    for i in range(4, 8):
        rows.append({"location_id": i, "parent_id": i // 2, "level": 2})
    for i in range(8, 16):
        rows.append({"location_id": i, "parent_id": i // 2, "level": 3})
    for i in range(16, 32):
        rows.append({"location_id": i, "parent_id": i // 2, "level": 4})
    for i in range(32, 64):
        rows.append({"location_id": i, "parent_id": i // 2, "level": 5})
    return pd.DataFrame(rows)


@pytest.fixture
def admin2_pop_all(hier):
    l5 = hier[hier.level == 5]["location_id"].tolist()
    return pd.DataFrame([{"location_id": loc, "year_id": 2020, "population": 1000.0}
                         for loc in l5])


@pytest.fixture
def full_pop(hier, admin2_pop_all):
    """Full population at EVERY level = roll-up of ALL admin-2 (incl. non-endemic)."""
    return roll_up_hierarchy(admin2_pop_all, hier, "population")


@pytest.fixture
def endemic_rate_draws():
    """16 endemic admin-2 (locs 32..47, all under super-region 2), 1 year, 2 draws."""
    rows = []
    for loc in range(32, 48):
        for draw in [0, 1]:
            rows.append({"location_id": loc, "year_id": 2020, "draw": draw,
                         "inc_rate": 0.01, "mort_rate": 0.001})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# finalize_all_age
# ---------------------------------------------------------------------------

def test_all_age_count_rolls_up(hier, admin2_pop_all, full_pop, endemic_rate_draws):
    out = finalize_all_age(endemic_rate_draws, admin2_pop_all, hier, full_pop)
    # 16 endemic × (0.01 × 1000) = 160 at global, per draw
    g = out[(out.location_id == 1) & (out.draw == 0)]
    assert np.isclose(g["inc_count"].iloc[0], 160.0)
    assert np.isclose(g["mort_count"].iloc[0], 16.0)  # 16 × (0.001 × 1000)


def test_all_age_full_pop_denominator_not_summed_endemic(
    hier, admin2_pop_all, full_pop, endemic_rate_draws
):
    out = finalize_all_age(endemic_rate_draws, admin2_pop_all, hier, full_pop)
    g = out[(out.location_id == 1) & (out.draw == 0)]
    # Global: count 160 ÷ FULL global pop 32000 = 0.005 — NOT ÷ summed-endemic 16000 (0.01).
    assert np.isclose(g["inc_rate"].iloc[0], 160.0 / 32000.0)
    assert not np.isclose(g["inc_rate"].iloc[0], 160.0 / 16000.0)
    # Super-region 2 (all 16 endemic under it): count 160 ÷ its own full pop 16000 = 0.01.
    sr2 = out[(out.location_id == 2) & (out.draw == 0)]
    assert np.isclose(sr2["inc_rate"].iloc[0], 160.0 / 16000.0)


def test_all_age_preserves_draws(hier, admin2_pop_all, full_pop, endemic_rate_draws):
    out = finalize_all_age(endemic_rate_draws, admin2_pop_all, hier, full_pop)
    assert set(out["draw"].unique()) == {0, 1}


def test_all_age_2023_continuity_denominator_invariant(
    hier, admin2_pop_all, full_pop, endemic_rate_draws
):
    """Adding zero-count (masked/non-endemic) locations to the numerator set must not
    move the aggregate rate — because they contribute 0 to the count and the denominator
    is the fixed full-level population. This is exactly what makes an observed series
    (over a slightly different loc set) meet the forecast series at the rake year."""
    forecast_out = finalize_all_age(endemic_rate_draws, admin2_pop_all, hier, full_pop)

    # "observed"-style set: same 16 endemic + 4 masked locs (under SR 3) with rate 0.
    extra = pd.DataFrame([{"location_id": loc, "year_id": 2020, "draw": draw,
                           "inc_rate": 0.0, "mort_rate": 0.0}
                          for loc in range(48, 52) for draw in [0, 1]])
    observed_like = pd.concat([endemic_rate_draws, extra], ignore_index=True)
    observed_out = finalize_all_age(observed_like, admin2_pop_all, hier, full_pop)

    fg = forecast_out[(forecast_out.location_id == 1) & (forecast_out.draw == 0)]["inc_rate"].iloc[0]
    og = observed_out[(observed_out.location_id == 1) & (observed_out.draw == 0)]["inc_rate"].iloc[0]
    assert np.isclose(fg, og)


def test_all_age_missing_admin2_pop_raises(hier, full_pop, endemic_rate_draws):
    incomplete_pop = pd.DataFrame([{"location_id": 32, "year_id": 2020, "population": 1000.0}])
    with pytest.raises(ValueError, match="missing population"):
        finalize_all_age(endemic_rate_draws, incomplete_pop, hier, full_pop)


# ---------------------------------------------------------------------------
# square_to_locations
# ---------------------------------------------------------------------------

def test_square_to_locations_true_zeros(hier, admin2_pop_all, full_pop, endemic_rate_draws):
    out = finalize_all_age(endemic_rate_draws, admin2_pop_all, hier, full_pop)
    all_locs = hier["location_id"].tolist()
    squared = square_to_locations(out, all_locs, ["inc_count", "mort_count", "inc_rate", "mort_rate"])
    # every hierarchy node now present for every (year, draw)
    assert set(squared["location_id"].unique()) == set(all_locs)
    # a non-endemic admin-2 (loc 48) and super-region 3 (loc 3) are true zeros
    for loc in (48, 3):
        z = squared[(squared.location_id == loc) & (squared.draw == 0)]
        assert (z[["inc_count", "mort_count", "inc_rate", "mort_rate"]] == 0.0).all(axis=None)


def test_square_to_locations_preserves_existing(hier, admin2_pop_all, full_pop, endemic_rate_draws):
    out = finalize_all_age(endemic_rate_draws, admin2_pop_all, hier, full_pop)
    squared = square_to_locations(out, hier["location_id"].tolist(),
                                  ["inc_count", "mort_count", "inc_rate", "mort_rate"])
    g = squared[(squared.location_id == 1) & (squared.draw == 0)]
    assert np.isclose(g["inc_count"].iloc[0], 160.0)  # unchanged by squaring


# ---------------------------------------------------------------------------
# Age/sex cores — fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def as_pop_all(hier):
    """Age/sex population at every level (roll-up of admin-2 age/sex pop). 2 ages × 2 sexes."""
    l5 = hier[hier.level == 5]["location_id"].tolist()
    rows = [{"location_id": loc, "year_id": 2020, "age_group_id": age, "sex_id": sex,
             "population": 250.0}
            for loc in l5 for age in [3, 10] for sex in [1, 2]]
    return roll_up_hierarchy(pd.DataFrame(rows), hier, "population", preserve_age_sex=True)


@pytest.fixture
def rr_as():
    """Static age/sex rr at the 16 endemic admin-2: age 3 twice age 10."""
    rows = [{"location_id": loc, "age_group_id": age, "sex_id": sex,
             "rr_inc_as": ri, "rr_mort_as": ri}
            for loc in range(32, 48) for age, ri in [(3, 2.0), (10, 1.0)] for sex in [1, 2]]
    return pd.DataFrame(rows)


@pytest.fixture
def aa_count_draws_as():
    """Admin-2 endemic all-age count draws — 4 draws with varying counts (non-trivial UI)."""
    rows = [{"location_id": loc, "year_id": 2020, "draw": draw,
             "aa_malaria_inc_count": 100.0 + 10 * draw,
             "aa_malaria_mort_count": 10.0 + draw}
            for loc in range(32, 48) for draw in range(4)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# finalize_age_sex_draws
# ---------------------------------------------------------------------------

def test_age_sex_draws_totals_preserve_all_age(hier, rr_as, as_pop_all, aa_count_draws_as):
    """Age/sex counts summed over age×sex return the all-age count (no age 2 here)."""
    # keep at super-region 2 (aggregate node holding all 16 endemic admin-2)
    out = finalize_age_sex_draws(aa_count_draws_as, rr_as, as_pop_all, hier, node_ids=[2])
    tot = out.groupby(["location_id", "year_id", "draw"])["inc_count"].sum().reset_index()
    # SR2 all-age = 16 endemic × aa_inc; draw 0 -> 16×100, draw 3 -> 16×130
    for draw, aa in [(0, 100.0), (3, 130.0)]:
        v = tot[tot.draw == draw]["inc_count"].iloc[0]
        assert np.isclose(v, 16 * aa)


def test_age_sex_draws_node_filter(hier, rr_as, as_pop_all, aa_count_draws_as):
    out = finalize_age_sex_draws(aa_count_draws_as, rr_as, as_pop_all, hier, node_ids=[1, 2])
    assert set(out["location_id"].unique()) == {1, 2}


def test_age_sex_draws_has_rate(hier, rr_as, as_pop_all, aa_count_draws_as):
    out = finalize_age_sex_draws(aa_count_draws_as, rr_as, as_pop_all, hier, node_ids=[2])
    assert {"inc_rate", "mort_rate"}.issubset(out.columns)
    assert (out["inc_rate"] >= 0).all()


# ---------------------------------------------------------------------------
# finalize_age_sex_summary — the monotone-scaling shortcut
# ---------------------------------------------------------------------------

def test_age_sex_summary_shortcut_equals_bruteforce_at_leaf(hier, rr_as, as_pop_all, aa_count_draws_as):
    """The load-bearing shortcut test: at admin-2, f·quantile(aa) must equal the
    brute-force quantile(f·aa) computed from the full age/sex draw distribution."""
    summary = finalize_age_sex_summary(aa_count_draws_as, rr_as, as_pop_all, hier,
                                       quantiles=(0.025, 0.975))

    # brute force: disaggregate ALL draws at admin-2, then quantiles over draws
    as_draws = disaggregate_malaria_draws(aa_count_draws_as, rr_as, as_pop_all)
    brute = (as_draws.groupby(_KEYS_AS)
             .agg(inc_mean=("malaria_inc_count_pred", "mean"),
                  inc_lo=("malaria_inc_count_pred", lambda s: s.quantile(0.025)),
                  inc_hi=("malaria_inc_count_pred", lambda s: s.quantile(0.975)),
                  mort_mean=("malaria_mort_count_pred", "mean"),
                  mort_lo=("malaria_mort_count_pred", lambda s: s.quantile(0.025)),
                  mort_hi=("malaria_mort_count_pred", lambda s: s.quantile(0.975)))
             .reset_index())

    leaf_ids = set(hier[hier.level == 5]["location_id"])
    s = (summary[summary.location_id.isin(leaf_ids)]
         .merge(brute, on=_KEYS_AS, suffixes=("", "_b")))
    assert len(s) > 0
    for col in ("inc_mean", "inc_lo", "inc_hi", "mort_mean", "mort_lo", "mort_hi"):
        assert np.allclose(s[col], s[f"{col}_b"]), col


def test_age_sex_summary_aggregate_matches_perdraw_bruteforce(hier, rr_as, as_pop_all, aa_count_draws_as):
    """At an aggregate node (SR2, the shortcut does NOT apply) the summary must equal
    brute-force per-draw disaggregate → roll up → quantiles over draws."""
    summary = finalize_age_sex_summary(aa_count_draws_as, rr_as, as_pop_all, hier,
                                       quantiles=(0.025, 0.975))
    as_draws = disaggregate_malaria_draws(aa_count_draws_as, rr_as, as_pop_all)
    rolled = roll_up_hierarchy(
        as_draws.rename(columns={"malaria_inc_count_pred": "c"})[[*_KEYS_AS, "draw", "c"]],
        hier, "c", preserve_age_sex=True, extra_group_cols=["draw"])
    sr2 = rolled[rolled.location_id == 2]
    brute = (sr2.groupby(_KEYS_AS)["c"]
             .agg(inc_mean="mean",
                  inc_lo=lambda s: s.quantile(0.025),
                  inc_hi=lambda s: s.quantile(0.975))
             .reset_index())
    s = summary[summary.location_id == 2].merge(brute, on=_KEYS_AS, suffixes=("", "_b"))
    assert len(s) > 0
    for col in ("inc_mean", "inc_lo", "inc_hi"):
        assert np.allclose(s[col], s[f"{col}_b"]), col


def test_age_sex_summary_covers_leaf_and_aggregate(hier, rr_as, as_pop_all, aa_count_draws_as):
    summary = finalize_age_sex_summary(aa_count_draws_as, rr_as, as_pop_all, hier)
    levels_present = set(hier.set_index("location_id").loc[
        summary.location_id.unique(), "level"].unique())
    assert 5 in levels_present   # admin-2 leaf (shortcut path)
    assert 0 in levels_present   # global (aggregate path)
