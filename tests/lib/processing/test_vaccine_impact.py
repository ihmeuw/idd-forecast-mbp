"""Tests for the post-disaggregation impact logic.

The property that matters most here: cumulation happens PER DRAW before any
collapse across draws. Cumulating published quantiles instead would understate
the uncertainty on cumulative series, because draw-level errors would not
accumulate. Several tests pin that directly.
"""
import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.vaccine_impact import (
    _aligned,
    apply_protection_to_age_sex,
    burden_weighted_reduction,
    scenario_totals,
    _with_cumulative,
    draw_level,
    eligible_locations,
    summarize,
    super_region_map,
)

LOCS = [11, 22]
YEARS = [2024, 2025, 2026]


def _totals() -> pd.DataFrame:
    """Two draws whose trajectories differ, so cumulative spread must widen."""
    rows = []
    for draw, scale in ((0, 1.0), (1, 3.0)):
        for i, year in enumerate(YEARS):
            rows.append(dict(ssp_scenario="ssp245", measure="mortality", year_id=year,
                             draw=draw, count_novacc=100.0 * scale * (i + 1),
                             count_vacc=90.0 * scale * (i + 1)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _aligned
# ---------------------------------------------------------------------------
def _grid(value=1.0) -> pd.DataFrame:
    return pd.DataFrame([dict(location_id=l, year_id=y, v=value)
                         for l in LOCS for y in YEARS])


def test_aligned_returns_the_forecast_grid():
    da = _aligned(_grid(), "v", LOCS, YEARS, fill=None)
    assert da.dims == ("location_id", "year_id")
    assert list(da.location_id.values) == LOCS
    assert list(da.year_id.values) == YEARS
    assert (da.to_numpy() == 1.0).all()


def test_aligned_fills_cells_the_frame_does_not_cover():
    partial = _grid().iloc[1:]
    da = _aligned(partial, "v", LOCS, YEARS, fill=0.0)
    assert da.to_numpy()[0, 0] == 0.0


def test_aligned_raises_when_a_gap_must_not_be_filled():
    """Population must never be silently zero-filled -- a missing cell is a bug
    upstream, not a zero."""
    with pytest.raises(ValueError, match="missing for"):
        _aligned(_grid().iloc[1:], "v", LOCS, YEARS, fill=None)


# ---------------------------------------------------------------------------
# cumulation
# ---------------------------------------------------------------------------
def test_cumulation_is_per_draw():
    out = _with_cumulative(_totals())
    d0 = out[(out.draw == 0)].sort_values("year_id")
    assert d0.count_novacc_cum.tolist() == pytest.approx(np.cumsum(d0.count_novacc).tolist())
    # draws never mix
    d1 = out[(out.draw == 1)].sort_values("year_id")
    assert d1.count_novacc_cum.iloc[-1] == pytest.approx(3 * d0.count_novacc_cum.iloc[-1])


def test_averted_is_the_difference_of_the_two_scenarios():
    out = _with_cumulative(_totals())
    assert out.averted.equals(out.count_novacc - out.count_vacc)
    assert (out.averted > 0).all()


def test_draw_level_keeps_every_draw_year_row():
    out = draw_level(_totals())
    assert len(out) == len(_totals())
    assert {"count_novacc_cum", "count_vacc_cum", "averted_cum"} <= set(out.columns)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_emits_annual_and_cumulative_statistics():
    out = summarize(_totals())
    assert len(out) == len(YEARS)
    for stem in ("novacc", "vacc", "averted", "novacc_cum", "vacc_cum", "averted_cum"):
        for stat in ("mean", "lo", "hi"):
            assert f"{stem}_{stat}" in out.columns


def test_cumulative_interval_widens_because_draws_accumulate():
    """The reason cumulation must precede the collapse: summing the published
    per-year quantiles would not widen like this."""
    out = summarize(_totals()).sort_values("year_id")
    spread = out.averted_cum_hi - out.averted_cum_lo
    assert spread.is_monotonic_increasing
    assert spread.iloc[-1] > spread.iloc[0]


def test_cumulative_mean_equals_the_mean_of_per_draw_cumulative():
    out = summarize(_totals()).sort_values("year_id")
    per_draw = _with_cumulative(_totals())
    expected = per_draw.groupby("year_id").averted_cum.mean().sort_index()
    assert out.averted_cum_mean.to_numpy() == pytest.approx(expected.to_numpy())


# ---------------------------------------------------------------------------
# geography helpers (hierarchy injected, so no shared storage needed)
# ---------------------------------------------------------------------------
def _hierarchy() -> pd.DataFrame:
    """166 is a super-region; 200 a country under it; 300/301 admin2 leaves."""
    return pd.DataFrame([
        dict(location_id=166, path_to_top_parent="1,166", level=1, location_name="Sub-Saharan Africa"),
        dict(location_id=200, path_to_top_parent="1,166,200", level=3, location_name="Country A"),
        dict(location_id=250, path_to_top_parent="1,166,200,250", level=4, location_name="Admin1 A"),
        dict(location_id=300, path_to_top_parent="1,166,200,250,300", level=5, location_name="Leaf 1"),
        dict(location_id=301, path_to_top_parent="1,166,200,250,301", level=5, location_name="Leaf 2"),
        dict(location_id=999, path_to_top_parent="1,4,999", level=5, location_name="Elsewhere"),
    ])


def test_eligible_locations_maps_leaves_to_their_coverage_parent(tmp_path):
    csv = tmp_path / "cov.csv"
    pd.DataFrame({"subnat_id": [250]}).to_csv(csv, index=False)
    out = eligible_locations(csv, {300, 301, 999}, hierarchy=_hierarchy())
    assert sorted(out.location_id) == [300, 301]           # 999 is not under 250
    assert set(out.coverage_location_id) == {250}


def test_eligible_locations_raises_when_nothing_overlaps(tmp_path):
    csv = tmp_path / "cov.csv"
    pd.DataFrame({"subnat_id": [250]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="no forecast locations"):
        eligible_locations(csv, {999}, hierarchy=_hierarchy())


def test_super_region_map_finds_the_level_one_ancestor():
    out = super_region_map([300, 301], hierarchy=_hierarchy())
    assert set(out.super_region_id) == {166}
    assert set(out.super_region_name) == {"Sub-Saharan Africa"}
    assert sorted(out.location_id) == [300, 301]


def test_super_region_map_raises_when_no_location_has_one():
    with pytest.raises(ValueError, match="no super-region"):
        super_region_map([12345], hierarchy=_hierarchy())


# ---------------------------------------------------------------------------
# burden-weighted reduction and scenario totals, with inputs injected so the
# arithmetic is checkable without touching the 2.97 GB population file
# ---------------------------------------------------------------------------
AGES, SEXES = (238, 34), (1, 2)


def _mapping() -> pd.DataFrame:
    return pd.DataFrame({"location_id": [300, 301], "coverage_location_id": [250, 250]})


def _as_malaria() -> pd.DataFrame:
    """Anchor-year age/sex rates. Age 34 carries twice the rate of 238, so the
    burden weighting must favour it."""
    rows = []
    for loc in (300, 301):
        for age, mult in zip(AGES, (1.0, 2.0)):
            for sex in SEXES:
                rows.append(dict(location_id=loc, year_id=2023, age_group_id=age, sex_id=sex,
                                 malaria_inc_rate=0.10 * mult, aa_malaria_inc_rate=0.10,
                                 malaria_mort_rate=0.01 * mult, aa_malaria_mort_rate=0.01))
    return pd.DataFrame(rows)


def _as_population(years) -> pd.DataFrame:
    return pd.DataFrame([
        dict(location_id=loc, year_id=y, age_group_id=age, sex_id=sex, population=1000.0)
        for loc in (300, 301) for y in years for age in AGES for sex in SEXES])


def _protection(years, case=0.5, death=0.4) -> pd.DataFrame:
    """Protection is reported at admin1 (250) and must broadcast to 300/301."""
    return pd.DataFrame([
        dict(location_id=250, year_id=y, age_group_id=age, sex_id=sex,
             effective_protection_case=case, effective_protection_death=death)
        for y in years for age in AGES for sex in SEXES])


def test_reduction_broadcasts_admin1_protection_to_children():
    years = [2024]
    out = burden_weighted_reduction(_protection(years), _mapping(), years,
                                    as_malaria=_as_malaria(), as_population=_as_population(years))
    assert sorted(out.location_id) == [300, 301]
    # uniform protection => the burden-weighted average is that same value
    assert out.r_inc.tolist() == pytest.approx([0.5, 0.5])
    assert out.r_mort.tolist() == pytest.approx([0.4, 0.4])


def test_reduction_is_burden_weighted_not_population_weighted():
    """Protect only the high-burden age group. Population is equal across ages, so
    a population-weighted answer would be 0.5; burden weighting must exceed it."""
    years = [2024]
    prot = _protection(years)
    prot.loc[prot.age_group_id == 238, ["effective_protection_case",
                                        "effective_protection_death"]] = 0.0
    out = burden_weighted_reduction(prot, _mapping(), years,
                                    as_malaria=_as_malaria(), as_population=_as_population(years))
    assert (out.r_inc > 0.30).all()          # 2/3 of burden is in age 34
    assert out.r_inc.iloc[0] == pytest.approx(0.5 * (2 / 3))


def test_reduction_rejects_a_factor_outside_the_unit_interval():
    years = [2024]
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        burden_weighted_reduction(_protection(years, case=1.8), _mapping(), years,
                                  as_malaria=_as_malaria(), as_population=_as_population(years))


def _forecast_ds(locs, years, draws=2, rate=0.02):
    import xarray as xr
    shape = (len(locs), len(years), draws)
    arr = np.log(np.full(shape, rate))
    coords = {"location_id": locs, "year_id": years, "draw": list(range(draws))}
    dims = ("location_id", "year_id", "draw")
    return xr.Dataset({"log_malaria_inc_rate_pred": (dims, arr),
                       "log_malaria_mort_rate_pred": (dims, arr)}, coords=coords)


def test_scenario_totals_applies_one_minus_reduction_to_every_draw():
    locs, years, rate, pop = [300, 301], [2024, 2025], 0.02, 500.0
    reduction = pd.DataFrame([dict(location_id=l, year_id=y, r_inc=0.25, r_mort=0.10)
                              for l in locs for y in years])
    aa_pop = pd.DataFrame([dict(location_id=l, year_id=y, population=pop)
                           for l in locs for y in years])
    out = scenario_totals("ssp245", reduction, aa_pop, locs, years,
                          dataset=_forecast_ds(locs, years, rate=rate))

    expected_novacc = rate * pop * len(locs)
    inc = out[out.measure == "incidence"]
    assert inc.count_novacc.unique() == pytest.approx([expected_novacc])
    assert inc.count_vacc.unique() == pytest.approx([expected_novacc * 0.75])
    mort = out[out.measure == "mortality"]
    assert mort.count_vacc.unique() == pytest.approx([expected_novacc * 0.90])
    assert set(out.draw) == {0, 1}          # reduction is draw-free, applied to all


def test_scenario_totals_treats_locations_without_a_reduction_as_unaffected():
    locs, years = [300, 301], [2024]
    reduction = pd.DataFrame([dict(location_id=300, year_id=2024, r_inc=0.5, r_mort=0.5)])
    aa_pop = pd.DataFrame([dict(location_id=l, year_id=2024, population=100.0) for l in locs])
    out = scenario_totals("ssp245", reduction, aa_pop, locs, years,
                          dataset=_forecast_ds(locs, years, rate=0.01))
    inc = out[out.measure == "incidence"]
    # loc 300 halved, loc 301 untouched -> 25% off the two-location total
    assert (inc.count_vacc / inc.count_novacc).unique() == pytest.approx([0.75])


def test_clock_is_called_when_supplied():
    """Progress reporting is optional, but must actually fire when a logger is given."""
    seen = []
    years = [2024]
    burden_weighted_reduction(_protection(years), _mapping(), years, clock=seen.append,
                              as_malaria=_as_malaria(), as_population=_as_population(years))
    assert any("as_rr source read" in m for m in seen)
    assert any("as fractions" in m for m in seen)


def test_scenario_totals_rejects_an_unsorted_file_coord():
    """The positional numpy subset is only valid when the FILE's coord is sorted
    ascending; otherwise searchsorted mis-attributes cells silently."""
    locs, years = [300, 301], [2024]
    ds = _forecast_ds([301, 300], years)      # the file, not the query, is unsorted
    reduction = pd.DataFrame([dict(location_id=l, year_id=2024, r_inc=0.0, r_mort=0.0) for l in locs])
    aa_pop = pd.DataFrame([dict(location_id=l, year_id=2024, population=1.0) for l in locs])
    with pytest.raises(ValueError, match="not sorted ascending"):
        scenario_totals("ssp245", reduction, aa_pop, locs, years, dataset=ds)


@pytest.mark.parametrize("bad_locs, bad_years, expected", [
    ([300, 999], [2024], "does not carry location_id"),
    ([300, 301], [2024, 2099], "does not carry year_id"),
])
def test_scenario_totals_rejects_cells_the_forecast_lacks(bad_locs, bad_years, expected):
    """Membership is checked before the positional lookup: searchsorted on an
    absent value returns an out-of-range index and would raise IndexError."""
    ds = _forecast_ds([300, 301], [2024, 2025])
    reduction = pd.DataFrame([dict(location_id=l, year_id=y, r_inc=0.0, r_mort=0.0)
                              for l in bad_locs for y in bad_years])
    aa_pop = pd.DataFrame([dict(location_id=l, year_id=y, population=1.0)
                           for l in bad_locs for y in bad_years])
    with pytest.raises(ValueError, match=expected):
        scenario_totals("ssp245", reduction, aa_pop, bad_locs, bad_years, dataset=ds)


# ---------------------------------------------------------------------------
# cell-wise protection for delivery products
# ---------------------------------------------------------------------------
def _as_counts() -> pd.DataFrame:
    """Two age groups with equal counts, so a uniform-vs-cell-wise reduction is
    distinguishable."""
    return pd.DataFrame([
        dict(location_id=300, year_id=2030, age_group_id=age, sex_id=1, draw=0,
             malaria_inc_count_pred=100.0, malaria_mort_count_pred=10.0)
        for age in (34, 8)])


def _cell_protection(young=0.5, old=0.0) -> pd.DataFrame:
    return pd.DataFrame([
        dict(location_id=300, year_id=2030, age_group_id=34, sex_id=1,
             effective_protection_case=young, effective_protection_death=young),
        dict(location_id=300, year_id=2030, age_group_id=8, sex_id=1,
             effective_protection_case=old, effective_protection_death=old)])


def test_protection_is_applied_per_cell_not_spread_across_ages():
    """The whole reason delivery cannot reuse the burden-weighted collapse: an
    age group with no protection must be left untouched."""
    out = apply_protection_to_age_sex(_as_counts(), _cell_protection()).set_index("age_group_id")
    assert out.loc[34, "malaria_inc_count_pred"] == pytest.approx(50.0)   # halved
    assert out.loc[8, "malaria_inc_count_pred"] == pytest.approx(100.0)   # untouched


def test_case_and_death_protection_hit_their_own_measure():
    prot = _cell_protection()
    prot.loc[prot.age_group_id == 34, "effective_protection_death"] = 0.9
    out = apply_protection_to_age_sex(_as_counts(), prot).set_index("age_group_id")
    assert out.loc[34, "malaria_inc_count_pred"] == pytest.approx(50.0)   # case 0.5
    assert out.loc[34, "malaria_mort_count_pred"] == pytest.approx(1.0)   # death 0.9


def test_cells_without_protection_are_left_unreduced():
    out = apply_protection_to_age_sex(_as_counts(), _cell_protection().iloc[:1])
    assert out.set_index("age_group_id").loc[8, "malaria_inc_count_pred"] == pytest.approx(100.0)


def test_protection_columns_are_dropped_from_the_result():
    out = apply_protection_to_age_sex(_as_counts(), _cell_protection())
    assert "effective_protection_case" not in out.columns


def test_out_of_range_protection_raises():
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        apply_protection_to_age_sex(_as_counts(), _cell_protection(young=1.4))


def test_uniform_collapse_and_cellwise_agree_on_the_total_but_not_by_age():
    """Applying the burden-weighted R to the all-age total gives the same TOTAL as
    cell-wise reduction -- which is why the impact figures can use the collapse --
    but a different age distribution, which is why delivery cannot."""
    counts, prot = _as_counts(), _cell_protection()
    cellwise = apply_protection_to_age_sex(counts, prot)
    total_cellwise = cellwise.malaria_inc_count_pred.sum()

    # the collapse: burden-weighted average protection applied to the total
    weights = counts.malaria_inc_count_pred / counts.malaria_inc_count_pred.sum()
    r = float((weights.to_numpy() * prot.effective_protection_case.to_numpy()).sum())
    total_collapsed = counts.malaria_inc_count_pred.sum() * (1 - r)

    assert total_cellwise == pytest.approx(total_collapsed)
    uniform_by_age = counts.malaria_inc_count_pred * (1 - r)
    assert not np.allclose(cellwise.malaria_inc_count_pred.to_numpy(),
                           uniform_by_age.to_numpy())
