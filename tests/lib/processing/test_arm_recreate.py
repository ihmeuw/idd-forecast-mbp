"""2,A,X recreation: exact against the all-age factor, draw-aligned differences, refusals."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from idd_forecast_mbp.lib.data.malaria_arms import Arm
from idd_forecast_mbp.lib.processing.arm_factors import all_age_factors
from idd_forecast_mbp.lib.processing.arm_recreate import (
    admin2_all_age_counts,
    arm_weights,
    draw_aligned_difference,
    forecast_nc,
    read_admin2_log_rates,
    recreate_admin2_age_sex,
    roll_up_age_sex_to_nodes,
)

if TYPE_CHECKING:
    from pathlib import Path

LOCS = [101, 102]
YEARS = [2023, 2024, 2030]
DRAWS = [0, 1, 2]
AGES = [2, 3, 6]
SEXES = [1, 2]


BASE_ARM = Arm(
    name="Baseline",
    model="m",
    forecast_label="lab",
    dah="Baseline",
    vaccine="projected",
    population_hold_year=None,
    age_structure_hold_year=None,
)


def _arm(**kw: Any) -> Arm:
    return dataclasses.replace(BASE_ARM, **kw)


@pytest.fixture
def node(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    shape = (len(LOCS), len(YEARS), len(DRAWS))
    inc = rng.normal(-3, 0.2, shape)
    mort = rng.normal(-7, 0.2, shape)
    mort[1, :, :] = np.nan  # location 102 is incidence-only (mortality masked)
    ds = xr.Dataset(
        {
            "log_malaria_inc_rate_pred": (("location_id", "year_id", "draw"), inc),
            "log_malaria_mort_rate_pred": (("location_id", "year_id", "draw"), mort),
        },
        coords={"location_id": LOCS, "year_id": YEARS, "draw": DRAWS},
    )
    (tmp_path / "lab").mkdir()
    ds.to_netcdf(tmp_path / "lab" / "malaria_forecast_ssp245_Baseline.nc")
    return tmp_path


@pytest.fixture
def as_population() -> pd.DataFrame:
    rows = [
        (loc, y, a, s, 50.0 + loc % 7 + a + s + (y - 2023))
        for loc in LOCS
        for y in YEARS
        for a in AGES
        for s in SEXES
    ]
    return pd.DataFrame(
        rows, columns=["location_id", "year_id", "age_group_id", "sex_id", "population"]
    )


@pytest.fixture
def aa_population(as_population: pd.DataFrame) -> pd.DataFrame:
    return as_population.groupby(["location_id", "year_id"], as_index=False)[
        "population"
    ].sum()


@pytest.fixture
def as_malaria() -> pd.DataFrame:
    rows = [
        (loc, 2023, a, s, 0.1 * a, 0.2, 0.01 * s, 0.01)
        for loc in LOCS
        for a in AGES
        for s in SEXES
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "location_id",
            "year_id",
            "age_group_id",
            "sex_id",
            "malaria_inc_rate",
            "aa_malaria_inc_rate",
            "malaria_mort_rate",
            "aa_malaria_mort_rate",
        ],
    )


@pytest.fixture
def protection() -> pd.DataFrame:
    rows = [
        (loc, y, 3, s, 0.5, 0.4) for loc in LOCS for y in (2024, 2030) for s in SEXES
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "location_id",
            "year_id",
            "age_group_id",
            "sex_id",
            "effective_protection_case",
            "effective_protection_death",
        ],
    )


def test_read_admin2_log_rates_flattens_in_location_year_draw_order(node: Path) -> None:
    lr = read_admin2_log_rates(
        node / "lab" / "malaria_forecast_ssp245_Baseline.nc",
        years=[2024, 2030],
        draws=[2],
    )
    assert lr[["location_id", "year_id", "draw"]].to_numpy().tolist() == [
        [101, 2024, 2],
        [101, 2030, 2],
        [102, 2024, 2],
        [102, 2030, 2],
    ]
    with xr.open_dataset(node / "lab" / "malaria_forecast_ssp245_Baseline.nc") as ds:
        expect = float(
            ds["log_malaria_inc_rate_pred"].sel(location_id=102, year_id=2030, draw=2)
        )
    assert lr.loc[3, "inc_log_rate"] == pytest.approx(expect)
    assert np.isnan(lr.loc[3, "mort_log_rate"])


def test_recreation_sums_to_all_age_times_the_arm_factor(
    node: Path,
    as_population: pd.DataFrame,
    aa_population: pd.DataFrame,
    as_malaria: pd.DataFrame,
    protection: pd.DataFrame,
) -> None:
    arm = _arm()
    w = arm_weights(
        arm,
        LOCS,
        YEARS,
        "v",
        as_malaria=as_malaria,
        as_population=as_population,
        protection=protection,
    )
    rec = recreate_admin2_age_sex(
        arm, "ssp245", ve_variant="v", node=node, weights=w, aa_population=aa_population
    )
    summed = rec.groupby(["location_id", "year_id", "draw"], as_index=False)[
        ["inc_count", "mort_count"]
    ].sum(min_count=1)
    rates = read_admin2_log_rates(forecast_nc(arm, "ssp245", node))
    aa = admin2_all_age_counts(rates, aa_population).merge(
        all_age_factors(w), on=["location_id", "year_id"]
    )
    m = summed.merge(aa, on=["location_id", "year_id", "draw"], suffixes=("", "_aa"))
    np.testing.assert_allclose(m.inc_count, m.inc_count_aa * m.inc_factor, rtol=1e-12)
    keep = m.location_id == 101
    np.testing.assert_allclose(
        m.mort_count[keep], m.mort_count_aa[keep] * m.mort_factor[keep], rtol=1e-12
    )
    assert m.mort_count[~keep].isna().all()  # a masked outcome stays masked
    # vaccine protection exists only after 2023, so 2023 is the unprotected all-age count
    at_anchor = m[m.year_id == 2023]
    np.testing.assert_allclose(at_anchor.inc_count, at_anchor.inc_count_aa, rtol=1e-12)


def test_population_hold_uses_the_hold_year_population_everywhere(
    node: Path,
    as_population: pd.DataFrame,
    aa_population: pd.DataFrame,
    as_malaria: pd.DataFrame,
) -> None:
    arm = _arm(vaccine="none", population_hold_year=2023)
    w = arm_weights(
        arm, LOCS, YEARS, "v", as_malaria=as_malaria, as_population=as_population
    )
    rec = recreate_admin2_age_sex(
        arm, "ssp245", ve_variant="v", node=node, weights=w, aa_population=aa_population
    )
    tot = rec.groupby(["location_id", "year_id", "draw"], as_index=False)[
        "inc_count"
    ].sum()
    rates = read_admin2_log_rates(forecast_nc(arm, "ssp245", node))
    pop23 = (
        aa_population[aa_population.year_id == 2023].set_index("location_id").population
    )
    expect = np.exp(rates.inc_log_rate) * rates.location_id.map(pop23)
    np.testing.assert_allclose(
        tot.sort_values(["location_id", "year_id", "draw"]).inc_count.to_numpy(),
        expect.to_numpy(),
        rtol=1e-12,
    )


def test_admin2_all_age_counts_refuses_missing_population(
    node: Path, aa_population: pd.DataFrame
) -> None:
    rates = read_admin2_log_rates(node / "lab" / "malaria_forecast_ssp245_Baseline.nc")
    with pytest.raises(ValueError, match="have no population"):
        admin2_all_age_counts(rates, aa_population[aa_population.year_id != 2030])


def test_draw_aligned_difference_and_its_refusals() -> None:
    a = pd.DataFrame({"location_id": [1, 1], "draw": [0, 1], "inc_count": [5.0, 7.0]})
    b = pd.DataFrame({"location_id": [1, 1], "draw": [0, 1], "inc_count": [2.0, 3.0]})
    d = draw_aligned_difference(a, b, ["inc_count"])
    assert d.inc_count.tolist() == [3.0, 4.0]
    with pytest.raises(ValueError, match="not aligned"):
        draw_aligned_difference(a, b.iloc[:1], ["inc_count"])
    with pytest.raises(ValueError, match="different key columns"):
        draw_aligned_difference(a, b.rename(columns={"draw": "d"}), ["inc_count"])


def test_roll_up_age_sex_to_nodes_sums_children_and_zero_fills() -> None:
    # contiguous levels 0-5, as in the real hierarchy (the root is its own parent)
    hier = pd.DataFrame(
        {
            "location_id": [1, 2, 3, 4, 10, 20, 101, 102, 201],
            "parent_id": [1, 1, 2, 3, 4, 4, 10, 10, 20],
            "level": [0, 1, 2, 3, 4, 4, 5, 5, 5],
        }
    )
    rows = [
        (101, 2030, 3, 1, 0, 2.0, 1.0),
        (102, 2030, 3, 1, 0, 3.0, np.nan),  # mortality masked at 102
    ]
    counts = pd.DataFrame(
        rows,
        columns=[
            "location_id",
            "year_id",
            "age_group_id",
            "sex_id",
            "draw",
            "inc_count",
            "mort_count",
        ],
    )
    out = roll_up_age_sex_to_nodes(
        counts, hier, [1, 10, 20], years=[2030], draws=[0], ages=[3, 6], sexes=[1]
    )
    got = out.set_index(["location_id", "age_group_id"])
    assert got.loc[(10, 3), "inc_count"] == 5.0
    assert got.loc[(10, 3), "mort_count"] == 1.0
    assert got.loc[(1, 3), "inc_count"] == 5.0
    assert got.loc[(20, 3), "inc_count"] == 0.0  # no endemic child: a true zero
    assert (
        got.loc[(10, 6), "inc_count"] == 0.0
    )  # an age with no counts is zero, not missing
    assert len(out) == 3 * 2


def test_recreation_accepts_pre_read_log_rates(
    node: Path,
    as_population: pd.DataFrame,
    aa_population: pd.DataFrame,
    as_malaria: pd.DataFrame,
    protection: pd.DataFrame,
) -> None:
    arm = _arm()
    w = arm_weights(
        arm,
        LOCS,
        YEARS,
        "v",
        as_malaria=as_malaria,
        as_population=as_population,
        protection=protection,
    )
    rates = read_admin2_log_rates(forecast_nc(arm, "ssp245", node), draws=[1])
    a = recreate_admin2_age_sex(
        arm,
        "ssp245",
        ve_variant="v",
        node=node,
        weights=w,
        aa_population=aa_population,
        draws=[1],
    )
    b = recreate_admin2_age_sex(
        arm,
        "ssp245",
        ve_variant="v",
        node=node / "nowhere",
        weights=w,
        aa_population=aa_population,
        log_rates=rates,
    )
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
