"""All-age products: counts from log rates, roll-up, summaries, own-population rates, anchor check."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from idd_forecast_mbp.lib.processing.all_age_products import (
    admin2_counts_one_year,
    build_all_age_summary,
    build_anchor_check,
    summarize_one_year,
)

if TYPE_CHECKING:
    from pathlib import Path

# levels 0-5, one chain, two admin-2 children
HIER = pd.DataFrame(
    {
        "location_id": [1, 2, 3, 4, 5, 101, 102],
        "parent_id": [1, 1, 2, 3, 4, 5, 5],
        "level": [0, 1, 2, 3, 4, 5, 5],
    }
)
YEARS = [2023, 2030]
DRAWS = [0, 1]


@pytest.fixture
def nc(tmp_path: Path) -> Path:
    inc = np.log(
        np.array([[[0.1, 0.3], [0.2, 0.4]], [[0.5, 0.5], [0.6, 0.6]]])
    )  # loc x year x draw
    mort = np.full_like(inc, np.nan)
    mort[0] = np.log(0.01)  # location 102 mortality masked
    ds = xr.Dataset(
        {
            "log_malaria_inc_rate_pred": (("location_id", "year_id", "draw"), inc),
            "log_malaria_mort_rate_pred": (("location_id", "year_id", "draw"), mort),
        },
        coords={"location_id": [101, 102], "year_id": YEARS, "draw": DRAWS},
    )
    path = tmp_path / "f.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def population() -> pd.DataFrame:
    rows = [
        (loc, y, 100.0 if loc == 101 else 200.0) for loc in [101, 102] for y in YEARS
    ]
    rows += [(loc, y, 1000.0) for loc in [1, 2, 3, 4, 5] for y in YEARS]
    return pd.DataFrame(rows, columns=["location_id", "year_id", "population"])


def test_admin2_counts_one_year_scales_rates_and_drops_masks(
    nc: Path, population: pd.DataFrame
) -> None:
    with xr.open_dataset(nc) as ds:
        inc = admin2_counts_one_year(
            ds["log_malaria_inc_rate_pred"], 2030, population, "inc_count"
        )
        mort = admin2_counts_one_year(
            ds["log_malaria_mort_rate_pred"], 2030, population, "mort_count"
        )
        with pytest.raises(ValueError, match="no population row"):
            admin2_counts_one_year(
                ds["log_malaria_inc_rate_pred"],
                2030,
                population[population.location_id != 102],
                "c",
            )
    assert inc.set_index(["location_id", "draw"]).inc_count.round(9).to_dict() == {
        (101, 0): 20.0,
        (101, 1): 40.0,
        (102, 0): 120.0,
        (102, 1): 120.0,
    }
    assert set(mort.location_id) == {101}


def test_summary_rolls_up_applies_factors_and_uses_own_denominators(
    nc: Path, population: pd.DataFrame
) -> None:
    factors = {
        "inc": pd.DataFrame({"location_id": [101], "year_id": [2030], "factor": [0.5]})
    }
    held = population.assign(population=population.population * 2)
    s = build_all_age_summary(
        nc,
        HIER,
        population,
        [0, 5],
        (0.0, 1.0),
        factors=factors,
        denominator_population=held,
    )
    g = s.set_index(["location_id", "year_id"])
    # 2030 global inc: draw 0 = 0.5*20 + 120 = 130, draw 1 = 0.5*40 + 120 = 140
    assert g.loc[(1, 2030), "inc_count_mean"] == pytest.approx(135.0)
    assert g.loc[(1, 2030), "inc_count_upper"] == pytest.approx(140.0)
    assert g.loc[(1, 2030), "inc_rate_mean"] == pytest.approx(
        135.0 / 2000.0
    )  # the arm world's own population
    assert (
        g.loc[(102, 2030), "mort_count_mean"] != g.loc[(102, 2030), "mort_count_mean"]
    )  # masked: NaN
    assert set(s.level) == {0, 5}
    with pytest.raises(ValueError, match="no population denominator"):
        build_all_age_summary(
            nc,
            HIER,
            population,
            [0],
            (0.0, 1.0),
            denominator_population=held[held.location_id != 1],
        )


def test_summary_refuses_when_nothing_summarises(
    nc: Path, population: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="no summarised rows"):
        build_all_age_summary(nc, HIER, population, [9], (0.0, 1.0))
    with xr.open_dataset(nc) as ds:
        assert summarize_one_year(
            ds[["log_malaria_inc_rate_pred"]].isel(location_id=[]),
            2030,
            HIER,
            population,
            [0],
            (0.0, 1.0),
        ).empty


def test_anchor_check_reports_numbers(nc: Path, population: pd.DataFrame) -> None:
    s = build_all_age_summary(nc, HIER, population, [0, 5], (0.0, 1.0))
    obs = pd.DataFrame(
        {
            "location_id": [101, 102, 1],
            "inc_count": [20.0, 100.0, 0.0],
            "inc_rate": [0.2, 0.5, 0.0],
            "mort_count": [1.0, 0.0, 0.0],
            "mort_rate": [0.01, 0.0, 0.0],
        }
    )
    check, verdict = build_anchor_check(s, 2023, 1e-6, observed=obs)
    assert int(str(verdict["n_failed"])) >= 1
    assert verdict["passed"] is False
    assert verdict["worst"] is not None
    row = check[
        (check.location_id == 101)
        & (check.measure == "inc")
        & (check.metric == "count")
    ].iloc[0]
    assert row.predicted == pytest.approx(20.0)
    assert bool(row.within_tol)
    with pytest.raises(ValueError, match="no rows for anchor year"):
        build_anchor_check(s, 1999, 1e-6, observed=obs)
    zero = obs.assign(inc_count=0.0, inc_rate=0.0, mort_count=0.0, mort_rate=0.0)
    _, v0 = build_anchor_check(
        s[s.location_id == 1].assign(
            inc_count_mean=0.0,
            inc_rate_mean=0.0,
            mort_count_mean=0.0,
            mort_rate_mean=0.0,
        ),
        2023,
        1e-6,
        observed=zero,
    )
    assert v0["max_abs_rel_diff"] is None
    assert v0["worst"] is None


def test_a_file_with_incidence_only_gives_incidence_columns_only(
    tmp_path: Path, nc: Path, population: pd.DataFrame
) -> None:
    with xr.open_dataset(nc) as ds:
        ds[["log_malaria_inc_rate_pred"]].to_netcdf(tmp_path / "inc_only.nc")
    s = build_all_age_summary(
        tmp_path / "inc_only.nc", HIER, population, [0], (0.0, 1.0)
    )
    assert "inc_rate_mean" in s.columns
    assert not any(c.startswith("mort_") for c in s.columns)
    obs = pd.DataFrame({"location_id": [1], "inc_count": [1.0], "inc_rate": [0.1]})
    check, _ = build_anchor_check(s, 2023, 1e-6, observed=obs)
    assert set(check.measure) == {"inc"}
