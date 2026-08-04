"""Build the forecast prediction frame from the stage-08b covariate netCDFs.

08b is written at admin-2 (LSAE) with dims ``(location_id, year_id, draw)``. The
dengue models are fitted at FHS, so the covariates have to be rolled up
**population-weighted** before predicting. That roll-up reproduces the observed
past FHS covariates to ~1e-16, which is what makes it legitimate rather than an
approximation.

Two things here are correctness-critical rather than merely fiddly.

**Factor codings must come from the fit, never be recomputed.** ``A0_af`` and
``as_id`` are contiguous integer codes assigned by
``lib/data/dengue_inputs.attach_age_sex_rr`` over whatever was in the *fit* frame.
Recomputing them over the *forecast* frame — which covers 382 FHS parents against
the 305 that were fitted — silently renumbers every country, so each location gets
another country's effect. The result looks entirely plausible. So the fit's mapping
is passed in, and locations whose country the fit never saw are dropped and
counted rather than quietly mapped to something.

**The year covariate is a supplied path, not the calendar.** Whatever
``lib/modeling/year_path`` produces is what the model is handed, which is how a
decayed time effect is applied without refitting anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.dengue_inputs import _URBAN_EPS, URBAN_COLUMN
from idd_forecast_mbp.lib.modeling.year_path import effective_year

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Covariates in 08b that carry a draw dimension.
DRAW_COVARIATES = ("dengue_suitability", "relative_humidity", "total_precipitation")

#: Covariates in 08b that are a single realisation per (location, year).
SCALAR_COVARIATES = (URBAN_COLUMN, "gdppc_mean", "people_flood_days_per_capita")


@dataclass(frozen=True)
class ForecastFrame:
    """The prediction frame for one ssp, plus what was dropped building it."""

    frame: pd.DataFrame
    ssp_scenario: str
    grain: str
    n_locations: int
    n_dropped_unseen_country: int
    year_sr_columns: tuple[str, ...]


def _roll_up_population_weighted(
    values: np.ndarray[Any, Any],
    population: np.ndarray[Any, Any],
    group_starts: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Population-weighted mean over contiguous groups along axis 0.

    ``values`` is ``(n_child, n_year[, n_draw])`` and ``population`` is
    ``(n_child, n_year)``; rows must already be sorted so each parent's children
    are contiguous, with ``group_starts`` giving each parent's first row.

    A weighted MEAN, not a sum: these are covariates (rates, indices, dollars),
    not counts. Summing them would be meaningless.
    """
    weights = population if values.ndim == 2 else population[:, :, None]
    numerator = np.add.reduceat(values * weights, group_starts, axis=0)
    denominator = np.add.reduceat(population, group_starts, axis=0)
    if values.ndim == 3:
        denominator = denominator[:, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def load_forecast_covariates(  # noqa: PLR0913
    ssp_scenario: str,
    *,
    grain: str = "fhs",
    inputs_path: Path | None = None,
    years: Sequence[int] | None = None,
    draws: Sequence[int] | None = None,
    hierarchy: pd.DataFrame | None = None,
    population: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Covariates for one ssp at ``grain``, as a tidy ``(location, year, draw)`` frame.

    Each draw-varying covariate is read and rolled up one at a time — the full
    admin-2 array is ~1.2 GB per covariate, so holding all three at once is
    avoidable waste.
    """
    from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy

    inputs_path = Path(inputs_path or mbpc.DEN_FORECAST_INPUTS_READ_PATH)
    path = inputs_path / f"dengue_forecast_inputs_{ssp_scenario}.nc"
    hierarchy = load_hierarchy() if hierarchy is None else hierarchy

    if population is None:
        population = pd.read_parquet(
            mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
            columns=["location_id", "year_id", "population"])

    with xr.open_dataset(path) as ds:
        admin2 = ds["location_id"].values.astype("int64")
        all_years = ds["year_id"].values.astype("int64")
        all_draws = ds["draw"].values.astype("int64")

        keep_years = (np.isin(all_years, list(years)) if years is not None
                      else np.ones(all_years.size, bool))
        keep_draws = (np.isin(all_draws, list(draws)) if draws is not None
                      else np.ones(all_draws.size, bool))
        sel_years, sel_draws = all_years[keep_years], all_draws[keep_draws]

        if grain == "lsae":
            parents = admin2
        else:
            fhs = hierarchy.set_index("location_id")["fhs_location_id"]
            parents = admin2_parent = fhs.reindex(admin2).to_numpy()
            if np.isnan(admin2_parent.astype(float)).any():
                msg = "some 08b locations have no fhs_location_id in the hierarchy"
                raise KeyError(msg)

        order = np.argsort(parents, kind="stable")
        sorted_parents = parents[order]
        starts = np.flatnonzero(np.r_[True, sorted_parents[1:] != sorted_parents[:-1]])
        out_locations = sorted_parents[starts]

        pop_wide = (population[population.location_id.isin(admin2)]
                    .pivot(index="location_id", columns="year_id", values="population")
                    .reindex(index=admin2, columns=sel_years)
                    .to_numpy(dtype="float32"))
        pop_sorted = pop_wide[order]

        pieces: dict[str, np.ndarray[Any, Any]] = {}
        for name in DRAW_COVARIATES:
            if name not in ds:
                continue
            arr = ds[name].values[:, keep_years, :][:, :, keep_draws]
            pieces[name] = (arr[order] if grain == "lsae"
                            else _roll_up_population_weighted(
                                arr[order], pop_sorted, starts))
            del arr
        for name in SCALAR_COVARIATES:
            if name not in ds:
                continue
            arr = ds[name].values[:, keep_years]
            rolled = (arr[order] if grain == "lsae"
                      else _roll_up_population_weighted(arr[order], pop_sorted, starts))
            pieces[name] = np.repeat(rolled[:, :, None], sel_draws.size, axis=2)

        a0 = ds["A0_location_id"].values.astype("int64")[order]
        # A0 is constant within a parent by construction, so the first child's is it.
        a0_by_parent = a0[starts]

    n_loc, n_year, n_draw = out_locations.size, sel_years.size, sel_draws.size
    frame = pd.DataFrame({
        "location_id": np.repeat(out_locations, n_year * n_draw),
        "year_id": np.tile(np.repeat(sel_years, n_draw), n_loc),
        "draw": np.tile(sel_draws, n_loc * n_year),
        "A0_location_id": np.repeat(a0_by_parent, n_year * n_draw),
    })
    for name, arr in pieces.items():
        frame[name] = arr.reshape(-1)

    return _derive_forecast_columns(frame)


def _derive_forecast_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the fit-time covariate transforms, exactly.

    These must match ``dengue_inputs.derive_model_columns`` — the same clip bounds
    and the same log — or the model is evaluated on a differently-scaled covariate
    than it was fitted on.
    """
    out = frame.copy()
    if URBAN_COLUMN in out.columns:
        out["urban_fraction"] = np.clip(
            out[URBAN_COLUMN].astype(float), _URBAN_EPS, 1 - _URBAN_EPS)
    if "gdppc_mean" in out.columns:
        out["log_gdppc_mean"] = np.log(out["gdppc_mean"].astype(float))
    return out


def build_prediction_frame(  # noqa: PLR0913
    covariates: pd.DataFrame,
    hierarchy: pd.DataFrame,
    *,
    a0_code_map: pd.Series,
    anchor_year: int,
    year_center: float,
    year_path: str | Callable[..., np.ndarray[Any, Any]] = "identity",
    year_path_kwargs: dict[str, Any] | None = None,
    ssp_scenario: str = "",
    grain: str = "fhs",
) -> ForecastFrame:
    """Attach the country code, super-region and year columns a model needs.

    ``a0_code_map`` must be the mapping the FIT used (``A0_location_id`` ->
    ``A0_af``). Locations whose country the fit never saw are dropped and counted;
    predicting them would require an orphan-country fallback, and silently reusing
    some other country's code is worse than dropping.

    The ``year_sr_*`` columns are built from the **effective** year, so a decayed
    time path flows straight through to the prediction without touching the fit.
    """
    out = covariates.copy()
    by_location = hierarchy.set_index("location_id")
    out["super_region_location_id"] = out["location_id"].map(
        by_location["super_region_id"])

    before = out["location_id"].nunique()
    out["A0_af"] = out["A0_location_id"].map(a0_code_map)
    out = out[out["A0_af"].notna()]
    out["A0_af"] = out["A0_af"].astype("int64")
    dropped = before - out["location_id"].nunique()

    years = np.sort(out["year_id"].unique())
    path = effective_year(years, anchor_year, year_path, **(year_path_kwargs or {}))
    out["effective_year"] = out["year_id"].map(dict(zip(years, path, strict=True)))

    # CENTRED, using the fit's constant. The fit built year_sr_* as
    # (year - year_center) inside the group and 0 outside; predicting on raw year
    # instead would offset every year term by ~2011.5. Centring is also what makes
    # the out-of-group 0 coincide with the in-group mean rather than sitting 2000
    # units away -- see DECISIONS 2026-08-03.
    out["year_centered"] = out["effective_year"] - year_center

    year_sr_columns = []
    for super_region in sorted(out["super_region_location_id"].dropna().unique()):
        column = f"year_sr_{int(super_region)}"
        out[column] = np.where(
            out["super_region_location_id"] == super_region, out["year_centered"], 0.0)
        year_sr_columns.append(column)
    # A model with a plain year term reads this one.
    out["year_id_effective"] = out["effective_year"]

    return ForecastFrame(
        frame=out,
        ssp_scenario=ssp_scenario,
        grain=grain,
        n_locations=int(out["location_id"].nunique()),
        n_dropped_unseen_country=int(dropped),
        year_sr_columns=tuple(year_sr_columns),
    )
