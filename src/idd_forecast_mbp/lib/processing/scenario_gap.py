"""Exact attribution of a gap between two forecast scenarios to the inputs that differ between them.

The malaria forecast (``forecast_malaria_admin_2s_rocket.r``) is additive on the link scale and shifted so each
admin-2 equals its observation in the rake year ``ry``:

    logit PfPR(t) = L_obs + sum_k [f_k(x_k(t)) - f_k(x_k(ry))]
    log rate(t)   = obs_log_rate + g(logit PfPR(t)) - g(L_obs) + gamma * [log gdp(t) - log gdp(ry)]

so any mix of inputs (some taken from scenario A, the rest from B) can be rebuilt from the fitted term shapes.
The Shapley value then splits ``value(A) - value(B)`` into one part per input, and the parts sum to the gap
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import factorial
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence
    from pathlib import Path

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
AnyArray = npt.NDArray[Any]
Forecast = dict[str, FloatArray]
YearParts = dict[str, dict[str, FloatArray]]

SUITABILITY_DAYS = 365.0
# the rocket clamps suitability/365 to [1e-3, 1 - 1e-3] before the logit
SUITABILITY_CLAMP = 1e-3


def interp_term(
    x_grid: FloatArray, y_grid: FloatArray, x: FloatArray, *, log_x: bool = False
) -> FloatArray:
    """A term's value at ``x`` by linear interpolation on its exported grid.

    A finite ``x`` outside the grid raises: np.interp would clamp it silently. NaN stays NaN.
    ``log_x`` interpolates in log(x), for a log-spaced grid.
    """
    xg = np.log(x_grid) if log_x else np.asarray(x_grid, dtype=float)
    xx = np.log(x) if log_x else np.asarray(x, dtype=float)
    finite = np.isfinite(xx)
    if finite.any() and (xx[finite].min() < xg[0] or xx[finite].max() > xg[-1]):
        msg = (
            f"values outside the term grid: [{np.nanmin(x)}, {np.nanmax(x)}] vs "
            f"[{x_grid[0]}, {x_grid[-1]}]"
        )
        raise ValueError(msg)
    out: FloatArray = np.interp(xx, xg, y_grid)
    out[~finite] = np.nan
    return out


def logit_suitability(suitability_days: FloatArray) -> FloatArray:
    """The PfPR model's suitability covariate: logit of the clamped fraction of suitable days."""
    p = np.clip(
        np.asarray(suitability_days, dtype=float) / SUITABILITY_DAYS,
        SUITABILITY_CLAMP,
        1 - SUITABILITY_CLAMP,
    )
    out: FloatArray = np.log(p / (1 - p))
    return out


def log_rate_from_terms(  # noqa: PLR0913 - the anchor, the PfPR path and the GDP term
    obs_log_rate: FloatArray,
    logit_pfpr: FloatArray,
    obs_logit_pfpr: FloatArray,
    g: Callable[[FloatArray], FloatArray],
    *,
    gdp_slope: float,
    log_gdp_change: FloatArray,
) -> FloatArray:
    """Anchored incidence or mortality log rate from the shifted logit PfPR and the GDP change since ``ry``."""
    out: FloatArray = (
        obs_log_rate + g(logit_pfpr) - g(obs_logit_pfpr) + gdp_slope * log_gdp_change
    )
    return out


def shapley(
    value: Callable[[frozenset[str]], FloatArray], factors: Sequence[str]
) -> dict[str, FloatArray]:
    """Shapley split of ``value(all factors) - value(no factors)``.

    ``value(S)`` is the outcome with the factors in ``S`` taken from scenario A and the rest from B. Each
    coalition is evaluated once; the returned parts sum to the full difference exactly.
    """
    n = len(factors)
    if len(set(factors)) != n:
        msg = f"factors must be distinct: {factors}"
        raise ValueError(msg)
    values = {
        frozenset(c): value(frozenset(c))
        for r in range(n + 1)
        for c in combinations(factors, r)
    }
    parts: dict[str, FloatArray] = {}
    for k in factors:
        others = [f for f in factors if f != k]
        total = np.zeros_like(values[frozenset()], dtype=float)
        for r in range(n):
            weight = factorial(r) * factorial(n - r - 1) / factorial(n)
            for c in combinations(others, r):
                s = frozenset(c)
                total = total + weight * (values[s | {k}] - values[s])
        parts[k] = total
    return parts


# ---------------------------------------------------------------------------
# Rebuilding the malaria forecast for any mix of scenario inputs
# ---------------------------------------------------------------------------

INPUTS_NC = (
    mbpc.MODEL_ROOT
    / "04-forecasting_data/malaria/forecast_inputs/lsae_1285/current/malaria_forecast_inputs_{ssp}.nc"
)
FORECAST_NC = (
    mbpc.MODEL_ROOT
    / "04-forecasting_data/malaria/forecast_outputs/lsae_1285/{label}/malaria_forecast_{ssp}_{dah}.nc"
)
LOG_RATE_VARS = {
    "inc": "log_malaria_inc_rate_pred",
    "mort": "log_malaria_mort_rate_pred",
}
OBS_RATE_COLS = {"inc": "malaria_inc_rate", "mort": "malaria_mort_rate"}
PFPR_SCALE = 0.999  # the fit and the rocket take logit(0.999 * pfpr)
INPUT_FACTORS = ("gdp", "suitability", "wind")
ROLL_UP = (("A0_location_id", 3), ("region_id", 2), ("super_region_id", 1))
# the rebuild pieces holding an input at the rake year zeroes (its change since then)
HELD_PIECES = {
    "gdp": ("gdp_pfpr", "log_gdp"),
    "dah": ("dah",),
    "suitability": ("suitability",),
    "wind": ("wind",),
}
MEASURES = ("inc", "mort")
ADMIN2_LEVEL = 5
PRODUCTS = mbpc.MODEL_ROOT / "05-products/malaria/lsae_1285"


def align(index_from: AnyArray, wanted: AnyArray) -> IntArray:
    """Positions of ``wanted`` in ``index_from``; a missing id raises."""
    pos: IntArray = pd.Index(index_from).get_indexer(wanted).astype(np.int64)
    if (pos < 0).any():
        msg = f"{int((pos < 0).sum())} wanted ids are missing"
        raise ValueError(msg)
    return pos


def read_inputs(  # pragma: no cover - netCDF I/O
    ssp: str, locs: IntArray, years: IntArray, dah: str, *, wind: bool
) -> dict[str, FloatArray]:
    """The model's scenario inputs for ``locs`` x ``years`` (x draw), read contiguously then subset."""
    with xr.open_dataset(str(INPUTS_NC).format(ssp=ssp)) as ds:
        li = align(ds.location_id.to_numpy(), locs)
        yi = align(ds.year_id.to_numpy(), years)
        out = {
            "gdp": ds.gdppc_mean.transpose("location_id", "year_id").to_numpy()[li][
                :, yi
            ],
            "dah": ds.mal_DAH_total_per_capita.sel(dah_scenario=dah)
            .transpose("location_id", "year_id")
            .to_numpy()[li][:, yi],
            "suit": ds.malaria_suitability.transpose(
                "location_id", "year_id", "draw"
            ).to_numpy()[li][:, yi],
        }
        if wind:
            out["wind"] = ds.wind_speed.transpose(
                "location_id", "year_id", "draw"
            ).to_numpy()[li][:, yi]
    return {k: v.astype("float64") for k, v in out.items()}


def read_forecast(  # pragma: no cover - netCDF I/O
    label: str, ssp: str, dah: str
) -> tuple[IntArray, IntArray, Forecast]:
    with xr.open_dataset(str(FORECAST_NC).format(label=label, ssp=ssp, dah=dah)) as ds:
        locs = ds.location_id.to_numpy().astype("int64")
        years = ds.year_id.to_numpy().astype("int64")
        rates = {
            m: ds[v]
            .transpose("location_id", "year_id", "draw")
            .to_numpy()
            .astype("float64")
            for m, v in LOG_RATE_VARS.items()
        }
    return locs, years, rates


def read_observed(locs: IntArray, year: int) -> pd.DataFrame:  # pragma: no cover
    obs = read_parquet_with_integer_ids(
        mbpc.MAL_RAKED_AA_READ_PATH / "aa_full_malaria_df.parquet",
        columns=["location_id", "year_id", *OBS_RATE_COLS.values(), "malaria_pfpr"],
        filters=[("year_id", "==", year)],
    ).set_index("location_id")
    return obs.reindex(locs)


def read_population(locs: IntArray, years: IntArray) -> FloatArray:  # pragma: no cover
    pop = read_parquet_with_integer_ids(
        mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
        columns=["location_id", "year_id", "population"],
        filters=[("location_id", "in", locs.tolist())],
    )
    # pivot (not pivot_table) so a duplicated location-year raises instead of being averaged
    wide = pop.pivot(index="location_id", columns="year_id", values="population")  # noqa: PD010
    out: FloatArray = wide.reindex(index=locs, columns=years).to_numpy(dtype="float64")
    return out


def load_terms(
    term_dir: Path,
) -> tuple[
    dict[tuple[str, str], tuple[FloatArray, FloatArray]], dict[tuple[str, str], float]
]:  # pragma: no cover - parquet I/O
    grids = pd.read_parquet(term_dir / "term_grids.parquet")
    slopes = pd.read_parquet(term_dir / "linear_terms.parquet")
    g = {
        (m, c): (d.x.to_numpy(), d.y.to_numpy())
        for (m, c), d in grids.groupby(["model", "covariate"])
    }
    s = {(r.model, r.covariate): float(r.slope) for r in slopes.itertuples()}
    return g, s


@dataclass
class Rebuild:
    """The forecast's anchored log rates for any mix of scenario inputs (module docstring).

    ``inputs[ssp]`` holds ``gdp`` and ``dah`` (loc x year) and ``suit`` / ``wind`` (loc x year x draw);
    ``l_obs`` and ``obs_log[measure]`` are the rake-year anchors as (loc, 1) columns.
    """

    grids: dict[tuple[str, str], tuple[FloatArray, FloatArray]]
    slopes: dict[tuple[str, str], float]
    inputs: YearParts
    l_obs: FloatArray
    obs_log: dict[str, FloatArray]
    ry: int
    wind: bool = field(init=False)

    def __post_init__(self) -> None:
        self.wind = abs(self.slopes.get(("pfpr", "wind_speed"), 0.0)) > 0

    def term(
        self, model: str, cov: str, x: FloatArray, *, log_x: bool = False
    ) -> FloatArray:
        gx, gy = self.grids[(model, cov)]
        return interp_term(gx, gy, x, log_x=log_x)

    def year_parts(self, t: int) -> YearParts:
        """Each scenario's change since the rake year in every scenario-dependent piece, at year index ``t``."""
        ry, parts = self.ry, {}
        for ssp, x in self.inputs.items():
            gdp_pfpr = self.term(
                "pfpr", "gdppc_mean", x["gdp"][:, t], log_x=True
            ) - self.term("pfpr", "gdppc_mean", x["gdp"][:, ry], log_x=True)
            dah = self.term(
                "pfpr", "mal_DAH_total_per_capita", x["dah"][:, t]
            ) - self.term("pfpr", "mal_DAH_total_per_capita", x["dah"][:, ry])
            suit = logit_suitability(x["suit"][:, t]) - logit_suitability(
                x["suit"][:, ry]
            )
            wind = (
                x["wind"][:, t] - x["wind"][:, ry] if self.wind else np.zeros_like(suit)
            )
            parts[ssp] = {
                "dah": dah[:, None],
                "gdp_pfpr": gdp_pfpr[:, None],
                "log_gdp": np.log(x["gdp"][:, t] / x["gdp"][:, ry])[:, None],
                "suitability": self.slopes[("pfpr", "logit_malaria_suitability")]
                * suit,
                "wind": self.slopes.get(("pfpr", "wind_speed"), 0.0) * wind,
            }
        return parts

    def log_rate(
        self, parts: YearParts, measure: str, src: dict[str, str]
    ) -> FloatArray:
        """Log rate with each input in ``INPUT_FACTORS`` taken from scenario ``src[input]``.

        DAH per capita is the same in every scenario, so it is taken from the GDP source.
        """
        p_g, p_s, p_w = (parts[src[f]] for f in INPUT_FACTORS)
        pfpr = (
            self.l_obs + p_g["dah"] + p_g["gdp_pfpr"] + p_s["suitability"] + p_w["wind"]
        )
        return log_rate_from_terms(
            self.obs_log[measure],
            pfpr,
            self.l_obs,
            lambda x: self.term(measure, "logit_malaria_pfpr", x),
            gdp_slope=self.slopes[(measure, "log_gdppc_mean")],
            log_gdp_change=p_g["log_gdp"],
        )


def pair_parts(  # noqa: PLR0913 - the rebuild, one year's pieces, the pair and the count frame
    rebuild: Rebuild,
    parts: YearParts,
    *,
    measure: str,
    a: str,
    b: str,
    population: FloatArray,
    mask: FloatArray,
    factors: list[str],
) -> dict[str, FloatArray]:
    """Draw-mean counts in A and B and the Shapley part of ``count_A - count_B`` for each factor."""

    def value(coalition: frozenset[str]) -> FloatArray:
        src = {f: (a if f in coalition else b) for f in INPUT_FACTORS}
        c = np.exp(rebuild.log_rate(parts, measure, src)) * population[:, None]
        c[mask] = np.nan
        return c

    split = shapley(value, factors)
    out = {
        "count_a": value(frozenset(factors)).mean(axis=1),
        "count_b": value(frozenset()).mean(axis=1),
    }
    out.update({f"part_{f}": split[f].mean(axis=1) for f in factors})
    return out


def roll_up(
    admin2: pd.DataFrame,
    keys: pd.DataFrame,
    value_cols: list[str],
    *,
    by: tuple[str, ...] = ("pair", "measure", "year_id"),
) -> pd.DataFrame:
    """Sum admin-2 rows to country (level 3), region (2), super-region (1) and global (0, location 1).

    ``keys`` maps ``location_id`` to ``A0_location_id``, ``region_id`` and ``super_region_id``. An all-NaN
    group stays NaN (an outcome masked everywhere in it).
    """
    keyed = admin2.merge(keys, on="location_id", how="left")
    if keyed[[c for c, _ in ROLL_UP]].isna().any().any():
        msg = "admin-2 rows without a parent in the hierarchy"
        raise ValueError(msg)
    group_by = list(by)
    levels = []
    for col, level in ROLL_UP:
        agg = keyed.groupby([*group_by, col], as_index=False)[value_cols].sum(
            min_count=1
        )
        levels.append(agg.rename(columns={col: "location_id"}).assign(level=level))
    glob = keyed.groupby(group_by, as_index=False)[value_cols].sum(min_count=1)
    levels.append(glob.assign(location_id=1, level=0))
    out = pd.concat(levels, ignore_index=True)
    out["location_id"] = out.location_id.astype("int64")
    return out


def check_rebuild(
    rebuild: Rebuild, parts: YearParts, forecasts: dict[str, Forecast], t: int
) -> list[dict[str, Any]]:
    """Max |log diff| between each pure scenario's rebuild and its saved forecast at year index ``t``."""
    return [
        {
            "ssp": s,
            "measure": m,
            "max_abs_log_diff": float(
                np.nanmax(
                    np.abs(
                        rebuild.log_rate(parts, m, dict.fromkeys(INPUT_FACTORS, s))
                        - fc[m][:, t]
                    )
                )
            ),
        }
        for s, fc in forecasts.items()
        for m in LOG_RATE_VARS
    ]


def load_run(  # noqa: PLR0913  # pragma: no cover - reads every input
    spec: str,
    *,
    forecast_label: str,
    term_root: Path,
    ssps: list[str],
    dah: str,
    rake_year: int,
) -> tuple[Rebuild, dict[str, Forecast], IntArray, IntArray, FloatArray]:
    """Everything one model's rebuild needs: term grids, each SSP's inputs and saved forecast, anchors, population."""
    grids, slopes = load_terms(term_root / spec)
    wind = abs(slopes.get(("pfpr", "wind_speed"), 0.0)) > 0
    locs, years, first = read_forecast(forecast_label, ssps[0], dah)
    forecasts = {ssps[0]: first}
    for ssp in ssps[1:]:
        lo, yr, forecasts[ssp] = read_forecast(forecast_label, ssp, dah)
        if not (np.array_equal(lo, locs) and np.array_equal(yr, years)):
            msg = f"{ssp} forecast grid differs from {ssps[0]}'s"
            raise ValueError(msg)
    inputs = {s: read_inputs(s, locs, years, dah, wind=wind) for s in ssps}
    for s in ssps[1:]:
        if not np.array_equal(inputs[s]["dah"], inputs[ssps[0]]["dah"], equal_nan=True):
            msg = f"DAH per capita differs between {ssps[0]} and {s}"
            raise ValueError(msg)
    obs = read_observed(locs, rake_year)
    pfpr = PFPR_SCALE * obs.malaria_pfpr.to_numpy()
    rebuild = Rebuild(
        grids=grids,
        slopes=slopes,
        inputs=inputs,
        l_obs=np.log(pfpr / (1 - pfpr))[:, None],
        obs_log={
            m: np.log(obs[c].to_numpy())[:, None] for m, c in OBS_RATE_COLS.items()
        },
        ry=int(np.flatnonzero(years == rake_year)[0]),
    )
    return rebuild, forecasts, locs, years, read_population(locs, years)


def hold_pieces(
    parts: dict[str, FloatArray], held: Collection[str]
) -> dict[str, FloatArray]:
    """One SSP's rebuild pieces (``Rebuild.year_parts(t)[ssp]``) with each held input's change since the rake year zeroed."""
    unknown = set(held) - set(HELD_PIECES)
    if unknown:
        msg = f"unknown held input(s): {sorted(unknown)}"
        raise ValueError(msg)
    zero = {p for h in held for p in HELD_PIECES[h]}
    return {k: (np.zeros_like(v) if k in zero else v) for k, v in parts.items()}


def subset_rebuild(rebuild: Rebuild, rows: IntArray) -> Rebuild:
    """The same rebuild restricted to admin-2 ``rows`` (inputs and anchors)."""
    return Rebuild(
        grids=rebuild.grids,
        slopes=rebuild.slopes,
        inputs={
            s: {k: v[rows] for k, v in x.items()} for s, x in rebuild.inputs.items()
        },
        l_obs=rebuild.l_obs[rows],
        obs_log={m: v[rows] for m, v in rebuild.obs_log.items()},
        ry=rebuild.ry,
    )


def factor_frame(baseline: pd.DataFrame, no_vaccine: pd.DataFrame) -> pd.DataFrame:
    """``[location_id, year_id, measure, factor]`` from admin-2 product rows; a zero no-vaccine count gets 1."""
    m = baseline.merge(no_vaccine, on=["location_id", "year_id"], suffixes=("_b", "_n"))
    rows = []
    for k in MEASURES:
        b, n = m[f"{k}_count_mean_b"], m[f"{k}_count_mean_n"]
        rows.append(
            pd.DataFrame(
                {
                    "location_id": m.location_id,
                    "year_id": m.year_id,
                    "measure": k,
                    "factor": np.where(
                        n > 0, b / n.where(n > 0), np.where(n.isna(), np.nan, 1.0)
                    ),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def read_factors(  # pragma: no cover - I/O
    spec: str, years: Sequence[int], ssp: str = "ssp245"
) -> pd.DataFrame:
    cols = ["location_id", "year_id", "inc_count_mean", "mort_count_mean"]
    frames = {}
    for arm in ("Baseline", "no_vaccine"):
        frames[arm] = pd.read_parquet(
            PRODUCTS / f"{spec}__{arm}/current/all_age_summary_{ssp}.parquet",
            columns=cols,
            filters=[("level", "==", ADMIN2_LEVEL), ("year_id", "in", list(years))],
        )
    return factor_frame(frames["Baseline"], frames["no_vaccine"])


def weighted_mean(values: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted mean over axis 0 (locations); NaN cells are left out and the weights renormalised.

    ``values`` is (loc, ...) and ``weights`` (loc,). A slice with no finite value is NaN.
    """
    w = np.broadcast_to(weights.reshape(-1, *([1] * (values.ndim - 1))), values.shape)
    ok = np.isfinite(values) & np.isfinite(w)
    num = np.where(ok, values * w, 0.0).sum(axis=0)
    den = np.where(ok, w, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def by_group(
    values: FloatArray, weights: FloatArray, groups: FloatArray
) -> dict[int, FloatArray]:
    """``weighted_mean`` within each group id, plus the all-location mean under key 1 (global)."""
    out = {1: weighted_mean(values, weights)}
    for g in np.unique(groups):
        sel = groups == g
        out[int(g)] = weighted_mean(values[sel], weights[sel])
    return out
