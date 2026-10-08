"""All-age summary products from one forecast netCDF: count-space roll-up, draw summary, anchor check.

Moved from ``05_aggregation/finish_run.py`` (deleted 2026-10-07/08 after its hold code was found to
differ from the 2025 first-submission definitions). Only the parts verified against the July
products are kept, with their logic unchanged: per-year admin-2 counts from log rates, count-space
roll-up per draw, mean/lower/upper, rates = count / the level's OWN population row, and the
anchor-year check against raked observed. Arm changes (vaccine, holds) arrive as per-(location, year)
count factors from ``arm_factors``; they are never computed here.

Counts and rates use two population frames on purpose: admin-2 counts are always rate x the real
population; ``denominator_population`` is the population of the arm's world (the 2023 population for
the population hold), defaulting to the real one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.processing.aggregation import roll_up_hierarchy
from idd_forecast_mbp.lib.processing.summarize import summarize_draws

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

PRED_VARS = {"inc": "log_malaria_inc_rate_pred", "mort": "log_malaria_mort_rate_pred"}
STATS = ("mean", "lower", "upper")
ADMIN2_LEVEL = 5


def load_population(
    years: Sequence[int],
) -> pd.DataFrame:  # pragma: no cover - constant-path I/O
    """All-age population for every hierarchy level and ``years``; read, never summed from children."""
    path = mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet"
    pop = read_parquet_with_integer_ids(
        path, columns=["location_id", "year_id", "population"]
    )
    pop = pop[pop.year_id.isin([int(y) for y in years])]
    if pop.empty:
        msg = f"population artifact {path} has no rows for years {years}"
        raise ValueError(msg)
    return pop.reset_index(drop=True)


def observed_all_age(year: int) -> pd.DataFrame:  # pragma: no cover - constant-path I/O
    """Raked observed all-age counts and rates at every level for one year."""
    path = mbpc.MAL_RAKED_AA_READ_PATH / "aa_full_malaria_df.parquet"
    obs = read_parquet_with_integer_ids(
        path,
        columns=[
            "location_id",
            "year_id",
            "malaria_inc_count",
            "malaria_inc_rate",
            "malaria_mort_count",
            "malaria_mort_rate",
        ],
    )
    obs = obs[obs.year_id == year]
    if obs.empty:
        msg = f"observed artifact {path} has no rows for year {year}"
        raise ValueError(msg)
    return obs.rename(
        columns={
            "malaria_inc_count": "inc_count",
            "malaria_inc_rate": "inc_rate",
            "malaria_mort_count": "mort_count",
            "malaria_mort_rate": "mort_rate",
        }
    ).reset_index(drop=True)


def admin2_counts_one_year(
    da: xr.DataArray, year: int, pop_admin2: pd.DataFrame, count_col: str
) -> pd.DataFrame:
    """Draw-level admin-2 counts for one year: exp(log rate) x that year's population; masked rows dropped."""
    slab = da.sel(year_id=year).transpose("location_id", "draw")
    loc_ids = slab.location_id.to_numpy()
    draw_ids = slab.draw.to_numpy()
    pop_year = pop_admin2[pop_admin2.year_id == year].set_index("location_id")[
        "population"
    ]
    missing = np.setdiff1d(loc_ids, pop_year.index.to_numpy())
    if missing.size:
        msg = (
            f"{missing.size} forecast locations have no population row for {year} "
            f"(first few: {missing[:5].tolist()})"
        )
        raise ValueError(msg)
    counts = np.exp(slab.to_numpy()) * pop_year.reindex(loc_ids).to_numpy()[:, None]
    frame = pd.DataFrame(
        {
            "location_id": np.repeat(loc_ids, draw_ids.size).astype("int64"),
            "year_id": np.int64(year),
            "draw": np.tile(draw_ids, loc_ids.size).astype("int64"),
            count_col: counts.ravel(),
        }
    )
    return frame.dropna(subset=[count_col]).reset_index(drop=True)


def summarize_one_year(  # noqa: PLR0913, PLR0917 - the year's six inputs and the optional arm factors
    ds: xr.Dataset,
    year: int,
    hierarchy_df: pd.DataFrame,
    pop_admin2: pd.DataFrame,
    levels: Sequence[int],
    quantiles: tuple[float, float],
    *,
    factors: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Aggregate and summarise both measures for one year; arm factors scale admin-2 counts first."""
    per_measure = []
    for measure, var in PRED_VARS.items():
        if var not in ds.data_vars:
            continue
        count_col = f"{measure}_count"
        draws = admin2_counts_one_year(ds[var], year, pop_admin2, count_col)
        if draws.empty:
            continue
        if factors is not None and measure in factors:
            f = factors[measure]
            f_year = f.loc[f.year_id == year].set_index("location_id")["factor"]
            draws[count_col] = (
                draws[count_col].to_numpy()
                * draws.location_id.map(f_year).fillna(1.0).to_numpy()
            )
        rolled = roll_up_hierarchy(
            draws,
            hierarchy_df,
            count_col,
            extra_group_cols=["draw"],
            start_level=ADMIN2_LEVEL,
        )
        rolled = rolled.merge(
            hierarchy_df[["location_id", "level"]], on="location_id", how="left"
        )
        rolled = rolled[rolled.level.isin(list(levels))]
        if rolled.empty:
            continue
        per_measure.append(
            summarize_draws(
                rolled,
                [count_col],
                ["location_id", "year_id", "level"],
                quantiles=quantiles,
            )
        )
    if not per_measure:
        return pd.DataFrame()
    out = per_measure[0]
    for extra in per_measure[1:]:
        out = out.merge(extra, on=["location_id", "year_id", "level"], how="outer")
    return out


def build_all_age_summary(  # noqa: PLR0913 - inputs, output shape, and the arm's factors/denominators
    nc_path: Path,
    hierarchy_df: pd.DataFrame,
    population: pd.DataFrame,
    levels: Sequence[int],
    quantiles: tuple[float, float],
    *,
    factors: dict[str, pd.DataFrame] | None = None,
    denominator_population: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The all-age product for one forecast netCDF: counts, rates and the denominator used, per level."""
    denom = population if denominator_population is None else denominator_population
    with xr.open_dataset(nc_path) as ds:
        years = [int(y) for y in ds.year_id.to_numpy()]
        pop_admin2 = population.merge(
            hierarchy_df.loc[hierarchy_df.level == ADMIN2_LEVEL, ["location_id"]],
            on="location_id",
        )
        pieces = [
            p
            for year in years
            if not (
                p := summarize_one_year(
                    ds,
                    year,
                    hierarchy_df,
                    pop_admin2,
                    levels,
                    quantiles,
                    factors=factors,
                )
            ).empty
        ]
    if not pieces:
        msg = f"{nc_path} produced no summarised rows"
        raise ValueError(msg)
    summary = pd.concat(pieces, ignore_index=True).merge(
        denom, on=["location_id", "year_id"], how="left"
    )
    if summary.population.isna().any():
        msg = f"{int(summary.population.isna().sum())} summarised rows have no population denominator"
        raise ValueError(msg)
    for measure in PRED_VARS:
        count_col = f"{measure}_count"
        if f"{count_col}_mean" not in summary.columns:
            continue
        for stat in STATS:
            summary[f"{measure}_rate_{stat}"] = np.where(
                summary.population > 0,
                summary[f"{count_col}_{stat}"] / summary.population,
                0.0,
            )
    value_cols = [
        f"{m}_{space}_{s}"
        for m in PRED_VARS
        for space in ("count", "rate")
        for s in STATS
        if f"{m}_{space}_{s}" in summary.columns
    ]
    summary = summary[
        ["location_id", "year_id", "level", "population", *value_cols]
    ].astype(
        {
            "location_id": "int32",
            "year_id": "int16",
            "level": "int8",
            "population": "float64",
            **dict.fromkeys(value_cols, "float32"),
        }
    )
    return summary.sort_values(["level", "location_id", "year_id"]).reset_index(
        drop=True
    )


def build_anchor_check(
    summary: pd.DataFrame,
    anchor_year: int,
    rel_tol: float,
    observed: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Our anchor-year values against the raked observed, per location, measure and metric."""
    obs = observed if observed is not None else observed_all_age(anchor_year)
    ours = summary[summary.year_id == anchor_year]
    if ours.empty:
        msg = f"summary has no rows for anchor year {anchor_year}"
        raise ValueError(msg)
    rows = []
    for measure in PRED_VARS:
        for space in ("count", "rate"):
            mean_col, obs_col = f"{measure}_{space}_mean", f"{measure}_{space}"
            if mean_col not in ours.columns or obs_col not in obs.columns:
                continue
            merged = ours[["location_id", "level", mean_col]].merge(
                obs[["location_id", obs_col]], on="location_id"
            )
            rows.append(
                pd.DataFrame(
                    {
                        "location_id": merged.location_id.astype("int32"),
                        "level": merged.level.astype("int8"),
                        "year_id": np.int16(anchor_year),
                        "measure": measure,
                        "metric": space,
                        "observed": merged[obs_col].astype("float64"),
                        "predicted": merged[mean_col].astype("float64"),
                    }
                )
            )
    check = pd.concat(rows, ignore_index=True)
    check["abs_diff"] = check.predicted - check.observed
    denom = check.observed.abs()
    check["rel_diff"] = np.where(denom > 0, check.abs_diff / denom, np.nan)
    check["within_tol"] = (check.rel_diff.abs() <= rel_tol) | (
        (denom == 0) & (check.predicted.abs() == 0)
    )
    has_rel = check.rel_diff.notna().any()
    worst = check.loc[check.rel_diff.abs().idxmax()] if has_rel else None
    verdict: dict[str, object] = {
        "anchor_year": anchor_year,
        "rel_tol": rel_tol,
        "n_compared": len(check),
        "n_failed": int((~check.within_tol).sum()),
        "passed": bool(check.within_tol.all()),
        "max_abs_rel_diff": float(check.rel_diff.abs().max()) if has_rel else None,
        "worst": None
        if worst is None
        else {
            "location_id": int(worst.location_id),
            "level": int(worst.level),
            "measure": str(worst.measure),
            "metric": str(worst.metric),
            "observed": float(worst.observed),
            "predicted": float(worst.predicted),
            "rel_diff": float(worst.rel_diff),
        },
    }
    return check, verdict
