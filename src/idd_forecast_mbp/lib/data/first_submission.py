"""Read the first-submission upload run, for old-vs-new comparison.

``05-upload_data/upload_folders/2025_08_28`` is the run behind the FIRST
SUBMISSION (``.claude/DECISIONS.md`` 2026-07-31). It is not the newest directory
under ``upload_folders`` — a later Goalkeepers run sits there too — so the date is
pinned explicitly rather than picked by recency.

Layout: one directory per *arm*, named
``aa_cause_{cause}_measure_{measure}_metric_{metric}_ssp_scenario_{ssp}[_hold_{x}]``,
each holding a single ``draws.nc`` with dims ``(location_id, year_id, draw_id)``
and variable ``val``.

Two properties make it directly comparable without re-deriving anything:

- Values are **all-age counts at every hierarchy level**, so global and
  super-region rows already exist. No re-aggregation, and no chance of applying a
  different denominator than the original run did.
- Only counts are stored, so a *rate* comparison has to divide by the population
  artifact — and must use the same population the new run uses, or the difference
  includes a denominator change.

Caveats from that decision entry, which matter when reading a comparison plot:
this run starts at **2022** while the current one starts at 2023, and its 2023
global mortality differs from the current vintage (706,112 vs 669,712 for
malaria), so it is anchored to a different observed vintage. An old-vs-new gap
therefore contains a baseline shift as well as a model difference.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The run behind the first submission. Pinned, not resolved by recency.
FIRST_SUBMISSION_RUN = "2025_08_28"

#: Measure name in the arm directory, per outcome.
ARM_MEASURE = {"inc": "incidence", "mort": "mortality"}


def upload_root(run_date: str = FIRST_SUBMISSION_RUN) -> Path:
    """Directory holding one subdirectory per arm."""
    return mbpc.MODEL_ROOT / "05-upload_data" / "upload_folders" / run_date


def arm_name(
    measure: str,
    ssp_scenario: str,
    *,
    cause: str = "dengue",
    metric: str = "count",
    hold: str | None = None,
) -> str:
    """The arm directory name for one (cause, measure, metric, ssp[, hold])."""
    measure_name = ARM_MEASURE.get(measure, measure)
    name = (f"aa_cause_{cause}_measure_{measure_name}_metric_{metric}"
            f"_ssp_scenario_{ssp_scenario}")
    return f"{name}_hold_{hold}" if hold else name


def load_arm(  # noqa: PLR0913
    measure: str,
    ssp_scenario: str,
    *,
    locations: Sequence[int],
    years: Sequence[int] | None = None,
    cause: str = "dengue",
    hold: str | None = None,
    run_date: str = FIRST_SUBMISSION_RUN,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> pd.DataFrame:
    """All-age counts for one arm at ``locations``, collapsed to mean/lower/upper.

    Subset before collapsing: each ``draws.nc`` is ~700 MB and we normally want a
    handful of aggregate rows out of 51,590 locations.

    Returns ``(location_id, year_id, count_mean, count_lower, count_upper)``.
    """
    path = upload_root(run_date) / arm_name(
        measure, ssp_scenario, cause=cause, hold=hold) / "draws.nc"
    if not path.exists():
        msg = f"no first-submission arm at {path}"
        raise FileNotFoundError(msg)

    lower_q, upper_q = quantiles
    with xr.open_dataset(path) as ds:
        available = set(ds["location_id"].to_numpy().tolist())
        keep = [int(loc) for loc in locations if int(loc) in available]
        if not keep:
            msg = f"none of {list(locations)[:5]}... present in {path.parent.name}"
            raise KeyError(msg)
        block = ds["val"].sel(location_id=keep)
        if years is not None:
            block = block.sel(year_id=[int(y) for y in years])
        draw_dim = "draw_id" if "draw_id" in block.dims else "draw"
        # Each .quantile() attaches its own scalar `quantile` coord; combining two
        # of them into one Dataset conflicts, so drop it from each.
        summary = xr.Dataset({
            "count_mean": block.mean(dim=draw_dim),
            "count_lower": block.quantile(lower_q, dim=draw_dim).drop_vars(
                "quantile", errors="ignore"),
            "count_upper": block.quantile(upper_q, dim=draw_dim).drop_vars(
                "quantile", errors="ignore"),
        })
        out = summary.to_dataframe().reset_index()

    return out[["location_id", "year_id", "count_mean", "count_lower", "count_upper"]]


def load_comparison(  # noqa: PLR0913
    ssp_scenarios: Sequence[str],
    *,
    locations: Sequence[int],
    years: Sequence[int] | None = None,
    measures: Sequence[str] = ("inc", "mort"),
    cause: str = "dengue",
    run_date: str = FIRST_SUBMISSION_RUN,
) -> pd.DataFrame:
    """Every (measure, ssp) arm stacked, for plotting against a new run."""
    frames = []
    for ssp in ssp_scenarios:
        for measure in measures:
            piece = load_arm(measure, ssp, locations=locations, years=years,
                             cause=cause, run_date=run_date)
            piece["measure"] = measure
            piece["ssp_scenario"] = ssp
            frames.append(piece)
    return pd.concat(frames, ignore_index=True)
