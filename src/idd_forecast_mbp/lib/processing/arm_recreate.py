"""Recreate any arm's admin-2 draw-level age/sex values (2,A,X) in memory, and differences of them.

2,A,X is never saved (DECISIONS 2026-09-28): it is rebuilt here, exactly, from the saved forecast
draws, the population artifacts, the age/sex pattern and the vaccine protection, with the same arm
weights the saved all-age products used (``arm_factors``), so a recreated frame rolled up to any
level equals the saved summary's draws there. Typical uses: match draws and scenario across two
arms (or two scenarios within an arm), difference at admin-2, then roll the difference up.

    from idd_forecast_mbp.lib.data.malaria_arms import load_arms
    arm = load_arms(ARMS_FILE).arm("spec7426_best", "Baseline")
    base = recreate_admin2_age_sex(arm, "ssp245", ve_variant="loglinear_severe0", draws=[0, 1])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.processing.aggregation import roll_up_hierarchy
from idd_forecast_mbp.lib.processing.arm_factors import (
    KEYS_AA,
    KEYS_AS,
    admin2_protection,
    arm_age_sex_counts,
    cell_weights,
    hold_population,
    structure_multiplier,
)
from idd_forecast_mbp.lib.processing.disaggregation import (
    compute_as_rr,
    malaria_as_fractions,
)
from idd_forecast_mbp.lib.processing.vaccine_impact import eligible_locations

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from idd_forecast_mbp.lib.data.malaria_arms import Arm

ANCHOR_YEAR = 2023
LOG_RATE_VARS = {
    "inc": "log_malaria_inc_rate_pred",
    "mort": "log_malaria_mort_rate_pred",
}
FORECAST_NODE = mbpc.MAL_FORECAST_OUTPUTS_READ_PATH.parent


def forecast_nc(arm: Arm, ssp: str, node: Path = FORECAST_NODE) -> Path:
    """The arm's forecast netCDF for ``ssp``: the launch's label on the forecast node."""
    return node / arm.forecast_label / f"malaria_forecast_{ssp}_{arm.dah}.nc"


def _read_as_malaria(
    locations: Sequence[int],
) -> pd.DataFrame:  # pragma: no cover - constant-path I/O
    return read_parquet_with_integer_ids(
        mbpc.MAL_RAKED_AS_READ_PATH / "as_full_malaria_df.parquet",
        columns=[
            *KEYS_AS,
            "malaria_inc_rate",
            "aa_malaria_inc_rate",
            "malaria_mort_rate",
            "aa_malaria_mort_rate",
        ],
        filters=[
            ("location_id", "in", list(locations)),
            ("year_id", "==", ANCHOR_YEAR),
        ],
    )


def read_population(
    name: str, keys: list[str], locations: Sequence[int], years: Sequence[int]
) -> pd.DataFrame:  # pragma: no cover
    return read_parquet_with_integer_ids(
        mbpc.POPULATION_READ_PATH / name,
        columns=[*keys, "population"],
        filters=[
            ("location_id", "in", list(locations)),
            ("year_id", "in", list(years)),
        ],
    )


def _read_protection(
    ve_variant: str, product: str, locations: Sequence[int]
) -> pd.DataFrame:  # pragma: no cover
    prot = read_parquet_with_integer_ids(
        mbpc.MAL_VACCINE_COHORTS_READ_PATH
        / mbpc.MAL_VACCINE_COHORTS_FILENAME_TEMPLATE.format(
            variant=ve_variant, products=product
        ),
        columns=[*KEYS_AS, "effective_protection_case", "effective_protection_death"],
    )
    mapping = eligible_locations(mbpc.VACCINE_COVERAGE_FILE, set(locations))
    return admin2_protection(prot, mapping)


def arm_weights(  # noqa: PLR0913 - the arm plus four injectable inputs
    arm: Arm,
    locations: Sequence[int],
    years: Sequence[int],
    ve_variant: str,
    *,
    as_malaria: pd.DataFrame | None = None,
    as_population: pd.DataFrame | None = None,
    protection: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The arm's per-cell weights ``w = f * m * (1 - p)`` at ``locations`` x ``years``.

    ``f`` is the forecast's own split (``as_rr`` frozen at 2023 x the year's age/sex population, or
    the hold-year population for the population hold). Inputs default to the canonical artifacts;
    tests inject frames.
    """
    years = [int(y) for y in years]
    hold = arm.population_hold_year or arm.age_structure_hold_year
    want_years = sorted(set(years) | ({int(hold)} if hold else set()))
    as_mal = as_malaria if as_malaria is not None else _read_as_malaria(locations)
    rr = compute_as_rr(as_mal, anchor_year=ANCHOR_YEAR)
    as_pop = (
        as_population
        if as_population is not None
        else read_population(
            "as_2023_full_population_df.parquet", KEYS_AS, locations, want_years
        )
    )
    if arm.population_hold_year is not None:
        as_pop = hold_population(as_pop, arm.population_hold_year, keys=KEYS_AS)
    multiplier = (
        structure_multiplier(as_pop, arm.age_structure_hold_year)
        if arm.age_structure_hold_year is not None
        else None
    )
    fractions = malaria_as_fractions(rr, as_pop[as_pop.year_id.isin(years)])
    if arm.vaccine == "none":
        prot = None
    else:
        prot = (
            protection
            if protection is not None
            else _read_protection(ve_variant, arm.vaccine, locations)
        )
    return cell_weights(fractions, protection=prot, multiplier=multiplier)


def read_admin2_log_rates(
    nc_path: Path,
    *,
    years: Sequence[int] | None = None,
    draws: Sequence[int] | None = None,
) -> pd.DataFrame:
    """``[location_id, year_id, draw, inc_log_rate, mort_log_rate]`` from one forecast netCDF.

    Reads each variable CONTIGUOUSLY and subsets in numpy (DECISIONS 2026-08-24: scattered ``.sel``
    on these files is orders of magnitude slower).
    """
    with xr.open_dataset(nc_path) as ds:
        locs = ds.location_id.to_numpy()
        yrs = ds.year_id.to_numpy()
        drs = ds.draw.to_numpy()
        yi = (
            np.arange(yrs.size)
            if years is None
            else np.flatnonzero(np.isin(yrs, list(years)))
        )
        di = (
            np.arange(drs.size)
            if draws is None
            else np.flatnonzero(np.isin(drs, list(draws)))
        )
        cols = {}
        for m, var in LOG_RATE_VARS.items():
            arr = ds[var].transpose("location_id", "year_id", "draw").to_numpy()
            cols[f"{m}_log_rate"] = arr[:, yi][:, :, di].ravel()
    n_l, n_y, n_d = locs.size, yi.size, di.size
    return pd.DataFrame(
        {
            "location_id": np.repeat(locs, n_y * n_d).astype("int64"),
            "year_id": np.tile(np.repeat(yrs[yi], n_d), n_l).astype("int64"),
            "draw": np.tile(drs[di], n_l * n_y).astype("int64"),
            **cols,
        }
    )


def admin2_all_age_counts(
    log_rates: pd.DataFrame, aa_population: pd.DataFrame
) -> pd.DataFrame:
    """All-age counts per draw: ``exp(log rate) x population``; a masked (NaN) outcome stays NaN."""
    out = log_rates.merge(
        aa_population[[*KEYS_AA, "population"]], on=KEYS_AA, how="left"
    )
    if out.population.isna().any():
        msg = f"{int(out.population.isna().sum())} forecast rows have no population"
        raise ValueError(msg)
    for m in LOG_RATE_VARS:
        out[f"{m}_count"] = np.exp(out[f"{m}_log_rate"]) * out.population
    return out[[*KEYS_AA, "draw", "inc_count", "mort_count"]]


def recreate_admin2_age_sex(  # noqa: PLR0913 - arm, scenario and the optional subsets/injections
    arm: Arm,
    ssp: str,
    *,
    ve_variant: str,
    years: Sequence[int] | None = None,
    draws: Sequence[int] | None = None,
    node: Path = FORECAST_NODE,
    weights: pd.DataFrame | None = None,
    aa_population: pd.DataFrame | None = None,
    log_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The arm's admin-2 age/sex counts per draw (2,A,X) for ``ssp``; never written anywhere.

    Returns ``[location_id, year_id, age_group_id, sex_id, draw, inc_count, mort_count]``. A
    masked outcome (the forecast's zero-burden policy) is NaN; non-endemic locations have no rows.
    ``log_rates`` (from ``read_admin2_log_rates``) skips the read when a caller loops over draws.
    """
    rates = (
        log_rates
        if log_rates is not None
        else read_admin2_log_rates(
            forecast_nc(arm, ssp, node), years=years, draws=draws
        )
    )
    locs = sorted(rates.location_id.unique().tolist())
    yrs = sorted(rates.year_id.unique().tolist())
    if aa_population is None:  # pragma: no cover - constant-path I/O
        want = sorted(
            set(yrs)
            | ({arm.population_hold_year} if arm.population_hold_year else set())
        )
        aa_population = read_population(
            "aa_2023_full_population_df.parquet", KEYS_AA, locs, want
        )
    if arm.population_hold_year is not None:
        aa_population = hold_population(
            aa_population, arm.population_hold_year, keys=KEYS_AA
        )
    if weights is None:  # pragma: no cover - reads the canonical artifacts
        weights = arm_weights(arm, locs, yrs, ve_variant)
    return arm_age_sex_counts(admin2_all_age_counts(rates, aa_population), weights)


def draw_aligned_difference(
    a: pd.DataFrame, b: pd.DataFrame, value_cols: Sequence[str]
) -> pd.DataFrame:
    """``a - b`` cell by cell; refuses unless both carry exactly the same cells and draws.

    The point of differencing at admin-2 is that draw d of one arm/scenario is compared with draw d
    of the other; a silent outer join would compare a cell with nothing.
    """
    keys = [c for c in a.columns if c not in value_cols]
    if sorted(keys) != sorted(c for c in b.columns if c not in value_cols):
        msg = "the two frames have different key columns"
        raise ValueError(msg)
    merged = a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        msg = f"{unmatched} cell(s) present in only one frame; draws/cells are not aligned"
        raise ValueError(msg)
    out = merged[keys].copy()
    for c in value_cols:
        out[c] = merged[f"{c}_a"] - merged[f"{c}_b"]
    return out


def roll_up_age_sex_to_nodes(  # noqa: PLR0913 - the counts, the hierarchy and the full output grid
    as_counts: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    node_ids: Sequence[int],
    *,
    years: Sequence[int],
    draws: Sequence[int],
    ages: Sequence[int],
    sexes: Sequence[int],
) -> pd.DataFrame:
    """Admin-2 age/sex counts rolled up in count space to ``node_ids``, on the complete grid.

    A node with no endemic admin-2 below it is a true zero, not a gap, so the result is
    zero-filled over node x year x age x sex x draw. A masked outcome (NaN) contributes nothing.
    """
    grid = pd.MultiIndex.from_product(
        [
            sorted(int(n) for n in node_ids),
            list(years),
            list(ages),
            list(sexes),
            list(draws),
        ],
        names=[*KEYS_AS, "draw"],
    )
    out = pd.DataFrame(index=grid)
    for m in ("inc", "mort"):
        col = f"{m}_count"
        part = as_counts.loc[as_counts[col].notna(), [*KEYS_AS, "draw", col]]
        rolled = roll_up_hierarchy(
            part,
            hierarchy_df,
            col,
            preserve_age_sex=True,
            extra_group_cols=["draw"],
            start_level=5,
        )
        rolled = rolled[rolled.location_id.isin({int(n) for n in node_ids})]
        out[col] = (
            rolled.set_index([*KEYS_AS, "draw"])[col]
            .reindex(grid)
            .fillna(0.0)
            .to_numpy()
        )
    return out.reset_index()
