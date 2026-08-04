"""Finish one forecast run into all-age hierarchy products.

Reads the stage-04 draw-level admin-2 log-rate netCDFs for a SINGLE forecast run
directory — named explicitly, never resolved through a ``current`` symlink — and
writes, per ``(ssp, dah)``:

``all_age_summary_{ssp}_{dah}.parquet``
    Mean / lower / upper incidence and mortality, in both count and rate space, at
    the requested hierarchy levels (default global, super-region, region, country).

``anchor_{year}_{ssp}_{dah}.parquet``
    Our value against the raked observed value at the anchor year, per location and
    measure, with a numeric tolerance flag. Written alongside a small JSON verdict.

Invariants
----------
Counts aggregate up the hierarchy in count space; they are never derived from an
aggregated rate. An aggregate rate is a count divided by that level's own population
row, read from the population artifact — population is never summed from children.
Locations the forecast dropped are absent rather than zero-filled, so they contribute
nothing to a parent's count while the parent's population denominator still covers
them.

Why this is cheap
-----------------
Work is chunked by year. One year of draws is about a million rows per measure, and
the draw axis is collapsed before any year is concatenated, so peak memory is tens
of megabytes rather than the shape of the output hierarchy. Rate summaries are
derived by dividing the summarised counts: population carries no draw dimension, so
summarising and then dividing is exactly equal to dividing and then summarising.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.processing.aggregation import roll_up_hierarchy
from idd_forecast_mbp.lib.processing.summarize import summarize_draws
from idd_forecast_mbp.lib.versioning import finalize_artifact

# Stage-04 netCDF variable per measure. The rocket writes log rates.
PRED_VARS = {
    "inc": "log_malaria_inc_rate_pred",
    "mort": "log_malaria_mort_rate_pred",
}
STATS = ("mean", "lower", "upper")
ADMIN2_LEVEL = 5


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_population(
    years: Sequence[int], hold_year: int | None = None
) -> pd.DataFrame:
    """All-age population for every hierarchy level, restricted to ``years``.

    This is the single source of every denominator in this module. It is read, never
    derived, and never summed from children.

    ``hold_year`` gives the ACCOUNTING population sensitivity: every year's population
    is replaced by its ``hold_year`` value. Because this one frame supplies both the
    admin-2 rate-to-count multiplier and the aggregate denominator, freezing it here
    is the whole intervention — nothing downstream needs to know.

    Note what this is NOT: it does not touch the forecast, so predicted admin-2 RATES
    are unchanged. Aggregate rates do move, because an aggregate rate is a count over
    that level's own population and both terms change. This is deliberately the
    narrower of the two population experiments; the demographic one has to rebuild the
    per-capita DAH covariate and re-forecast.
    """
    path = mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet"
    pop = read_parquet_with_integer_ids(
        path, columns=["location_id", "year_id", "population"]
    )
    wanted = sorted(set(int(y) for y in years) | ({int(hold_year)} if hold_year else set()))
    pop = pop[pop.year_id.isin(wanted)]
    if pop.empty:
        raise ValueError(f"population artifact {path} has no rows for years {years}")

    if hold_year is not None:
        ref = (
            pop.loc[pop.year_id == int(hold_year), ["location_id", "population"]]
            .set_index("location_id")["population"]
        )
        if ref.empty:
            raise ValueError(f"population artifact has no rows for hold year {hold_year}")
        pop = pop.copy()
        pop["population"] = pop.location_id.map(ref)
        missing = int(pop.population.isna().sum())
        if missing:
            raise ValueError(
                f"{missing} location-years have no {hold_year} population to hold at"
            )
    return pop[pop.year_id.isin([int(y) for y in years])].reset_index(drop=True)


def age_structure_hold_factor(
    measure: str,
    hold_year: int,
    location_ids: Sequence[int],
    years: Sequence[int],
) -> pd.DataFrame:
    """Per-(location, year) multiplier that freezes the age/sex STRUCTURE at ``hold_year``.

    The old chain (``OLD_make_population_hold_variables_by_draw.py``) defined this on
    age/sex counts as ``count * (as_pop[ref]/as_pop[y]) / (aa_pop[ref]/aa_pop[y])``.
    Total population cancels out of that ratio, leaving ``as_share[ref]/as_share[y]`` —
    each age/sex group reweighted by the change in its SHARE of the population, with
    total population still free to grow. So this is not a redistribution: the per-group
    weights differ, so the all-age total moves.

    Because our disaggregation is normalised to the all-age total, the all-age effect
    collapses to a scalar per (location, year) and no age/sex product is needed:

        factor[l, y] = sum over (a, s) of  rr_share[l, a, s] * as_share[l, ref, a, s]
                                                             / as_share[l, y, a, s]

    where ``rr_share`` is the observed age/sex share of burden at ``hold_year``. At
    ``y == hold_year`` every term collapses and the factor is exactly 1, so the anchor
    year is untouched — which is the check to look at first if this ever looks wrong.

    Must be applied at admin-2 BEFORE the count-space roll-up, matching the old chain,
    not to an already-aggregated total.
    """
    locs = [int(x) for x in location_ids]
    want_years = sorted(set(int(y) for y in years) | {int(hold_year)})
    loc_filter = [("location_id", "in", locs)]

    # Observed age/sex share of burden at the hold year: as count / all-age count.
    burden = read_parquet_with_integer_ids(
        mbpc.MAL_RAKED_AS_READ_PATH / "as_full_malaria_df.parquet",
        columns=[
            "location_id", "year_id", "age_group_id", "sex_id",
            f"malaria_{measure}_count", f"aa_malaria_{measure}_count",
        ],
        filters=loc_filter + [("year_id", "==", int(hold_year))],
    )
    if burden.empty:
        raise ValueError(
            f"no age/sex {measure} burden rows at {hold_year} for the forecast locations"
        )
    denom = burden[f"aa_malaria_{measure}_count"]
    burden["rr_share"] = np.where(
        denom > 0, burden[f"malaria_{measure}_count"] / denom, 0.0
    )
    burden = burden[["location_id", "age_group_id", "sex_id", "rr_share"]]

    # The factor is a burden-share-WEIGHTED AVERAGE of share ratios, so the weights must
    # sum to 1 per location. Normalise them: it corrects any small inconsistency between
    # the raked age/sex counts and their all-age total, and it makes the hold-year
    # identity exact. Locations with NO burden at all (non-endemic admin-2, which is most
    # of them) sum to zero and are handled below by leaving their factor at 1.
    share_sum = burden.groupby("location_id", observed=True)["rr_share"].transform("sum")
    burden["rr_share"] = np.where(share_sum > 0, burden.rr_share / share_sum, 0.0)
    no_burden = set(
        burden.loc[share_sum <= 0, "location_id"].unique().tolist()
    )

    # Age/sex population shares: the hold year, and every forecast year.
    as_pop = read_parquet_with_integer_ids(
        mbpc.POPULATION_READ_PATH / "as_2023_full_population_df.parquet",
        columns=[
            "location_id", "year_id", "age_group_id", "sex_id",
            "as_population_fraction",
        ],
        filters=loc_filter + [("year_id", "in", want_years)],
    )
    as_pop["as_population_fraction"] = as_pop.as_population_fraction.astype("float32")

    ref = as_pop.loc[as_pop.year_id == int(hold_year)].drop(columns="year_id").rename(
        columns={"as_population_fraction": "share_ref"}
    )
    merged = as_pop.merge(
        ref, on=["location_id", "age_group_id", "sex_id"], how="inner"
    ).merge(burden, on=["location_id", "age_group_id", "sex_id"], how="inner")

    # A group with no population in year y contributes nothing rather than dividing by 0.
    merged["term"] = np.where(
        merged.as_population_fraction > 0,
        merged.rr_share * merged.share_ref / merged.as_population_fraction,
        0.0,
    )
    factor = (
        merged.groupby(["location_id", "year_id"], observed=True)["term"]
        .sum()
        .reset_index(name="factor")
    )
    # A location with no burden is left unscaled rather than zeroed. Numerically it makes
    # no difference (zero times anything is zero), but it keeps the factor interpretable
    # and lets the hold-year identity below be an exact check.
    factor.loc[factor.location_id.isin(no_burden), "factor"] = 1.0

    at_hold = factor[factor.year_id == int(hold_year)]["factor"]
    if len(at_hold) and not np.allclose(at_hold, 1.0, atol=1e-6):
        raise ValueError(
            f"age-structure factor at the hold year {hold_year} is not 1 "
            f"(min {at_hold.min():.6f}, max {at_hold.max():.6f}); the burden shares and "
            f"population shares are inconsistent"
        )
    return factor[factor.year_id.isin([int(y) for y in years])]


def anchor_vintage_factor(
    measure: str,
    anchor_year: int,
    location_ids: Sequence[int],
    years: Sequence[int],
    alt_observed: Path,
) -> pd.DataFrame:
    """Per-location rescale that swaps the ANCHOR to a different observed vintage.

    The rocket shifts each location so its anchor-year prediction equals that location's
    observed anchor-year value:

        rate(y) = obs[anchor] * exp( pred_log(y) - pred_log(anchor) )

    so the whole trajectory is PROPORTIONAL to the anchor value. Swapping the anchor for
    a different observed vintage is therefore an exact per-location multiplication by
    ``obs_alt[anchor] / obs_current[anchor]`` — no re-forecast, and no approximation.

    That makes this the cheap arm of the old-vs-new decomposition: it isolates how much
    of the gap between two runs is purely the observed data vintage the forecast was
    anchored to, holding the model and the covariates fixed.

    Locations with no value in the alternative vintage keep a factor of 1 and are
    counted, since the two vintages sit on different hierarchy generations.
    """
    cols = ["location_id", "year_id", f"malaria_{measure}_rate"]
    cur = read_parquet_with_integer_ids(
        mbpc.MAL_RAKED_AA_READ_PATH / "aa_full_malaria_df.parquet", columns=cols
    )
    alt = read_parquet_with_integer_ids(alt_observed, columns=cols)
    locs = {int(x) for x in location_ids}
    cur = cur[(cur.year_id == int(anchor_year)) & cur.location_id.isin(locs)]
    alt = alt[(alt.year_id == int(anchor_year)) & alt.location_id.isin(locs)]

    m = cur.merge(alt, on="location_id", suffixes=("_cur", "_alt"))
    c, a = m[f"malaria_{measure}_rate_cur"], m[f"malaria_{measure}_rate_alt"]
    m["factor"] = np.where(c > 0, a / c, 1.0)
    n_missing = len(locs) - len(m)
    click.echo(
        f"  anchor vintage [{measure}]: matched {len(m):,} of {len(locs):,} locations "
        f"({n_missing:,} unmatched keep factor 1); ratio median "
        f"{m.factor.median():.4f}, mean {m.factor.mean():.4f}"
    )
    per_loc = m[["location_id", "factor"]]
    return pd.MultiIndex.from_product(
        [per_loc.location_id, [int(y) for y in years]], names=["location_id", "year_id"]
    ).to_frame(index=False).merge(per_loc, on="location_id")


def observed_all_age(year: int) -> pd.DataFrame:
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
        raise ValueError(f"observed artifact {path} has no rows for year {year}")
    return obs.rename(
        columns={
            "malaria_inc_count": "inc_count",
            "malaria_inc_rate": "inc_rate",
            "malaria_mort_count": "mort_count",
            "malaria_mort_rate": "mort_rate",
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-year draw work
# ---------------------------------------------------------------------------

def admin2_counts_one_year(
    da: xr.DataArray,
    year: int,
    pop_admin2: pd.DataFrame,
    count_col: str,
) -> pd.DataFrame:
    """Draw-level admin-2 counts for one year, as a long frame.

    The netCDF holds log rates. Exponentiate, multiply by the admin-2 population for
    that year, and drop locations whose outcome is masked (the rocket writes NaN for
    every draw of a masked outcome, so these are whole locations, not stray cells).
    """
    slab = da.sel(year_id=year).transpose("location_id", "draw")
    loc_ids = slab.location_id.values
    draw_ids = slab.draw.values

    pop_year = pop_admin2[pop_admin2.year_id == year].set_index("location_id")[
        "population"
    ]
    missing = np.setdiff1d(loc_ids, pop_year.index.values)
    if missing.size:
        raise ValueError(
            f"{missing.size} forecast locations have no population row for {year} "
            f"(first few: {missing[:5].tolist()}); the denominator artifact is "
            f"inconsistent with the forecast location set"
        )
    pop_vec = pop_year.reindex(loc_ids).to_numpy()

    counts = np.exp(slab.values) * pop_vec[:, None]

    frame = pd.DataFrame(
        {
            "location_id": np.repeat(loc_ids, draw_ids.size).astype("int64"),
            "year_id": np.int64(year),
            "draw": np.tile(draw_ids, loc_ids.size).astype("int64"),
            count_col: counts.ravel(),
        }
    )
    return frame.dropna(subset=[count_col]).reset_index(drop=True)


def summarize_one_year(
    ds: xr.Dataset,
    year: int,
    hierarchy_df: pd.DataFrame,
    pop_admin2: pd.DataFrame,
    levels: Sequence[int],
    quantiles: tuple[float, float],
    as_factors: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Aggregate and summarise both measures for a single year."""
    per_measure = []
    for measure, var in PRED_VARS.items():
        if var not in ds.data_vars:
            continue
        count_col = f"{measure}_count"
        draws = admin2_counts_one_year(ds[var], year, pop_admin2, count_col)
        if draws.empty:
            continue

        # Age/sex-structure hold: scale admin-2 counts BEFORE the roll-up, matching the
        # old chain's order of operations. A location with no factor row keeps 1.0.
        if as_factors is not None and measure in as_factors:
            f = as_factors[measure]
            f_year = f.loc[f.year_id == year].set_index("location_id")["factor"]
            scale = draws.location_id.map(f_year).fillna(1.0).to_numpy()
            draws[count_col] = draws[count_col].to_numpy() * scale

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


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def build_all_age_summary(
    nc_path: Path,
    hierarchy_df: pd.DataFrame,
    population: pd.DataFrame,
    levels: Sequence[int],
    quantiles: tuple[float, float],
    as_factors: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """The all-age product for one (ssp, dah) netCDF."""
    with xr.open_dataset(nc_path) as ds:
        years = [int(y) for y in ds.year_id.values]
        pop_admin2 = population.merge(
            hierarchy_df.loc[hierarchy_df.level == ADMIN2_LEVEL, ["location_id"]],
            on="location_id",
        )
        pieces = []
        for year in years:
            piece = summarize_one_year(
                ds, year, hierarchy_df, pop_admin2, levels, quantiles,
                as_factors=as_factors,
            )
            if not piece.empty:
                pieces.append(piece)

    if not pieces:
        raise ValueError(f"{nc_path} produced no summarised rows")
    summary = pd.concat(pieces, ignore_index=True)

    # Denominators: read, joined on location_id, never summed from children.
    summary = summary.merge(population, on=["location_id", "year_id"], how="left")
    if summary.population.isna().any():
        n = int(summary.population.isna().sum())
        raise ValueError(f"{n} summarised rows have no population denominator")

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
    summary = summary[["location_id", "year_id", "level", "population", *value_cols]]
    summary = summary.astype(
        {
            "location_id": "int32",
            "year_id": "int16",
            "level": "int8",
            "population": "float64",
            **{c: "float32" for c in value_cols},
        }
    )
    return summary.sort_values(["level", "location_id", "year_id"]).reset_index(
        drop=True
    )


def build_anchor_check(
    summary: pd.DataFrame, anchor_year: int, rel_tol: float
) -> tuple[pd.DataFrame, dict]:
    """Our anchor-year values against the raked observed, per location and measure.

    The forecast shifts every admin-2 location so its anchor year equals the observed
    value, so a mismatch here is a statement about aggregation, not about the
    regression.
    """
    obs = observed_all_age(anchor_year)
    ours = summary[summary.year_id == anchor_year]
    if ours.empty:
        raise ValueError(f"summary has no rows for anchor year {anchor_year}")

    rows = []
    for measure in PRED_VARS:
        for space in ("count", "rate"):
            mean_col = f"{measure}_{space}_mean"
            obs_col = f"{measure}_{space}"
            if mean_col not in ours.columns or obs_col not in obs.columns:
                continue
            merged = ours[["location_id", "level", mean_col]].merge(
                obs[["location_id", obs_col]], on="location_id", how="inner"
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

    worst = (
        check.loc[check.rel_diff.abs().idxmax()]
        if check.rel_diff.notna().any()
        else None
    )
    verdict = {
        "anchor_year": anchor_year,
        "rel_tol": rel_tol,
        "n_compared": int(len(check)),
        "n_failed": int((~check.within_tol).sum()),
        "passed": bool(check.within_tol.all()),
        "max_abs_rel_diff": (
            float(check.rel_diff.abs().max()) if check.rel_diff.notna().any() else None
        ),
        "worst": (
            None
            if worst is None
            else {
                "location_id": int(worst.location_id),
                "level": int(worst.level),
                "measure": str(worst.measure),
                "metric": str(worst.metric),
                "observed": float(worst.observed),
                "predicted": float(worst.predicted),
                "rel_diff": float(worst.rel_diff),
            }
        ),
    }
    return check, verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--forecast-run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="stage-04 output dir for ONE run, e.g. "
    ".../forecast_outputs/lsae_1285/<model_key>. Named explicitly, never `current`.",
)
@click.option(
    "--ssp-scenario",
    "ssp_scenarios",
    multiple=True,
    default=("ssp126", "ssp245", "ssp585"),
    show_default=True,
    help="repeatable; missing netCDFs are skipped with a warning",
)
@click.option("--dah-scenario", default="Baseline", show_default=True)
@click.option(
    "--levels",
    default="0,1,2,3",
    show_default=True,
    help="hierarchy levels to save: 0 global, 1 super-region, 2 region, 3 country",
)
@click.option("--anchor-year", default=2023, show_default=True, type=int)
@click.option(
    "--rel-tol",
    default=1e-6,
    show_default=True,
    help="relative tolerance for the anchor-year match",
)
@click.option(
    "--lower-quantile", default=0.025, show_default=True, type=float
)
@click.option(
    "--upper-quantile", default=0.975, show_default=True, type=float
)
@click.option(
    "--run-key",
    default=None,
    help="product node name; defaults to the forecast run dir's basename, with a "
         "hold suffix appended when --population-hold-year is set",
)
@click.option(
    "--anchor-vintage",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="alternative raked-observed parquet to anchor on. Rescales each location by "
         "the ratio of that vintage's anchor-year rate to the current one -- EXACT, no "
         "re-forecast. Use to isolate how much of an old-vs-new gap is data vintage.",
)
@click.option(
    "--age-structure-hold-year",
    default=None,
    type=int,
    help="AGE/SEX-STRUCTURE sensitivity: hold each age/sex group's SHARE of the "
         "population at this year, letting total population grow. Needs no "
         "re-forecast and no age/sex product -- it reduces to a per-admin-2 scalar. "
         "Unlike the population hold this is NOT pure accounting: it reweights age "
         "groups with different burden shares, so the all-age total moves.",
)
@click.option(
    "--population-hold-year",
    default=None,
    type=int,
    help="ACCOUNTING population sensitivity: use this year's population for every "
         "year, in both the rate-to-count step and the aggregate denominator. Needs "
         "no re-forecast. Does not change admin-2 rates; does change aggregate rates.",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="override the versioned product dir (skips finalize)",
)
@click.option(
    "--set-current/--no-set-current",
    default=True,
    show_default=True,
    help="repoint this run's product `current` symlink. Safe for a sensitivity: each "
         "run key is its own artifact node, so this never touches the baseline.",
)
@click.option(
    "--strict-anchor/--no-strict-anchor",
    default=False,
    show_default=True,
    help="exit nonzero when the anchor-year match is outside tolerance. Off by "
         "default because the ~0.1%% aggregate gap is structural, not a bug.",
)
def main(  # noqa: PLR0913
    forecast_run_dir: Path,
    ssp_scenarios: tuple[str, ...],
    dah_scenario: str,
    levels: str,
    anchor_year: int,
    rel_tol: float,
    lower_quantile: float,
    upper_quantile: float,
    run_key: str | None,
    population_hold_year: int | None,
    age_structure_hold_year: int | None,
    anchor_vintage: Path | None,
    output_dir: Path | None,
    set_current: bool,
    strict_anchor: bool,
) -> None:
    """Turn one stage-04 forecast run into all-age hierarchy products."""
    level_ids = [int(x) for x in levels.split(",") if x.strip() != ""]
    quantiles = (lower_quantile, upper_quantile)
    # A held-population finish must never land on top of the unheld product, so the
    # hold goes into the product node name unless a run key was named explicitly.
    if run_key is None:
        run_key = forecast_run_dir.name
        if population_hold_year is not None:
            run_key = f"{run_key}__denom_hold{population_hold_year}"
        if age_structure_hold_year is not None:
            run_key = f"{run_key}__asstruct_hold{age_structure_hold_year}"
        if anchor_vintage is not None:
            run_key = f"{run_key}__anchor_{anchor_vintage.parent.name}"

    if output_dir is None:
        out_dir = mbpc.mal_products_write_path(run_key)
        node_root = mbpc.mal_products_root(run_key)
    else:
        out_dir = output_dir
        node_root = None
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"run_key          : {run_key}")
    click.echo(f"forecast run dir : {forecast_run_dir}")
    click.echo(f"levels           : {level_ids}")
    if population_hold_year is not None:
        click.echo(f"population hold  : {population_hold_year} (accounting only)")
    if age_structure_hold_year is not None:
        click.echo(f"age-struct hold  : {age_structure_hold_year}")
    click.echo(f"output dir       : {out_dir}")

    hierarchy_df = load_hierarchy()
    verdicts: dict[str, dict] = {}
    wrote_any = False

    for ssp in ssp_scenarios:
        nc_path = forecast_run_dir / f"malaria_forecast_{ssp}_{dah_scenario}.nc"
        if not nc_path.exists():
            click.echo(f"  [{ssp}] SKIP — no {nc_path.name}")
            continue

        click.echo(f"  [{ssp}] aggregating {nc_path.name}")
        with xr.open_dataset(nc_path) as ds:
            years = [int(y) for y in ds.year_id.values]
        population = load_population(
            sorted(set(years) | {anchor_year}), hold_year=population_hold_year
        )

        as_factors = None
        if anchor_vintage is not None:
            a2v = hierarchy_df.loc[
                hierarchy_df.level == ADMIN2_LEVEL, "location_id"
            ].tolist()
            as_factors = {
                m: anchor_vintage_factor(m, anchor_year, a2v, years, anchor_vintage)
                for m in PRED_VARS
            }
        if age_structure_hold_year is not None:
            a2 = hierarchy_df.loc[
                hierarchy_df.level == ADMIN2_LEVEL, "location_id"
            ].tolist()
            as_factors = {}
            for measure in PRED_VARS:
                as_factors[measure] = age_structure_hold_factor(
                    measure, age_structure_hold_year, a2, years
                )
                f = as_factors[measure]["factor"]
                click.echo(
                    f"  [{ssp}] {measure} age-structure factor: "
                    f"min {f.min():.4f} max {f.max():.4f} mean {f.mean():.4f}"
                )

        summary = build_all_age_summary(
            nc_path, hierarchy_df, population, level_ids, quantiles,
            as_factors=as_factors,
        )
        summary_path = out_dir / f"all_age_summary_{ssp}_{dah_scenario}.parquet"
        write_parquet(summary, summary_path, validate=True)
        click.echo(f"  [{ssp}] wrote {summary_path.name}  ({len(summary):,} rows)")

        check, verdict = build_anchor_check(summary, anchor_year, rel_tol)
        check_path = out_dir / f"anchor_{anchor_year}_{ssp}_{dah_scenario}.parquet"
        write_parquet(check, check_path, validate=True)
        verdicts[ssp] = verdict
        flag = "PASS" if verdict["passed"] else "FAIL"
        click.echo(
            f"  [{ssp}] anchor {anchor_year}: {flag} "
            f"({verdict['n_failed']}/{verdict['n_compared']} outside tol, "
            f"max |rel diff| = {verdict['max_abs_rel_diff']})"
        )
        wrote_any = True

    if not wrote_any:
        raise click.ClickException(
            f"no forecast netCDFs found in {forecast_run_dir} for {ssp_scenarios}"
        )

    (out_dir / f"anchor_{anchor_year}_verdict.json").write_text(
        json.dumps(verdicts, indent=2) + "\n"
    )

    if set_current and node_root is not None:
        finalize_artifact(node_root, out_dir.name)

    # The anchor gap is REPORTED, not fatal. At aggregate levels it cannot be zero by
    # construction: locations the forecast drops are excluded from the sum while the
    # full population stays in the denominator, which costs ~0.1% of global incidence.
    # That is a property of the zero-burden policy, so failing the run on it would
    # mean every run fails for a known reason. --strict-anchor restores the old
    # behaviour for when the check is being used to hunt an actual aggregation bug.
    failed = [s for s, v in verdicts.items() if not v["passed"]]
    if failed:
        worst = max(
            (v["max_abs_rel_diff"] or 0.0) for v in verdicts.values()
        )
        click.echo(
            f"\nanchor {anchor_year}: {len(failed)} scenario(s) outside tol "
            f"({', '.join(failed)}); max |rel diff| = {worst:.4g}. See "
            f"anchor_{anchor_year}_verdict.json.",
            err=True,
        )
        if strict_anchor:
            raise SystemExit(1)
    else:
        click.echo("\nall scenarios finished and anchor-matched.")


if __name__ == "__main__":
    main()
