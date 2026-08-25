"""
Post-disaggregation vaccine impact for malaria.

Turns the cohort protection table into scenario totals. The forecast is age-less
`(location, year, draw)`; age enters via `lib/processing/disaggregation.py`'s
as_rr -> fractions, and protection is applied as a burden-weighted collapse

    R(loc, yr) = SUM_as f(loc, yr, as) * protection(loc, yr, as)
    vaccine_count = novacc_count * (1 - R)

which is exact, not an approximation: the fractions sum to 1 within each
(location, year), so this equals applying protection cell-wise and
re-aggregating, without materialising the age-specific array.

Protection is reported at admin1 (the coverage geography) and broadcast to each
admin1's admin2 children -- also exact, since the fractions carry no population
weighting, so a child's value equals its parent's.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.processing.disaggregation import compute_as_rr, malaria_as_fractions

MODEL_RUN = "2026_07_31_full_model_selection_results"
DAH_SCENARIO = "Baseline"
SSP_SCENARIOS = ("ssp126", "ssp245", "ssp585")
ANCHOR_YEAR = 2023

# Super-region is level 1 of the LSAE hierarchy.
SUPER_REGION_LEVEL = 1


from contextlib import contextmanager


def _tick(clock, message: str) -> None:
    """Progress reporting is optional: lib code should not require a logger."""
    if clock is not None:
        clock(message)


@contextmanager
def _null_ctx(obj):
    """Use an already-open dataset without closing it on exit."""
    yield obj


def _read_hierarchy(columns: list[str]) -> pd.DataFrame:  # pragma: no cover - thin
    # I/O wrapper over a constant path; callers inject a frame in tests instead
    return read_parquet_with_integer_ids(
        rfc.HIERARCHY_READ_PATH / f"full_hierarchy_2023_{rfc.LSAE_HIERARCHY}.parquet",
        columns=columns,
    )


def eligible_locations(coverage_csv: Path, forecast_locations: set[int],
                       hierarchy: pd.DataFrame | None = None) -> pd.DataFrame:
    """[location_id (admin2), coverage_location_id (admin1)] for admin2s that sit
    under a coverage location AND appear in the forecast."""
    cov = pd.read_csv(coverage_csv, usecols=["subnat_id"])
    admin1 = {int(x) for x in cov["subnat_id"].unique()}
    hier = hierarchy if hierarchy is not None else _read_hierarchy(
        ["location_id", "path_to_top_parent"])
    rows = []
    for lid, path in zip(hier["location_id"], hier["path_to_top_parent"]):
        lid = int(lid)
        if lid not in forecast_locations:
            continue
        hit = {int(x) for x in str(path).split(",") if x} & admin1
        if hit:
            rows.append((lid, int(next(iter(hit)))))
    if not rows:
        raise ValueError("no forecast locations fall under any coverage location")
    return pd.DataFrame(rows, columns=["location_id", "coverage_location_id"])

def burden_weighted_reduction(protection: pd.DataFrame, mapping: pd.DataFrame,
                              years: list[int], clock=None,
                              as_malaria: pd.DataFrame | None = None,
                              as_population: pd.DataFrame | None = None) -> pd.DataFrame:
    """One (location_id, year_id) row per eligible admin2 with the draw-free
    reduction factors for incidence and mortality."""
    locs = mapping["location_id"].tolist()

    as_mal = as_malaria if as_malaria is not None else read_parquet_with_integer_ids(
        rfc.MAL_RAKED_AS_READ_PATH / "as_full_malaria_df.parquet",
        columns=["location_id", "year_id", "age_group_id", "sex_id", "malaria_inc_rate",
                 "aa_malaria_inc_rate", "malaria_mort_rate", "aa_malaria_mort_rate"],
        filters=[("year_id", "==", ANCHOR_YEAR), ("location_id", "in", locs)],
    )
    _tick(clock, f"as_rr source read: {len(as_mal):,} rows")
    rr = compute_as_rr(as_mal, anchor_year=ANCHOR_YEAR)

    as_pop = as_population if as_population is not None else read_parquet_with_integer_ids(
        rfc.POPULATION_READ_PATH / "as_2023_full_population_df.parquet",
        columns=["location_id", "year_id", "age_group_id", "sex_id", "population"],
        filters=[("location_id", "in", locs), ("year_id", "in", years)],
    )
    _tick(clock, f"as population read: {len(as_pop):,} rows")
    fracs = malaria_as_fractions(rr, as_pop)
    _tick(clock, f"as fractions: {len(fracs):,} rows")

    # protection is admin1; attach it to each admin2 child
    prot = protection.merge(mapping, left_on="location_id", right_on="coverage_location_id",
                            suffixes=("_a1", ""))
    prot = prot[["location_id", "year_id", "age_group_id", "sex_id",
                 "effective_protection_case", "effective_protection_death"]]

    merged = fracs.merge(prot, on=["location_id", "year_id", "age_group_id", "sex_id"], how="left")
    merged[["effective_protection_case", "effective_protection_death"]] = merged[
        ["effective_protection_case", "effective_protection_death"]].fillna(0.0)

    merged["r_inc"] = merged["inc_fraction"] * merged["effective_protection_case"]
    merged["r_mort"] = merged["mort_fraction"] * merged["effective_protection_death"]
    out = merged.groupby(["location_id", "year_id"], as_index=False)[["r_inc", "r_mort"]].sum()
    bad = out[(out.r_inc < 0) | (out.r_inc > 1) | (out.r_mort < 0) | (out.r_mort > 1)]
    if not bad.empty:
        raise ValueError(f"reduction factor outside [0,1] on {len(bad)} location-years")
    return out

def _aligned(df: pd.DataFrame, value_col: str, locs: list[int],
             years: list[int], fill: float | None) -> xr.DataArray:
    """(location_id, year_id) frame -> DataArray on the forecast's own grid.

    `fill` is used for cells the frame does not cover; None means a gap is an
    error rather than something to paper over.
    """
    wide = df.pivot_table(index="location_id", columns="year_id", values=value_col)
    wide = wide.reindex(index=locs, columns=years)
    if fill is None:
        if wide.isna().any().any():
            raise ValueError(
                f"{value_col} missing for {int(wide.isna().sum().sum())} location-years"
            )
    else:
        wide = wide.fillna(fill)
    return xr.DataArray(wide.to_numpy(), dims=("location_id", "year_id"),
                        coords={"location_id": locs, "year_id": years})

def scenario_totals(ssp: str, reduction: pd.DataFrame, aa_pop: pd.DataFrame,
                    locs: list[int], years: list[int], clock=None,
                    dataset: "xr.Dataset | None" = None) -> pd.DataFrame:
    """Totals over eligible locations per (year, draw) for both scenarios.

    Reads each variable CONTIGUOUSLY and subsets positionally in numpy. Measured
    on this file: a contiguous read of the whole 317 MB variable takes ~3.0s and
    the numpy subset ~0.4ms, whereas `.sel(location_id=[...])` on 60 scattered
    locations takes 36s and on 1,986 did not finish in 600s. HDF5 fancy-indexing
    pulls whole chunks, so naming fewer locations does not read less -- it just
    reads them badly. Do not "optimize" this back into a lazy .sel.
    """
    path = (rfc.MAL_FORECAST_OUTPUTS_READ_PATH
            / f"malaria_forecast_{ssp}_{DAH_SCENARIO}.nc")
    pop = _aligned(aa_pop, "population", locs, years, fill=None).to_numpy()

    frames = []
    with (xr.open_dataset(path) if dataset is None else _null_ctx(dataset)) as ds:
        loc_coord = ds["location_id"].to_numpy()
        year_coord = ds["year_id"].to_numpy()
        # Membership first: searchsorted on an absent value returns an index past
        # the end, which would raise IndexError before any check could fire.
        for name, coord, wanted in (("location_id", loc_coord, locs),
                                    ("year_id", year_coord, years)):
            absent = sorted(set(wanted) - set(coord.tolist()))
            if absent:
                raise ValueError(
                    f"{path}: forecast does not carry {name} {absent[:5]}"
                    f"{'...' if len(absent) > 5 else ''}")
            if not np.all(np.diff(coord) > 0):
                raise ValueError(
                    f"{path}: {name} coord is not sorted ascending; the positional "
                    "numpy subset would mis-attribute cells")
        lpos = np.searchsorted(loc_coord, locs)
        ypos = np.searchsorted(year_coord, years)

        for measure, var, rcol in (("incidence", "log_malaria_inc_rate_pred", "r_inc"),
                                   ("mortality", "log_malaria_mort_rate_pred", "r_mort")):
            dims = ds[var].dims
            arr = ds[var].to_numpy()
            _tick(clock, f"{ssp}/{measure}: contiguous read {arr.nbytes / 1e6:.0f} MB {arr.shape}")
            arr = np.transpose(arr, [dims.index(d) for d in ("location_id", "year_id", "draw")])
            counts = np.exp(arr[lpos][:, ypos, :]) * pop[:, :, None]
            del arr
            r = _aligned(reduction, rcol, locs, years, fill=0.0).to_numpy()
            tot_novacc = counts.sum(axis=0)
            tot_vacc = (counts * (1.0 - r[:, :, None])).sum(axis=0)
            del counts
            yy, dd = np.meshgrid(np.asarray(years), ds["draw"].to_numpy(), indexing="ij")
            frames.append(pd.DataFrame({
                "year_id": yy.ravel(),
                "draw": dd.ravel(),
                "count_novacc": tot_novacc.ravel(),
                "count_vacc": tot_vacc.ravel(),
                "measure": measure,
            }))
            _tick(clock, f"{ssp}/{measure}: aggregated")
    out = pd.concat(frames, ignore_index=True)
    out["ssp_scenario"] = ssp
    return out

def _with_cumulative(totals: pd.DataFrame) -> pd.DataFrame:
    """Add per-draw cumulative columns. Cumulation is PER DRAW, before any
    collapse across draws: the cumulative sum of a quantile is not the quantile
    of the cumulative sum, and cumulating published quantiles would understate
    the uncertainty because draw-level errors would not accumulate."""
    totals = totals.copy()
    totals["averted"] = totals["count_novacc"] - totals["count_vacc"]
    totals = totals.sort_values(["ssp_scenario", "measure", "draw", "year_id"])
    per_draw = totals.groupby(["ssp_scenario", "measure", "draw"])
    for col in ("count_novacc", "count_vacc", "averted"):
        totals[f"{col}_cum"] = per_draw[col].cumsum()
    return totals

def draw_level(totals: pd.DataFrame) -> pd.DataFrame:
    """Draw-level annual AND cumulative totals, unsummarized.

    Kept as draws so downstream figures can (a) show real distributions in box
    plots and (b) difference two runs DRAW-WISE. Draw indices are paired across
    runs -- same underlying forecast draw -- so a draw-wise difference is valid
    and gives a far tighter, correct interval; differencing two sets of
    summarized quantiles would not.
    """
    return _with_cumulative(totals).reset_index(drop=True)

def summarize(totals: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% UI across draws, for annual and cumulative series."""
    totals = _with_cumulative(totals)

    lo = lambda s: s.quantile(0.025)
    hi = lambda s: s.quantile(0.975)
    spec = {
        "novacc": "count_novacc", "vacc": "count_vacc", "averted": "averted",
        "novacc_cum": "count_novacc_cum", "vacc_cum": "count_vacc_cum",
        "averted_cum": "averted_cum",
    }
    agg = {}
    for name, col in spec.items():
        agg[f"{name}_mean"] = (col, "mean")
        agg[f"{name}_lo"] = (col, lo)
        agg[f"{name}_hi"] = (col, hi)
    return (totals.groupby(["ssp_scenario", "measure", "year_id"])
            .agg(**agg).reset_index())


def super_region_map(location_ids: list[int],
                     hierarchy: pd.DataFrame | None = None) -> pd.DataFrame:
    """[location_id, super_region_id, super_region_name] for the given locations.

    Super-region is level 1 of the hierarchy; a location's super-region is the
    level-1 entry on its path to the top parent.
    """
    hier = hierarchy if hierarchy is not None else _read_hierarchy(
        ["location_id", "path_to_top_parent", "level", "location_name"])
    supers = hier[hier["level"] == SUPER_REGION_LEVEL].set_index("location_id")["location_name"]
    wanted = set(location_ids)
    rows = []
    for lid, path in zip(hier["location_id"], hier["path_to_top_parent"]):
        lid = int(lid)
        if lid not in wanted:
            continue
        hit = [int(x) for x in str(path).split(",") if x and int(x) in supers.index]
        if hit:
            rows.append((lid, hit[0], supers.loc[hit[0]]))
    if not rows:
        raise ValueError("no super-region found for any requested location")
    return pd.DataFrame(rows, columns=["location_id", "super_region_id", "super_region_name"])


# Column each measure's disaggregated count arrives in, and the protection column
# that reduces it. Incidence is reduced by efficacy against clinical disease;
# mortality by efficacy against severe/fatal outcomes -- they are different curves.
PROTECTION_FOR_MEASURE = {
    "malaria_inc_count_pred": "effective_protection_case",
    "malaria_mort_count_pred": "effective_protection_death",
}


def apply_protection_to_age_sex(as_counts: pd.DataFrame, protection: pd.DataFrame,
                                columns: dict[str, str] | None = None) -> pd.DataFrame:
    """Reduce age/sex counts CELL-WISE by the vaccine protection for that cell.

    This is NOT the same as scaling all-age counts by the burden-weighted `R` and
    then disaggregating. That collapse is exact for TOTALS -- the fractions sum to
    1 -- but it spreads the reduction uniformly across ages, which is wrong for an
    age-specific product: it would show a reduction among 15-19 year olds who have
    no protection, and understate it among the under-fives who have most of it.
    Delivery products are age-specific, so the reduction has to be applied per cell.

    `protection` is [location_id, year_id, age_group_id, sex_id,
    effective_protection_case, effective_protection_death], at the SAME location
    grain as `as_counts` (broadcast admin1 -> admin2 first if needed). Cells with
    no protection row are left unreduced.
    """
    columns = columns or PROTECTION_FOR_MEASURE
    keys = ["location_id", "year_id", "age_group_id", "sex_id"]
    prot_cols = sorted(set(columns.values()))
    out = as_counts.merge(protection[keys + prot_cols], on=keys, how="left")
    out[prot_cols] = out[prot_cols].fillna(0.0)

    bad = out[(out[prot_cols] < 0).any(axis=1) | (out[prot_cols] > 1).any(axis=1)]
    if not bad.empty:
        raise ValueError(f"protection outside [0,1] on {len(bad)} cell(s)")

    for count_col, prot_col in columns.items():
        if count_col in out.columns:
            out[count_col] = out[count_col] * (1.0 - out[prot_col])
    return out.drop(columns=prot_cols)
