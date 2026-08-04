"""
Hybrid SDG malaria incidence-rate deliverable (one-off, 2026-06-03).

Age-sex incidence RATE per 1,000 population, per country, mean over draws, one SSP,
raked to the goalkeepers/SDG GBD2025 malaria incidence at the 2025 overlap year.

CORE PRINCIPLE (the only sane one): counts come from our malaria (non-zero) locations;
every RATE -- at the rake step and at the end -- is count / the ACTUAL FULL population
of that location (the SDG/goalkeepers population, including non-malarial places whose
people belong in the denominator with zero count). Nothing is divided by a partial
"covered" population.

Outputs
-------
1. Deliverable parquet [location_id (country), year_id, sex_id, age_group_id,
   rate_per_thousand] -- every admin-0 country, 2025-2045 (0 where we don't forecast).
2. Plotdata parquet [location_id (country), year_id, age_group_id, sex_id, inc_count,
   population, period] -- past (MAL) + future (forecast), country counts + FULL country
   population, for the plotting notebook to roll up to region/super-region/global.

Pipeline (everything in COUNT space; rates only as count/full-pop):
  1. rocket forecast -> admin-2 all-age incidence rate (mean over draws)
  2. admin-2 all-age COUNT = rate x our admin-2 population (lsae_1285); sum -> FHS-loc count
  3. MAL all-age COUNT @2025 at FHS-loc = sum_as(MAL_rate_as x SDG_fhs_pop_as)
  4. rake: factor[fhs] = MAL_count_2025 / our_count_2025 (held forward); raked FHS count
  5. disaggregate raked count to age-sex with MAL's age structure + SDG FHS population
  6. aggregate FHS age-sex counts -> country; rate = count / FULL SDG country pop * 1000

Populations: our admin-2 pop builds the malaria counts only; the SDG/goalkeepers
population is the denominator for every rate and the FHS-loc disaggregation weight.
External SDG files are one-off inputs (hardcoded by design -- not our artifacts);
all of OUR inputs resolve through versioned *_READ_PATH.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.processing.disaggregation import disaggregate_age_sex_malaria

# ---- external (FHS / goalkeepers) one-off inputs ----
SDG_POP_PAST_PATH = "/mnt/share/forecasting/data/37/past/population/20260518_goalkeepers2026_run_id_444/population.nc"
SDG_POP_FUTURE_PATH = "/mnt/share/forecasting/data/37/future/population/20260518_goalkeepers2026_shifted_run_id_444/summary_agg/population.nc"
SDG_MAL_PATH = "/mnt/share/forecasting/data/37/past/incidence/20260521_malaria_incidence_goalkeepers/malaria.nc"
SDG_FUTURE_SCENARIO = 130

AGES_25 = [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
           30, 31, 32, 34, 235, 238, 388, 389]   # our most-detailed set (== MAL)
SEXES = [1, 2]
RAKE_YEAR = 2025
KEYS = ["location_id", "year_id", "age_group_id", "sex_id"]


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def _forecast_inc_rate(ssp_scenario: str, years: list[int],
                       forecast_run_date: str | None = None) -> pd.DataFrame:
    """Rocket forecast -> mean-over-draws admin-2 all-age incidence RATE.

    ``forecast_run_date`` selects the forecast_outputs subdir (a model key such as
    ``2026_07_14_hybrid``); ``None`` reads the ``current`` symlink (default)."""
    run = forecast_run_date or "current"
    path = (mbpc.MODEL_ROOT / "04-forecasting_data" / "malaria" / "forecast_outputs"
            / mbpc.LSAE_HIERARCHY / run / f"malaria_forecast_{ssp_scenario}_Baseline.nc")
    ds = xr.open_dataset(path)
    yrs = [y for y in years if y in ds.year_id.values]
    da = np.exp(ds["log_malaria_inc_rate_pred"].sel(year_id=yrs)).mean(dim="draw")
    df = da.to_dataframe(name="inc_rate").reset_index()[["location_id", "year_id", "inc_rate"]]
    ds.close()
    return df.dropna(subset=["inc_rate"])


def _our_admin2_aa_pop(location_ids, years: list[int]) -> pd.DataFrame:
    """lsae_1285 versioned all-age admin-2 population (used to build malaria counts)."""
    ap = read_parquet_with_integer_ids(
        mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
        columns=["location_id", "year_id", "population"],
        filters=[("location_id", "in", list(location_ids)),
                 ("year_id", "in", list(years))],
    )
    return (ap.rename(columns={"population": "aa_population"})
            .drop_duplicates(["location_id", "year_id"]))


def _sdg_fhs(path: str, var: str, years, value_name: str, extra=None) -> pd.DataFrame:
    """SDG variable at its native (FHS most-detailed) locations, age-sex."""
    sel = dict(age_group_id=AGES_25, sex_id=SEXES, year_id=list(years))
    if extra:
        sel.update(extra)
    with xr.open_dataset(path) as d:
        df = d[var].sel(**sel).to_dataframe(name=value_name).reset_index()
    return (df.rename(columns={"location_id": "fhs_location_id"})
            [["fhs_location_id", "year_id", "age_group_id", "sex_id", value_name]])


def _full_country_pop(path: str, var: str, years, fhs_md: pd.DataFrame,
                      value_name: str = "population", extra=None) -> pd.DataFrame:
    """FULL country population = sum of SDG pop over ALL most-detailed-FHS locations of
    each country. The admin-0/nation row is ABSENT from the SDG files for subnational
    countries (Brazil, India, ...), so we aggregate the pieces rather than read the
    nation directly (reading it gave missing pop -> rate 0 for every subnational country)."""
    p = _sdg_fhs(path, var, years, value_name, extra)
    p = p.merge(fhs_md, on="fhs_location_id", how="inner")
    return (p.groupby(["A0_location_id", "year_id", "age_group_id", "sex_id"], as_index=False)
            [value_name].sum().rename(columns={"A0_location_id": "location_id"}))


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def _disaggregate_fhs(raked_fhs_count: pd.DataFrame, rr: pd.DataFrame,
                      sdg_fhs_pop: pd.DataFrame) -> pd.DataFrame:
    """Split all-age FHS counts to age-sex via MAL structure + SDG FHS pop (canonical fn)."""
    grid = (sdg_fhs_pop.merge(rr, on=["fhs_location_id", "age_group_id", "sex_id"], how="left")
            .merge(raked_fhs_count, on=["fhs_location_id", "year_id"], how="inner")
            .rename(columns={"fhs_location_id": "location_id",
                             "population": "population",
                             "raked_count": "aa_malaria_inc_count"}))
    grid["aa_malaria_mort_count"] = grid["aa_malaria_inc_count"]   # dummy; keep only inc
    grid["rr_mort_as"] = grid["rr_inc_as"]
    out = disaggregate_age_sex_malaria(grid)
    return (out.rename(columns={"location_id": "fhs_location_id"})
            [["fhs_location_id", "year_id", "age_group_id", "sex_id", "malaria_inc_count_pred"]]
            .rename(columns={"malaria_inc_count_pred": "inc_count"}))


def main(ssp_scenario: str = "ssp245", output_dir=None, output_name: str = "malaria.parquet",
         year_start: int = 2025, year_end: int = 2045, past_start: int = 2010,
         forecast_min: int = 2023, plotdata_dir=None, forecast_run_date=None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = (mbpc.MODEL_ROOT / "04-forecasting_data" / "malaria"
                      / "hybrid_deliverable" / mbpc.LSAE_HIERARCHY / mbpc.RUN_DATE)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if plotdata_dir is None:
        plotdata_dir = (mbpc.MODEL_ROOT / "04-forecasting_data" / "malaria"
                        / "hybrid_deliverable" / mbpc.LSAE_HIERARCHY / mbpc.RUN_DATE)
    plotdata_dir = Path(plotdata_dir); plotdata_dir.mkdir(parents=True, exist_ok=True)

    # Our raked forecast is computed back to forecast_min (2023) so the plotdata can show our
    # 2023/2024 estimates under the SAME rake-to-2025 machinery (the "adjusted" past). The
    # DELIVERABLE stays year_start..year_end (2025-2045). MAL observed past is past_start..2024.
    calc_years = list(range(forecast_min, year_end + 1))   # raked forecast: 2023-2045
    deliv_years = list(range(year_start, year_end + 1))    # deliverable: 2025-2045
    past_years = list(range(past_start, RAKE_YEAR))        # MAL observed: 2010-2024

    h = read_parquet_with_integer_ids(
        mbpc.HIERARCHY_READ_PATH / f"full_hierarchy_2023_{mbpc.LSAE_HIERARCHY}.parquet")
    all_a0 = sorted(int(x) for x in h.loc[h["level"] == 3, "location_id"].unique())
    fhs_md = (h[h["most_detailed_fhs"] == 1][["location_id", "A0_location_id"]]
              .drop_duplicates().rename(columns={"location_id": "fhs_location_id"}))

    # --- 1-2. forecast -> our admin-2 malaria COUNT -> FHS-loc count ---
    fc = _forecast_inc_rate(ssp_scenario, calc_years, forecast_run_date)
    a2_ids = fc["location_id"].unique().tolist()
    xwalk = (h[h["location_id"].isin(a2_ids)][["location_id", "fhs_location_id", "A0_location_id"]]
             .drop_duplicates("location_id"))
    fc = fc.merge(xwalk, on="location_id", how="left")
    if fc["fhs_location_id"].isna().any():
        raise ValueError("forecast admin-2 with no fhs_location_id")
    aapop = _our_admin2_aa_pop(a2_ids, calc_years)
    fc = fc.merge(aapop, on=["location_id", "year_id"], how="left")
    if fc["aa_population"].isna().any():
        bad = fc.loc[fc["aa_population"].isna(), "location_id"].unique()[:5]
        raise ValueError(f"admin-2 missing population (stale pop vintage?), e.g. {bad}")
    fc["inc_count"] = fc["inc_rate"] * fc["aa_population"]
    our_fhs = (fc.groupby(["fhs_location_id", "year_id"], as_index=False)
               .agg(our_count=("inc_count", "sum")))
    fhs_ids = sorted(int(x) for x in our_fhs["fhs_location_id"].unique())
    fhs_a0 = (h[h["location_id"].isin(fhs_ids)][["location_id", "A0_location_id"]]
              .drop_duplicates().rename(columns={"location_id": "fhs_location_id"}))

    # --- SDG FHS population over ALL needed years (full population of each FHS loc) ---
    sdg_fhs_pop = pd.concat([
        _sdg_fhs(SDG_POP_PAST_PATH, "population", [y for y in calc_years if y <= RAKE_YEAR], "population"),
        _sdg_fhs(SDG_POP_FUTURE_PATH, "value", [y for y in calc_years if y > RAKE_YEAR],
                 "population", extra=dict(statistic="mean", scenario=SDG_FUTURE_SCENARIO)),
    ], ignore_index=True)
    sdg_fhs_pop = sdg_fhs_pop[sdg_fhs_pop["fhs_location_id"].isin(fhs_ids)]

    # --- 3. MAL @2025 rate + 2025 age structure at FHS-loc ---
    mal = _sdg_fhs(SDG_MAL_PATH, "point_estimate", [RAKE_YEAR], "mal_rate")
    mal = mal[mal["fhs_location_id"].isin(fhs_ids)]
    pop25 = sdg_fhs_pop[sdg_fhs_pop["year_id"] == RAKE_YEAR]
    mal = mal.merge(pop25[["fhs_location_id", "age_group_id", "sex_id", "population"]],
                    on=["fhs_location_id", "age_group_id", "sex_id"], how="left")
    mal_count = (mal.assign(c=mal["mal_rate"] * mal["population"])
                 .groupby("fhs_location_id", as_index=False).agg(mal_count=("c", "sum"),
                                                                 pop=("population", "sum")))
    mal_count["mal_allage_rate"] = mal_count["mal_count"] / mal_count["pop"]
    mal = mal.merge(mal_count[["fhs_location_id", "mal_allage_rate"]], on="fhs_location_id", how="left")
    mal["rr_inc_as"] = mal["mal_rate"] / mal["mal_allage_rate"]
    rr = mal[["fhs_location_id", "age_group_id", "sex_id", "rr_inc_as"]]

    # --- 4. rake (count space): factor = MAL_count / our_count @2025, held forward ---
    base = (our_fhs[our_fhs["year_id"] == RAKE_YEAR][["fhs_location_id", "our_count"]]
            .merge(mal_count[["fhs_location_id", "mal_count"]], on="fhs_location_id", how="inner"))
    base["factor"] = base["mal_count"] / base["our_count"]
    our_fhs = our_fhs.merge(base[["fhs_location_id", "factor"]], on="fhs_location_id", how="inner")
    our_fhs["raked_count"] = our_fhs["our_count"] * our_fhs["factor"]

    # --- 5. disaggregate future raked counts to age-sex (MAL structure, SDG FHS pop) ---
    future_fhs_as = _disaggregate_fhs(
        our_fhs[["fhs_location_id", "year_id", "raked_count"]], rr, sdg_fhs_pop)

    # --- 6. aggregate FHS age-sex counts -> country; FULL country population denominator ---
    def _to_country(fhs_as: pd.DataFrame) -> pd.DataFrame:
        return (fhs_as.merge(fhs_a0, on="fhs_location_id", how="left")
                .groupby(["A0_location_id", "year_id", "age_group_id", "sex_id"], as_index=False)
                .agg(inc_count=("inc_count", "sum"))
                .rename(columns={"A0_location_id": "location_id"}))

    future_country = _to_country(future_fhs_as)

    full_a0_future = pd.concat([
        _full_country_pop(SDG_POP_PAST_PATH, "population", [y for y in calc_years if y <= RAKE_YEAR], fhs_md),
        _full_country_pop(SDG_POP_FUTURE_PATH, "value", [y for y in calc_years if y > RAKE_YEAR],
                          fhs_md, extra=dict(statistic="mean", scenario=SDG_FUTURE_SCENARIO)),
    ], ignore_index=True)

    # deliverable: every admin-0, year_start..year_end (2025-2045), rate = count / FULL country pop * 1000
    full_grid = pd.MultiIndex.from_product(
        [all_a0, deliv_years, AGES_25, SEXES], names=KEYS).to_frame(index=False)
    out = (full_grid.merge(future_country, on=KEYS, how="left")
           .merge(full_a0_future, on=KEYS, how="left"))
    out["inc_count"] = out["inc_count"].fillna(0.0)
    out["rate_per_thousand"] = np.where(out["population"] > 0,
                                        out["inc_count"] / out["population"] * 1000.0, 0.0)
    deliverable = (out[KEYS + ["rate_per_thousand"]]
                   .astype({k: "int64" for k in KEYS} | {"rate_per_thousand": "float32"})
                   .sort_values(KEYS).reset_index(drop=True))
    out_path = output_dir / output_name
    write_parquet(deliverable, out_path)
    print(f"[deliverable] {len(deliverable):,} rows -> {out_path}")

    # --- plotdata: past (MAL) + future, country counts + FULL country population ---
    # past: MAL age-sex rate (per year) x SDG FHS pop (per year) -> FHS count -> country
    mal_past = _sdg_fhs(SDG_MAL_PATH, "point_estimate", past_years, "mal_rate")
    mal_past = mal_past[mal_past["fhs_location_id"].isin(fhs_ids)]
    pop_past_fhs = _sdg_fhs(SDG_POP_PAST_PATH, "population", past_years, "population")
    past_fhs_as = mal_past.merge(
        pop_past_fhs, on=["fhs_location_id", "year_id", "age_group_id", "sex_id"], how="left")
    past_fhs_as["inc_count"] = past_fhs_as["mal_rate"] * past_fhs_as["population"]
    past_country = _to_country(past_fhs_as[["fhs_location_id", "year_id", "age_group_id",
                                            "sex_id", "inc_count"]])
    full_a0_past = _full_country_pop(SDG_POP_PAST_PATH, "population", past_years, fhs_md)

    def _plot_block(country_cnt, full_pop, years, period):
        g = pd.MultiIndex.from_product([all_a0, years, AGES_25, SEXES], names=KEYS).to_frame(index=False)
        b = (g.merge(country_cnt, on=KEYS, how="left").merge(full_pop, on=KEYS, how="left"))
        b["inc_count"] = b["inc_count"].fillna(0.0)
        b["period"] = period
        return b[KEYS + ["inc_count", "population", "period"]]

    # future block covers calc_years (2023-2045) so the notebook can show our raked
    # 2023/2024 estimates ("adjusted past") alongside the MAL observed past.
    plotdata = pd.concat([
        _plot_block(past_country, full_a0_past, past_years, "past"),
        _plot_block(future_country, full_a0_future, calc_years, "future"),
    ], ignore_index=True)
    plotdata = (plotdata.astype({k: "int64" for k in KEYS}
                                | {"inc_count": "float64", "population": "float64"})
                .sort_values(["period"] + KEYS).reset_index(drop=True))
    plot_path = plotdata_dir / f"plotdata_{ssp_scenario}.parquet"
    write_parquet(plotdata, plot_path)
    print(f"[plotdata]    {len(plotdata):,} rows -> {plot_path}")
    return deliverable


# =========================================================================== #
# SENSITIVITY: same machinery, but anchor (rake) to 2023 / 2024 / 2025, plus a
# NO-RAKE variant (our forecast's own level, age-split by MAL's 2023 fractions,
# no leveling). One combined plotdata with a `variant` column for overlaying.
# Everything per FULL SDG population, so the variants are directly comparable.
# =========================================================================== #
def _sens_to_country(fhs_as, fhs_a0):
    return (fhs_as.merge(fhs_a0, on="fhs_location_id", how="left")
            .groupby(["A0_location_id", "year_id", "age_group_id", "sex_id"], as_index=False)
            .agg(inc_count=("inc_count", "sum")).rename(columns={"A0_location_id": "location_id"}))


def _sens_block(country_cnt, full_pop, years, variant, all_a0):
    g = pd.MultiIndex.from_product([all_a0, years, AGES_25, SEXES], names=KEYS).to_frame(index=False)
    b = g.merge(country_cnt, on=KEYS, how="left").merge(full_pop, on=KEYS, how="left")
    b["inc_count"] = b["inc_count"].fillna(0.0)
    b["variant"] = variant
    return b[KEYS + ["inc_count", "population", "variant"]]


def _mal_year(year, sdg_fhs_pop, fhs_ids):
    """MAL all-age count (by FHS-loc) and age structure rr (by FHS-loc, age, sex) at `year`."""
    mal = _sdg_fhs(SDG_MAL_PATH, "point_estimate", [year], "mal_rate")
    mal = mal[mal["fhs_location_id"].isin(fhs_ids)]
    pop = sdg_fhs_pop[sdg_fhs_pop["year_id"] == year]
    mal = mal.merge(pop[["fhs_location_id", "age_group_id", "sex_id", "population"]],
                    on=["fhs_location_id", "age_group_id", "sex_id"], how="left")
    mc = (mal.assign(c=mal["mal_rate"] * mal["population"])
          .groupby("fhs_location_id", as_index=False).agg(mal_count=("c", "sum"), pop=("population", "sum")))
    mc["allage"] = mc["mal_count"] / mc["pop"]
    mal = mal.merge(mc[["fhs_location_id", "allage"]], on="fhs_location_id", how="left")
    mal["rr_inc_as"] = mal["mal_rate"] / mal["allage"]
    return mc[["fhs_location_id", "mal_count"]], mal[["fhs_location_id", "age_group_id", "sex_id", "rr_inc_as"]]


def _our_rr(year, h, fhs_ids, a2):
    """Our OWN observed age/sex structure (rate ratios) at `year`, from the raked-AS data,
    aggregated to FHS-loc. Lets us disaggregate the forecast with OUR age pattern, not MAL's.
    `a2` = most-detailed admin-2 set; the raked-AS file also carries parent rows we must drop."""
    a = read_parquet_with_integer_ids(
        mbpc.MAL_RAKED_AS_READ_PATH / "as_full_malaria_df.parquet",
        columns=["location_id", "year_id", "age_group_id", "sex_id", "malaria_inc_count", "population"],
        filters=[("year_id", "==", year)])
    a = a[a["location_id"].isin(a2)]
    a = a.merge(h[["location_id", "fhs_location_id"]].drop_duplicates(), on="location_id", how="left")
    a = a[a["fhs_location_id"].isin(fhs_ids)]
    g = a.groupby(["fhs_location_id", "age_group_id", "sex_id"], as_index=False).agg(
        cnt=("malaria_inc_count", "sum"), pop=("population", "sum"))
    g["as_rate"] = np.where(g["pop"] > 0, g["cnt"] / g["pop"], 0.0)
    aa = g.groupby("fhs_location_id", as_index=False).agg(aacnt=("cnt", "sum"), aapop=("pop", "sum"))
    aa["aa_rate"] = np.where(aa["aapop"] > 0, aa["aacnt"] / aa["aapop"], 0.0)
    g = g.merge(aa[["fhs_location_id", "aa_rate"]], on="fhs_location_id", how="left")
    g["rr_inc_as"] = np.where(g["aa_rate"] > 0, g["as_rate"] / g["aa_rate"], 0.0)
    return g[["fhs_location_id", "age_group_id", "sex_id", "rr_inc_as"]]


def _our_model_past(model_past_years, a2, h, all_a0, full_a0_past):
    """Our pipeline's OBSERVED age-sex malaria, aggregated to country (rate per full SDG pop).
    These years are real data (raked-AS), so the 'model' line has an observed past, not a stub.
    `a2` = most-detailed admin-2 set; the raked-AS file also carries parent rows we must drop."""
    a = read_parquet_with_integer_ids(
        mbpc.MAL_RAKED_AS_READ_PATH / "as_full_malaria_df.parquet",
        columns=["location_id", "year_id", "age_group_id", "sex_id", "malaria_inc_count"],
        filters=[("year_id", "in", list(model_past_years))])
    a = a[a["location_id"].isin(a2)]
    a = a.merge(h[["location_id", "A0_location_id"]].drop_duplicates(), on="location_id", how="left")
    cnt = (a.groupby(["A0_location_id", "year_id", "age_group_id", "sex_id"], as_index=False)
           .agg(inc_count=("malaria_inc_count", "sum")).rename(columns={"A0_location_id": "location_id"}))
    return _sens_block(cnt, full_a0_past, list(model_past_years), "model", all_a0)


def sensitivity(ssp_scenario="ssp245", rake_years=(2023, 2024, 2025), norake_rr_year=2023,
                year_end=2045, past_start=2010, forecast_min=2023, output_dir=None,
                forecast_run_date=None, old_delivered_plotdata=None):
    if output_dir is None:
        output_dir = (mbpc.MODEL_ROOT / "04-forecasting_data" / "malaria"
                      / "hybrid_deliverable" / mbpc.LSAE_HIERARCHY / mbpc.RUN_DATE)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    calc_years = list(range(forecast_min, year_end + 1))
    past_years = list(range(past_start, RAKE_YEAR + 1))   # MAL observed incl 2025

    h = read_parquet_with_integer_ids(
        mbpc.HIERARCHY_READ_PATH / f"full_hierarchy_2023_{mbpc.LSAE_HIERARCHY}.parquet")
    all_a0 = sorted(int(x) for x in h.loc[h["level"] == 3, "location_id"].unique())
    fhs_md = (h[h["most_detailed_fhs"] == 1][["location_id", "A0_location_id"]]
              .drop_duplicates().rename(columns={"location_id": "fhs_location_id"}))

    # our forecast -> FHS-loc all-age counts (shared across all variants)
    fc = _forecast_inc_rate(ssp_scenario, calc_years, forecast_run_date)
    a2 = fc["location_id"].unique().tolist()
    xw = (h[h["location_id"].isin(a2)][["location_id", "fhs_location_id", "A0_location_id"]]
          .drop_duplicates("location_id"))
    fc = fc.merge(xw, on="location_id", how="left").merge(
        _our_admin2_aa_pop(a2, calc_years), on=["location_id", "year_id"], how="left")
    fc["inc_count"] = fc["inc_rate"] * fc["aa_population"]
    our_fhs = fc.groupby(["fhs_location_id", "year_id"], as_index=False).agg(our_count=("inc_count", "sum"))
    fhs_ids = sorted(int(x) for x in our_fhs["fhs_location_id"].unique())
    fhs_a0 = (h[h["location_id"].isin(fhs_ids)][["location_id", "A0_location_id"]]
              .drop_duplicates().rename(columns={"location_id": "fhs_location_id"}))

    sdg_fhs_pop = pd.concat([
        _sdg_fhs(SDG_POP_PAST_PATH, "population", [y for y in calc_years if y <= RAKE_YEAR], "population"),
        _sdg_fhs(SDG_POP_FUTURE_PATH, "value", [y for y in calc_years if y > RAKE_YEAR], "population",
                 extra=dict(statistic="mean", scenario=SDG_FUTURE_SCENARIO)),
    ], ignore_index=True)
    sdg_fhs_pop = sdg_fhs_pop[sdg_fhs_pop["fhs_location_id"].isin(fhs_ids)]
    full_a0_future = pd.concat([
        _full_country_pop(SDG_POP_PAST_PATH, "population", [y for y in calc_years if y <= RAKE_YEAR], fhs_md),
        _full_country_pop(SDG_POP_FUTURE_PATH, "value", [y for y in calc_years if y > RAKE_YEAR], fhs_md,
                          extra=dict(statistic="mean", scenario=SDG_FUTURE_SCENARIO)),
    ], ignore_index=True)
    full_a0_past = _full_country_pop(SDG_POP_PAST_PATH, "population", past_years, fhs_md)

    # observed past (MAL)
    mp = _sdg_fhs(SDG_MAL_PATH, "point_estimate", past_years, "mal_rate")
    mp = mp[mp["fhs_location_id"].isin(fhs_ids)].merge(
        _sdg_fhs(SDG_POP_PAST_PATH, "population", past_years, "population"),
        on=["fhs_location_id", "year_id", "age_group_id", "sex_id"], how="left")
    mp["inc_count"] = mp["mal_rate"] * mp["population"]
    past_country = _sens_to_country(mp[["fhs_location_id", "year_id", "age_group_id", "sex_id", "inc_count"]], fhs_a0)

    blocks = [_sens_block(past_country, full_a0_past, past_years, "observed", all_a0)]
    # rake-to-Y variants (level + age structure both from MAL year Y)
    for Y in rake_years:
        mc, rr = _mal_year(Y, sdg_fhs_pop, fhs_ids)
        base = (our_fhs[our_fhs["year_id"] == Y][["fhs_location_id", "our_count"]]
                .merge(mc, on="fhs_location_id", how="inner"))
        base["factor"] = base["mal_count"] / base["our_count"]
        of = our_fhs.merge(base[["fhs_location_id", "factor"]], on="fhs_location_id", how="inner")
        of["raked_count"] = of["our_count"] * of["factor"]
        fas = _disaggregate_fhs(of[["fhs_location_id", "year_id", "raked_count"]], rr, sdg_fhs_pop)
        blocks.append(_sens_block(_sens_to_country(fas, fhs_a0), full_a0_future, calc_years, f"rake{Y}", all_a0))
    # NO-RAKE variant: our own forecast level (factor = 1), age structure from MAL norake_rr_year
    _, rr_nr = _mal_year(norake_rr_year, sdg_fhs_pop, fhs_ids)
    of = our_fhs.rename(columns={"our_count": "raked_count"})
    fas = _disaggregate_fhs(of[["fhs_location_id", "year_id", "raked_count"]], rr_nr, sdg_fhs_pop)
    blocks.append(_sens_block(_sens_to_country(fas, fhs_a0), full_a0_future, calc_years, f"norake_rr{norake_rr_year}", all_a0))

    # MODEL variant: our pipeline's own estimate end-to-end. OBSERVED raked-AS past
    # (past_start .. forecast_min-1) spliced with our forecast (factor = 1) age-split by OUR
    # OWN norake_rr_year structure. Past is real data; everything per full SDG pop.
    blocks.append(_our_model_past(range(past_start, forecast_min), a2, h, all_a0, full_a0_past))
    own_rr = _our_rr(norake_rr_year, h, fhs_ids, a2)
    fas_m = _disaggregate_fhs(of[["fhs_location_id", "year_id", "raked_count"]], own_rr, sdg_fhs_pop)
    blocks.append(_sens_block(_sens_to_country(fas_m, fhs_a0), full_a0_future, calc_years, "model", all_a0))

    # OLD_DELIVERED variant: the previously-delivered forecast, read from its own
    # plotdata (future block). Already in count + FULL-SDG-pop space per country/
    # year/age/sex, so it overlays directly against the variants above. Defaults to
    # the prior delivery at hybrid_deliverable/<LSAE>/<RUN_DATE>/plotdata_<ssp>.parquet.
    if old_delivered_plotdata is None:
        old_delivered_plotdata = (mbpc.MODEL_ROOT / "04-forecasting_data" / "malaria"
                                  / "hybrid_deliverable" / mbpc.LSAE_HIERARCHY / mbpc.RUN_DATE
                                  / f"plotdata_{ssp_scenario}.parquet")
    old_delivered_plotdata = Path(old_delivered_plotdata)
    if old_delivered_plotdata.exists():
        od = read_parquet_with_integer_ids(
            old_delivered_plotdata, columns=KEYS + ["inc_count", "population", "period"])
        od = od[od["period"] == "future"][KEYS + ["inc_count", "population"]].copy()
        od["variant"] = "old_delivered"
        blocks.append(od[KEYS + ["inc_count", "population", "variant"]])
        print(f"[sensitivity] old_delivered line from {old_delivered_plotdata}")
    else:
        print(f"[sensitivity] WARNING: old_delivered plotdata not found, skipping "
              f"old_delivered line: {old_delivered_plotdata}")

    sens = pd.concat(blocks, ignore_index=True).astype(
        {k: "int64" for k in KEYS} | {"inc_count": "float64", "population": "float64"})
    out = output_dir / f"sensitivity_plotdata_{ssp_scenario}.parquet"
    write_parquet(sens, out)
    print(f"[sensitivity] {len(sens):,} rows | variants={sorted(sens.variant.unique())} -> {out}")
    return sens


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hybrid SDG malaria age-sex incidence rate per 1,000 by country")
    p.add_argument("--ssp_scenario", default="ssp245")
    p.add_argument("--output_dir", default=None,
                   help="default: project staging; pass the /ihme dir for the real deliverable")
    p.add_argument("--output_name", default="malaria.parquet")
    p.add_argument("--plotdata_dir", default=None, help="default: project staging")
    p.add_argument("--year_start", type=int, default=2025)
    p.add_argument("--year_end", type=int, default=2045)
    p.add_argument("--past_start", type=int, default=2010)
    p.add_argument("--forecast_min", type=int, default=2023,
                   help="earliest year of our raked forecast (for the plotdata 2023/24 adjustment)")
    p.add_argument("--mode", choices=["deliverable", "sensitivity"], default="deliverable")
    p.add_argument("--rake_years", type=int, nargs="+", default=[2023, 2024, 2025],
                   help="sensitivity: single-year pins to compare (each a SEPARATE single-year pin)")
    p.add_argument("--norake_rr_year", type=int, default=2023,
                   help="sensitivity: MAL year used for age/sex fractions in the no-rake variant")
    p.add_argument("--forecast_run_date", default=None,
                   help="forecast_outputs subdir / model key to read (e.g. 2026_07_14_hybrid); "
                        "default reads the `current` symlink")
    p.add_argument("--old_delivered_plotdata", default=None,
                   help="sensitivity: path to the previously-delivered plotdata_<ssp>.parquet for the "
                        "'old_delivered' overlay line; default = hybrid_deliverable/<LSAE>/<RUN_DATE>/")
    a = p.parse_args()
    if a.mode == "sensitivity":
        sensitivity(ssp_scenario=a.ssp_scenario, rake_years=tuple(a.rake_years),
                    norake_rr_year=a.norake_rr_year, year_end=a.year_end,
                    past_start=a.past_start, forecast_min=a.forecast_min, output_dir=a.output_dir,
                    forecast_run_date=a.forecast_run_date,
                    old_delivered_plotdata=a.old_delivered_plotdata)
    else:
        main(ssp_scenario=a.ssp_scenario, output_dir=a.output_dir, output_name=a.output_name,
             year_start=a.year_start, year_end=a.year_end, past_start=a.past_start,
             forecast_min=a.forecast_min, plotdata_dir=a.plotdata_dir,
             forecast_run_date=a.forecast_run_date)
