"""QUICK GBD2023 forecast time-series — compute + driver.

No re-fit, no re-forecast, no external/GBD2025 inputs. The rocket forecast is
already anchored to GBD2023 observed at the 2023 rake year, so we aggregate the
existing per-(ssp) forecast netCDFs up to global + super-region in COUNT space and
overlay the GBD2023 observed past.

Rate basis: COUNT / FULL GBD2023 population (aa_2023_full_population over ALL
admin-2 locs in the group), consistent for observed and predicted, so the lines
meet at 2023. Incidence rate per 1,000; mortality rate per 100,000.

This module is the ANALYSIS + DRIVER tier. The figure painters/layouts live in
`idd_forecast_mbp.lib.viz.forecast_timeseries` (the scrapable, reusable tier);
this script computes the prepared `panel_data` frame, calls those layouts, and
owns IO (savefig). See that module for the painter/layout convention and the
`panel_data` column contract.

Outputs to {MODEL_ROOT}/04-forecasting_data/malaria/gbd2023_timeseries/{RUN_DATE}/.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.viz.forecast_timeseries import (
    METRIC_TITLE,
    forecast_groups,
    plot_forecast_metric_grid,
    plot_forecast_ssp_2x2,
    plot_forecast_superregion_grid,
)

SSPS = ["ssp126", "ssp245", "ssp585"]
RUN_DATE = date.today().strftime("%Y%m%d")
OUT_DIR = (Path(mbpc.MODEL_ROOT) / "04-forecasting_data" / "malaria"
           / "gbd2023_timeseries" / RUN_DATE)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FC_DIR = (Path(mbpc.MODEL_ROOT) / "04-forecasting_data" / "malaria"
          / "forecast_outputs" / mbpc.LSAE_HIERARCHY / "current")
POP_PATH = Path(mbpc.POPULATION_READ_PATH) / "aa_2023_full_population_df.parquet"
AA_PATH = (Path(mbpc.MODEL_ROOT) / "02-processed_data" / "malaria" / "raked_aa"
           / mbpc.LSAE_HIERARCHY / "current" / "aa_full_malaria_df.parquet")
HIER_PATH = (Path(mbpc.HIERARCHY_READ_PATH)
             / f"full_hierarchy_2023_{mbpc.LSAE_HIERARCHY}.parquet")

INC_PER = 1_000.0
MORT_PER = 100_000.0

PANEL_COLS = ["group", "measure", "metric", "series", "year_id", "mid", "lo", "hi"]


# ─────────────────────────────── compute ──────────────────────────────────────
def _loc_super_region() -> pd.DataFrame:
    """admin-2 (most-detailed) location_id -> super_region_id/name."""
    h = pd.read_parquet(HIER_PATH, columns=["location_id", "level",
                                            "super_region_id", "super_region_name"])
    h = h[h["level"] == 5][["location_id", "super_region_id", "super_region_name"]]
    return h.astype({"location_id": "int64", "super_region_id": "int64"})


def _full_pop() -> pd.DataFrame:
    """FULL GBD2023 admin-2 population (all locs) -> the rate denominator."""
    p = pd.read_parquet(POP_PATH, columns=["location_id", "year_id", "population"])
    return p.astype({"location_id": "int64", "year_id": "int64"})


def _observed_counts() -> pd.DataFrame:
    """GBD2023 observed admin-2 counts (the past line)."""
    a = pd.read_parquet(AA_PATH, columns=["location_id", "year_id",
                                          "malaria_inc_count", "malaria_mort_count"])
    a = a.rename(columns={"malaria_inc_count": "inc_count",
                          "malaria_mort_count": "mort_count"})
    return a.astype({"location_id": "int64", "year_id": "int64"})


def _aggregate(counts: pd.DataFrame, pop: pd.DataFrame, lsr: pd.DataFrame,
               period: str) -> pd.DataFrame:
    """Sum counts (group) + FULL pop (group), at super-region AND global; rate=count/pop."""
    c = counts.merge(lsr, on="location_id", how="inner")
    p = pop.merge(lsr, on="location_id", how="inner")
    rows = []
    for keycols, _label in [(["super_region_id", "super_region_name"], None),
                            (None, "Global")]:
        if keycols:
            cc = (c.groupby(keycols + ["year_id"], as_index=False)
                  [["inc_count", "mort_count"]].sum())
            pp = (p.groupby(keycols + ["year_id"], as_index=False)["population"].sum())
            m = cc.merge(pp, on=keycols + ["year_id"], how="left")
            m["group"] = m["super_region_name"]
        else:
            cc = (c.groupby(["year_id"], as_index=False)
                  [["inc_count", "mort_count"]].sum())
            pp = (p.groupby(["year_id"], as_index=False)["population"].sum())
            m = cc.merge(pp, on=["year_id"], how="left")
            m["group"] = "Global"
        rows.append(m[["group", "year_id", "inc_count", "mort_count", "population"]])
    out = pd.concat(rows, ignore_index=True)
    out[["inc_count", "mort_count"]] = out[["inc_count", "mort_count"]].fillna(0.0)
    out["inc_rate"] = out["inc_count"] / out["population"] * INC_PER
    out["mort_rate"] = out["mort_count"] / out["population"] * MORT_PER
    out["period"] = period
    return out


def _pop_group_full(pop: pd.DataFrame, lsr: pd.DataFrame) -> pd.DataFrame:
    """FULL GBD2023 pop summed to super-region + global (the rate denominator)."""
    p = pop.merge(lsr, on="location_id", how="inner")
    sr = (p.groupby(["super_region_name", "year_id"], as_index=False)["population"].sum()
          .rename(columns={"super_region_name": "group"}))
    gl = p.groupby(["year_id"], as_index=False)["population"].sum()
    gl["group"] = "Global"
    return pd.concat([sr, gl], ignore_index=True)[["group", "year_id", "population"]]


def build_draw_quantiles(lsr: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Per-draw group aggregates -> central (mean) + 2.5/97.5% bands, for level-0
    (Global) and level-1 (super-region) groups, by measure (inc/mort) & metric (rate/count).
    Numerator = sum over kept admin-2 of (per-draw rate x admin-2 pop); denominator = FULL pop.
    Returns long rows [group, year_id, mid, lo, hi, ssp, measure, metric]."""
    pop_group = _pop_group_full(pop, lsr)
    sr_map = lsr.set_index("location_id")["super_region_name"]
    out = []
    for ssp in SSPS:
        ds = xr.open_dataset(FC_DIR / f"malaria_forecast_{ssp}_Baseline.nc")
        locs = ds.location_id.values.astype("int64")
        years = ds.year_id.values.astype("int64")
        sr_vals = sr_map.reindex(locs).values
        pk = pop[pop["location_id"].isin(locs.tolist())]
        pop_da = (pk.set_index(["location_id", "year_id"]).to_xarray()["population"]
                  .reindex(location_id=locs, year_id=years))
        for measure, var in [("inc", "log_malaria_inc_rate_pred"),
                             ("mort", "log_malaria_mort_rate_pred")]:
            count = (np.exp(ds[var]) * pop_da).assign_coords(sr=("location_id", sr_vals))
            gsr = (count.groupby("sr").sum("location_id")
                   .to_dataframe(name="count").reset_index().rename(columns={"sr": "group"}))
            ggl = count.sum("location_id").to_dataframe(name="count").reset_index()
            ggl["group"] = "Global"
            cdf = pd.concat([gsr[["group", "year_id", "draw", "count"]],
                             ggl[["group", "year_id", "draw", "count"]]], ignore_index=True)
            cdf = cdf.merge(pop_group, on=["group", "year_id"], how="left")
            per = INC_PER if measure == "inc" else MORT_PER
            for metric in ["rate", "count"]:
                cdf["val"] = (cdf["count"] / cdf["population"] * per
                              if metric == "rate" else cdf["count"])
                q = (cdf.groupby(["group", "year_id"])["val"]
                     .agg(mid="mean",
                          lo=lambda s: s.quantile(0.025),
                          hi=lambda s: s.quantile(0.975)).reset_index())
                q["ssp"], q["measure"], q["metric"] = ssp, measure, metric
                out.append(q)
        ds.close()
    return pd.concat(out, ignore_index=True)


def build_panel_data() -> pd.DataFrame:
    """Tidy per-panel frame consumed by lib.viz.forecast_timeseries:
        [group, measure, metric, series, year_id, mid, lo, hi]
    'observed' series carries a point estimate (lo/hi = NaN); ssp series carry the
    draw mean (mid) + 95% interval. Compute-once: every figure reads this one frame."""
    lsr, pop = _loc_super_region(), _full_pop()

    # ssp series: draw mean + 95% interval
    draws = (build_draw_quantiles(lsr, pop)
             .rename(columns={"ssp": "series"})[PANEL_COLS])

    # observed series: point estimate (no band), melted from the wide aggregate
    obs_wide = _aggregate(_observed_counts(), pop, lsr, "observed")
    obs_blocks = []
    for measure, metric, col in [("inc", "rate", "inc_rate"), ("mort", "rate", "mort_rate"),
                                 ("inc", "count", "inc_count"), ("mort", "count", "mort_count")]:
        b = obs_wide[["group", "year_id", col]].rename(columns={col: "mid"})
        b["series"], b["measure"], b["metric"] = "observed", measure, metric
        b["lo"] = np.nan
        b["hi"] = np.nan
        obs_blocks.append(b[PANEL_COLS])

    panel = pd.concat([draws, *obs_blocks], ignore_index=True)
    panel.to_parquet(OUT_DIR / "gbd2023_panel_data.parquet", index=False)
    return panel


# ─────────────────────────────── driver ───────────────────────────────────────
def _slug(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


def main() -> None:
    panel = build_panel_data()
    n = 0

    # global 2x2 (measure x metric)
    fig = plot_forecast_metric_grid(panel, group="Global")
    fig.savefig(OUT_DIR / "global_timeseries.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    n += 1

    # super-region small multiples, one per (measure, metric)
    for measure, metric in METRIC_TITLE:
        fig = plot_forecast_superregion_grid(panel, measure=measure, metric=metric)
        fig.savefig(OUT_DIR / f"superregion_{measure}_{metric}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
        n += 1

    # ssp 2x2, per (group, measure, metric) for Global + each forecast super-region
    out2 = OUT_DIR / "ssp_2x2"
    out2.mkdir(exist_ok=True)
    groups = ["Global"] + forecast_groups(panel)
    n2 = 0
    for group in groups:
        for measure, metric in METRIC_TITLE:
            fig = plot_forecast_ssp_2x2(panel, group=group, measure=measure, metric=metric)
            fig.savefig(out2 / f"{_slug(group)}_{measure}_{metric}_2x2.png",
                        dpi=130, bbox_inches="tight")
            plt.close(fig)
            n2 += 1

    print(f"[done] {OUT_DIR}")
    print(f"  {n} top-level figures + ssp_2x2/: {n2} figures")
    print("  gbd2023_panel_data.parquet")


if __name__ == "__main__":
    main()
