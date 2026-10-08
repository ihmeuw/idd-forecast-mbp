"""Acceptance checks for one model's finished arms (MALARIA_RUN_PLAN step 7). Numbers, not asserts.

    .venv/bin/python src/idd_forecast_mbp/05_aggregation/check_malaria_run.py --model spec7426_best \\
        [--ssp ssp245] [--n-countries 3] [--fhs/--no-fhs] [--current --description "why"]

A. Anchor: every arm's 2023 values against the raked observed (GBD 2023) all-age values, by level,
   from the anchor tables the finishing driver wrote (max |relative difference|).
B. Recreation: Baseline admin-2 draws recreated in memory for the ``n`` highest-burden countries,
   summed to each country, mean over draws, against the saved paper summary's country mean.
C. FHS deliverable (best model only): FHS age/sex rates x age/sex population summed to all-age,
   mean over draws, against the paper summary at the same countries (2023-2100).
Writes ``checks.json`` into the product node ``<model>__acceptance``.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
import xarray as xr
from idd_tools.versions import VersionsOptions, versions_options

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.data.malaria_arms import load_arms
from idd_forecast_mbp.lib.processing.arm_factors import KEYS_AA, KEYS_AS
from idd_forecast_mbp.lib.processing.arm_recreate import (
    arm_weights,
    forecast_nc,
    read_admin2_log_rates,
    read_population,
    recreate_admin2_age_sex,
)
from idd_forecast_mbp.lib.versioning import finish_stage, stage_target

REPO = Path(mbpc.__file__).resolve().parents[2]
ARMS_FILE = REPO / "reports" / "model_runs" / "malaria_arms.yaml"
SSPS = ("ssp126", "ssp245", "ssp585")
COUNTRY_LEVEL = 3
ADMIN2_LEVEL = 5
ANCHOR_YEAR = 2023


def rel_diff(ours: pd.Series, ref: pd.Series) -> float:
    """max |ours - ref| / |ref| over cells where ref != 0 (NaN when there are none)."""
    ok = ref.abs() > 0
    return (
        float(((ours[ok] - ref[ok]).abs() / ref[ok].abs()).max())
        if ok.any()
        else float("nan")
    )


def check_anchor(model: str, arms_file: str) -> dict[str, dict[str, float]]:
    """``{arm: {"level_L_measure": max |rel diff|}}`` from each arm's frozen anchor tables (counts)."""
    cfg = load_arms(arms_file)
    out: dict[str, dict[str, float]] = {}
    for arm in cfg.arms_for(model):
        node = mbpc.mal_products_root(arm.run_key) / "current"
        tabs = [pd.read_parquet(p) for p in sorted(node.glob("anchor_2023_*.parquet"))]
        if not tabs:
            out[arm.name] = {"missing": float("nan")}
            continue
        t = pd.concat(tabs)
        t = t[t.metric == "count"]
        out[arm.name] = {
            f"level_{int(lv)}_{m}": float(g.rel_diff.abs().max())
            for (lv, m), g in t.groupby(["level", "measure"])
        }
    return out


def top_countries(summary: pd.DataFrame, n: int) -> list[int]:
    lvl3 = summary[(summary.level == COUNTRY_LEVEL) & (summary.year_id == ANCHOR_YEAR)]
    return [int(x) for x in lvl3.nlargest(n, "inc_count_mean").location_id]


def check_recreate(model: str, ssp: str, n: int, arms_file: str) -> dict[str, object]:
    cfg = load_arms(arms_file)
    arm = cfg.arm(model, "Baseline")
    summary = pd.read_parquet(
        mbpc.mal_products_root(arm.run_key)
        / "current"
        / f"all_age_summary_{ssp}.parquet"
    )
    countries = top_countries(summary, n)
    hier = load_hierarchy()
    a2 = hier[hier.level == ADMIN2_LEVEL]
    country_of = {}
    for lid, path in zip(a2.location_id, a2.path_to_top_parent, strict=True):
        hit = {int(x) for x in str(path).split(",") if x} & set(countries)
        if hit:
            country_of[int(lid)] = next(iter(hit))
    rates = read_admin2_log_rates(forecast_nc(arm, ssp))
    rates = rates[rates.location_id.isin(country_of)]
    locs = sorted(rates.location_id.unique().tolist())
    years = sorted(rates.year_id.unique().tolist())
    aa_pop = read_population("aa_2023_full_population_df.parquet", KEYS_AA, locs, years)
    weights = arm_weights(arm, locs, years, cfg.ve_variant)
    rec = recreate_admin2_age_sex(
        arm,
        ssp,
        ve_variant=cfg.ve_variant,
        weights=weights,
        aa_population=aa_pop,
        log_rates=rates,
    )
    rec["country"] = rec.location_id.map(country_of)
    by = rec.groupby(["country", "year_id", "draw"], as_index=False)[
        ["inc_count", "mort_count"]
    ].sum(min_count=1)
    mean = by.groupby(["country", "year_id"], as_index=False)[
        ["inc_count", "mort_count"]
    ].mean()
    saved = summary[summary.location_id.isin(countries)].rename(
        columns={"location_id": "country"}
    )
    m = mean.merge(saved, on=["country", "year_id"])
    return {
        "ssp": ssp,
        "countries": countries,
        "n_cells": len(m),
        "inc_count_mean_max_rel_diff": rel_diff(m.inc_count, m.inc_count_mean),
        "mort_count_mean_max_rel_diff": rel_diff(m.mort_count, m.mort_count_mean),
    }


def check_fhs(
    model: str, ssp: str, countries: list[int], arms_file: str
) -> dict[str, object]:
    cfg = load_arms(arms_file)
    arm = cfg.arm(model, "Baseline")
    nc = (
        mbpc.mal_products_root(f"{model}__fhs_deliverable")
        / "current"
        / f"malaria_fhs_{ssp}.nc"
    )
    with xr.open_dataset(nc) as ds:
        fhs_locs = {int(x) for x in ds.location_id.to_numpy()}
        keep = [c for c in countries if c in fhs_locs]
        sub = ds.sel(location_id=keep).to_dataframe().reset_index()
    pop = read_population(
        "as_2023_full_population_df.parquet",
        KEYS_AS,
        keep,
        sorted(sub.year_id.unique()),
    )
    sub = sub.merge(pop, on=KEYS_AS)
    sub["inc_count"] = sub.incidence_rate * sub.population
    sub["mort_count"] = sub.mortality_rate * sub.population
    tot = sub.groupby(["location_id", "year_id", "draw"], as_index=False)[
        ["inc_count", "mort_count"]
    ].sum()
    mean = tot.groupby(["location_id", "year_id"], as_index=False)[
        ["inc_count", "mort_count"]
    ].mean()
    summary = pd.read_parquet(
        mbpc.mal_products_root(arm.run_key)
        / "current"
        / f"all_age_summary_{ssp}.parquet"
    )
    m = mean.merge(summary, on=["location_id", "year_id"])
    return {
        "ssp": ssp,
        "countries_in_fhs": keep,
        "n_cells": len(m),
        "inc_count_mean_max_rel_diff": rel_diff(m.inc_count, m.inc_count_mean),
        "mort_count_mean_max_rel_diff": rel_diff(m.mort_count, m.mort_count_mean),
    }


@click.command(help=__doc__)
@click.option("--model", required=True)
@click.option("--ssp", default="ssp245", show_default=True, type=click.Choice(SSPS))
@click.option("--n-countries", default=3, show_default=True, type=int)
@click.option(
    "--fhs/--no-fhs",
    default=False,
    show_default=True,
    help="also check the FHS deliverable",
)
@click.option(
    "--arms-file",
    default=str(ARMS_FILE),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@versions_options
def main(  # noqa: PLR0913 - one option per check knob
    *,
    model: str,
    ssp: str,
    n_countries: int,
    fhs: bool,
    arms_file: str,
    versions: VersionsOptions,
) -> None:
    report: dict[str, object] = {
        "model": model,
        "anchor": check_anchor(model, arms_file),
    }
    rec = check_recreate(model, ssp, n_countries, arms_file)
    report["recreate"] = rec
    if fhs:
        report["fhs"] = check_fhs(model, ssp, list(rec["countries"]), arms_file)  # type: ignore[call-overload]
    root = mbpc.mal_products_root(f"{model}__acceptance")
    out = stage_target(root, versions)
    (out / "checks.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    click.echo(json.dumps(report, indent=1, default=float))
    finish_stage(root, versions)


if __name__ == "__main__":
    main()
