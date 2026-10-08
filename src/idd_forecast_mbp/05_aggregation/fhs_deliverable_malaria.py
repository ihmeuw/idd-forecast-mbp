"""The malaria FHS deliverable (3,A,X,beta,a,m): reference arm, every FHS location, draws, age/sex, rates.

    # one SSP x a draw range (run many in parallel; each writes its own chunk file)
    .venv/bin/python src/idd_forecast_mbp/05_aggregation/fhs_deliverable_malaria.py chunk \\
        --model spec7426_best --ssp ssp245 --draw-start 0 --draw-end 10
    # after every chunk of every SSP exists: one netCDF per SSP, then freeze
    .venv/bin/python src/idd_forecast_mbp/05_aggregation/fhs_deliverable_malaria.py combine \\
        --model spec7426_best [--current --description "why" --label L]

The reference arm is Baseline (DECISIONS 2026-10-07). For each draw the admin-2 age/sex counts are
recreated exactly as every other product builds them (``lib/processing/arm_recreate``: the arm's
weights on the forecast's own split, vaccine protection cell-wise), rolled up in count space to the
513 FHS locations (zero-filled: a location with no endemic admin-2 is a true zero), and divided by
that location's own age/sex population. Years 2023-2100. Output node: ``<model>__fhs_deliverable``,
one ``malaria_fhs_{ssp}.nc`` per SSP with ``incidence_rate`` and ``mortality_rate`` over
(location_id, year_id, age_group_id, sex_id, draw).
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr
from idd_tools.versions import VersionsOptions, versions_options

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.data.malaria_arms import load_arms
from idd_forecast_mbp.lib.processing.arm_factors import (
    KEYS_AA,
    KEYS_AS,
)
from idd_forecast_mbp.lib.processing.arm_recreate import (
    arm_weights,
    forecast_nc,
    read_admin2_log_rates,
    read_population,
    recreate_admin2_age_sex,
    roll_up_age_sex_to_nodes,
)
from idd_forecast_mbp.lib.versioning import finish_stage, stage_target

REPO = Path(mbpc.__file__).resolve().parents[2]
ARMS_FILE = REPO / "reports" / "model_runs" / "malaria_arms.yaml"
REFERENCE_ARM = "Baseline"
YEARS = list(range(2023, 2101))  # beta: each SSP 2023-2100
SSPS = ("ssp126", "ssp245", "ssp585")
CHUNK_RE = re.compile(r"^chunk_(?P<ssp>ssp\d+)_(?P<start>\d{3})_(?P<end>\d{3})\.nc$")
RATE_VARS = {"inc": "incidence_rate", "mort": "mortality_rate"}


def node_root(model: str) -> Path:
    return mbpc.mal_products_root(f"{model}__fhs_deliverable")


def chunk_name(ssp: str, start: int, end: int) -> str:
    return f"chunk_{ssp}_{start:03d}_{end:03d}.nc"


def rates_to_dataset(counts: pd.DataFrame, as_pop: pd.DataFrame) -> xr.Dataset:
    """Counts at FHS nodes -> float32 rates on (location, year, age, sex, draw); population 0 -> rate 0."""
    df = counts.merge(as_pop[[*KEYS_AS, "population"]], on=KEYS_AS, how="left")
    if df.population.isna().any():
        msg = f"{int(df.population.isna().sum())} FHS cells have no age/sex population"
        raise ValueError(msg)
    data = {}
    for m, name in RATE_VARS.items():
        df[name] = np.where(
            df.population > 0, df[f"{m}_count"] / df.population, 0.0
        ).astype("float32")
        data[name] = df.set_index([*KEYS_AS, "draw"])[name].to_xarray()
    return xr.Dataset(data)


@click.group()
def cli() -> None:
    """FHS deliverable chunks and their combination."""


@cli.command()
@click.option("--model", required=True)
@click.option("--ssp", required=True, type=click.Choice(SSPS))
@click.option("--draw-start", required=True, type=int)
@click.option("--draw-end", required=True, type=int, help="exclusive")
@click.option(
    "--arms-file",
    default=str(ARMS_FILE),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
def chunk(model: str, ssp: str, draw_start: int, draw_end: int, arms_file: str) -> None:
    """Rates for one SSP and draws [start, end) at every FHS location, into the node's working/."""
    cfg = load_arms(arms_file)
    arm = cfg.arm(model, REFERENCE_ARM)
    draws = list(range(draw_start, draw_end))
    out_dir = stage_target(node_root(model), VersionsOptions())
    rates = read_admin2_log_rates(forecast_nc(arm, ssp), years=YEARS, draws=draws)
    locs = sorted(rates.location_id.unique().tolist())
    aa_pop = read_population("aa_2023_full_population_df.parquet", KEYS_AA, locs, YEARS)
    weights = arm_weights(arm, locs, YEARS, cfg.ve_variant)
    hierarchy = load_hierarchy()
    fhs_ids = (
        hierarchy.loc[hierarchy.in_fhs_hierarchy.astype(bool), "location_id"]
        .astype(int)
        .tolist()
    )
    as_pop = read_population(
        "as_2023_full_population_df.parquet", KEYS_AS, fhs_ids, YEARS
    )
    ages = sorted(as_pop.age_group_id.unique().tolist())
    sexes = sorted(as_pop.sex_id.unique().tolist())
    parts = []
    for d in draws:
        as_d = recreate_admin2_age_sex(
            arm,
            ssp,
            ve_variant=cfg.ve_variant,
            weights=weights,
            aa_population=aa_pop,
            log_rates=rates[rates.draw == d],
        )
        parts.append(
            roll_up_age_sex_to_nodes(
                as_d, hierarchy, fhs_ids, years=YEARS, draws=[d], ages=ages, sexes=sexes
            )
        )
        click.echo(f"  [{ssp}] draw {d} done")
    ds = rates_to_dataset(pd.concat(parts, ignore_index=True), as_pop)
    path = out_dir / chunk_name(ssp, draw_start, draw_end)
    tmp = path.with_suffix(".nc.tmp")
    ds.to_netcdf(
        tmp, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    )
    tmp.replace(path)
    click.echo(f"wrote {path}")


@cli.command()
@click.option("--model", required=True)
@click.option("--n-draws", default=100, show_default=True, type=int)
@versions_options
def combine(model: str, n_draws: int, versions: VersionsOptions) -> None:
    """One ``malaria_fhs_{ssp}.nc`` per SSP from its chunks; refuses unless every draw is present."""
    root = node_root(model)
    work = root / "working"
    chunks: dict[str, list[Path]] = {s: [] for s in SSPS}
    for p in sorted(work.glob("chunk_*.nc")):
        m = CHUNK_RE.match(p.name)
        if m:
            chunks[m["ssp"]].append(p)
    for ssp, paths in chunks.items():
        if not paths:
            msg = f"no chunks for {ssp} in {work}"
            raise click.ClickException(msg)
        ds = xr.open_mfdataset(paths, combine="nested", concat_dim="draw").sortby(
            "draw"
        )
        got = [int(d) for d in ds.draw.to_numpy()]
        if got != list(range(n_draws)):
            msg = f"{ssp}: draws present are not 0..{n_draws - 1} (got {len(got)})"
            raise click.ClickException(msg)
        path = work / f"malaria_fhs_{ssp}.nc"
        tmp = path.with_suffix(".nc.tmp")
        ds.to_netcdf(
            tmp, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        )
        ds.close()
        tmp.replace(path)
        for p in paths:
            p.unlink()
        click.echo(f"wrote {path} ({n_draws} draws; {len(paths)} chunks removed)")
    finish_stage(root, versions)


if __name__ == "__main__":
    cli()
