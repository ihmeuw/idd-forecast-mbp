"""Finish one malaria result arm into its all-age products (the paper output, 1,B,Y,alpha,a).

    .venv/bin/python src/idd_forecast_mbp/05_aggregation/finalize_malaria_arm.py \\
        --model spec7426_best --arm Baseline [--ssp-scenario ssp245 ...] \\
        [--current --description "why"] [--dry-run]

Resolves the arm in ``reports/model_runs/malaria_arms.yaml`` (``lib/data/malaria_arms.py``), reads the
forecast launch that arm uses (by its label on the forecast_outputs node), builds the arm's draw-free
all-age factors (``lib/processing/arm_factors.py``: vaccine protection, population hold, age-structure
hold) and runs finish_run's count-space roll-up and draw summary at every LSAE level (0-5) for every
year the forecast holds (historical 2000-2022 plus the SSP years). Writes, per SSP,
``all_age_summary_{ssp}.parquet`` (counts and rates, mean/lower/upper, with the level's own population)
and the 2023 anchor check, into the product node ``<model>__<arm>``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np
import pandas as pd
import xarray as xr
from idd_tools.versions import VersionsOptions, versions_options

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.data.malaria_arms import Arm, load_arms
from idd_forecast_mbp.lib.io.parquet import write_parquet
from idd_forecast_mbp.lib.processing.arm_factors import all_age_factors
from idd_forecast_mbp.lib.processing.arm_recreate import arm_weights, forecast_nc
from idd_forecast_mbp.lib.versioning import finish_stage, stage_target

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(mbpc.__file__).resolve().parents[2]
ARMS_FILE = REPO / "reports" / "model_runs" / "malaria_arms.yaml"
ANCHOR_YEAR = 2023
LEVELS = (0, 1, 2, 3, 4, 5)
SSPS = ("ssp126", "ssp245", "ssp585")


def _finish_run() -> ModuleType:
    """The sibling stage script ``finish_run.py``, loaded as a module (stage dirs are not packages)."""
    path = Path(__file__).with_name("finish_run.py")
    spec = importlib.util.spec_from_file_location("finish_run", path)
    if spec is None or spec.loader is None:
        msg = f"cannot load {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arm_factor_frames(
    arm: Arm,
    locations: list[int],
    years: list[int],
    ve_variant: str,
) -> dict[str, pd.DataFrame] | None:
    """``{measure: [location_id, year_id, factor]}`` for the arm, or None when every factor is 1.

    The factor is ``sum_as(w)`` from the same ``arm_weights`` the admin-2 recreation uses. A location
    whose weights all vanish has no age/sex pattern (no observed burden at 2023, so the forecast
    dropped it; protection never reaches 1) and keeps factor 1: it carries no counts either way.
    """
    if (
        arm.vaccine == "none"
        and arm.age_structure_hold_year is None
        and arm.population_hold_year is None
    ):
        return None
    factors = all_age_factors(arm_weights(arm, locations, years, ve_variant))
    out = {}
    for m in ("inc", "mort"):
        f = factors[f"{m}_factor"].to_numpy()
        out[m] = pd.DataFrame(
            {
                "location_id": factors.location_id,
                "year_id": factors.year_id,
                "factor": np.where(f > 0, f, 1.0),
            }
        )
    return out


@click.command(help=__doc__)
@click.option("--model", required=True, help="model key in malaria_arms.yaml")
@click.option("--arm", "arm_name", required=True, help="arm name in malaria_arms.yaml")
@click.option(
    "--arms-file",
    default=str(ARMS_FILE),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--ssp-scenario", "ssps", multiple=True, default=SSPS, show_default=True)
@click.option("--lower-quantile", default=0.025, show_default=True, type=float)
@click.option("--upper-quantile", default=0.975, show_default=True, type=float)
@click.option(
    "--rel-tol", default=1e-6, show_default=True, help="anchor-year relative tolerance"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Resolve and print the arm and its inputs; write nothing.",
)
@versions_options
def main(  # noqa: PLR0913 - one option per knob
    *,
    model: str,
    arm_name: str,
    arms_file: str,
    ssps: tuple[str, ...],
    lower_quantile: float,
    upper_quantile: float,
    rel_tol: float,
    dry_run: bool,
    versions: VersionsOptions,
) -> None:
    cfg = load_arms(arms_file)
    arm = cfg.arm(model, arm_name)
    ncs = {ssp: forecast_nc(arm, ssp) for ssp in ssps}
    click.echo(
        f"arm {arm.run_key}: forecast {arm.forecast_label}, dah {arm.dah}, vaccine {arm.vaccine}, "
        f"population hold {arm.population_hold_year}, age-structure hold {arm.age_structure_hold_year}"
    )
    missing = [str(p) for p in ncs.values() if not p.is_file()]
    if missing:
        msg = f"forecast netCDF(s) missing: {missing}"
        raise click.ClickException(msg)
    node_root = mbpc.mal_products_root(arm.run_key)
    if dry_run:
        click.echo(
            f"would write into {node_root}; inputs: {[str(p) for p in ncs.values()]}"
        )
        return
    out_dir = stage_target(node_root, versions)
    fr = _finish_run()
    hierarchy = load_hierarchy()
    verdicts = {}
    for ssp, nc in ncs.items():
        with xr.open_dataset(nc) as ds:
            years = [int(y) for y in ds.year_id.to_numpy()]
            locations = [int(x) for x in ds.location_id.to_numpy()]
        population = fr.load_population(
            sorted(set(years) | {ANCHOR_YEAR}), hold_year=arm.population_hold_year
        )
        factors = arm_factor_frames(arm, locations, years, cfg.ve_variant)
        if factors is not None:
            for m, f in factors.items():
                click.echo(
                    f"  [{ssp}] {m} factor: min {f.factor.min():.4f} max {f.factor.max():.4f}"
                )
        summary = fr.build_all_age_summary(
            nc,
            hierarchy,
            population,
            list(LEVELS),
            (lower_quantile, upper_quantile),
            as_factors=factors,
        )
        write_parquet(
            summary, out_dir / f"all_age_summary_{ssp}.parquet", validate=True
        )
        check, verdict = fr.build_anchor_check(summary, ANCHOR_YEAR, rel_tol)
        write_parquet(
            check, out_dir / f"anchor_{ANCHOR_YEAR}_{ssp}.parquet", validate=True
        )
        verdicts[ssp] = verdict
        click.echo(
            f"  [{ssp}] {len(summary):,} rows; anchor {'PASS' if verdict['passed'] else 'FAIL'} "
            f"(max |rel diff| {verdict['max_abs_rel_diff']})"
        )
    (out_dir / "arm.json").write_text(
        json.dumps(
            {
                "arm": arm.name,
                "model": arm.model,
                "forecast_label": arm.forecast_label,
                "dah": arm.dah,
                "vaccine": arm.vaccine,
                "ve_variant": cfg.ve_variant,
                "population_hold_year": arm.population_hold_year,
                "age_structure_hold_year": arm.age_structure_hold_year,
                "forecast_files": {s: str(p.resolve()) for s, p in ncs.items()},
                "arms_file": str(Path(arms_file).resolve()),
                "levels": list(LEVELS),
                "anchor": verdicts,
            },
            indent=1,
        )
        + "\n"
    )
    finish_stage(node_root, versions)


if __name__ == "__main__":
    main()
