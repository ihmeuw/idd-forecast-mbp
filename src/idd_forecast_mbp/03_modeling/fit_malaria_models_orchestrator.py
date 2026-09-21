"""Jobmon orchestrator for the malaria model-selection workflow (thin CLI).

    .venv/bin/python src/idd_forecast_mbp/03_modeling/fit_malaria_models_orchestrator.py \\
        --spec-table <run_dir>/spec_table.parquet --output-dir <run_dir> \\
        --r-image <singularity .img> --r-shell <execRscript.sh> [--probe | --full] [...]

The body lives in ``idd_forecast_mbp.lib.modeling.malaria_fit_run.submit_malaria_fit_run``
(cells, bundling, templates, resources, done-check, finalize); this script only parses
options and calls it. Defaults reproduce the 2026-07 selection run: selection thresholds
(inc_count >= 1, pfpr >= 0.0001), ten temporal windows, efs / maxit 30, no saved fits.
"""

from __future__ import annotations

from pathlib import Path

import click

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.modeling.malaria_fit_run import (
    DEFAULT_PREP_SCRIPT,
    DEFAULT_WORKER,
    FitRunResources,
    submit_malaria_fit_run,
)

DEFAULT_PAST_INPUTS = mbpc.MAL_PAST_INPUTS_READ_PATH / mbpc.MAL_PAST_INPUTS_FILENAME


@click.command(help=__doc__)
@click.option(
    "--spec-table",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="parquet: spec_index, n_smooths, n_scams, formula_text [, suit_variant]",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="run root (receives spec_table.parquet, manifest.json and every task's output)",
)
@click.option(
    "--worker",
    default=str(DEFAULT_WORKER),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="select_malaria_models_rocket.r",
)
@click.option("--r-image", required=True, help="singularity .img for R")
@click.option("--r-shell", required=True, help="execRscript.sh wrapper")
@click.option(
    "--past-inputs",
    default=str(DEFAULT_PAST_INPUTS),
    show_default=True,
    type=click.Path(dir_okay=False),
    help="malaria_past_inputs.parquet every task reads",
)
@click.option(
    "--prep-script",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=f"R file defining prepare_malaria_fit_frame(); default {DEFAULT_PREP_SCRIPT.name}",
)
@click.option(
    "--probe/--full",
    default=False,
    show_default=True,
    help="probe = N specs per (n_scams, n_smooths) level; full = whole table",
)
@click.option(
    "--n-per-level", default=2, show_default=True, help="probe: specs per level"
)
@click.option("--optimizer", default="efs", show_default=True)
@click.option("--maxit", default=30, show_default=True, help="EFS max iterations")
@click.option(
    "--cv-strategy",
    type=click.Choice(["temporal", "random"]),
    default="temporal",
    show_default=True,
    help="temporal = IS + temporal-window OOS; random = OOS-only k-fold",
)
@click.option(
    "--cv-n-folds", default=10, show_default=True, help="random: number of CV folds"
)
@click.option(
    "--inc-count-min",
    default=1.0,
    show_default=True,
    help="row filter: malaria_inc_count >= this",
)
@click.option(
    "--pfpr-min",
    default=0.0001,
    show_default=True,
    help="row filter: malaria_pfpr >= this",
)
@click.option(
    "--save-fits/--no-save-fits",
    default=False,
    show_default=True,
    help="write each fitted object as fits/<cell>/spec_<i>.rds with a JSON sidecar",
)
@click.option(
    "--save-predictions/--no-save-predictions",
    default=False,
    show_default=True,
    help="write each cell's predictions as predictions/<cell>/spec_<i>.parquet",
)
@click.option("--cores", default=16, show_default=True)
@click.option("--max-concurrent", default=500, show_default=True)
@click.option("--project", default="proj_rapidresponse", show_default=True)
@click.option("--queue", default="all.q", show_default=True)
def main(  # noqa: PLR0913 - one option per launcher knob
    *,
    spec_table: str,
    output_dir: str,
    worker: str,
    r_image: str,
    r_shell: str,
    past_inputs: str,
    prep_script: str | None,
    probe: bool,
    n_per_level: int,
    optimizer: str,
    maxit: int,
    cv_strategy: str,
    cv_n_folds: int,
    inc_count_min: float,
    pfpr_min: float,
    save_fits: bool,
    save_predictions: bool,
    cores: int,
    max_concurrent: int,
    project: str,
    queue: str,
) -> None:
    resources = FitRunResources(
        cores=cores, max_concurrent=max_concurrent, project=project, queue=queue
    )
    submit_malaria_fit_run(
        Path(spec_table),
        Path(output_dir),
        worker=Path(worker),
        r_image=r_image,
        r_shell=r_shell,
        past_inputs=Path(past_inputs),
        prep_script=Path(prep_script) if prep_script else None,
        optimizer=optimizer,
        maxit=maxit,
        cv_strategy=cv_strategy,  # type: ignore[arg-type]  # click.Choice guarantees the literal
        cv_n_folds=cv_n_folds,
        inc_count_min=inc_count_min,
        pfpr_min=pfpr_min,
        save_fits=save_fits,
        save_predictions=save_predictions,
        probe_n_per_level=n_per_level if probe else None,
        resources=resources,
        log=click.echo,
    )


if __name__ == "__main__":
    main()
