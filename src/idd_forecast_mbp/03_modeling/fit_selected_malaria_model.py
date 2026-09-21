"""Fit the selected malaria model into the models node and, if asked, freeze and promote it.

    .venv/bin/python src/idd_forecast_mbp/03_modeling/fit_selected_malaria_model.py \\
        --config reports/model_selection/malaria_selection_config.yaml \\
        --r-image <singularity .img> --r-shell <execRscript.sh> \\
        [--run-dir <selection run dir>] [--current --description "why" --label <name>] [--dry-run]

Reads the pick from ``<run_dir>/selection_result.json``, runs fit_selected_malaria_model.r
into the node's ``working/`` slot with the ``final_fit:`` settings of the config, verifies
the two outputs exist, then ``finish_if_requested``: with ``--current`` the snapshot is
frozen and promoted in one go (a chained-stage launch); without it the fit sits in
``working/`` until ``idd-versions <node> freeze`` or the gate's Flag-best promotes it.
One run per slot: finish a fit before launching the next.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

import click
from idd_tools.versions import VersionsOptions, resolve_target, versions_options

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.versioning import finish_stage
from idd_forecast_mbp.select.rank import (
    RESULT_FILE,
    load_config,
    read_result,
    resolve_run_dir,
)

WORKER = Path(__file__).resolve().with_name("fit_selected_malaria_model.r")


def build_worker_args(
    result_path: Path,
    out_dir: Path,
    past_inputs: Path,
    final_fit: dict[str, Any],
    prep_script: Path | None = None,
) -> list[str]:
    """The worker's flags from the config's ``final_fit:`` section; nothing defaulted here.

    ``prep_script`` is passed through only when given; the worker's own default is the
    package's ``lib/malaria_fit_frame.R``.
    """
    filt = final_fit["data_filter"]
    args = [
        "--result",
        str(result_path),
        "--out-dir",
        str(out_dir),
        "--past-inputs",
        str(past_inputs),
        "--inc-count-min",
        str(filt["malaria_inc_count_min"]),
        "--pfpr-min",
        str(filt["malaria_pfpr_min"]),
        "--optimizer",
        str(final_fit["optimizer"]),
        "--maxit",
        str(int(final_fit["maxit"])),
        "--suit-variant",
        str(final_fit["suit_variant"]),
        "--inc-mort-rhs",
        str(final_fit["inc_mort_rhs"]),
    ]
    if prep_script is not None:
        args += ["--prep-script", str(prep_script)]
    return args


def wrapper_quote(arg: str) -> str:
    """Quote one argument for execRscript.sh.

    The wrapper joins its arguments into a string and ``eval``s ``bash -c "... $args"``, so an
    argument is parsed twice: once inside that double-quoted string (where ``"``, ``$``, backtick
    and backslash are special) and once by ``bash -c``. Single-quote it for the second parse
    and escape the double-quote-context characters for the first. Plain paths pass unchanged.
    """
    quoted = shlex.quote(arg)
    return (
        quoted.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )


def worker_command(
    r_shell: str, r_image: str, worker: Path, args: list[str]
) -> list[str]:
    """The IHME singularity R shell invocation, as the jobmon orchestrators spell it, with
    every worker argument quoted for the wrapper's double parse (see wrapper_quote)."""
    return [
        r_shell,
        "-i",
        r_image,
        "-s",
        str(worker),
        *(wrapper_quote(a) for a in args),
    ]


@click.command(help=__doc__)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--run-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Override the config's selection run dir.",
)
@click.option("--r-image", required=True, help="singularity .img for R")
@click.option("--r-shell", required=True, help="execRscript.sh wrapper")
@click.option(
    "--worker",
    default=str(WORKER),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--node",
    default=None,
    type=click.Path(file_okay=False),
    help="Models node; default constants.MAL_MODELS_NODE.",
)
@click.option(
    "--past-inputs",
    default=None,
    type=click.Path(dir_okay=False),
    help="Past-inputs parquet; default the current past_inputs snapshot.",
)
@click.option(
    "--prep-script",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="R file defining prepare_malaria_fit_frame(); default the package's lib/malaria_fit_frame.R.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the worker command and the target; run nothing.",
)
@versions_options
def main(  # noqa: PLR0913 - one option per launcher knob
    *,
    config_path: str,
    run_dir: str | None,
    r_image: str,
    r_shell: str,
    worker: str,
    node: str | None,
    past_inputs: str | None,
    prep_script: str | None,
    versions: VersionsOptions,
    dry_run: bool,
) -> None:
    cfg = load_config(config_path)
    selection_dir = resolve_run_dir(cfg.run_dir, run_dir, root=mbpc._MODELING_STAGE)  # noqa: SLF001 - stage roots are underscore-named in constants
    result_path = selection_dir / RESULT_FILE
    record = read_result(selection_dir)  # refuses when no result was written
    models_node = Path(node) if node else mbpc.MAL_MODELS_NODE
    inputs = (
        Path(past_inputs)
        if past_inputs
        else mbpc.MAL_PAST_INPUTS_READ_PATH / mbpc.MAL_PAST_INPUTS_FILENAME
    )
    target = resolve_target(models_node, versions)
    cmd = worker_command(
        r_shell,
        r_image,
        Path(worker),
        build_worker_args(
            result_path,
            target,
            inputs,
            cfg.final_fit,
            Path(prep_script) if prep_script else None,
        ),
    )
    click.echo(
        f"selected spec {record['pick']['spec_index']} ({record['status']}) from {selection_dir}"
    )
    click.echo(f"target: {target}")
    click.echo("command: " + " ".join(cmd))
    if dry_run:
        click.echo("(--dry-run: nothing run)")
        return
    proc = subprocess.run(cmd, check=False)  # noqa: S603 - fixed argv from options
    if proc.returncode != 0:
        msg = f"fit worker failed with exit code {proc.returncode}; nothing frozen"
        raise click.ClickException(msg)
    for name in (mbpc.MAL_MODELS_RDATA, mbpc.MAL_MODELS_RUN_JSON):
        if not (target / name).is_file():
            msg = f"worker reported success but {target / name} is missing; nothing frozen"
            raise click.ClickException(msg)
    finished = finish_stage(models_node, versions)
    if finished is None:
        click.echo(
            f"fit left in {target}; freeze with `idd-versions {models_node} freeze '<why>' [--current]` or promote from the gate"
        )


if __name__ == "__main__":
    main()
