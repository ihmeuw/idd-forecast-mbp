"""Rank a finalized malaria selection run and write the tentative pick into the run dir.

    .venv/bin/python src/idd_forecast_mbp/03_modeling/rank_selection_run.py \
        --config reports/model_selection/malaria_selection_config.yaml [--run-dir <dir>] [--no-write]

Reads ``<run_dir>/selection_summary.parquet`` (from finalize_selection_run.py), applies the
``rank:`` parameters of the config, and writes ``selection_result.json``, ``ranking.parquet``
and ``candidates.parquet`` beside it with ``status: tentative``, then renders
``reports/model_selection/malaria_model_selection.qmd`` to ``report.html`` in the run dir. Touches no registry; the
gate does that after the report has been read.
"""

from __future__ import annotations

from pathlib import Path

import click

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.select.malaria_spec_design import build_universe
from idd_forecast_mbp.select.rank import (
    format_pick,
    load_config,
    load_summary,
    render_report,
    run_selection,
    write_result,
)

QMD = (
    Path(__file__).resolve().parents[3]
    / "reports"
    / "model_selection"
    / "malaria_model_selection.qmd"
)


def resolve_run_dir(config_run_dir: str, override: str | None) -> Path:
    """Absolute run dir: an absolute override as given, otherwise relative to the modeling stage root."""
    chosen = Path(override) if override else Path(config_run_dir)
    return chosen if chosen.is_absolute() else mbpc._MODELING_STAGE / chosen  # noqa: SLF001 - stage roots are underscore-named in constants by convention


@click.command(help=__doc__)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Committed selection config (fit: and rank: sections).",
)
@click.option(
    "--run-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Override the config's run_dir; absolute, or relative to the 03-modeling_data root.",
)
@click.option(
    "--no-write",
    is_flag=True,
    help="Rank and print the pick; write nothing into the run dir.",
)
@click.option(
    "--render/--no-render",
    default=True,
    show_default=True,
    help="Render the Quarto report into the run dir after writing the result.",
)
@click.option(
    "--quarto",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="quarto binary; default: the user-level install, else PATH.",
)
def main(
    config_path: str,
    run_dir: str | None,
    quarto: str | None,
    *,
    no_write: bool,
    render: bool,
) -> None:
    cfg = load_config(config_path)
    target = resolve_run_dir(cfg.run_dir, run_dir)
    summary = load_summary(target)
    result = run_selection(summary, build_universe(), cfg.rank)
    click.echo(f"run dir: {target}")
    click.echo(format_pick(result))
    if no_write:
        click.echo("(--no-write: nothing written)")
        return
    paths = write_result(result, cfg, target)
    for name, path in paths.items():
        click.echo(f"wrote {name}: {path}")
    if render:
        out = render_report(QMD, target, quarto=quarto)
        click.echo(f"wrote report: {out}")


if __name__ == "__main__":
    main()
