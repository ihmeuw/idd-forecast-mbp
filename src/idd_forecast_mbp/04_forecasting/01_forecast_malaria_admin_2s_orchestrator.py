"""Jobmon orchestrator for the malaria admin-2 forecast (stage 04).

Replaces the raw ``sbatch --array`` R launcher
(``01_forecast_malaria_admin_2s_launcher.r``) with an idd-tools jobmon workflow.

Unlike the 03 model-selection orchestrator -- which hand-rolls its cell/partition
logic because it needs per-``n_smooths`` bundle sizes the library can't yet
express -- stage 04 is the clean case and uses the idd-tools cell layer directly:

  * ONE fitted model per invocation. ``--model-version`` names a snapshot or label
    on the fitted-models node (default: the one that is ``current``); its directory
    is resolved here and passed to the workers as ``--model-dir``. The rare
    multi-model comparison = invoke this N times, once per model.
  * The cell space is the scenario grid ``{ssp_scenario x dah_scenario}``
    (``build_factorial_cellset``). Every forecast cell is ~25 min -- well above
    the ~5-min bundling floor and homogeneous -- so ``trivial_partition`` emits
    one task per cell. No CALIB table, no bundling, no param_map, no
    ``inflate_cells`` bridge (contrast 03).
  * Output goes to the forecast_outputs node's ``working/`` slot (``idd_tools.versions``;
    ``--scratch LABEL`` for a test run). Nothing is frozen unless the launch said so:
    ``--current --description "..."`` freezes and promotes on success, ``--freeze``
    freezes without promoting, ``--label NAME`` names the snapshot (the model key and
    any hold arm belong in the label; file names carry only (ssp, dah)). ONE run per
    slot: finish a run before launching the next into the same node. A probe never
    freezes.

Worker = forecast_malaria_admin_2s_rocket.r, invoked per task via the IHME
singularity R shell (jobmon only sees a shell command string; the R worker needs
no Python). The rocket reads its cell as 1:1 CLI flags (``--ssp-scenario`` /
``--dah-scenario``) plus the run-level constants. Image / shell / worker paths are
CLI args, so no /ihme or /mnt paths are committed here.

Sensitivities: a covariate hold (``--hold-covariate``) is a separate run of the same
model; launch it with its own ``--label`` after the baseline run has been finished.
"""

from __future__ import annotations

import click

from idd_tools.jobmon import (
    Task,
    TaskTemplateSpec,
    build_factorial_cellset,
    filter_already_done,
    submit_with_manifest,
    trivial_partition,
)
from idd_tools.versions import VersionsOptions, resolve_target, versions_options

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.versioning import finish_stage

# Output node (.../malaria/forecast_outputs/lsae_1285); `current` lives directly
# under it. Taken as the parent of the public read (current) path so no absolute
# path is committed in this file.
OUTPUT_NODE = mbpc.MAL_FORECAST_OUTPUTS_READ_PATH.parent

FORECAST_TEMPLATE = "forecast"

# Which netCDF variables each --outcomes choice must produce. pfpr always runs
# (it feeds inc/mort) but is never written, so it is not an outcome here.
OUTCOME_VARS = {
    "inc":  ["log_malaria_inc_rate_pred"],
    "mort": ["log_malaria_mort_rate_pred"],
    "both": ["log_malaria_inc_rate_pred", "log_malaria_mort_rate_pred"],
}


def _nc_variables(path) -> set:
    """Variable names in a netCDF file (metadata only — no data read)."""
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        return set(ds.variables.keys())


def forecast_task_id(cell: dict) -> str:
    """Stable, human-readable id per scenario cell (= log/nc suffix key)."""
    return f"forecast_{cell['ssp_scenario']}_{cell['dah_scenario']}"


def build_forecast_manifest(ssp_scenarios, dah_scenarios, run_level, *, workflow_name):
    """Factorial cell space over the scenario grid -> one task per (ssp, dah).

    The run-level constants (model_run_date, years, rake-year, policy) are
    identical for every cell, so they are merged into each task's task_args
    after partitioning rather than carried as pseudo-axes of the cell space.
    """
    cellset = build_factorial_cellset({
        "ssp_scenario": list(ssp_scenarios),
        "dah_scenario": list(dah_scenarios),
    })
    manifest = trivial_partition(
        cellset,
        workflow_name=workflow_name,
        task_template=FORECAST_TEMPLATE,
        task_id_fn=forecast_task_id,
    )
    for t in manifest.tasks:
        t.task_args.update(run_level)
    return manifest


@click.command()
@click.option("--model-version", default=None,
              help="snapshot name or label on the fitted-models node "
                   "(default: the model that is current)")
@click.option("--worker", required=True, type=click.Path(exists=True, dir_okay=False),
              help="path to forecast_malaria_admin_2s_rocket.r")
@click.option("--r-image", required=True, help="singularity .img for R")
@click.option("--r-shell", required=True, help="execRscript.sh wrapper")
@click.option("--ssp-scenario", "ssp_scenarios", multiple=True,
              default=("ssp126", "ssp245", "ssp585"), show_default=True,
              help="repeatable; all three SSPs run by default")
@click.option("--dah-scenario", "dah_scenarios", multiple=True,
              default=("Baseline",), show_default=True, help="repeatable")
@click.option("--forecast-start", default=2023, show_default=True, type=int)
@click.option("--forecast-end", default=2100, show_default=True, type=int)
@click.option("--rake-year", default="2023", show_default=True,
              help="4-digit scalar OR path to a per-loc rake-year parquet")
@click.option("--zero-burden-policy", default="drop", show_default=True,
              type=click.Choice(["drop", "impute"]))
@click.option("--outcomes", default="both", show_default=True,
              type=click.Choice(["inc", "mort", "both"]),
              help="which outcomes to predict+write; pfpr always runs (it feeds both)")
@click.option("--cores", default=10, show_default=True,
              help="in-task mclapply over draws; predict is bandwidth-bound past ~10")
@click.option("--memory", default="60G", show_default=True,
              help="~12G over the measured 47.5G peak; see memory/forecast-04-resourcing")
@click.option("--runtime", default="45m", show_default=True,
              help="<= 2x the ~25-31min measured (project 2x rule)")
@click.option("--probe-only", type=int, default=None,
              help="scope to N tasks for a first run through the new machinery; "
                   "a probe is never frozen")
@click.option("--max-concurrent", default=500, show_default=True)
@click.option("--project", default="proj_rapidresponse", show_default=True)
@click.option("--queue", default="all.q", show_default=True)
@click.option("--version-tag", default=None,
              help="optional; must be declared in the repo's .jobmon_versions.toml")
@click.option("--output-key", default=None, hidden=True,
              help="gone: the run's identity is the --label it is frozen under")
@click.option("--hold-covariate", "hold_covariates", multiple=True,
              type=click.Choice(["gdppc", "suitability", "temp", "flood", "dah"]),
              help="repeatable; hold this covariate constant at --hold-year. Does NOT "
                   "change the fitted model, only the forecast trajectory.")
@click.option("--hold-year", default=2023, show_default=True, type=int,
              help="year at which held covariates are frozen")
@versions_options
def main(model_version, worker, r_image, r_shell, ssp_scenarios, dah_scenarios,  # noqa: PLR0913
         forecast_start, forecast_end, rake_year, zero_burden_policy, outcomes,
         cores, memory, runtime, probe_only,
         max_concurrent, project, queue, version_tag,
         output_key, hold_covariates, hold_year, versions: VersionsOptions):
    if output_key:
        raise click.UsageError(
            "--output-key is gone: the run writes to the node's working/ slot and its "
            "identity is the --label it is frozen under (idd_tools.versions)"
        )
    # A hold run reuses the baseline's file names, so it needs its own snapshot name.
    if hold_covariates and versions.finishes and not versions.label:
        raise click.UsageError("--hold-covariate with --freeze/--current needs --label naming the arm")
    if probe_only is not None and versions.finishes:
        raise click.UsageError("a --probe-only run is never frozen; drop --freeze/--current")
    model_dir = mbpc.malaria_model_dir(model_version)
    model_name = model_version or model_dir.name
    run_dir = resolve_target(OUTPUT_NODE, versions)   # working/ or scratch/<label>/
    run_key = versions.label or model_name
    workflow_name = f"malaria_forecast_{run_key}"

    run_level = {
        "model_dir": str(model_dir),
        "model_version": model_name,
        "out_dir": str(run_dir),
        "forecast_start": int(forecast_start),
        "forecast_end": int(forecast_end),
        "rake_year": str(rake_year),
        "zero_burden_policy": zero_burden_policy,
        "outcomes": outcomes,
    }
    # Only present when a hold is requested, so a baseline run's task args are exactly
    # what they were before this flag existed.
    if hold_covariates:
        run_level["hold_covariate"] = ",".join(hold_covariates)
        run_level["hold_year"] = int(hold_year)

    manifest = build_forecast_manifest(ssp_scenarios, dah_scenarios, run_level,
                                       workflow_name=workflow_name)

    hold_note = (
        f" hold={'+'.join(hold_covariates)}@{hold_year}" if hold_covariates else ""
    )
    finish_note = (
        f"; on success: {'freeze + promote' if versions.current else 'freeze'}"
        f"{f' as {versions.label}' if versions.label else ''}"
        if versions.finishes else "; nothing frozen at the end"
    )
    click.echo(f"model {model_name} ({model_dir}){hold_note}: {len(ssp_scenarios)} ssp x "
               f"{len(dah_scenarios)} dah (outcomes={outcomes}) -> "
               f"{len(manifest.tasks)} task(s); writing to {run_dir}{finish_note}")

    # "Done" = the per-cell netCDF and its location-status sidecar exist AND the
    # netCDF carries the variables THIS run's --outcomes asked for. Requiring the
    # sidecar forces a clean re-run after a crash between the two (atomic) writes;
    # the variable check stops an earlier smaller-outcome file (e.g. inc-only)
    # from counting as done for a later `both` run.
    def done(task: Task) -> bool:
        if task.task_template != FORECAST_TEMPLATE:
            return False
        stem = (f"malaria_forecast_{task.task_args['ssp_scenario']}"
                f"_{task.task_args['dah_scenario']}")
        nc = run_dir / f"{stem}.nc"
        sidecar = run_dir / f"{stem}_location_status.parquet"
        if not (nc.exists() and sidecar.exists()):
            return False
        want = set(OUTCOME_VARS[task.task_args["outcomes"]])
        return want.issubset(_nc_variables(nc))

    manifest = filter_already_done(manifest, done)
    if not manifest.tasks:
        click.echo("All forecasts already present in the slot.")
        finished = finish_stage(OUTPUT_NODE, versions) if probe_only is None else None
        if finished is None:
            click.echo(f"Nothing frozen; freeze by hand with `idd-versions {OUTPUT_NODE} freeze` if wanted.")
        return

    # First line is an f-string (baked image/shell/worker paths); the rest are
    # plain strings so the {arg} jobmon placeholders survive verbatim.
    # The hold flags are appended ONLY when a hold is requested. Two reasons: an
    # empty `--hold-covariate ''` would need shell quoting to survive, and jobmon
    # does not guarantee a shell; and a baseline command stays byte-identical to what
    # it was before this flag existed, so baseline task names and done() are unchanged.
    forecast_cmd = (
        f"{r_shell} -i {r_image} -s {worker} "
        "--model-dir {model_dir} --model-version {model_version} --out-dir {out_dir} "
        "--ssp-scenario {ssp_scenario} --dah-scenario {dah_scenario} "
        "--forecast-start {forecast_start} --forecast-end {forecast_end} "
        "--rake-year {rake_year} --zero-burden-policy {zero_burden_policy} "
        "--outcomes {outcomes}"
    )
    forecast_task_args = [
        "model_dir", "model_version", "out_dir", "forecast_start", "forecast_end",
        "rake_year", "zero_burden_policy", "outcomes",
    ]
    if hold_covariates:
        forecast_cmd += " --hold-covariate {hold_covariate} --hold-year {hold_year}"
        forecast_task_args += ["hold_covariate", "hold_year"]

    templates = {
        FORECAST_TEMPLATE: TaskTemplateSpec(
            command_template=forecast_cmd,
            node_args=["ssp_scenario", "dah_scenario"],   # vary per task
            task_args=forecast_task_args,                 # constant across the run
        ),
    }
    def resources(task: Task) -> dict:
        return {"cores": int(cores), "memory": memory, "runtime": runtime}

    result = submit_with_manifest(
        manifest,
        output_dir=run_dir,
        templates=templates,
        resources=resources,
        concurrency_limit=int(max_concurrent),
        probe_only=probe_only,
        project=project,
        queue=queue,
        version_tag=version_tag,
        tool_name="idd-forecast-mbp",
        log_method=click.echo,
    )
    click.echo(
        f"workflow {result.workflow_id} status {result.status} "
        f"({result.n_tasks_submitted} tasks); run record at {result.run_record_path}"
    )
    # The success path: freeze (and promote) only when every task finished ("D") and
    # this was not a probe. Anything else leaves working/ as it is for inspection.
    if result.status != "D":
        click.echo(f"workflow status {result.status!r}: nothing frozen; working/ left as is.")
        return
    if probe_only is not None:
        click.echo("probe run: nothing frozen.")
        return
    if finish_stage(OUTPUT_NODE, versions) is None:
        click.echo(f"run complete in {run_dir}; freeze with `idd-versions {OUTPUT_NODE} freeze '<why>' [--current] [--label L]`.")


if __name__ == "__main__":
    main()
