"""Jobmon orchestrator for the malaria admin-2 forecast (stage 04).

Replaces the raw ``sbatch --array`` R launcher
(``01_forecast_malaria_admin_2s_launcher.r``) with an idd-tools jobmon workflow.

Unlike the 03 model-selection orchestrator -- which hand-rolls its cell/partition
logic because it needs per-``n_smooths`` bundle sizes the library can't yet
express -- stage 04 is the clean case and uses the idd-tools cell layer directly:

  * ONE formulation per invocation. ``--model-run-date`` (e.g. ``2026_07_08_f1``)
    is a run-level parameter, NOT a cell axis; it is both the model-registry key
    and the output dir name. The rare multi-formulation comparison = invoke this
    N times, once per formulation.
  * The cell space is the scenario grid ``{ssp_scenario x dah_scenario}``
    (``build_factorial_cellset``). Every forecast cell is ~25 min -- well above
    the ~5-min bundling floor and homogeneous -- so ``trivial_partition`` emits
    one task per cell. No CALIB table, no bundling, no param_map, no
    ``inflate_cells`` bridge (contrast 03).
  * ``current`` is OPT-IN. Every run writes its dated
    ``<forecast_outputs>/<model_run_date>/`` dir regardless; only ``--set-current``
    appends a ``finalize`` task (``depends_on`` every forecast task) that repoints
    the output node's ``current`` symlink. Without it, the run saves but leaves
    ``current`` untouched.

Worker = forecast_malaria_admin_2s_rocket.r, invoked per task via the IHME
singularity R shell (jobmon only sees a shell command string; the R worker needs
no Python). The rocket reads its cell as 1:1 CLI flags (``--ssp-scenario`` /
``--dah-scenario``) plus the run-level constants. Image / shell / worker paths are
CLI args, so no /ihme or /mnt paths are committed here.

Sensitivities (deferred): a covariate hold-constant sweep is just another cell
axis (``{ssp x dah x hold_constant}``) plus a freeze transform in the rocket;
factorial + trivial_partition absorbs it with no structural change to this file.
"""

from __future__ import annotations

import click

from idd_tools.jobmon import (
    Task,
    TaskManifest,
    TaskTemplateSpec,
    build_factorial_cellset,
    filter_already_done,
    submit_with_manifest,
    trivial_partition,
)

from idd_forecast_mbp import constants as mbpc

# Output node (.../malaria/forecast_outputs/lsae_1285); `current` lives directly
# under it. Taken as the parent of the public read (current) path so no absolute
# path is committed in this file.
OUTPUT_NODE = mbpc.MAL_FORECAST_OUTPUTS_READ_PATH.parent

FORECAST_TEMPLATE = "forecast"
FINALIZE_TEMPLATE = "finalize"

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


def add_finalize_task(manifest, run_level, *, workflow_name):
    """Append a finalize task that depends on every forecast task.

    filter_already_done prunes completed forecast tasks from this task's
    depends_on, so a re-run whose forecasts are all on disk finalizes
    immediately (empty depends_on -> root).
    """
    forecast_ids = [t.task_id for t in manifest.tasks]
    finalize = Task(
        index=len(manifest.tasks),
        task_id="finalize_set_current",
        task_template=FINALIZE_TEMPLATE,
        # Keyed on output_key, not model_run_date: finalize repoints the OUTPUT node's
        # `current`, so it must follow where the run wrote, which --output-key can move
        # away from the registry key.
        task_args={"output_key": run_level["output_key"]},
        depends_on=forecast_ids,
    )
    return TaskManifest(workflow_name=workflow_name, tasks=[*manifest.tasks, finalize])


@click.command()
@click.option("--model-run-date", required=True,
              help="formulation run date = model-registry key AND output dir name, "
                   "e.g. 2026_07_08_f1")
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
@click.option("--set-current/--no-set-current", default=False, show_default=True,
              help="opt-in: append a finalize task that repoints the output node "
                   "`current` symlink to this run")
@click.option("--finalize-worker", type=click.Path(exists=True, dir_okay=False),
              help="path to finalize_malaria_forecast.r (required with --set-current)")
@click.option("--cores", default=10, show_default=True,
              help="in-task mclapply over draws; predict is bandwidth-bound past ~10")
@click.option("--memory", default="60G", show_default=True,
              help="~12G over the measured 47.5G peak; see memory/forecast-04-resourcing")
@click.option("--runtime", default="45m", show_default=True,
              help="<= 2x the ~25-31min measured (project 2x rule)")
@click.option("--probe-only", type=int, default=None,
              help="scope to N tasks for a first run through the new machinery; "
                   "finalize is skipped in probe mode")
@click.option("--max-concurrent", default=500, show_default=True)
@click.option("--project", default="proj_rapidresponse", show_default=True)
@click.option("--queue", default="all.q", show_default=True)
@click.option("--version-tag", default=None,
              help="optional; must be declared in the repo's .jobmon_versions.toml")
@click.option("--output-key", default=None,
              help="output dir name; defaults to --model-run-date. One fitted model "
                   "can write several runs (the covariate-hold sensitivities), each in "
                   "its own dir with unchanged file names. REQUIRED with --hold-covariate.")
@click.option("--hold-covariate", "hold_covariates", multiple=True,
              type=click.Choice(["gdppc", "suitability", "temp", "flood", "dah"]),
              help="repeatable; hold this covariate constant at --hold-year. Does NOT "
                   "change the fitted model, only the forecast trajectory.")
@click.option("--hold-year", default=2023, show_default=True, type=int,
              help="year at which held covariates are frozen")
def main(model_run_date, worker, r_image, r_shell, ssp_scenarios, dah_scenarios,
         forecast_start, forecast_end, rake_year, zero_burden_policy, outcomes,
         set_current, finalize_worker, cores, memory, runtime, probe_only,
         max_concurrent, project, queue, version_tag,
         output_key, hold_covariates, hold_year):
    # A hold run MUST land somewhere other than the baseline's dir, or it overwrites
    # the baseline netCDFs -- the file names carry only (ssp, dah), by design.
    if hold_covariates and not output_key:
        raise click.UsageError(
            "--hold-covariate requires --output-key so the sensitivity does not "
            "overwrite the baseline run's output files"
        )
    run_key = output_key or model_run_date
    run_dir = OUTPUT_NODE / run_key
    workflow_name = f"malaria_forecast_{run_key}"

    run_level = {
        "model_run_date": model_run_date,
        "output_key": run_key,
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

    # `current` is opt-in AND never set from a probe (a partial run must not
    # become the pointed-to version).
    add_finalize = set_current and probe_only is None
    if set_current and probe_only is not None:
        click.echo("[probe] --set-current ignored: finalize is skipped in probe mode.")
    if add_finalize:
        if finalize_worker is None:
            raise click.UsageError("--set-current requires --finalize-worker")
        manifest = add_finalize_task(manifest, run_level, workflow_name=workflow_name)

    hold_note = (
        f" hold={'+'.join(hold_covariates)}@{hold_year}" if hold_covariates else ""
    )
    click.echo(f"model {model_run_date}{hold_note}: {len(ssp_scenarios)} ssp x "
               f"{len(dah_scenarios)} dah (outcomes={outcomes}) -> "
               f"{len(manifest.tasks)} task(s)"
               f"{' (+finalize)' if add_finalize else ''}; run dir {run_dir}")

    # "Done" = the per-cell netCDF and its location-status sidecar exist AND the
    # netCDF carries the variables THIS run's --outcomes asked for. Requiring the
    # sidecar forces a clean re-run after a crash between the two (atomic) writes;
    # the variable check stops an earlier smaller-outcome file (e.g. inc-only)
    # from counting as done for a later `both` run. Finalize always runs (its own
    # idempotent symlink swap); returning False keeps it in the manifest.
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
        click.echo("Nothing to do: all forecasts present and --set-current not requested.")
        return

    # First line is an f-string (baked image/shell/worker paths); the rest are
    # plain strings so the {arg} jobmon placeholders survive verbatim.
    # The hold flags are appended ONLY when a hold is requested. Two reasons: an
    # empty `--hold-covariate ''` would need shell quoting to survive, and jobmon
    # does not guarantee a shell; and a baseline command stays byte-identical to what
    # it was before this flag existed, so baseline task names and done() are unchanged.
    forecast_cmd = (
        f"{r_shell} -i {r_image} -s {worker} "
        "--model-run-date {model_run_date} --output-key {output_key} "
        "--ssp-scenario {ssp_scenario} --dah-scenario {dah_scenario} "
        "--forecast-start {forecast_start} --forecast-end {forecast_end} "
        "--rake-year {rake_year} --zero-burden-policy {zero_burden_policy} "
        "--outcomes {outcomes}"
    )
    forecast_task_args = [
        "model_run_date", "output_key", "forecast_start", "forecast_end",
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
    if add_finalize:
        # run_date passed as an explicit --run-date flag (not a VAR=val env
        # prefix), so the command does not depend on jobmon executing it through
        # a shell. finalize_malaria_forecast.r reads it via optparse.
        # It repoints the node `current`, so it keys off the OUTPUT dir, not the model.
        finalize_cmd = (
            f"{r_shell} -i {r_image} -s {finalize_worker} "
            "--run-date {output_key}"
        )
        templates[FINALIZE_TEMPLATE] = TaskTemplateSpec(
            command_template=finalize_cmd,
            node_args=[],
            task_args=["output_key"],
        )

    def resources(task: Task) -> dict:
        if task.task_template == FINALIZE_TEMPLATE:
            return {"cores": 1, "memory": "2G", "runtime": "5m"}  # symlink swap
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


if __name__ == "__main__":
    main()
