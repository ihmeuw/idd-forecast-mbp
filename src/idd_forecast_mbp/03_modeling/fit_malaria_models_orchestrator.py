"""Jobmon orchestrator for the malaria model-selection workflow.

Two *kinds of cell*, so the in-sample fit isn't redone per OOS experiment:
  - IS cell        : one per spec  -> 1 fit on all data (is_* metrics + summary).
  - OOS-exp cell   : one per (spec x OOS experiment) -> that experiment's folds
                     only, FE-present (oos_*/cv_* metrics + per-fold summaries).

So a spec that goes through IS + the OOS experiments below is
  1 IS fit + sum(folds over experiments)   -- IS fit computed ONCE per spec.

The OOS experiments are defined in OOS_VERSIONS (edit the years there). Cells are
bundled into serial tasks per (cell, n_smooths) so the one-time data load is
amortized; bundle size + runtime come from the CALIB table (edit there, regenerate
from a probe). The orchestrator reads spec_table and fans out, so it scales with
the spec count with no code change.

Worker = select_malaria_models_rocket.r, invoked per task via the IHME
singularity R shell. jobmon only sees a shell command string; the R worker needs
no Python. Inputs the R prep must have written into --output-dir:
  spec_table.parquet       spec_index, n_smooths, n_scams, formula_text
                           (from malaria_spec_design.py; the worker fits formula_text)

Aggregation (IS + experiments -> per-spec) is a post-hoc join on spec_index, no
jobmon stage needed. Image/shell/worker are CLI args (no committed /ihme paths).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd

from idd_tools.jobmon import (
    Task,
    TaskManifest,
    TaskTemplateSpec,
    build_hierarchical_cellset,
    filter_already_done,
    rectangular_partition,
    submit_with_manifest,
)

from idd_forecast_mbp import constants as mbpc

MODELING_YEARS = mbpc.MODELING_YEARS


# --- OOS experiments (edit the years here) --------------------------------
# FE-only run: max_lag=0 so train_lo=MODELING_YEARS[0] (full data, no lag-availability
# reservation). This resurrects the vwide_preC/vwide_full windows the 2003 start had
# excluded -> 10 temporal windows. MUST match the worker's `lags` (empty) -- the worker
# asserts length(lags)==0 iff max_lag==0. Passed to the worker via --max-lag below.
max_lag = 0
min_training_years = 5
train_lo = MODELING_YEARS[0] + max_lag

gaps = {
    'narrow': 1,
    'wide': 3,
    'vwide': 7,
    'exwide': 10,
}

test_windows = {
    'preC': (2013, 2019),
    'full': (2013, MODELING_YEARS[-1]),
    'recent': (2019, MODELING_YEARS[-1]),
}

OOS_VERSIONS = []
for gap_name, gap_years in gaps.items():
    for test_name, (test_lo, test_hi) in test_windows.items():
        train_hi = test_lo - gap_years
        if train_hi - train_lo < min_training_years - 1:
            continue
        OOS_VERSIONS.append({
            "name": f"{gap_name}_{test_name}",
            "strategy": "temporal",
            "train_lo": train_lo,
            "train_hi": train_hi,
            "test_lo": test_lo,
            "test_hi": test_hi,
        })

# --- Resources (domain values; machinery is idd-tools cells/partition below) --------
# `rectangular_partition` takes ONE `max_per_task`, so bundling is uniform per template
# rather than per-n_smooths like the old CALIB table. The resources() callable in main()
# recovers n_smooths-awareness by sizing each task's runtime from PER_SPEC_SEC x n_specs.
# The resources() callable recovers n_smooths-awareness by sizing each task's runtime from
# PER_SPEC_SEC x n_specs.
MEM = {"is_cell": "6G", "oos_temporal": "5G", "oos_random": "6G"}  # per-template mem ask; census peak RSS ~4.65/4.34G
MAX_PER_TASK = {"is_cell": 4, "oos_temporal": 3, "oos_random": 1}  # uniform serial-bundle chunk / template
# per-spec wall-seconds upper bound by (template, n_smooths), CALIBRATED from the full census
# run (wf 599407, dir 20260710_efs; 17,820 per-spec fit times) = max observed per-spec fit
# elapsed per tier. Supersedes the earlier small-probe (598876/598878) values, which
# over-allocated ~1.3-20x. Regenerate from any completed run's select_summary_*
# {is,cv}_elapsed_sec grouped by (template, n_smooths).
PER_SPEC_SEC = {
    ("is_cell", 0): 5,   ("is_cell", 1): 172, ("is_cell", 2): 236, ("is_cell", 3): 316,
    ("is_cell", 4): 396, ("is_cell", 5): 307, ("is_cell", 6): 284, ("is_cell", 7): 168,
    ("oos_temporal", 0): 5,   ("oos_temporal", 1): 115, ("oos_temporal", 2): 203,
    ("oos_temporal", 3): 221, ("oos_temporal", 4): 284, ("oos_temporal", 5): 251,
    ("oos_temporal", 6): 250, ("oos_temporal", 7): 186,
}
PER_SPEC_DEFAULT = 450   # unmeasured (template, n_smooths); census overall max ~396
LOAD_SEC = 120           # one-time per-task data load + jobmon overhead (not captured in per-spec fit time)

IS_TEMPLATE = "is_cell"
OOS_BY_NAME = {v["name"]: v for v in OOS_VERSIONS}   # window name -> its train/test bounds
FINALIZE_SCRIPT = Path(__file__).resolve().parent / "finalize_selection_run.py"


def select_probe_specs(spec_table: pd.DataFrame, n_per_level: int) -> pd.DataFrame:
    """N specs per distinct (n_scams, n_smooths) cell (deterministic) — for a small test
    run that spans all three fitting engines. Grouping on (n_scams, n_smooths) rather than
    n_smooths alone guarantees coverage of every engine x complexity combination the worker
    dispatches on: lm (0,0), gam (0,>=1: unconstrained smooths only), and scam (>=1 scam
    term) at each smooth/scam count. n_smooths-only grouping can miss an engine entirely
    (e.g. two scam specs fill the n_smooths=1 level and the lone gam spec never gets probed)."""
    grp = ["n_scams", "n_smooths"] if "n_scams" in spec_table.columns else ["n_smooths"]
    return (
        spec_table.sort_values("spec_index")
        .groupby(grp, sort=True)
        .head(n_per_level)
        .reset_index(drop=True)
    )


def build_cellsets(selected: pd.DataFrame, cv_strategy: str = "temporal"):
    """One idd-tools cell per (spec, experiment). temporal: an IS cell + one per temporal
    window. random: OOS-only, a single 'random' cell per spec (k-fold CV inside the worker),
    no IS. Returns (is_cellset, oos_cellset), each rectangular-partitioned under its template.
    `n_scams` rides on every cell so partition can split gam from scam."""
    is_rows: list[dict] = []
    oos_rows: list[dict] = []
    for _, r in selected.iterrows():
        base = {"n_smooths": int(r["n_smooths"]), "n_scams": int(r["n_scams"]),
                "spec_index": int(r["spec_index"])}
        if cv_strategy == "temporal":
            is_rows.append({"cell": "IS", **base})
            for v in OOS_VERSIONS:   # all temporal in the current OOS_VERSIONS config
                oos_rows.append({"cell": v["name"], **base})
        else:   # random k-fold: OOS-only, one 'random' cell per spec
            oos_rows.append({"cell": "random", **base})
    axes = ["cell", "n_smooths", "n_scams", "spec_index"]
    is_cs = build_hierarchical_cellset(is_rows, axes=axes) if is_rows else None
    oos_cs = build_hierarchical_cellset(oos_rows, axes=axes) if oos_rows else None
    return is_cs, oos_cs


def _feature_fn(group_key: dict) -> dict:
    return {"cell": group_key["cell"], "n_smooths": group_key["n_smooths"],
            "n_scams": group_key["n_scams"]}


def _task_id_fn(group_key: dict, chunk_idx: int) -> str:
    # keep the cell parseable as `task_id.rsplit('_n', 1)[0]` (finalize/notebook rely on it);
    # n_scams goes after `_n` so it doesn't disturb that split.
    return f"{group_key['cell']}_n{group_key['n_smooths']}_s{group_key['n_scams']}_bin{chunk_idx}"


def partition_all(is_cs, oos_cs, *, workflow_name: str, oos_template: str = "oos_temporal") -> TaskManifest:
    """rectangular_partition each cellset under its template (fix on cell/n_smooths/n_scams,
    uniform chunk = MAX_PER_TASK) and combine into ONE analytic manifest. Each task carries
    its cell list under task_args['cells']; that survives only in the SAVED manifest (the R
    worker reads it by task_id). `n_specs` is stamped onto task_features for resources()."""
    collected: list[Task] = []
    for cs, tmpl in ((is_cs, "is_cell"), (oos_cs, oos_template)):
        if cs is None:
            continue
        m = rectangular_partition(
            cs, fix=["cell", "n_smooths", "n_scams"], workflow_name=workflow_name,
            task_template=tmpl, max_per_task=MAX_PER_TASK[tmpl],
            task_id_fn=_task_id_fn, features_fn=_feature_fn)
        collected.extend(m.tasks)
    tasks = [
        Task(index=i, task_id=t.task_id, task_template=t.task_template,
             task_args=t.task_args, depends_on=t.depends_on, shared_axes=t.shared_axes,
             task_features={**t.task_features, "n_specs": len(t.task_args["cells"])})
        for i, t in enumerate(collected)
    ]
    return TaskManifest(workflow_name=workflow_name, tasks=tasks)


def to_submit_manifest(analytic: TaskManifest, *, output_dir: Path, manifest_path: Path,
                       optimizer: str, maxit: int) -> TaskManifest:
    """Command-args-only manifest for submit_with_manifest: drop the 'cells' list (jobmon
    create_task(**task_args) can't take it) and add the args the R worker's command needs.
    The worker recovers its cell list from the saved manifest by --task-id."""
    subm: list[Task] = []
    for t in analytic.tasks:
        args = {"task_id": t.task_id, "output_dir": str(output_dir),
                "manifest": str(manifest_path), "optimizer": optimizer, "maxit": int(maxit)}
        if t.task_template == "oos_temporal":
            v = OOS_BY_NAME[t.task_args["cell"]]
            args.update({"train_lo": int(v["train_lo"]), "train_hi": int(v["train_hi"]),
                         "test_lo": int(v["test_lo"]), "test_hi": int(v["test_hi"])})
        subm.append(Task(index=t.index, task_id=t.task_id, task_template=t.task_template,
                         task_args=args, task_features=t.task_features,
                         depends_on=t.depends_on, shared_axes=t.shared_axes))
    return TaskManifest(workflow_name=analytic.workflow_name, tasks=subm)


@click.command()
@click.option("--spec-table", required=True, type=click.Path(exists=True, dir_okay=False),
              help="parquet: spec_index, n_smooths, n_scams, formula_text (from malaria_spec_design.py)")
@click.option("--output-dir", required=True, type=click.Path(file_okay=False),
              help="run root (holds spec_table.parquet + the written manifest)")
@click.option("--worker", required=True, type=click.Path(exists=True, dir_okay=False),
              help="path to select_malaria_models_rocket.r")
@click.option("--r-image", required=True, help="singularity .img for R")
@click.option("--r-shell", required=True, help="execRscript.sh wrapper")
@click.option("--probe/--full", default=False, show_default=True,
              help="probe = N specs per n_smooths level (test run); full = whole neighborhood")
@click.option("--n-per-level", default=2, show_default=True, help="probe: specs per n_smooths level")
@click.option("--optimizer", default="efs", show_default=True)
@click.option("--maxit", default=30, show_default=True,
              help="EFS max iterations. Fits freeze by ~20 iters here; 300 only "
                   "spins on constrained smooths whose sp collapses to a boundary.")
@click.option("--cores", default=16, show_default=True)
@click.option("--max-concurrent", default=500, show_default=True)
@click.option("--project", default="proj_rapidresponse", show_default=True)
@click.option("--queue", default="all.q", show_default=True)
@click.option("--cv-strategy", type=click.Choice(["temporal", "random"]), default="temporal",
              show_default=True, help="temporal = IS + temporal-window OOS; random = OOS-only k-fold")
@click.option("--cv-n-folds", default=10, show_default=True, help="random: number of CV folds")
def main(spec_table, output_dir, worker, r_image, r_shell, probe, n_per_level,
         optimizer, maxit, cores, max_concurrent, project, queue, cv_strategy, cv_n_folds):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = pd.read_parquet(spec_table)
    selected = (select_probe_specs(specs, n_per_level) if probe
                else specs.sort_values("spec_index").reset_index(drop=True))

    # cells -> partition -> analytic manifest (each task carries its cell list). Save it so
    # the R worker can recover its spec list by --task-id (idd-tools inflate_cells is Python).
    oos_template = "oos_temporal" if cv_strategy == "temporal" else "oos_random"
    is_cs, oos_cs = build_cellsets(selected, cv_strategy)
    analytic = partition_all(is_cs, oos_cs, workflow_name="malaria_select", oos_template=oos_template)
    manifest_path = output_dir / "manifest.json"
    analytic.save(manifest_path)
    cell_names = (["IS"] + [v["name"] for v in OOS_VERSIONS]) if cv_strategy == "temporal" \
        else [f"random({cv_n_folds}-fold)"]
    click.echo(f"{'Probe' if probe else 'Full'} [{cv_strategy}]: {len(selected)} specs -> "
               f"{len(analytic.tasks)} tasks (cells: {cell_names}); manifest -> {manifest_path}")

    # done-check: select_summary_<task_id>.parquet has a row for every spec in the task's cells.
    expected = {t.task_id: {int(c["spec_index"]) for c in t.task_args["cells"]}
                for t in analytic.tasks}

    # BLAS/OpenMP threads pinned to cores (no --cleanenv on the wrapper, so a plain env
    # prefix reaches R inside the container).
    threads = int(cores)
    prefix = f"OPENBLAS_NUM_THREADS={threads} OMP_NUM_THREADS={threads} "
    # --write-summary FALSE: summary.scam's testStat throws/segfaults on degenerate smooths;
    # metrics come from direct field access, so summaries are never needed here.
    base = (prefix + f"{r_shell} -i {r_image} -s {worker} "
            "--task-id {task_id} --output-dir {output_dir} --manifest {manifest} "
            f"--optimizer {{optimizer}} --maxit {{maxit}} --max-lag {max_lag} --write-summary FALSE ")
    common_args = ["output_dir", "manifest", "optimizer", "maxit"]
    if cv_strategy == "temporal":
        oos_tmpl = TaskTemplateSpec(
            command_template=base + ("--cv-strategy temporal "
                                     "--train-lo {train_lo} --train-hi {train_hi} "
                                     "--test-lo {test_lo} --test-hi {test_hi} "
                                     "--fit-is-fe FALSE --fit-oos TRUE"),
            node_args=["task_id"],
            task_args=common_args + ["train_lo", "train_hi", "test_lo", "test_hi"])
    else:   # random k-fold OOS: no year bounds; k folds inside the worker
        oos_tmpl = TaskTemplateSpec(
            command_template=base + f"--cv-strategy random --cv-n-folds {cv_n_folds} "
                                    "--fit-is-fe FALSE --fit-oos TRUE",
            node_args=["task_id"], task_args=common_args)
    templates = {
        "is_cell": TaskTemplateSpec(
            command_template=base + "--fit-is-fe TRUE --fit-oos FALSE",
            node_args=["task_id"], task_args=common_args),
        oos_template: oos_tmpl,
        # finalize: a light Python task (env python via sys.executable, on shared storage so
        # compute nodes can run it) joining per-cell outputs -> selection_summary.parquet.
        "finalize": TaskTemplateSpec(
            command_template=f"{sys.executable} {FINALIZE_SCRIPT} --run-dir {{output_dir}}",
            node_args=["output_dir"], task_args=[]),
    }

    # command-args-only manifest (drop the cells list; add worker command args + bounds)
    submit_manifest = to_submit_manifest(analytic, output_dir=output_dir,
                                         manifest_path=manifest_path,
                                         optimizer=optimizer, maxit=maxit)

    def spec_done(task: Task) -> bool:
        p = output_dir / f"select_summary_{task.task_id}.parquet"
        if not p.exists():
            return False
        col = "is_r_sq" if task.task_template == "is_cell" else "oos_r_sq"
        try:
            df = pd.read_parquet(p, columns=["spec_index", col])
        except Exception:
            return False  # missing column / partial / old schema -> re-run
        return expected.get(task.task_id, set()).issubset(set(df["spec_index"].astype(int)))

    submit_manifest = filter_already_done(submit_manifest, spec_done)

    # Finalize stage (temporal full runs only; the probe is a fit-only smoke test, and random
    # OOS is read straight from the per-spec files). One task joining every per-cell
    # select_summary_* into selection_summary.parquet, depending on all surviving fit tasks.
    if not probe and cv_strategy == "temporal":
        fit_ids = [t.task_id for t in submit_manifest.tasks]
        finalize_task = Task(index=len(submit_manifest.tasks), task_id="finalize",
                             task_template="finalize", task_args={"output_dir": str(output_dir)},
                             depends_on=fit_ids)
        submit_manifest = TaskManifest(workflow_name=submit_manifest.workflow_name,
                                       tasks=[*submit_manifest.tasks, finalize_task])
        click.echo(f"+ finalize task depends_on {len(fit_ids)} fit tasks")

    # runtime = one-time load + (per-spec seconds x n_specs x 1.15 contention buffer), 8-min floor.
    def resources(task: Task) -> dict:
        if task.task_template == "finalize":
            return {"memory": "10G", "runtime": "20m", "cores": 1}
        tmpl = task.task_template
        ns = int(task.task_features["n_smooths"])
        n_specs = int(task.task_features["n_specs"])
        if tmpl == "oos_random":
            # k folds x one full-data fit each (~ IS per-spec cost) per spec, 1.15 buffer.
            per_fit = PER_SPEC_SEC.get(("is_cell", ns), PER_SPEC_DEFAULT)
            rt_min = max(8, -(-int(LOAD_SEC + cv_n_folds * per_fit * n_specs * 1.15) // 60))
            return {"memory": MEM["oos_random"], "runtime": f"{rt_min}m", "cores": threads}
        per = PER_SPEC_SEC.get((tmpl, ns), PER_SPEC_DEFAULT)
        rt_min = max(8, -(-int(LOAD_SEC + per * n_specs * 1.15) // 60))
        return {"memory": MEM[tmpl], "runtime": f"{rt_min}m", "cores": threads}

    result = submit_with_manifest(
        submit_manifest,
        output_dir=output_dir,
        templates=templates,
        resources=resources,
        concurrency_limit=int(max_concurrent),
        project=project,
        queue=queue,
        tool_name="idd-forecast-mbp",
        log_method=click.echo,
    )
    click.echo(
        f"workflow {result.workflow_id} status {result.status} "
        f"({result.n_tasks_submitted} tasks); run record at {result.run_record_path}"
    )


if __name__ == "__main__":
    main()
