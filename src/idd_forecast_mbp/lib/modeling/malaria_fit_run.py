"""Submit malaria model fits, one spec table across in-sample and out-of-sample cells, as a jobmon workflow.

This is the body of ``03_modeling/fit_malaria_models_orchestrator.py`` as an importable function,
so that another repo can drive refits of the malaria models (a term dropped, a constraint
relaxed, a subset of terms) on a chosen row set and, if it asks, keep every fitted object.

Two kinds of cell, so the in-sample fit is not redone per out-of-sample experiment: one IS cell
per spec, and one OOS cell per (spec, temporal window) or a single random k-fold cell per spec.
Cells sharing (cell, n_smooths, n_scams) are bundled into serial tasks so the one-time data
load is amortised; each task's runtime is sized from a per-spec table. The worker is
``select_malaria_models_rocket.r``, run through the IHME singularity R shell; it reads its cell
list from the saved manifest by ``--task-id`` and fits ``formula_text`` verbatim.

The function writes only under ``output_dir`` and refuses a directory that already holds a
selection result, that is a registered snapshot or the ``current`` target of its node, or that
lies under the fitted-models node.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from idd_tools.jobmon import (
    Task,
    TaskManifest,
    TaskTemplateSpec,
    WorkflowResult,
    build_hierarchical_cellset,
    filter_already_done,
    rectangular_partition,
    submit_with_manifest,
)
from idd_tools.versions import read_registry

import idd_forecast_mbp
from idd_forecast_mbp import constants as mbpc

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from idd_tools.jobmon.manifest import CellSet

PACKAGE_DIR = Path(idd_forecast_mbp.__file__).resolve().parent
FINALIZE_SCRIPT = PACKAGE_DIR / "03_modeling" / "finalize_selection_run.py"
DEFAULT_WORKER = PACKAGE_DIR / "03_modeling" / "select_malaria_models_rocket.r"
DEFAULT_PREP_SCRIPT = PACKAGE_DIR / "lib" / "malaria_fit_frame.R"

IS_TEMPLATE = "is_cell"
OOS_TEMPLATES = {"temporal": "oos_temporal", "random": "oos_random"}
SPEC_TABLE_COLUMNS = ("spec_index", "n_smooths", "n_scams", "formula_text")
SPEC_TABLE_NAME = "spec_table.parquet"
MANIFEST_NAME = "manifest.json"
SUMMARY_NAME = "selection_summary.parquet"
SELECTION_RESULT_NAME = "selection_result.json"
CvStrategy = Literal["temporal", "random"]


# --- out-of-sample windows -----------------------------------------------------------------


@dataclass(frozen=True)
class OosWindow:
    """One temporal split: fit on [train_lo, train_hi], predict [test_lo, test_hi]."""

    name: str
    train_lo: int
    train_hi: int
    test_lo: int
    test_hi: int


DEFAULT_GAPS: Mapping[str, int] = {"narrow": 1, "wide": 3, "vwide": 7, "exwide": 10}


def default_test_windows(
    modeling_years: Sequence[int] = mbpc.MODELING_YEARS,
) -> dict[str, tuple[int, int]]:
    last = int(modeling_years[-1])
    return {"preC": (2013, 2019), "full": (2013, last), "recent": (2019, last)}


def temporal_windows(
    modeling_years: Sequence[int] = mbpc.MODELING_YEARS,
    gaps: Mapping[str, int] = DEFAULT_GAPS,
    test_windows: Mapping[str, tuple[int, int]] | None = None,
    *,
    min_training_years: int = 5,
    max_lag: int = 0,
) -> list[OosWindow]:
    """Every (gap, test window) pair with at least ``min_training_years`` of training data.

    With the defaults this is the ten windows the 2026-07 selection run used. ``max_lag``
    reserves the first years for lag availability; the worker asserts it against its own lags.
    """
    test_windows = (
        default_test_windows(modeling_years) if test_windows is None else test_windows
    )
    train_lo = int(modeling_years[0]) + max_lag
    out: list[OosWindow] = []
    for gap_name, gap_years in gaps.items():
        for test_name, (test_lo, test_hi) in test_windows.items():
            train_hi = test_lo - gap_years
            if train_hi - train_lo < min_training_years - 1:
                continue
            out.append(
                OosWindow(
                    f"{gap_name}_{test_name}", train_lo, train_hi, test_lo, test_hi
                )
            )
    return out


# --- resources ------------------------------------------------------------------------------

# per-spec wall-seconds upper bound by (template, n_smooths), calibrated from the full census
# run (wf 599407, dir 20260710_efs; 17,820 per-spec fit times).
PER_SPEC_SEC: Mapping[tuple[str, int], int] = {
    ("is_cell", 0): 5,
    ("is_cell", 1): 172,
    ("is_cell", 2): 236,
    ("is_cell", 3): 316,
    ("is_cell", 4): 396,
    ("is_cell", 5): 307,
    ("is_cell", 6): 284,
    ("is_cell", 7): 168,
    ("oos_temporal", 0): 5,
    ("oos_temporal", 1): 115,
    ("oos_temporal", 2): 203,
    ("oos_temporal", 3): 221,
    ("oos_temporal", 4): 284,
    ("oos_temporal", 5): 251,
    ("oos_temporal", 6): 250,
    ("oos_temporal", 7): 186,
}


@dataclass(frozen=True)
class FitRunResources:
    """Cluster asks and bundling knobs; the defaults are the 2026-07 selection run's."""

    cores: int = 16
    memory: Mapping[str, str] = field(
        default_factory=lambda: {
            "is_cell": "6G",
            "oos_temporal": "5G",
            "oos_random": "6G",
        }
    )
    max_per_task: Mapping[str, int] = field(
        default_factory=lambda: {"is_cell": 4, "oos_temporal": 3, "oos_random": 1}
    )
    per_spec_sec: Mapping[tuple[str, int], int] = field(
        default_factory=lambda: dict(PER_SPEC_SEC)
    )
    per_spec_default: int = 450
    load_sec: int = 120
    contention: float = 1.15
    min_runtime_min: int = 8
    finalize: Mapping[str, str | int] = field(
        default_factory=lambda: {"memory": "10G", "runtime": "20m", "cores": 1}
    )
    max_concurrent: int = 500
    project: str = "proj_rapidresponse"
    queue: str = "all.q"


# --- pure helpers (the orchestrator's, unchanged in behaviour) --------------------------------


def select_probe_specs(spec_table: pd.DataFrame, n_per_level: int) -> pd.DataFrame:
    """N specs per distinct (n_scams, n_smooths) cell, deterministic, spanning every engine."""
    grp = ["n_scams", "n_smooths"] if "n_scams" in spec_table.columns else ["n_smooths"]
    return (
        spec_table.sort_values("spec_index")
        .groupby(grp, sort=True)
        .head(n_per_level)
        .reset_index(drop=True)
    )


def build_cellsets(
    selected: pd.DataFrame, cv_strategy: CvStrategy, windows: Sequence[OosWindow]
) -> tuple[CellSet | None, CellSet | None]:
    """One cell per (spec, experiment): IS + one per window (temporal) or one random cell (random)."""
    is_rows: list[dict[str, str | int]] = []
    oos_rows: list[dict[str, str | int]] = []
    for _, r in selected.iterrows():
        base = {
            "n_smooths": int(r["n_smooths"]),
            "n_scams": int(r["n_scams"]),
            "spec_index": int(r["spec_index"]),
        }
        if cv_strategy == "temporal":
            is_rows.append({"cell": "IS", **base})
            oos_rows.extend({"cell": w.name, **base} for w in windows)
        else:
            oos_rows.append({"cell": "random", **base})
    axes = ["cell", "n_smooths", "n_scams", "spec_index"]
    is_cs = build_hierarchical_cellset(is_rows, axes=axes) if is_rows else None
    oos_cs = build_hierarchical_cellset(oos_rows, axes=axes) if oos_rows else None
    return is_cs, oos_cs


def _feature_fn(group_key: dict[str, Any]) -> dict[str, Any]:
    return {k: group_key[k] for k in ("cell", "n_smooths", "n_scams")}


def _task_id_fn(group_key: dict[str, Any], chunk_idx: int) -> str:
    # cell stays parseable as task_id.rsplit('_n', 1)[0] (finalize relies on it)
    return f"{group_key['cell']}_n{group_key['n_smooths']}_s{group_key['n_scams']}_bin{chunk_idx}"


def partition_all(
    is_cs: CellSet | None,
    oos_cs: CellSet | None,
    *,
    workflow_name: str,
    oos_template: str,
    max_per_task: Mapping[str, int],
) -> TaskManifest:
    """Rectangular-partition each cellset under its template into one analytic manifest."""
    collected: list[Task] = []
    for cs, tmpl in ((is_cs, IS_TEMPLATE), (oos_cs, oos_template)):
        if cs is None:
            continue
        m = rectangular_partition(
            cs,
            fix=["cell", "n_smooths", "n_scams"],
            workflow_name=workflow_name,
            task_template=tmpl,
            max_per_task=max_per_task[tmpl],
            task_id_fn=_task_id_fn,
            features_fn=_feature_fn,
        )
        collected.extend(m.tasks)
    tasks = [
        Task(
            index=i,
            task_id=t.task_id,
            task_template=t.task_template,
            task_args=t.task_args,
            depends_on=t.depends_on,
            shared_axes=t.shared_axes,
            task_features={**t.task_features, "n_specs": len(t.task_args["cells"])},
        )
        for i, t in enumerate(collected)
    ]
    return TaskManifest(workflow_name=workflow_name, tasks=tasks)


def r_flag(value: bool) -> str:  # noqa: FBT001 - the R worker reads TRUE/FALSE strings
    return "TRUE" if value else "FALSE"


def to_submit_manifest(
    analytic: TaskManifest,
    *,
    common_args: Mapping[str, str | int | float],
    windows: Mapping[str, OosWindow],
) -> TaskManifest:
    """Command-args-only manifest: drop the cells list, add the worker's arguments per task."""
    subm: list[Task] = []
    for t in analytic.tasks:
        args: dict[str, str | int | float] = {"task_id": t.task_id, **common_args}
        if t.task_template == OOS_TEMPLATES["temporal"]:
            w = windows[t.task_args["cell"]]
            args.update(
                train_lo=w.train_lo,
                train_hi=w.train_hi,
                test_lo=w.test_lo,
                test_hi=w.test_hi,
            )
        subm.append(
            Task(
                index=t.index,
                task_id=t.task_id,
                task_template=t.task_template,
                task_args=args,
                task_features=t.task_features,
                depends_on=t.depends_on,
                shared_axes=t.shared_axes,
            )
        )
    return TaskManifest(workflow_name=analytic.workflow_name, tasks=subm)


def worker_templates(  # noqa: PLR0913 - one argument per command-line knob
    *,
    r_shell: str,
    r_image: str,
    worker: Path,
    cores: int,
    common_arg_names: Sequence[str],
    cv_strategy: CvStrategy,
    cv_n_folds: int,
    max_lag: int,
    python: str = sys.executable,
) -> dict[str, TaskTemplateSpec]:
    """The three task templates: IS fit, OOS fit (temporal or random), finalize join."""
    prefix = f"OPENBLAS_NUM_THREADS={cores} OMP_NUM_THREADS={cores} "
    base = (
        prefix
        + f"{r_shell} -i {r_image} -s {worker} "
        + "--task-id {task_id} --output-dir {output_dir} --manifest {manifest} "
        + "--spec-table {spec_table} --past-inputs {past_inputs} --prep-script {prep_script} "
        + "--inc-count-min {inc_count_min} --pfpr-min {pfpr_min} "
        + "--save-fits {save_fits} --save-predictions {save_predictions} "
        + f"--optimizer {{optimizer}} --maxit {{maxit}} --max-lag {max_lag} --write-summary FALSE "
    )
    common = list(common_arg_names)
    if cv_strategy == "temporal":
        oos = TaskTemplateSpec(
            command_template=base
            + "--cv-strategy temporal --train-lo {train_lo} --train-hi {train_hi} "
            + "--test-lo {test_lo} --test-hi {test_hi} --fit-is-fe FALSE --fit-oos TRUE",
            node_args=["task_id"],
            task_args=[*common, "train_lo", "train_hi", "test_lo", "test_hi"],
        )
    else:
        oos = TaskTemplateSpec(
            command_template=base
            + f"--cv-strategy random --cv-n-folds {cv_n_folds} --fit-is-fe FALSE --fit-oos TRUE",
            node_args=["task_id"],
            task_args=common,
        )
    return {
        IS_TEMPLATE: TaskTemplateSpec(
            command_template=base + "--fit-is-fe TRUE --fit-oos FALSE",
            node_args=["task_id"],
            task_args=common,
        ),
        OOS_TEMPLATES[cv_strategy]: oos,
        "finalize": TaskTemplateSpec(
            command_template=f"{python} {FINALIZE_SCRIPT} --run-dir {{output_dir}}",
            node_args=["output_dir"],
            task_args=[],
        ),
    }


def make_spec_done(
    output_dir: Path, expected: Mapping[str, set[int]]
) -> Callable[[Task], bool]:
    """Done when select_summary_<task_id>.parquet has a row for every spec of the task."""

    def spec_done(task: Task) -> bool:
        p = output_dir / f"select_summary_{task.task_id}.parquet"
        if not p.exists():
            return False
        col = "is_r_sq" if task.task_template == IS_TEMPLATE else "oos_r_sq"
        try:
            df = pd.read_parquet(p, columns=["spec_index", col])
        except Exception:  # noqa: BLE001 - partial / old-schema file -> re-run the task
            return False
        return expected.get(task.task_id, set()).issubset(
            set(df["spec_index"].astype(int))
        )

    return spec_done


def _feature_int(task: Task, name: str) -> int:
    value = task.task_features[name]
    if value is None:
        msg = f"task {task.task_id} lacks feature {name!r}"
        raise ValueError(msg)
    return int(value)


def make_resources(
    res: FitRunResources, *, cv_n_folds: int
) -> Callable[[Task], dict[str, str | int]]:
    """runtime = load + per-spec seconds x n_specs x contention, with a floor; memory per template."""

    def minutes(seconds: float) -> int:
        return max(res.min_runtime_min, -(-int(seconds) // 60))

    def resources(task: Task) -> dict[str, str | int]:
        if task.task_template == "finalize":
            return dict(res.finalize)
        tmpl = task.task_template
        ns = _feature_int(task, "n_smooths")
        n_specs = _feature_int(task, "n_specs")
        if tmpl == OOS_TEMPLATES["random"]:
            per_fit = res.per_spec_sec.get((IS_TEMPLATE, ns), res.per_spec_default)
            rt = minutes(res.load_sec + cv_n_folds * per_fit * n_specs * res.contention)
        else:
            per = res.per_spec_sec.get((tmpl, ns), res.per_spec_default)
            rt = minutes(res.load_sec + per * n_specs * res.contention)
        return {"memory": res.memory[tmpl], "runtime": f"{rt}m", "cores": res.cores}

    return resources


# --- guards -----------------------------------------------------------------------------------


def check_output_dir(
    output_dir: Path, *, models_node: Path = mbpc.MAL_MODELS_NODE
) -> None:
    """Refuse to run into a selection result, a registered snapshot, a `current` target, or the models node."""
    out = output_dir.resolve()
    if (out / SELECTION_RESULT_NAME).exists():
        msg = f"{out} already holds {SELECTION_RESULT_NAME}: a finished selection run is never an output dir"
        raise ValueError(msg)
    mn = Path(models_node).resolve()
    if out == mn or mn in out.parents:
        msg = f"{out} lies under the fitted-models node {mn}; this workflow never writes there"
        raise ValueError(msg)
    node = out.parent
    if (node / "registry.json").exists():
        registered = {rec.version for rec in read_registry(node)}
        if out.name in registered:
            msg = f"{out} is a registered snapshot of {node}"
            raise ValueError(msg)
    current = node / "current"
    if current.is_symlink() and current.resolve() == out:
        msg = f"{out} is the `current` target of {node}"
        raise ValueError(msg)


def check_spec_table(spec_table: pd.DataFrame) -> None:
    missing = [c for c in SPEC_TABLE_COLUMNS if c not in spec_table.columns]
    if missing:
        msg = f"spec table lacks columns {missing}; needs {list(SPEC_TABLE_COLUMNS)}"
        raise ValueError(msg)
    if spec_table["spec_index"].duplicated().any():
        msg = "spec table has duplicated spec_index values"
        raise ValueError(msg)


# --- the run ----------------------------------------------------------------------------------


@dataclass
class MalariaFitRun:
    """What one call submitted, and where its outputs will be."""

    output_dir: Path
    spec_table_path: Path
    manifest_path: Path
    analytic_manifest: TaskManifest
    submit_manifest: TaskManifest
    windows: tuple[OosWindow, ...]
    n_specs: int
    expected_outputs: dict[str, set[int]]
    workflow: WorkflowResult
    summary_path: Path | None


def submit_malaria_fit_run(  # noqa: PLR0913 - one argument per run knob
    spec_table: pd.DataFrame | Path,
    output_dir: Path,
    *,
    worker: Path = DEFAULT_WORKER,
    r_image: str,
    r_shell: str,
    past_inputs: Path,
    prep_script: Path | None = None,
    optimizer: str = "efs",
    maxit: int = 30,
    cv_strategy: CvStrategy = "temporal",
    oos_windows: Sequence[OosWindow] | None = None,
    cv_n_folds: int = 10,
    inc_count_min: float = 1.0,
    pfpr_min: float = 1e-4,
    save_fits: bool = False,
    save_predictions: bool = False,
    probe_n_per_level: int | None = None,
    finalize: bool | None = None,
    resources: FitRunResources | None = None,
    workflow_name: str = "malaria_select",
    max_lag: int = 0,
    log: Callable[[str], None] = print,
    submit: Callable[..., WorkflowResult] = submit_with_manifest,
) -> MalariaFitRun:
    """Fan a spec table into IS and OOS cells, bundle them into tasks, and submit the workflow.

    ``spec_table`` is a frame (written to ``<output_dir>/spec_table.parquet``) or the path of one
    (copied there if elsewhere). ``past_inputs`` is the parquet every task reads; it is passed
    explicitly and nothing follows a ``current`` link. ``prep_script`` is the R file defining
    ``prepare_malaria_fit_frame(parquet_path, inc_count_min, pfpr_min, suit_variant)``; the
    package's own file when None. ``oos_windows`` None means the ten windows of the 2026-07 run.
    ``probe_n_per_level`` N fits N specs per (n_scams, n_smooths) level; None fits them all.
    ``finalize`` None follows the orchestrator's rule: a finalize task on full temporal runs only.
    """
    res = resources or FitRunResources()
    output_dir = Path(output_dir)
    check_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_table_path = output_dir / SPEC_TABLE_NAME
    if isinstance(spec_table, pd.DataFrame):
        specs = spec_table
        check_spec_table(specs)
        specs.to_parquet(spec_table_path, index=False)
    else:
        src = Path(spec_table)
        specs = pd.read_parquet(src)
        check_spec_table(specs)
        if src.resolve() != spec_table_path.resolve():
            specs.to_parquet(spec_table_path, index=False)

    selected = (
        select_probe_specs(specs, probe_n_per_level)
        if probe_n_per_level is not None
        else specs.sort_values("spec_index").reset_index(drop=True)
    )
    windows = tuple(
        temporal_windows(max_lag=max_lag) if oos_windows is None else oos_windows
    )
    if cv_strategy not in OOS_TEMPLATES:
        msg = f"cv_strategy must be one of {sorted(OOS_TEMPLATES)}, got {cv_strategy!r}"
        raise ValueError(msg)
    oos_template = OOS_TEMPLATES[cv_strategy]

    is_cs, oos_cs = build_cellsets(selected, cv_strategy, windows)
    analytic = partition_all(
        is_cs,
        oos_cs,
        workflow_name=workflow_name,
        oos_template=oos_template,
        max_per_task=res.max_per_task,
    )
    manifest_path = output_dir / MANIFEST_NAME
    analytic.save(manifest_path)
    cell_names = (
        ["IS", *(w.name for w in windows)]
        if cv_strategy == "temporal"
        else [f"random({cv_n_folds}-fold)"]
    )
    log(
        f"{'Probe' if probe_n_per_level is not None else 'Full'} [{cv_strategy}]: {len(selected)} specs -> "
        f"{len(analytic.tasks)} tasks (cells: {cell_names}); manifest -> {manifest_path}"
    )

    expected = {
        t.task_id: {int(c["spec_index"]) for c in t.task_args["cells"]}
        for t in analytic.tasks
    }
    common_args: dict[str, str | int | float] = {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "spec_table": str(spec_table_path),
        "past_inputs": str(past_inputs),
        "prep_script": str(
            prep_script if prep_script is not None else DEFAULT_PREP_SCRIPT
        ),
        "inc_count_min": inc_count_min,
        "pfpr_min": pfpr_min,
        "save_fits": r_flag(save_fits),
        "save_predictions": r_flag(save_predictions),
        "optimizer": optimizer,
        "maxit": int(maxit),
    }
    templates = worker_templates(
        r_shell=r_shell,
        r_image=r_image,
        worker=Path(worker),
        cores=res.cores,
        common_arg_names=list(common_args),
        cv_strategy=cv_strategy,
        cv_n_folds=cv_n_folds,
        max_lag=max_lag,
    )
    submit_manifest = to_submit_manifest(
        analytic, common_args=common_args, windows={w.name: w for w in windows}
    )
    submit_manifest = filter_already_done(
        submit_manifest, make_spec_done(output_dir, expected)
    )

    do_finalize = (
        (probe_n_per_level is None and cv_strategy == "temporal")
        if finalize is None
        else finalize
    )
    summary_path: Path | None = None
    if do_finalize:
        fit_ids = [t.task_id for t in submit_manifest.tasks]
        finalize_task = Task(
            index=len(submit_manifest.tasks),
            task_id="finalize",
            task_template="finalize",
            task_args={"output_dir": str(output_dir)},
            depends_on=fit_ids,
        )
        submit_manifest = TaskManifest(
            workflow_name=submit_manifest.workflow_name,
            tasks=[*submit_manifest.tasks, finalize_task],
        )
        summary_path = output_dir / SUMMARY_NAME
        log(f"+ finalize task depends_on {len(fit_ids)} fit tasks")

    result = submit(
        submit_manifest,
        output_dir=output_dir,
        templates=templates,
        resources=make_resources(res, cv_n_folds=cv_n_folds),
        concurrency_limit=int(res.max_concurrent),
        project=res.project,
        queue=res.queue,
        tool_name="idd-forecast-mbp",
        log_method=log,
    )
    log(
        f"workflow {result.workflow_id} status {result.status} "
        f"({result.n_tasks_submitted} tasks); run record at {result.run_record_path}"
    )
    return MalariaFitRun(
        output_dir=output_dir,
        spec_table_path=spec_table_path,
        manifest_path=manifest_path,
        analytic_manifest=analytic,
        submit_manifest=submit_manifest,
        windows=windows,
        n_specs=len(selected),
        expected_outputs=expected,
        workflow=result,
        summary_path=summary_path,
    )
