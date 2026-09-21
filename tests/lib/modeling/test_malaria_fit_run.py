"""submit_malaria_fit_run against a fake submit: windows, manifests, commands, done-check, guards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from idd_tools.jobmon import Task, TaskManifest, WorkflowResult

from idd_forecast_mbp.lib.modeling import malaria_fit_run as mfr


def spec_frame(extra_rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    rows = [
        {
            "spec_index": 1,
            "n_smooths": 0,
            "n_scams": 0,
            "formula_text": "logit_malaria_pfpr ~ A0_af",
        },
        {
            "spec_index": 2,
            "n_smooths": 1,
            "n_scams": 0,
            "formula_text": "logit_malaria_pfpr ~ s(gdppc_mean, k = 4) + A0_af",
        },
        {
            "spec_index": 3,
            "n_smooths": 2,
            "n_scams": 2,
            "formula_text": 'logit_malaria_pfpr ~ s(gdppc_mean, k = 4, bs = "mpd") + s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + A0_af',
        },
        *(extra_rows or []),
    ]
    return pd.DataFrame(rows).astype(
        {
            "spec_index": "int32",
            "n_smooths": "int32",
            "n_scams": "int32",
            "formula_text": "string",
        }
    )


class FakeSubmit:
    """Records what submit_with_manifest would have received and returns a result."""

    def __init__(self) -> None:
        self.calls: list[tuple[TaskManifest, dict[str, Any]]] = []

    def __call__(self, manifest: TaskManifest, **kwargs: Any) -> WorkflowResult:
        self.calls.append((manifest, kwargs))
        return WorkflowResult(
            workflow_id=7,
            status="D",
            n_tasks_submitted=len(manifest.tasks),
            run_record_path=Path(kwargs["output_dir"]) / ".jobmon_run_record.parquet",
            workflow_name=manifest.workflow_name,
        )


def run(
    tmp_path: Path, specs: pd.DataFrame | Path | None = None, **overrides: Any
) -> tuple[mfr.MalariaFitRun, FakeSubmit]:
    fake = FakeSubmit()
    kwargs: dict[str, Any] = {
        "worker": Path("/w/select_malaria_models_rocket.r"),
        "r_image": "r.img",
        "r_shell": "execRscript.sh",
        "past_inputs": tmp_path / "past.parquet",
        "submit": fake,
        "log": lambda _s: None,
    }
    kwargs.update(overrides)
    result = mfr.submit_malaria_fit_run(
        spec_frame() if specs is None else specs, tmp_path / "run", **kwargs
    )
    return result, fake


# --- windows ---------------------------------------------------------------------------------


def test_default_windows_are_the_ten_of_the_selection_run() -> None:
    windows = mfr.temporal_windows()
    assert len(windows) == 10
    by_name = {w.name: w for w in windows}
    assert by_name["narrow_full"] == mfr.OosWindow(
        "narrow_full", 2000, 2012, 2013, 2023
    )
    assert by_name["exwide_recent"] == mfr.OosWindow(
        "exwide_recent", 2000, 2009, 2019, 2023
    )
    assert "exwide_preC" not in by_name  # 2013 - 10 = 2003 -> only 4 training years


def test_windows_respect_min_training_years_and_lag() -> None:
    assert mfr.temporal_windows(min_training_years=30) == []
    lagged = {w.name: w for w in mfr.temporal_windows(max_lag=3)}
    assert lagged["narrow_full"].train_lo == 2003


# --- a full temporal run ----------------------------------------------------------------------


def test_full_temporal_run_writes_spec_table_and_manifest_and_bundles_by_level(
    tmp_path: Path,
) -> None:
    result, _fake = run(tmp_path)
    assert result.spec_table_path.exists()
    assert pd.read_parquet(result.spec_table_path)["spec_index"].tolist() == [1, 2, 3]
    saved = json.loads(result.manifest_path.read_text())
    assert len(saved["tasks"]) == len(result.analytic_manifest.tasks) == 3 + 3 * 10
    templates = {t.task_template for t in result.analytic_manifest.tasks}
    assert templates == {"is_cell", "oos_temporal"}
    assert result.n_specs == 3
    assert result.windows == tuple(mfr.temporal_windows())
    assert set(result.expected_outputs) == {
        t.task_id for t in result.analytic_manifest.tasks
    }
    assert all(
        len(v) == 1 for v in result.expected_outputs.values()
    )  # distinct levels -> one spec per task


def test_full_temporal_run_appends_finalize_depending_on_every_fit_task(
    tmp_path: Path,
) -> None:
    result, fake = run(tmp_path)
    manifest, kwargs = fake.calls[0]
    fit_ids = [t.task_id for t in manifest.tasks if t.task_template != "finalize"]
    finalize = [t for t in manifest.tasks if t.task_template == "finalize"]
    assert len(finalize) == 1
    assert sorted(finalize[0].depends_on) == sorted(fit_ids)
    assert result.summary_path == tmp_path / "run" / "selection_summary.parquet"
    assert kwargs["output_dir"] == tmp_path / "run"
    assert kwargs["concurrency_limit"] == 500
    assert kwargs["project"] == "proj_rapidresponse"
    assert result.workflow.workflow_id == 7


def test_worker_command_carries_every_new_flag_with_run_level_defaults(
    tmp_path: Path,
) -> None:
    result, fake = run(tmp_path)
    manifest, kwargs = fake.calls[0]
    is_cmd = kwargs["templates"]["is_cell"].command_template
    for flag in (
        "--spec-table {spec_table}",
        "--past-inputs {past_inputs}",
        "--prep-script {prep_script}",
        "--inc-count-min {inc_count_min}",
        "--pfpr-min {pfpr_min}",
        "--save-fits {save_fits}",
        "--save-predictions {save_predictions}",
        "--max-lag 0",
        "--write-summary FALSE",
        "--fit-is-fe TRUE --fit-oos FALSE",
    ):
        assert flag in is_cmd
    assert is_cmd.startswith(
        "OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 execRscript.sh -i r.img -s /w/"
    )
    oos_cmd = kwargs["templates"]["oos_temporal"].command_template
    assert "--cv-strategy temporal --train-lo {train_lo}" in oos_cmd
    is_task = next(t for t in manifest.tasks if t.task_template == "is_cell")
    assert is_task.task_args["prep_script"] == str(mfr.DEFAULT_PREP_SCRIPT)
    assert is_task.task_args["past_inputs"] == str(tmp_path / "past.parquet")
    assert is_task.task_args["spec_table"] == str(result.spec_table_path)
    assert (is_task.task_args["inc_count_min"], is_task.task_args["pfpr_min"]) == (
        1.0,
        0.0001,
    )
    assert (is_task.task_args["save_fits"], is_task.task_args["save_predictions"]) == (
        "FALSE",
        "FALSE",
    )
    assert (is_task.task_args["optimizer"], is_task.task_args["maxit"]) == ("efs", 30)
    oos_task = next(t for t in manifest.tasks if t.task_id.startswith("narrow_full_"))
    assert (oos_task.task_args["train_lo"], oos_task.task_args["train_hi"]) == (
        2000,
        2012,
    )
    assert (oos_task.task_args["test_lo"], oos_task.task_args["test_hi"]) == (
        2013,
        2023,
    )


def test_switches_thresholds_and_prep_script_pass_through(tmp_path: Path) -> None:
    prep = tmp_path / "my_prep.R"
    _result, fake = run(
        tmp_path,
        save_fits=True,
        save_predictions=True,
        inc_count_min=0,
        pfpr_min=0,
        prep_script=prep,
        optimizer="bfgs",
        maxit=300,
    )
    manifest, _ = fake.calls[0]
    args = manifest.tasks[0].task_args
    assert (args["save_fits"], args["save_predictions"]) == ("TRUE", "TRUE")
    assert (args["inc_count_min"], args["pfpr_min"]) == (0, 0)
    assert args["prep_script"] == str(prep)
    assert (args["optimizer"], args["maxit"]) == ("bfgs", 300)


def test_custom_windows_replace_the_default_set(tmp_path: Path) -> None:
    one = [mfr.OosWindow("only", 2000, 2015, 2016, 2023)]
    result, _fake = run(tmp_path, oos_windows=one)
    assert result.windows == tuple(one)
    cells = {t.task_args["cell"] for t in result.analytic_manifest.tasks}
    assert cells == {"IS", "only"}


# --- probe and random -------------------------------------------------------------------------


def test_probe_takes_n_per_level_and_skips_finalize(tmp_path: Path) -> None:
    specs = spec_frame(
        [
            {
                "spec_index": 4,
                "n_smooths": 0,
                "n_scams": 0,
                "formula_text": "logit_malaria_pfpr ~ gdppc_mean + A0_af",
            }
        ]
    )
    result, fake = run(tmp_path, specs=specs, probe_n_per_level=1)
    assert (
        result.n_specs == 3
    )  # spec 4 shares level (0, 0) with spec 1 and is not probed
    chosen = {s for v in result.expected_outputs.values() for s in v}
    assert chosen == {1, 2, 3}
    manifest, _ = fake.calls[0]
    assert all(t.task_template != "finalize" for t in manifest.tasks)
    assert result.summary_path is None


def test_random_strategy_is_oos_only_one_cell_per_spec(tmp_path: Path) -> None:
    result, fake = run(tmp_path, cv_strategy="random", cv_n_folds=5)
    assert {t.task_template for t in result.analytic_manifest.tasks} == {"oos_random"}
    assert len(result.analytic_manifest.tasks) == 3
    manifest, kwargs = fake.calls[0]
    assert (
        "--cv-strategy random --cv-n-folds 5"
        in kwargs["templates"]["oos_random"].command_template
    )
    assert all(t.task_template != "finalize" for t in manifest.tasks)


def test_unknown_strategy_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cv_strategy"):
        run(tmp_path, cv_strategy="bootstrap")


# --- done-check -------------------------------------------------------------------------------


def test_filter_already_done_drops_tasks_whose_summary_covers_their_specs(
    tmp_path: Path,
) -> None:
    first, _ = run(tmp_path)
    done_task = next(
        t for t in first.analytic_manifest.tasks if t.task_template == "is_cell"
    )
    spec = next(iter(first.expected_outputs[done_task.task_id]))
    out = tmp_path / "run"
    pd.DataFrame({"spec_index": [spec], "is_r_sq": [0.5]}).to_parquet(
        out / f"select_summary_{done_task.task_id}.parquet"
    )
    partial_task = next(
        t for t in first.analytic_manifest.tasks if t.task_template == "oos_temporal"
    )
    pd.DataFrame({"spec_index": [999], "oos_r_sq": [0.1]}).to_parquet(
        out / f"select_summary_{partial_task.task_id}.parquet"
    )
    wrong_schema = [
        t for t in first.analytic_manifest.tasks if t.task_template == "oos_temporal"
    ][1]
    pd.DataFrame({"spec_index": [1]}).to_parquet(
        out / f"select_summary_{wrong_schema.task_id}.parquet"
    )

    _second, fake = run(tmp_path)
    submitted = {t.task_id for t in fake.calls[0][0].tasks}
    assert done_task.task_id not in submitted
    assert partial_task.task_id in submitted
    assert wrong_schema.task_id in submitted
    assert "finalize" in submitted


# --- resources ---------------------------------------------------------------------------------


def test_resources_size_runtime_from_the_per_spec_table() -> None:
    res = mfr.make_resources(mfr.FitRunResources(), cv_n_folds=10)
    is_task = Task(
        index=0,
        task_id="IS_n3_s1_bin0",
        task_template="is_cell",
        task_args={},
        task_features={"n_smooths": 3, "n_scams": 1, "n_specs": 4, "cell": "IS"},
    )
    assert res(is_task) == {
        "memory": "6G",
        "runtime": "27m",
        "cores": 16,
    }  # ceil((120 + 316*4*1.15)/60)
    tiny = Task(
        index=1,
        task_id="IS_n0_s0_bin0",
        task_template="is_cell",
        task_args={},
        task_features={"n_smooths": 0, "n_scams": 0, "n_specs": 1, "cell": "IS"},
    )
    assert res(tiny)["runtime"] == "8m"  # floor
    rnd = Task(
        index=2,
        task_id="random_n1_s0_bin0",
        task_template="oos_random",
        task_args={},
        task_features={"n_smooths": 1, "n_scams": 0, "n_specs": 1, "cell": "random"},
    )
    assert res(rnd) == {
        "memory": "6G",
        "runtime": "35m",
        "cores": 16,
    }  # ceil((120 + 10*172*1.15)/60)
    fin = Task(index=3, task_id="finalize", task_template="finalize", task_args={})
    assert res(fin) == {"memory": "10G", "runtime": "20m", "cores": 1}


def test_resources_refuse_a_task_missing_a_sizing_feature() -> None:
    res = mfr.make_resources(mfr.FitRunResources(), cv_n_folds=10)
    task = Task(
        index=0,
        task_id="IS_n1_s0_bin0",
        task_template="is_cell",
        task_args={},
        task_features={"n_smooths": 1, "n_scams": 0, "n_specs": None, "cell": "IS"},
    )
    with pytest.raises(ValueError, match="n_specs"):
        res(task)


# --- guards ------------------------------------------------------------------------------------


def test_refuses_a_directory_holding_a_selection_result(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "selection_result.json").write_text("{}")
    with pytest.raises(ValueError, match=r"selection_result\.json"):
        run(tmp_path)


def test_refuses_anything_under_the_models_node(tmp_path: Path) -> None:
    node = tmp_path / "models"
    node.mkdir()
    with pytest.raises(ValueError, match="fitted-models node"):
        mfr.check_output_dir(node / "working", models_node=node)
    with pytest.raises(ValueError, match="fitted-models node"):
        mfr.check_output_dir(node, models_node=node)


def test_refuses_the_current_target_and_a_registered_snapshot(tmp_path: Path) -> None:
    node = tmp_path / "scam_prelim"
    snap = node / "20260727_efs"
    snap.mkdir(parents=True)
    (node / "current").symlink_to("20260727_efs")
    with pytest.raises(ValueError, match="current"):
        mfr.check_output_dir(snap, models_node=tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="current"):
        mfr.check_output_dir(node / "current", models_node=tmp_path / "elsewhere")
    (node / "current").unlink()
    (node / "registry.json").write_text(
        json.dumps(
            [
                {
                    "version": "20260727_efs",
                    "description": "the selection run",
                    "recorded_at": "2026-09-16 14:16:00",
                    "labels": ["selected_2026_07_31"],
                    "repo": "idd-forecast-mbp",
                    "git_commit": "abc1234",
                    "git_dirty": False,
                    "fingerprint": "sha256:0",
                    "current": False,
                    "legacy": True,
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="registered snapshot"):
        mfr.check_output_dir(snap, models_node=tmp_path / "elsewhere")
    mfr.check_output_dir(
        node / "scratch" / "probe", models_node=tmp_path / "elsewhere"
    )  # allowed


def test_spec_table_must_have_the_four_columns_and_unique_indices(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="lacks columns"):
        run(tmp_path, specs=spec_frame().drop(columns=["n_scams"]))
    dup = pd.concat([spec_frame(), spec_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        run(tmp_path, specs=dup)


def test_spec_table_given_as_a_path_elsewhere_is_copied_into_the_run_dir(
    tmp_path: Path,
) -> None:
    src = tmp_path / "elsewhere.parquet"
    spec_frame().to_parquet(src, index=False)
    result, _ = run(tmp_path, specs=src)
    assert result.spec_table_path == tmp_path / "run" / "spec_table.parquet"
    assert result.spec_table_path.exists()
    assert pd.read_parquet(result.spec_table_path).equals(pd.read_parquet(src))
