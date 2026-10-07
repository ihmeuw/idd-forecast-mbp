"""The fit-selected launcher: worker arguments from the config, dry-run wiring, refusals."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from idd_forecast_mbp.select.rank import load_config

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO / "src" / "idd_forecast_mbp" / "03_modeling" / "fit_selected_malaria_model.py"
)
CONFIG = REPO / "reports" / "model_selection" / "malaria_selection_config.yaml"


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("fit_selected_malaria_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def selection_dir(tmp_path):
    d = tmp_path / "sel"
    d.mkdir()
    (d / "selection_result.json").write_text(
        json.dumps(
            {
                "status": "tentative",
                "run_dir": str(d),
                "spec_table_fingerprint": "sha256:abc",
                "pick": {
                    "spec_index": 1486,
                    "formula_text": "logit_malaria_pfpr ~ A0_af",
                },
                "anchor": {"spec_index": 1585},
                "rank": {},
                "code": {"commit": "x", "dirty": False},
                "config_source": "cfg",
            }
        )
    )
    return d


def test_build_worker_args_carries_every_final_fit_setting(launcher):
    cfg = load_config(CONFIG)
    args = launcher.build_worker_args(
        Path("/r.json"), Path("/out"), Path("/p.parquet"), cfg.final_fit
    )
    pairs = dict(zip(args[::2], args[1::2], strict=True))
    assert pairs["--result"] == "/r.json"
    assert pairs["--out-dir"] == "/out"
    assert pairs["--past-inputs"] == "/p.parquet"
    assert pairs["--optimizer"] == str(cfg.final_fit["optimizer"])
    assert pairs["--maxit"] == str(cfg.final_fit["maxit"])
    assert pairs["--suit-variant"] == cfg.final_fit["suit_variant"]
    assert pairs["--inc-mort-rhs"] == cfg.final_fit["inc_mort_rhs"]
    assert pairs["--inc-count-min"] == str(
        cfg.final_fit["data_filter"]["malaria_inc_count_min"]
    )
    assert pairs["--pfpr-min"] == str(cfg.final_fit["data_filter"]["malaria_pfpr_min"])


def test_prep_script_is_passed_through_only_when_given(launcher, tmp_path):
    cfg = load_config(CONFIG)
    base = launcher.build_worker_args(
        tmp_path / "r.json", tmp_path / "out", tmp_path / "p.parquet", cfg.final_fit
    )
    assert "--prep-script" not in base
    with_prep = launcher.build_worker_args(
        tmp_path / "r.json",
        tmp_path / "out",
        tmp_path / "p.parquet",
        cfg.final_fit,
        prep_script=tmp_path / "my_prep.R",
    )
    assert with_prep[: len(base)] == base
    assert with_prep[len(base) :] == ["--prep-script", str(tmp_path / "my_prep.R")]


def test_worker_command_shell_quotes_arguments_the_wrapper_reparses(launcher):
    rhs = 's(logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af'
    cmd = launcher.worker_command(
        "execRscript.sh", "img.img", Path("w.r"), ["--inc-mort-rhs", rhs]
    )
    # single quotes for the bash -c parse; the inner double quotes escaped for the wrapper's eval
    assert (
        cmd[-1] == "'" + rhs.replace('"', '\\"') + "'"
    )  # -> 's(..., bs = \"mpi\") + ...'
    assert launcher.wrapper_quote("/a/plain/path.parquet") == "/a/plain/path.parquet"
    assert launcher.wrapper_quote("0.0001") == "0.0001"
    assert cmd[-2] == "--inc-mort-rhs"


def test_worker_command_uses_the_r_shell_idiom(launcher):
    cmd = launcher.worker_command(
        "execRscript.sh", "img.img", Path("w.r"), ["--a", "1"]
    )
    assert cmd == ["execRscript.sh", "-i", "img.img", "-s", "w.r", "--a", "1"]


def _common(tmp_path, selection_dir):
    img = tmp_path / "r.img"
    img.write_text("")
    sh = tmp_path / "execRscript.sh"
    sh.write_text("")
    return [
        "--config",
        str(CONFIG),
        "--run-dir",
        str(selection_dir),
        "--r-image",
        str(img),
        "--r-shell",
        str(sh),
        "--node",
        str(tmp_path / "models"),
        "--past-inputs",
        str(tmp_path / "past.parquet"),
    ]


def test_dry_run_prints_command_and_creates_working_slot(
    launcher, tmp_path, selection_dir
):
    result = CliRunner().invoke(
        launcher.main, [*_common(tmp_path, selection_dir), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "selected spec 1486" in result.output
    assert "--inc-count-min 0" in result.output
    assert "--optimizer bfgs" in result.output
    assert (tmp_path / "models" / "working").is_dir()
    assert "(--dry-run: nothing run)" in result.output


def test_current_without_description_is_refused_before_anything_runs(
    launcher, tmp_path, selection_dir
):
    result = CliRunner().invoke(
        launcher.main, [*_common(tmp_path, selection_dir), "--dry-run", "--current"]
    )
    assert result.exit_code != 0
    assert "--description" in result.output
    assert not (tmp_path / "models" / "working").exists()


def test_missing_result_is_refused(launcher, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    args = [*_common(tmp_path, empty), "--dry-run"]
    result = CliRunner().invoke(launcher.main, args)
    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)


def _spec_table(selection_dir: Path) -> None:
    import pandas as pd  # noqa: PLC0415 - only these tests need a spec table

    pd.DataFrame(
        {
            "spec_index": pd.array([1, 7426], dtype="int32"),
            "formula_text": pd.array(
                [
                    "logit_malaria_pfpr ~ A0_af",
                    "logit_malaria_pfpr ~ gdppc_mean + A0_af",
                ],
                dtype="string",
            ),
        }
    ).to_parquet(selection_dir / "spec_table.parquet", index=False)


def test_spec_index_adds_the_spec_table_and_index_to_the_worker_args(
    launcher, tmp_path
):
    cfg = load_config(CONFIG)
    base = launcher.build_worker_args(
        tmp_path / "sel" / "r.json",
        tmp_path / "out",
        tmp_path / "p.parquet",
        cfg.final_fit,
    )
    chosen = launcher.build_worker_args(
        tmp_path / "sel" / "r.json",
        tmp_path / "out",
        tmp_path / "p.parquet",
        cfg.final_fit,
        spec_index=7426,
    )
    assert "--spec-index" not in base
    assert chosen[: len(base)] == base
    assert chosen[len(base) :] == [
        "--spec-table",
        str(tmp_path / "sel" / "spec_table.parquet"),
        "--spec-index",
        "7426",
    ]


def test_chosen_spec_reads_the_formula_and_refuses_an_unknown_index(
    launcher, selection_dir
):
    _spec_table(selection_dir)
    assert (
        launcher.chosen_spec(selection_dir, 7426)
        == "logit_malaria_pfpr ~ gdppc_mean + A0_af"
    )
    with pytest.raises(launcher.click.ClickException, match="spec 99 is not in"):
        launcher.chosen_spec(selection_dir, 99)


def test_dry_run_with_spec_index_names_the_chosen_spec(
    launcher, tmp_path, selection_dir
):
    _spec_table(selection_dir)
    result = CliRunner().invoke(
        launcher.main,
        [*_common(tmp_path, selection_dir), "--spec-index", "7426", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "chosen spec 7426" in result.output
    assert "rank pick was 1486" in result.output
    assert "logit_malaria_pfpr ~ gdppc_mean + A0_af" in result.output
    assert "--spec-index 7426" in result.output


def test_dry_run_refuses_a_spec_index_the_run_lacks(launcher, tmp_path, selection_dir):
    _spec_table(selection_dir)
    result = CliRunner().invoke(
        launcher.main,
        [*_common(tmp_path, selection_dir), "--spec-index", "5", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "spec 5 is not in" in result.output
