"""The spec-table builder: the typed universe -> spec_table.parquet in a dated run dir."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.select.malaria_spec_design import (
    CURRENT_DESIGN,
    DESIGN_FILE,
    load_design,
)

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO / "src" / "idd_forecast_mbp" / "03_modeling" / "build_malaria_spec_design.py"
)
N_SPECS = 2 * 2 * 3 * 5 * 3 * 5 * 3 * 3  # 8,100: levels per group axis, "out" included


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_malaria_spec_design", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frame(builder) -> pd.DataFrame:
    return builder.spec_table_frame()


def test_default_output_root_is_the_selection_node(builder):
    assert builder.OUTPUT_ROOT == mbpc.MAL_SELECTION_NODE
    assert mbpc.MAL_SELECTION_NODE.name == mbpc.LSAE_HIERARCHY
    assert mbpc.MAL_SELECTION_NODE.parent.name == "scam_prelim"


def test_spec_table_frame_has_the_worker_schema(frame):
    assert list(frame.columns) == ["spec_index", "n_smooths", "n_scams", "formula_text"]
    assert dict(frame.dtypes.astype(str)) == {
        "spec_index": "int32",
        "n_smooths": "int32",
        "n_scams": "int32",
        "formula_text": "string",
    }
    assert len(frame) == N_SPECS
    assert frame["spec_index"].tolist() == list(range(1, N_SPECS + 1))
    assert (frame["n_scams"] <= frame["n_smooths"]).all()
    assert frame["formula_text"].str.startswith("logit_malaria_pfpr ~ ").all()
    assert (
        frame["formula_text"].str.endswith("A0_af").all()
    )  # the all-out spec is "~ A0_af"


def test_main_writes_a_dated_run_dir_and_never_overwrites(builder, tmp_path):
    runner = CliRunner()
    first = runner.invoke(builder.main, ["--output-root", str(tmp_path), "--tag", "t"])
    assert first.exit_code == 0, first.output
    run_dir = Path(first.output.strip().splitlines()[-1])
    assert run_dir.parent == tmp_path
    assert run_dir.name.endswith("_t")
    table = pd.read_parquet(run_dir / "spec_table.parquet")
    assert len(table) == N_SPECS
    assert (run_dir / DESIGN_FILE).is_file()
    assert load_design(run_dir).groups == CURRENT_DESIGN.groups

    second = runner.invoke(builder.main, ["--output-root", str(tmp_path), "--tag", "t"])
    assert second.exit_code == 0, second.output
    second_dir = Path(second.output.strip().splitlines()[-1])
    assert second_dir != run_dir
    assert second_dir.name.endswith("_t_v2")
