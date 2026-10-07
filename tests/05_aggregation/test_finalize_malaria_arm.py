"""The arm finishing driver: input resolution and refusals (numbers are checked on real runs)."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner

from idd_forecast_mbp.lib.data.malaria_arms import Arm
from idd_forecast_mbp.lib.processing.arm_recreate import forecast_nc

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO / "src" / "idd_forecast_mbp" / "05_aggregation" / "finalize_malaria_arm.py"
)


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("finalize_malaria_arm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_ARM = Arm(
    name="no_vaccine",
    model="m",
    forecast_label="m__base",
    dah="Baseline",
    vaccine="none",
    population_hold_year=None,
    age_structure_hold_year=None,
)


def _arm(**kw: Any) -> Arm:
    return dataclasses.replace(BASE_ARM, **kw)


def test_forecast_nc_is_the_launch_label_on_the_node(
    driver: ModuleType, tmp_path: Path
) -> None:
    arm = _arm(dah="Constant")
    assert (
        forecast_nc(arm, "ssp245", tmp_path)
        == tmp_path / "m__base" / "malaria_forecast_ssp245_Constant.nc"
    )


def test_an_arm_with_no_finishing_change_needs_no_factors(driver: ModuleType) -> None:
    # no file is read on this path, so any location/year list will do
    assert driver.arm_factor_frames(_arm(), [1, 2], [2023], "loglinear_severe0") is None


def test_missing_forecast_is_refused_before_anything_is_written(
    driver: ModuleType, tmp_path: Path
) -> None:
    arms = tmp_path / "arms.yaml"
    arms.write_text(
        "ve_variant: loglinear_severe0\nhold_year: 2023\nselection_run: x\n"
        "models:\n  m: {spec_index: 1, forecast_runs: {base: m__base_that_does_not_exist}}\n"
        "arms:\n  Baseline: {forecast: base, dah: Baseline, vaccine: projected}\n"
    )
    result = CliRunner().invoke(
        driver.main,
        ["--model", "m", "--arm", "Baseline", "--arms-file", str(arms), "--dry-run"],
    )
    assert result.exit_code != 0
    assert "forecast netCDF(s) missing" in result.output
