"""plot_run_comparison's product loader: both product file layouts, and the refusal."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "src" / "idd_forecast_mbp" / "05_aggregation" / "plot_run_comparison.py"


@pytest.fixture(scope="module")
def prc() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot_run_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary() -> pd.DataFrame:
    cols = {
        "location_id": [1, 2],
        "year_id": [2030, 2030],
        "level": [0, 1],
        "population": [10.0, 5.0],
    }
    for m in ("inc", "mort"):
        for space in ("count", "rate"):
            for stat in ("mean", "lower", "upper"):
                cols[f"{m}_{space}_{stat}"] = [1.0, 2.0]
    return pd.DataFrame(cols)


@pytest.mark.parametrize(
    "name",
    ["all_age_summary_ssp245_Baseline.parquet", "all_age_summary_ssp245.parquet"],
)
def test_load_current_reads_either_product_layout(
    prc: ModuleType, tmp_path: Path, name: str
) -> None:
    _summary().to_parquet(tmp_path / name)
    out = prc.load_current(tmp_path, ["ssp245"], "Baseline", [0])
    assert set(out.measure) == {"inc", "mort"}
    assert out.location_id.tolist() == [1, 1]


def test_load_current_refuses_a_missing_scenario(
    prc: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="missing for ssp126"):
        prc.load_current(tmp_path, ["ssp126"], "Baseline", [0])


def test_historical_is_drawn_once_and_warns_when_ssps_differ(
    prc: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    cur = pd.DataFrame(
        {
            "ssp": ["ssp126", "ssp245", "ssp126", "ssp245", "ssp126", "ssp245"],
            "year_id": [2022, 2022, 2023, 2023, 2024, 2024],
            "count_mean": [1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
        }
    )
    h = prc.historical_series(cur, ["ssp126", "ssp245"], "count_mean")
    assert h.year_id.tolist() == [2022, 2023]
    assert "WARNING" not in capsys.readouterr().out
    cur.loc[0, "count_mean"] = 5.0
    prc.historical_series(cur, ["ssp126", "ssp245"], "count_mean")
    assert "SSPs differ before 2023" in capsys.readouterr().out
    assert prc.historical_series(cur.iloc[0:0], ["ssp126"], "count_mean").empty
