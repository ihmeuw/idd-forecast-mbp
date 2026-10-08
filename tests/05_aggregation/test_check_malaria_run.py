"""Acceptance-check helpers (the checks themselves run on real outputs)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "src" / "idd_forecast_mbp" / "05_aggregation" / "check_malaria_run.py"


@pytest.fixture(scope="module")
def chk() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_malaria_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rel_diff_ignores_zero_references(chk: ModuleType) -> None:
    assert chk.rel_diff(pd.Series([1.1, 5.0]), pd.Series([1.0, 0.0])) == pytest.approx(
        0.1
    )
    assert np.isnan(chk.rel_diff(pd.Series([1.0]), pd.Series([0.0])))


def test_top_countries_ranks_level_3_by_2023_incidence(chk: ModuleType) -> None:
    s = pd.DataFrame(
        {
            "location_id": [1, 2, 3, 4],
            "level": [3, 3, 3, 2],
            "year_id": [2023, 2023, 2023, 2023],
            "inc_count_mean": [5.0, 9.0, 1.0, 100.0],
        }
    )
    assert chk.top_countries(s, 2) == [2, 1]
