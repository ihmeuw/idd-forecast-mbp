"""FHS deliverable: chunk naming, rates on the full grid, refusals."""

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
SCRIPT = (
    REPO / "src" / "idd_forecast_mbp" / "05_aggregation" / "fhs_deliverable_malaria.py"
)


@pytest.fixture(scope="module")
def fhs() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fhs_deliverable_malaria", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunk_names_round_trip_through_the_pattern(fhs: ModuleType) -> None:
    name = fhs.chunk_name("ssp245", 0, 10)
    assert name == "chunk_ssp245_000_010.nc"
    m = fhs.CHUNK_RE.match(name)
    assert (m["ssp"], m["start"], m["end"]) == ("ssp245", "000", "010")


def test_rates_to_dataset_divides_by_own_population(fhs: ModuleType) -> None:
    keys = ["location_id", "year_id", "age_group_id", "sex_id"]
    counts = pd.DataFrame(
        [
            (1, 2030, 3, 1, 0, 4.0, 2.0),
            (1, 2030, 3, 1, 1, 6.0, 0.0),
            (2, 2030, 3, 1, 0, 0.0, 0.0),
            (2, 2030, 3, 1, 1, 0.0, 0.0),
        ],
        columns=[*keys, "draw", "inc_count", "mort_count"],
    )
    pop = pd.DataFrame(
        [(1, 2030, 3, 1, 100.0), (2, 2030, 3, 1, 0.0)], columns=[*keys, "population"]
    )
    ds = fhs.rates_to_dataset(counts, pop)
    assert set(ds.data_vars) == {"incidence_rate", "mortality_rate"}
    assert ds.incidence_rate.dtype == np.float32
    assert float(ds.incidence_rate.sel(location_id=1, draw=1)) == pytest.approx(0.06)
    assert (
        float(ds.incidence_rate.sel(location_id=2, draw=0)) == 0.0
    )  # population 0 -> rate 0


def test_rates_to_dataset_refuses_missing_population(fhs: ModuleType) -> None:
    keys = ["location_id", "year_id", "age_group_id", "sex_id"]
    counts = pd.DataFrame(
        [(1, 2030, 3, 1, 0, 4.0, 2.0)],
        columns=[*keys, "draw", "inc_count", "mort_count"],
    )
    with pytest.raises(ValueError, match="no age/sex population"):
        fhs.rates_to_dataset(counts, pd.DataFrame(columns=[*keys, "population"]))
