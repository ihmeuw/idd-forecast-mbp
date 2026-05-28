"""Regression tests for 02a_fhs_population.py.

Strategy: run main() with real data inputs but write to tmp_path, then
compare outputs against the golden files that were produced under lsae_1209
and live in the population artifact dir.

Part A does not depend on lsae_hierarchy at all — the FHS inputs are the same
regardless of hierarchy. The golden files at the artifact root are therefore the
correct comparison target.

Marked slow because loading the FHS NetCDF and parquet files takes ~1–2 min.
Run with: pytest -m slow tests/02_data_prep/test_02a_fhs_population.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids

# ── Constants ─────────────────────────────────────────────────────────────────

GOLDEN_POPULATION_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/population/lsae_1209/current"
)
RAW_DATA_PATH = Path("/mnt/team/idd/pub/forecast-mbp/01-raw_data")

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/02a_fhs_population.py"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_main():
    """Import main() from the numbered script file via importlib."""
    spec = importlib.util.spec_from_file_location("02a_fhs_population", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _read_golden(filename: str, **kwargs) -> pd.DataFrame:
    return read_parquet_with_integer_ids(GOLDEN_POPULATION_ROOT / filename, **kwargs)


def _compare_values(result_path, golden_path, key_cols, value_cols):
    """Merge result and golden on key_cols, assert every value cell matches."""
    cols = key_cols + value_cols
    result = read_parquet_with_integer_ids(result_path, columns=cols)
    golden = read_parquet_with_integer_ids(golden_path, columns=cols)
    merged = result.merge(golden, on=key_cols, suffixes=("_r", "_g"))
    assert len(merged) == len(result) == len(golden), (
        f"Row count mismatch after merge: result={len(result)}, "
        f"golden={len(golden)}, merged={len(merged)}"
    )
    for col in value_cols:
        np.testing.assert_allclose(
            merged[f"{col}_r"].values,
            merged[f"{col}_g"].values,
            rtol=1e-5,
            err_msg=f"Value mismatch in column '{col}'",
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def part_a_outputs(tmp_path_factory):
    """Run Part A once for the whole module; return the output directory."""
    out_dir = tmp_path_factory.mktemp("02a_outputs")
    main = _load_main()
    main(population_write_path=out_dir, raw_data_path=RAW_DATA_PATH)
    return out_dir


# ── Tests: aa_2023_fhs_population_df ─────────────────────────────────────────

@pytest.mark.slow
def test_aa_fhs_population_schema(part_a_outputs):
    result = read_parquet_with_integer_ids(part_a_outputs / "aa_2023_fhs_population_df.parquet")
    golden = _read_golden("aa_2023_fhs_population_df.parquet")
    assert set(result.columns) == set(golden.columns), (
        f"Column mismatch.\n  Got:      {sorted(result.columns)}\n"
        f"  Expected: {sorted(golden.columns)}"
    )


@pytest.mark.slow
def test_aa_fhs_population_row_count(part_a_outputs):
    result = read_parquet_with_integer_ids(part_a_outputs / "aa_2023_fhs_population_df.parquet")
    golden = _read_golden("aa_2023_fhs_population_df.parquet")
    assert len(result) == len(golden), (
        f"Row count mismatch: got {len(result)}, expected {len(golden)}"
    )


@pytest.mark.slow
def test_aa_fhs_population_values(part_a_outputs):
    """aa_population matches golden cell by cell for all 51k rows."""
    _compare_values(
        result_path=part_a_outputs / "aa_2023_fhs_population_df.parquet",
        golden_path=GOLDEN_POPULATION_ROOT / "aa_2023_fhs_population_df.parquet",
        key_cols=["age_group_id", "location_id", "year_id", "sex_id"],
        value_cols=["aa_population"],
    )


# ── Tests: as_2023_fhs_population_df ─────────────────────────────────────────

@pytest.mark.slow
def test_as_fhs_population_schema(part_a_outputs):
    result = read_parquet_with_integer_ids(part_a_outputs / "as_2023_fhs_population_df.parquet")
    golden = _read_golden("as_2023_fhs_population_df.parquet")
    assert set(result.columns) == set(golden.columns), (
        f"Column mismatch.\n  Got:      {sorted(result.columns)}\n"
        f"  Expected: {sorted(golden.columns)}"
    )


@pytest.mark.slow
def test_as_fhs_population_row_count(part_a_outputs):
    result = read_parquet_with_integer_ids(part_a_outputs / "as_2023_fhs_population_df.parquet")
    golden = _read_golden("as_2023_fhs_population_df.parquet")
    assert len(result) == len(golden), (
        f"Row count mismatch: got {len(result)}, expected {len(golden)}"
    )


@pytest.mark.slow
def test_as_fhs_population_values(part_a_outputs):
    """population and as_population_fraction match golden cell by cell for all 2.6M rows."""
    _compare_values(
        result_path=part_a_outputs / "as_2023_fhs_population_df.parquet",
        golden_path=GOLDEN_POPULATION_ROOT / "as_2023_fhs_population_df.parquet",
        key_cols=["age_group_id", "location_id", "year_id", "sex_id"],
        value_cols=["population", "as_population_fraction"],
    )
