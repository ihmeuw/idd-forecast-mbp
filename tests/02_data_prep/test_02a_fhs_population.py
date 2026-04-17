"""Regression tests for 02a_fhs_population.py.

Strategy: run main() with real data inputs but write to tmp_path, then
compare outputs against the golden files that were produced under lsae_1209
and live in the flat stage root.

Part A does not depend on lsae_hierarchy at all — the FHS inputs are the same
regardless of hierarchy. The golden files at the flat root are therefore the
correct comparison target.

Marked slow because loading the FHS NetCDF and parquet files takes ~1–2 min.
Run with: pytest -m slow tests/02_data_prep/test_02a_fhs_population.py
"""
import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest

from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids

# ── Constants ─────────────────────────────────────────────────────────────────

GOLDEN_ROOT = Path("/mnt/team/idd/pub/forecast-mbp/02-processed_data")
RAW_DATA_PATH = Path("/mnt/team/idd/pub/forecast-mbp/01-raw_data")
# read_path is used for age_specific_fhs/age_metadata.parquet, which lives in
# the flat stage root (a static reference file, never versioned).
READ_PATH = GOLDEN_ROOT

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


def _read_golden(filename: str) -> pd.DataFrame:
    return read_parquet_with_integer_ids(GOLDEN_ROOT / filename)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def part_a_outputs(tmp_path_factory):
    """Run Part A once for the whole module; return the output directory."""
    out_dir = tmp_path_factory.mktemp("02a_outputs")
    main = _load_main()
    main(processed_data_path=out_dir, raw_data_path=RAW_DATA_PATH, read_path=READ_PATH)
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
def test_aa_fhs_population_location_ids(part_a_outputs):
    result = read_parquet_with_integer_ids(part_a_outputs / "aa_2023_fhs_population_df.parquet")
    golden = _read_golden("aa_2023_fhs_population_df.parquet")
    assert set(result["location_id"].unique()) == set(golden["location_id"].unique()), (
        "Location ID sets differ"
    )


@pytest.mark.slow
def test_aa_fhs_population_total_sum(part_a_outputs):
    result = read_parquet_with_integer_ids(part_a_outputs / "aa_2023_fhs_population_df.parquet")
    golden = _read_golden("aa_2023_fhs_population_df.parquet")
    result_sum = result["aa_population"].sum()
    golden_sum = golden["aa_population"].sum()
    assert result_sum == pytest.approx(golden_sum, rel=1e-4), (
        f"Total aa_population sum mismatch: got {result_sum:.2f}, expected {golden_sum:.2f}"
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
def test_as_fhs_population_location_ids(part_a_outputs):
    result = read_parquet_with_integer_ids(part_a_outputs / "as_2023_fhs_population_df.parquet")
    golden = _read_golden("as_2023_fhs_population_df.parquet")
    assert set(result["location_id"].unique()) == set(golden["location_id"].unique()), (
        "Location ID sets differ"
    )


@pytest.mark.slow
def test_as_fhs_population_total_sum(part_a_outputs):
    result = read_parquet_with_integer_ids(part_a_outputs / "as_2023_fhs_population_df.parquet")
    golden = _read_golden("as_2023_fhs_population_df.parquet")
    result_sum = result["population"].sum()
    golden_sum = golden["population"].sum()
    assert result_sum == pytest.approx(golden_sum, rel=1e-4), (
        f"Total population sum mismatch: got {result_sum:.2f}, expected {golden_sum:.2f}"
    )


@pytest.mark.slow
def test_as_fhs_population_fraction_range(part_a_outputs):
    """as_population_fraction should be in [0, 1] (sums to ~1 per location-year)."""
    result = read_parquet_with_integer_ids(part_a_outputs / "as_2023_fhs_population_df.parquet")
    assert result["as_population_fraction"].between(0, 1).all(), (
        "as_population_fraction values outside [0, 1]"
    )
