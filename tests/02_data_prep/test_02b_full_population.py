"""Regression tests for 02b_full_population.py.

Strategy: seed a temp directory with the lsae_1209 hierarchy file and the
Part A golden outputs (both from the flat stage root), run Part B main() with
lsae_hierarchy="lsae_1209", then compare against the full-population golden
files that live in the flat stage root.

Using lsae_1209 ensures the test can run against the existing LSAE CSV files
that are only present for that hierarchy. The golden files at the flat root
were also produced with lsae_1209.

Marked slow because Part B reads 24 years × 2 levels of LSAE CSVs and
performs large cross-joins for age-specific population.
Run with: pytest -m slow tests/02_data_prep/test_02b_full_population.py
"""
import importlib.util
import shutil
from pathlib import Path

import pytest

from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids

# ── Constants ─────────────────────────────────────────────────────────────────

GOLDEN_ROOT = Path("/mnt/team/idd/pub/forecast-mbp/02-processed_data")
RAW_DATA_PATH = Path("/mnt/team/idd/pub/forecast-mbp/01-raw_data")
READ_PATH = GOLDEN_ROOT  # age_specific_fhs/ lives here

# Part B is tested with lsae_1209: this is what the golden files were built with,
# and the LSAE CSV files exist for this hierarchy.
LSAE_HIERARCHY = "lsae_1209"

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/02b_full_population.py"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_main():
    """Import main() from the numbered script file via importlib."""
    spec = importlib.util.spec_from_file_location("02b_full_population", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _read_golden(filename: str):
    return read_parquet_with_integer_ids(GOLDEN_ROOT / filename)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def seeded_dir(tmp_path_factory):
    """Seed a temp dir with lsae_1209 inputs from the flat stage root."""
    seed = tmp_path_factory.mktemp("02b_seed")

    # Hierarchy file written by script 01
    shutil.copy(
        GOLDEN_ROOT / f"full_hierarchy_2023_{LSAE_HIERARCHY}.parquet",
        seed / f"full_hierarchy_2023_{LSAE_HIERARCHY}.parquet",
    )
    # Part A outputs (golden files, produced with lsae_1209 config)
    for filename in ("aa_2023_fhs_population_df.parquet", "as_2023_fhs_population_df.parquet"):
        shutil.copy(GOLDEN_ROOT / filename, seed / filename)

    return seed


@pytest.fixture(scope="module")
def part_b_outputs(seeded_dir):
    """Run Part B once for the whole module; return the output directory."""
    main = _load_main()
    main(
        processed_data_path=seeded_dir,
        lsae_hierarchy=LSAE_HIERARCHY,
        raw_data_path=RAW_DATA_PATH,
        read_path=READ_PATH,
    )
    return seeded_dir


# ── Tests: aa_2023_full_population_df ────────────────────────────────────────

@pytest.mark.slow
def test_aa_full_population_schema(part_b_outputs):
    result = read_parquet_with_integer_ids(part_b_outputs / "aa_2023_full_population_df.parquet")
    golden = _read_golden("aa_2023_full_population_df.parquet")
    assert set(result.columns) == set(golden.columns), (
        f"Column mismatch.\n  Got:      {sorted(result.columns)}\n"
        f"  Expected: {sorted(golden.columns)}"
    )


@pytest.mark.slow
def test_aa_full_population_row_count(part_b_outputs):
    result = read_parquet_with_integer_ids(part_b_outputs / "aa_2023_full_population_df.parquet")
    golden = _read_golden("aa_2023_full_population_df.parquet")
    assert len(result) == len(golden), (
        f"Row count mismatch: got {len(result)}, expected {len(golden)}"
    )


@pytest.mark.slow
def test_aa_full_population_location_ids(part_b_outputs):
    result = read_parquet_with_integer_ids(part_b_outputs / "aa_2023_full_population_df.parquet")
    golden = _read_golden("aa_2023_full_population_df.parquet")
    assert set(result["location_id"].unique()) == set(golden["location_id"].unique()), (
        "Location ID sets differ"
    )


@pytest.mark.slow
def test_aa_full_population_total_sum(part_b_outputs):
    result = read_parquet_with_integer_ids(part_b_outputs / "aa_2023_full_population_df.parquet")
    golden = _read_golden("aa_2023_full_population_df.parquet")
    result_sum = result["population"].sum()
    golden_sum = golden["population"].sum()
    assert result_sum == pytest.approx(golden_sum, rel=1e-4), (
        f"Total population sum mismatch: got {result_sum:.2f}, expected {golden_sum:.2f}"
    )


@pytest.mark.slow
def test_aa_full_population_no_negatives(part_b_outputs):
    result = read_parquet_with_integer_ids(part_b_outputs / "aa_2023_full_population_df.parquet")
    assert (result["population"] >= 0).all(), "Negative population values found"


# ── Tests: as_2023_full_population_df ────────────────────────────────────────

@pytest.mark.slow
def test_as_full_population_schema(part_b_outputs):
    result = read_parquet_with_integer_ids(part_b_outputs / "as_2023_full_population_df.parquet")
    golden = _read_golden("as_2023_full_population_df.parquet")
    assert set(result.columns) == set(golden.columns), (
        f"Column mismatch.\n  Got:      {sorted(result.columns)}\n"
        f"  Expected: {sorted(golden.columns)}"
    )


@pytest.mark.slow
def test_as_full_population_row_count(part_b_outputs):
    result = read_parquet_with_integer_ids(part_b_outputs / "as_2023_full_population_df.parquet")
    golden = _read_golden("as_2023_full_population_df.parquet")
    assert len(result) == len(golden), (
        f"Row count mismatch: got {len(result)}, expected {len(golden)}"
    )


@pytest.mark.slow
def test_as_full_population_location_ids(part_b_outputs):
    result = read_parquet_with_integer_ids(part_b_outputs / "as_2023_full_population_df.parquet")
    golden = _read_golden("as_2023_full_population_df.parquet")
    assert set(result["location_id"].unique()) == set(golden["location_id"].unique()), (
        "Location ID sets differ"
    )


@pytest.mark.slow
def test_as_full_population_total_sum(part_b_outputs):
    result = read_parquet_with_integer_ids(part_b_outputs / "as_2023_full_population_df.parquet")
    golden = _read_golden("as_2023_full_population_df.parquet")
    result_sum = result["population"].sum()
    golden_sum = golden["population"].sum()
    assert result_sum == pytest.approx(golden_sum, rel=1e-4), (
        f"Total population sum mismatch: got {result_sum:.2f}, expected {golden_sum:.2f}"
    )


@pytest.mark.slow
def test_as_full_population_age_sex_completeness(part_b_outputs):
    """Every location-year in the all-age file should have all age-sex combos."""
    aa = read_parquet_with_integer_ids(part_b_outputs / "aa_2023_full_population_df.parquet")
    as_ = read_parquet_with_integer_ids(part_b_outputs / "as_2023_full_population_df.parquet")
    age_sex_df = read_parquet_with_integer_ids(part_b_outputs / "age_sex_df.parquet")

    n_age_sex = len(age_sex_df)
    expected_rows = len(aa) * n_age_sex
    assert len(as_) == pytest.approx(expected_rows, rel=0.01), (
        f"as_ row count {len(as_)} doesn't match aa rows × age-sex combos "
        f"({len(aa)} × {n_age_sex} = {expected_rows})"
    )
