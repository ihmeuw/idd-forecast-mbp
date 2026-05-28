"""Regression tests for 02b_full_population.py.

Strategy: seed a temp directory with the lsae_1209 hierarchy file and the
Part A golden outputs, run Part B main() with lsae_hierarchy="lsae_1209",
then compare against the full-population golden files in the artifact dir.

Using lsae_1209 ensures the test can run against the existing LSAE CSV files
that are only present for that hierarchy.

Marked slow because Part B reads 24 years × 2 levels of LSAE CSVs and
performs large cross-joins for age-specific population.
Run with: pytest -m slow tests/02_data_prep/test_02b_full_population.py
"""
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids

# ── Constants ─────────────────────────────────────────────────────────────────

GOLDEN_HIERARCHY_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/hierarchy/lsae_1209/current"
)
GOLDEN_POPULATION_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/population/lsae_1209/current"
)
RAW_DATA_PATH = Path("/mnt/team/idd/pub/forecast-mbp/01-raw_data")

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


def _read_golden(filename: str, **kwargs):
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


def _compare_values_sampled(result_path, golden_path, key_cols, value_cols,
                             frac=0.05, random_state=42):
    """Sample frac of result rows, merge with golden, assert every value cell matches."""
    cols = key_cols + value_cols
    result = read_parquet_with_integer_ids(result_path, columns=cols)
    sample = result.sample(frac=frac, random_state=random_state)
    del result
    golden = read_parquet_with_integer_ids(golden_path, columns=cols)
    merged = sample.merge(golden, on=key_cols, suffixes=("_r", "_g"))
    assert len(merged) == len(sample), (
        f"Not all sampled rows found in golden: {len(merged)} of {len(sample)} matched"
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
def seeded_dir(tmp_path_factory):
    """Seed a temp dir with lsae_1209 inputs from the artifact dirs."""
    seed = tmp_path_factory.mktemp("02b_seed")

    shutil.copy(
        GOLDEN_HIERARCHY_ROOT / f"full_hierarchy_2023_{LSAE_HIERARCHY}.parquet",
        seed / f"full_hierarchy_2023_{LSAE_HIERARCHY}.parquet",
    )
    for filename in ("aa_2023_fhs_population_df.parquet", "as_2023_fhs_population_df.parquet"):
        shutil.copy(GOLDEN_POPULATION_ROOT / filename, seed / filename)

    return seed


@pytest.fixture(scope="module")
def part_b_outputs(seeded_dir):
    """Run Part B once for the whole module; return the output directory."""
    main = _load_main()
    main(
        population_write_path=seeded_dir,
        lsae_hierarchy=LSAE_HIERARCHY,
        raw_data_path=RAW_DATA_PATH,
        hierarchy_read_path=seeded_dir,
        population_read_path=seeded_dir,
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
def test_aa_full_population_values(part_b_outputs):
    """population matches golden cell by cell for all 5.2M rows."""
    _compare_values(
        result_path=part_b_outputs / "aa_2023_full_population_df.parquet",
        golden_path=GOLDEN_POPULATION_ROOT / "aa_2023_full_population_df.parquet",
        key_cols=["location_id", "year_id"],
        value_cols=["population"],
    )


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
def test_as_full_population_values(part_b_outputs):
    """population and as_population_fraction match golden on a 5% random sample (260M rows total)."""
    _compare_values_sampled(
        result_path=part_b_outputs / "as_2023_full_population_df.parquet",
        golden_path=GOLDEN_POPULATION_ROOT / "as_2023_full_population_df.parquet",
        key_cols=["age_group_id", "location_id", "year_id", "sex_id"],
        value_cols=["population", "as_population_fraction"],
        frac=0.05,
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


# ── Tests: age_sex_df ────────────────────────────────────────────────────────

@pytest.mark.slow
def test_age_sex_df_exact(part_b_outputs):
    """age_sex_df is a 50-row lookup table — must be exactly identical to golden."""
    result = read_parquet_with_integer_ids(part_b_outputs / "age_sex_df.parquet")
    golden = _read_golden("age_sex_df.parquet")
    assert result.equals(golden), (
        f"age_sex_df differs from golden.\n"
        f"  Result:\n{result}\n  Golden:\n{golden}"
    )
