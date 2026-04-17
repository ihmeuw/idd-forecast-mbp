"""Regression tests for 03_rake_aa_A2_to_GBD.py.

Strategy: call main() with lsae_1209 golden inputs, write to tmp_path, compare
aa_full_malaria_df.parquet and aa_full_dengue_df.parquet against golden files
on a 5% location sample.

Run with: pytest -m slow tests/02_data_prep/test_03_rake_aa_A2_to_GBD.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp import constants as mbpc

# ── Constants ─────────────────────────────────────────────────────────────────

GOLDEN_HIERARCHY_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/hierarchy/lsae_1209/current"
)
GOLDEN_POPULATION_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/population/lsae_1209/current"
)
GOLDEN_MAL_AA_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/malaria/raked_aa/lsae_1209/current"
)
GOLDEN_DEN_AA_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/dengue/raked_aa/lsae_1209/current"
)
LSAE_INPUT_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/lsae_1209"
)

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/03_rake_aa_A2_to_GBD.py"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_main():
    spec = importlib.util.spec_from_file_location("03_rake_aa_A2_to_GBD", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raked_aa_output(tmp_path_factory):
    """Run script 03 once; return (mal_dir, den_dir)."""
    out_dir = tmp_path_factory.mktemp("03_raked_aa")
    mal_dir = out_dir / "malaria"
    den_dir = out_dir / "dengue"
    main = _load_main()
    main(
        lsae_hierarchy="lsae_1209",
        hierarchy_read_path=GOLDEN_HIERARCHY_ROOT,
        population_read_path=GOLDEN_POPULATION_ROOT,
        mal_raked_aa_write_path=mal_dir,
        den_raked_aa_write_path=den_dir,
        gbd_data_path=mbpc.GBD_DATA_PATH,
        lsae_input_path=LSAE_INPUT_PATH,
    )
    return mal_dir, den_dir


# ── Tests — malaria ───────────────────────────────────────────────────────────

@pytest.mark.slow
def test_malaria_schema(raked_aa_output):
    mal_dir, _ = raked_aa_output
    result = pd.read_parquet(mal_dir / "aa_full_malaria_df.parquet")
    golden = pd.read_parquet(GOLDEN_MAL_AA_ROOT / "aa_full_malaria_df.parquet")
    assert set(result.columns) == set(golden.columns), (
        f"Column mismatch.\n  Got:      {sorted(result.columns)}\n"
        f"  Expected: {sorted(golden.columns)}"
    )


@pytest.mark.slow
def test_malaria_row_count(raked_aa_output):
    mal_dir, _ = raked_aa_output
    result = pd.read_parquet(mal_dir / "aa_full_malaria_df.parquet")
    golden = pd.read_parquet(GOLDEN_MAL_AA_ROOT / "aa_full_malaria_df.parquet")
    assert len(result) == len(golden), (
        f"Row count mismatch. Got {len(result)}, expected {len(golden)}"
    )


@pytest.mark.slow
def test_malaria_values(raked_aa_output):
    """All value columns match golden exactly (cell-by-cell) for all rows."""
    mal_dir, _ = raked_aa_output
    result = pd.read_parquet(mal_dir / "aa_full_malaria_df.parquet").sort_values(["location_id", "year_id"]).reset_index(drop=True)
    golden = pd.read_parquet(GOLDEN_MAL_AA_ROOT / "aa_full_malaria_df.parquet").sort_values(["location_id", "year_id"]).reset_index(drop=True)

    value_cols = ["malaria_inc_count", "malaria_mort_count", "malaria_inc_rate", "malaria_mort_rate", "malaria_pfpr"]
    for col in value_cols:
        np.testing.assert_array_equal(
            result[col].values, golden[col].values,
            err_msg=f"Value mismatch in column '{col}'",
        )


# ── Tests — dengue ────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_dengue_schema(raked_aa_output):
    _, den_dir = raked_aa_output
    result = pd.read_parquet(den_dir / "aa_full_dengue_df.parquet")
    golden = pd.read_parquet(GOLDEN_DEN_AA_ROOT / "aa_full_dengue_df.parquet")
    assert set(result.columns) == set(golden.columns), (
        f"Column mismatch.\n  Got:      {sorted(result.columns)}\n"
        f"  Expected: {sorted(golden.columns)}"
    )


@pytest.mark.slow
def test_dengue_row_count(raked_aa_output):
    _, den_dir = raked_aa_output
    result = pd.read_parquet(den_dir / "aa_full_dengue_df.parquet")
    golden = pd.read_parquet(GOLDEN_DEN_AA_ROOT / "aa_full_dengue_df.parquet")
    assert len(result) == len(golden), (
        f"Row count mismatch. Got {len(result)}, expected {len(golden)}"
    )


@pytest.mark.slow
def test_dengue_values(raked_aa_output):
    """All value columns match golden exactly (cell-by-cell) for all rows."""
    _, den_dir = raked_aa_output
    result = pd.read_parquet(den_dir / "aa_full_dengue_df.parquet").sort_values(["location_id", "year_id"]).reset_index(drop=True)
    golden = pd.read_parquet(GOLDEN_DEN_AA_ROOT / "aa_full_dengue_df.parquet").sort_values(["location_id", "year_id"]).reset_index(drop=True)

    value_cols = ["dengue_inc_count", "dengue_mort_count", "dengue_inc_rate", "dengue_mort_rate"]
    for col in value_cols:
        np.testing.assert_array_equal(
            result[col].values, golden[col].values,
            err_msg=f"Value mismatch in column '{col}'",
        )
