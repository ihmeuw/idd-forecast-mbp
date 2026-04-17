"""Regression tests for 04_rake_as_A2_to_GBD.py.

Strategy: call main() with lsae_1209 golden inputs (malaria + dengue), write to
tmp_path, compare as_full_malaria_df.parquet and as_full_dengue_df.parquet against
the golden files.

Outputs are ~59M rows each, so value comparison uses a 5% location sample
(all years/ages/sexes for those locations) with exact cell-by-cell equality.

Run with: pytest -m slow --no-cov tests/02_data_prep/test_04_rake_as_A2_to_GBD.py
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
GOLDEN_MAL_AS_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/malaria/raked_as/lsae_1209/current"
)
GOLDEN_DEN_AS_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/dengue/raked_as/lsae_1209/current"
)

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/04_rake_as_A2_to_GBD.py"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_main():
    spec = importlib.util.spec_from_file_location("04_rake_as_A2_to_GBD", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raked_as_output(tmp_path_factory):
    """Run script 04 (malaria + dengue) once; return (mal_dir, den_dir)."""
    out_dir = tmp_path_factory.mktemp("04_raked_as")
    mal_dir = out_dir / "malaria"
    den_dir = out_dir / "dengue"
    main = _load_main()
    main(
        lsae_hierarchy="lsae_1209",
        hierarchy_read_path=GOLDEN_HIERARCHY_ROOT,
        population_read_path=GOLDEN_POPULATION_ROOT,
        mal_raked_aa_read_path=GOLDEN_MAL_AA_ROOT,
        den_raked_aa_read_path=GOLDEN_DEN_AA_ROOT,
        mal_raked_as_write_path=mal_dir,
        den_raked_as_write_path=den_dir,
        gbd_data_path=mbpc.GBD_DATA_PATH,
        causes=['malaria', 'dengue'],
    )
    return mal_dir, den_dir


# ── Tests — malaria ───────────────────────────────────────────────────────────

@pytest.mark.slow
def test_malaria_schema(raked_as_output):
    mal_dir, _ = raked_as_output
    result = pd.read_parquet(mal_dir / "as_full_malaria_df.parquet")
    golden = pd.read_parquet(GOLDEN_MAL_AS_ROOT / "as_full_malaria_df.parquet")
    assert set(result.columns) == set(golden.columns), (
        f"Column mismatch.\n  Got:      {sorted(result.columns)}\n"
        f"  Expected: {sorted(golden.columns)}"
    )


@pytest.mark.slow
def test_malaria_row_count(raked_as_output):
    mal_dir, _ = raked_as_output
    result = pd.read_parquet(mal_dir / "as_full_malaria_df.parquet")
    golden = pd.read_parquet(GOLDEN_MAL_AS_ROOT / "as_full_malaria_df.parquet")
    assert len(result) == len(golden), (
        f"Row count mismatch. Got {len(result)}, expected {len(golden)}"
    )


@pytest.mark.slow
def test_malaria_values(raked_as_output):
    """5% location sample, all years/ages/sexes, exact cell-by-cell."""
    mal_dir, _ = raked_as_output
    result = pd.read_parquet(mal_dir / "as_full_malaria_df.parquet")
    golden = pd.read_parquet(GOLDEN_MAL_AS_ROOT / "as_full_malaria_df.parquet")

    rng = np.random.default_rng(seed=42)
    all_locs = golden["location_id"].unique()
    sample_locs = rng.choice(all_locs, size=max(1, int(len(all_locs) * 0.05)), replace=False)

    sort_cols = ["location_id", "year_id", "age_group_id", "sex_id"]
    value_cols = ["malaria_inc_count", "malaria_mort_count", "malaria_inc_rate", "malaria_mort_rate"]

    r = result[result["location_id"].isin(sample_locs)].sort_values(sort_cols).reset_index(drop=True)
    g = golden[golden["location_id"].isin(sample_locs)].sort_values(sort_cols).reset_index(drop=True)

    for col in value_cols:
        np.testing.assert_array_equal(
            r[col].values, g[col].values,
            err_msg=f"Value mismatch in column '{col}'",
        )


# ── Tests — dengue ────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_dengue_schema(raked_as_output):
    _, den_dir = raked_as_output
    result = pd.read_parquet(den_dir / "as_full_dengue_df.parquet")
    golden = pd.read_parquet(GOLDEN_DEN_AS_ROOT / "as_full_dengue_df.parquet")
    assert set(result.columns) == set(golden.columns), (
        f"Column mismatch.\n  Got:      {sorted(result.columns)}\n"
        f"  Expected: {sorted(golden.columns)}"
    )


@pytest.mark.slow
def test_dengue_row_count(raked_as_output):
    _, den_dir = raked_as_output
    result = pd.read_parquet(den_dir / "as_full_dengue_df.parquet")
    golden = pd.read_parquet(GOLDEN_DEN_AS_ROOT / "as_full_dengue_df.parquet")
    assert len(result) == len(golden), (
        f"Row count mismatch. Got {len(result)}, expected {len(golden)}"
    )


@pytest.mark.slow
def test_dengue_values(raked_as_output):
    """5% location sample, all years/ages/sexes, exact cell-by-cell."""
    _, den_dir = raked_as_output
    result = pd.read_parquet(den_dir / "as_full_dengue_df.parquet")
    golden = pd.read_parquet(GOLDEN_DEN_AS_ROOT / "as_full_dengue_df.parquet")

    rng = np.random.default_rng(seed=42)
    all_locs = golden["location_id"].unique()
    sample_locs = rng.choice(all_locs, size=max(1, int(len(all_locs) * 0.05)), replace=False)

    sort_cols = ["location_id", "year_id", "age_group_id", "sex_id"]
    value_cols = ["dengue_inc_count", "dengue_mort_count", "dengue_inc_rate", "dengue_mort_rate"]

    r = result[result["location_id"].isin(sample_locs)].sort_values(sort_cols).reset_index(drop=True)
    g = golden[golden["location_id"].isin(sample_locs)].sort_values(sort_cols).reset_index(drop=True)

    for col in value_cols:
        np.testing.assert_array_equal(
            r[col].values, g[col].values,
            err_msg=f"Value mismatch in column '{col}'",
        )
