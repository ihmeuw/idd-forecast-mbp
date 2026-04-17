"""Regression tests for 05_malaria_modeling_dataframe.py.

Strategy: call main() with lsae_1209 golden inputs, write to tmp_path, compare
all 5 output parquets against golden files.

Small files (<2M rows): full exact comparison.
Large files (as_md, rest_md ~13M rows): 5% location sample, exact cell-by-cell.

Run with: pytest -m slow --no-cov tests/02_data_prep/test_05_malaria_modeling_dataframe.py
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
GOLDEN_MAL_AS_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/malaria/raked_as/lsae_1209/current"
)
GOLDEN_MODELING_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/modeling_dfs/lsae_1209/current"
)
LSAE_INPUT_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/lsae_1209"
)
DAH_READ_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/covariates/dah/current"
)

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/05_malaria_modeling_dataframe.py"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_main():
    spec = importlib.util.spec_from_file_location("05_malaria_modeling_dataframe", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _sample_locs(golden_df, seed=42, frac=0.05):
    rng = np.random.default_rng(seed=seed)
    all_locs = golden_df["location_id"].unique()
    return rng.choice(all_locs, size=max(1, int(len(all_locs) * frac)), replace=False)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def modeling_output(tmp_path_factory):
    """Run script 05 once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("05_malaria_modeling")
    main = _load_main()
    main(
        lsae_hierarchy="lsae_1209",
        hierarchy_read_path=GOLDEN_HIERARCHY_ROOT,
        population_read_path=GOLDEN_POPULATION_ROOT,
        mal_raked_aa_read_path=GOLDEN_MAL_AA_ROOT,
        mal_raked_as_read_path=GOLDEN_MAL_AS_ROOT,
        mal_modeling_write_path=out_dir,
        dah_read_path=DAH_READ_PATH,
        lsae_input_path=LSAE_INPUT_PATH,
    )
    return out_dir


# ── Helper: schema + row count + values for one file ─────────────────────────

def _check_file(out_dir, filename, sort_cols, value_cols, golden_root, sample=False):
    result = pd.read_parquet(out_dir / filename)
    golden = pd.read_parquet(golden_root / filename)

    assert set(result.columns) == set(golden.columns), (
        f"{filename} column mismatch.\n  Got: {sorted(result.columns)}\n"
        f"  Expected: {sorted(golden.columns)}"
    )
    assert len(result) == len(golden), (
        f"{filename} row count mismatch. Got {len(result)}, expected {len(golden)}"
    )

    if sample:
        locs = _sample_locs(golden)
        result = result[result["location_id"].isin(locs)]
        golden = golden[golden["location_id"].isin(locs)]

    result = result.sort_values(sort_cols).reset_index(drop=True)
    golden = golden.sort_values(sort_cols).reset_index(drop=True)

    for col in value_cols:
        np.testing.assert_allclose(
            result[col].values, golden[col].values,
            rtol=1e-10, atol=1e-12, equal_nan=True,
            err_msg=f"{filename}: value mismatch in '{col}'",
        )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_aa_ge3_stage1(modeling_output):
    _check_file(
        modeling_output, "aa_ge3_malaria_stage_1_modeling_df.parquet",
        sort_cols=["location_id", "year_id"],
        value_cols=["malaria_inc_count", "malaria_mort_count", "malaria_pfpr", "mean_temperature", "malaria_suitability"],
        golden_root=GOLDEN_MODELING_ROOT,
        sample=False,
    )


@pytest.mark.slow
def test_aa_md_pfpr(modeling_output):
    _check_file(
        modeling_output, "aa_md_malaria_pfpr_modeling_df.parquet",
        sort_cols=["location_id", "year_id"],
        value_cols=["malaria_inc_count", "malaria_mort_count", "malaria_pfpr", "logit_malaria_pfpr", "A0_malaria_pfpr"],
        golden_root=GOLDEN_MODELING_ROOT,
        sample=False,
    )


@pytest.mark.slow
def test_as_md(modeling_output):
    _check_file(
        modeling_output, "as_md_malaria_modeling_df.parquet",
        sort_cols=["location_id", "year_id", "age_group_id", "sex_id"],
        value_cols=["malaria_mort_rate", "malaria_inc_rate", "log_malaria_mort_rate", "malaria_pfpr"],
        golden_root=GOLDEN_MODELING_ROOT,
        sample=True,
    )


@pytest.mark.slow
def test_base_md(modeling_output):
    _check_file(
        modeling_output, "base_md_malaria_modeling_df.parquet",
        sort_cols=["location_id", "year_id", "age_group_id", "sex_id"],
        value_cols=["base_malaria_mort_rate", "base_malaria_inc_rate", "base_malaria_pfpr", "base_logit_malaria_pfpr"],
        golden_root=GOLDEN_MODELING_ROOT,
        sample=False,
    )


@pytest.mark.slow
def test_rest_md(modeling_output):
    _check_file(
        modeling_output, "rest_md_malaria_modeling_df.parquet",
        sort_cols=["location_id", "year_id", "age_group_id", "sex_id"],
        value_cols=["malaria_mort_rate", "malaria_inc_rate", "log_malaria_mort_rate", "base_malaria_pfpr"],
        golden_root=GOLDEN_MODELING_ROOT,
        sample=True,
    )
