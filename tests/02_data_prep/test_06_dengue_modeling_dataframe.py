"""Regression tests for 06_dengue_modeling_dataframe.py.

Strategy: call main() with lsae_1209 golden inputs, write to tmp_path, compare
all 4 output parquets against golden files.

Small files (<2M rows): full comparison. Large files (~20M rows): 5% location sample.

Run with: pytest -m slow --no-cov tests/02_data_prep/test_06_dengue_modeling_dataframe.py
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
GOLDEN_DEN_AA_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/dengue/raked_aa/lsae_1209/current"
)
GOLDEN_DEN_AS_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/dengue/raked_as/lsae_1209/current"
)
GOLDEN_MODELING_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/dengue/modeling_dfs/lsae_1209/current"
)
LSAE_INPUT_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/lsae_1209"
)

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/06_dengue_modeling_dataframe.py"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_main():
    spec = importlib.util.spec_from_file_location("06_dengue_modeling_dataframe", SCRIPT_PATH)
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
    """Run script 06 once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("06_dengue_modeling")
    main = _load_main()
    main(
        lsae_hierarchy="lsae_1209",
        hierarchy_read_path=GOLDEN_HIERARCHY_ROOT,
        population_read_path=GOLDEN_POPULATION_ROOT,
        den_raked_aa_read_path=GOLDEN_DEN_AA_ROOT,
        den_raked_as_read_path=GOLDEN_DEN_AS_ROOT,
        den_modeling_write_path=out_dir,
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
        modeling_output, "aa_ge3_dengue_stage_1_modeling_df.parquet",
        sort_cols=["location_id", "year_id"],
        value_cols=["aa_dengue_inc_count", "aa_dengue_mort_count", "dengue_suitability", "log_gdppc_mean", "logit_urban_1km_threshold_300"],
        golden_root=GOLDEN_MODELING_ROOT,
        sample=False,
    )


@pytest.mark.slow
def test_as_md(modeling_output):
    _check_file(
        modeling_output, "as_md_dengue_modeling_df.parquet",
        sort_cols=["location_id", "year_id", "age_group_id", "sex_id"],
        value_cols=["log_dengue_inc_rate", "logit_dengue_cfr", "dengue_mort_rate", "dengue_suitability"],
        golden_root=GOLDEN_MODELING_ROOT,
        sample=True,
    )


@pytest.mark.slow
def test_base_md(modeling_output):
    _check_file(
        modeling_output, "base_md_dengue_modeling_df.parquet",
        sort_cols=["location_id", "year_id", "age_group_id", "sex_id"],
        value_cols=["base_log_dengue_inc_rate", "base_logit_dengue_cfr", "base_dengue_mort_rate", "dengue_suitability"],
        golden_root=GOLDEN_MODELING_ROOT,
        sample=False,
    )


@pytest.mark.slow
def test_rest_md(modeling_output):
    _check_file(
        modeling_output, "rest_md_dengue_modeling_df.parquet",
        sort_cols=["location_id", "year_id", "age_group_id", "sex_id"],
        value_cols=["log_dengue_inc_rate", "logit_dengue_cfr", "dengue_mort_rate", "base_log_dengue_inc_rate", "base_logit_dengue_cfr"],
        golden_root=GOLDEN_MODELING_ROOT,
        sample=True,
    )
