"""Regression tests for forecasted_draw_specific_dengue_dataframes.py.

Strategy: call main() with lsae_1209 golden inputs, ssp245, draw 000.
Compare output against golden parquet.

~2.9M rows: 5% location sample.

Run with: pytest -m slow --no-cov tests/02_data_prep/test_09_forecasted_dengue_draw_dataframes.py
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
GOLDEN_DEN_AA_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/dengue/raked_aa/lsae_1209/current"
)
GOLDEN_DEN_AS_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/dengue/raked_as/lsae_1209/current"
)
GOLDEN_FORECASTING_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data"
)

SSP_SCENARIO = "ssp245"
DRAW = "000"

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/forecasted_draw_specific_dengue_dataframes.py"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_main():
    spec = importlib.util.spec_from_file_location(
        "forecasted_draw_specific_dengue_dataframes", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _sample_locs(golden_df, seed=42, frac=0.05):
    rng = np.random.default_rng(seed=seed)
    all_locs = golden_df["location_id"].unique()
    return rng.choice(all_locs, size=max(1, int(len(all_locs) * frac)), replace=False)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dengue_draw_output(tmp_path_factory):
    """Run the dengue draw script once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("09_dengue_draw")
    main = _load_main()
    main(
        ssp_scenario=SSP_SCENARIO,
        draw=DRAW,
        lsae_hierarchy="lsae_1209",
        hierarchy_read_path=GOLDEN_HIERARCHY_ROOT,
        den_raked_aa_read_path=GOLDEN_DEN_AA_ROOT,
        den_raked_as_read_path=GOLDEN_DEN_AS_ROOT,
        forecasting_data_read_path=GOLDEN_FORECASTING_ROOT,
        forecasting_data_write_path=out_dir,
    )
    return out_dir


# ── Helper ────────────────────────────────────────────────────────────────────

def _check_file(out_dir, filename, sort_cols, value_cols, golden_root):
    result = pd.read_parquet(out_dir / filename)
    golden = pd.read_parquet(golden_root / filename)

    assert set(result.columns) == set(golden.columns), (
        f"{filename} column mismatch.\n  Got: {sorted(result.columns)}\n"
        f"  Expected: {sorted(golden.columns)}"
    )
    assert len(result) == len(golden), (
        f"{filename} row count mismatch. Got {len(result)}, expected {len(golden)}"
    )

    locs = _sample_locs(golden)
    result = result[result["location_id"].isin(locs)].sort_values(sort_cols).reset_index(drop=True)
    golden = golden[golden["location_id"].isin(locs)].sort_values(sort_cols).reset_index(drop=True)

    for col in value_cols:
        np.testing.assert_allclose(
            result[col].values, golden[col].values,
            rtol=1e-10, atol=1e-12, equal_nan=True,
            err_msg=f"{filename}: value mismatch in '{col}'",
        )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_dengue_draw(dengue_draw_output):
    filename = f"dengue_forecast_ssp_scenario_{SSP_SCENARIO}_draw_{DRAW}.parquet"
    _check_file(
        dengue_draw_output, filename,
        sort_cols=["location_id", "year_id", "age_group_id", "sex_id"],
        value_cols=["logit_dengue_cfr", "log_gdppc_mean", "base_log_dengue_inc_rate",
                    "dengue_suitability", "logit_urban_1km_threshold_300"],
        golden_root=GOLDEN_FORECASTING_ROOT,
    )
