"""Regression tests for 07a_forecasted_dataframes_non_draw_part.py.

Strategy: call main() with lsae_1209 golden inputs, write to tmp_path, compare
all 6 output parquets (3 SSP scenarios × 2 causes) against golden files.

~4.8M rows per file: full exact comparison.

Run with: pytest -m slow --no-cov tests/02_data_prep/test_07a_forecasted_dataframes_non_draw_part.py
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
LSAE_INPUT_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/lsae_1209"
)
DAH_READ_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/covariates/dah/current"
)
GOLDEN_FORECASTING_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data"
)

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/07a_forecasted_dataframes_non_draw_part.py"
)

SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_main():
    spec = importlib.util.spec_from_file_location("07a_forecasted_dataframes_non_draw_part", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def forecasting_output(tmp_path_factory):
    """Run script 07 once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("07_forecasting_non_draw")
    main = _load_main()
    main(
        lsae_hierarchy="lsae_1209",
        hierarchy_read_path=GOLDEN_HIERARCHY_ROOT,
        population_read_path=GOLDEN_POPULATION_ROOT,
        lsae_input_path=LSAE_INPUT_PATH,
        dah_read_path=DAH_READ_PATH,
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
@pytest.mark.parametrize("ssp_scenario", SSP_SCENARIOS)
def test_dengue_non_draw(forecasting_output, ssp_scenario):
    filename = f"dengue_forecast_scenario_{ssp_scenario}_non_draw_part.parquet"
    _check_file(
        forecasting_output, filename,
        sort_cols=["location_id", "year_id"],
        value_cols=["people_flood_days", "logit_urban_1km_threshold_300", "gdppc_mean", "population"],
        golden_root=GOLDEN_FORECASTING_ROOT,
    )


@pytest.mark.slow
@pytest.mark.parametrize("ssp_scenario", SSP_SCENARIOS)
def test_malaria_non_draw(forecasting_output, ssp_scenario):
    filename = f"malaria_forecast_scenario_{ssp_scenario}_non_draw_part.parquet"
    _check_file(
        forecasting_output, filename,
        sort_cols=["location_id", "year_id"],
        value_cols=["people_flood_days", "logit_urban_1km_threshold_300", "gdppc_mean", "mal_DAH_total"],
        golden_root=GOLDEN_FORECASTING_ROOT,
    )
