"""Regression tests for forecasted_draw_specific_malaria_dataframes.py.

Strategy: call main() with lsae_1209 golden inputs, ssp245, draw 000.
Compare main draw file + 4 DAH scenario files against golden parquets.

~1.7M rows per file: 5% location sample.

Run with: pytest -m slow --no-cov tests/02_data_prep/test_08_forecasted_malaria_draw_dataframes.py
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
GOLDEN_MAL_AA_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/malaria/raked_aa/lsae_1209/current"
)
GOLDEN_MAL_AS_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/malaria/raked_as/lsae_1209/current"
)
GOLDEN_FORECASTING_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data"
)

SSP_SCENARIO = "ssp245"
DRAW = "000"

DAH_SCENARIOS = ["Baseline", "Constant", "Increasing", "Decreasing"]

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/forecasted_draw_specific_malaria_dataframes.py"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_main():
    spec = importlib.util.spec_from_file_location(
        "forecasted_draw_specific_malaria_dataframes", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _sample_locs(golden_df, seed=42, frac=0.05):
    rng = np.random.default_rng(seed=seed)
    all_locs = golden_df["location_id"].unique()
    return rng.choice(all_locs, size=max(1, int(len(all_locs) * frac)), replace=False)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def malaria_draw_output(tmp_path_factory):
    """Run the malaria draw script once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("08_malaria_draw")
    main = _load_main()
    main(
        ssp_scenario=SSP_SCENARIO,
        draw=DRAW,
        lsae_hierarchy="lsae_1209",
        hierarchy_read_path=GOLDEN_HIERARCHY_ROOT,
        mal_raked_aa_read_path=GOLDEN_MAL_AA_ROOT,
        mal_raked_as_read_path=GOLDEN_MAL_AS_ROOT,
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
def test_malaria_draw_base(malaria_draw_output):
    filename = f"malaria_forecast_ssp_scenario_{SSP_SCENARIO}_draw_{DRAW}.parquet"
    _check_file(
        malaria_draw_output, filename,
        sort_cols=["location_id", "year_id"],
        value_cols=["logit_malaria_pfpr", "log_gdppc_mean", "log_mal_DAH_total_per_capita",
                    "malaria_suitability", "year_to_rake_to"],
        golden_root=GOLDEN_FORECASTING_ROOT,
    )


@pytest.mark.slow
@pytest.mark.parametrize("dah_scenario", DAH_SCENARIOS)
def test_malaria_dah_scenario(malaria_draw_output, dah_scenario):
    filename = f"malaria_forecast_ssp_scenario_{SSP_SCENARIO}_dah_scenario_{dah_scenario}_draw_{DRAW}.parquet"
    _check_file(
        malaria_draw_output, filename,
        sort_cols=["location_id", "year_id"],
        value_cols=["logit_malaria_pfpr", "log_gdppc_mean", "mal_DAH_total_per_capita",
                    "malaria_suitability"],
        golden_root=GOLDEN_FORECASTING_ROOT,
    )
