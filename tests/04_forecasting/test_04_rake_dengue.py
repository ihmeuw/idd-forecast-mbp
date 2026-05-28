"""Regression tests for rake_dengue.py.

Strategy: call main() with first_submission inputs (ssp245, draw 000).
Compare output NC against golden.

144M rows — checked via xarray .sel() slices:
  - full age × sex grids for 3 (location, year) combos  → 150 cells per data var
  - global mean of each data variable

Run with: pytest -m slow --no-cov tests/04_forecasting/test_04_rake_dengue.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# ── Constants ─────────────────────────────────────────────────────────────────

GOLDEN_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/dengue/lsae_1209/first_submission"
)
PROCESSED_DATA_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data"
)
DENGUE_RAKED_AS_READ_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/dengue/raked_as/lsae_1209/current"
)
MODELING_DATA_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/dengue_cfr_model/lsae_1209/current"
)

SSP = "ssp245"
DRAW = "000"

# (location_id, year_id) slices — two nonzero, one NaN location
SLICE_COORDS = [
    (574, 2000),
    (574, 2050),
    (54471, 2022),
]

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/04_forecasting/rake_dengue.py"
)

# ── Loader ────────────────────────────────────────────────────────────────────

def _load_main():
    spec = importlib.util.spec_from_file_location("rake_dengue", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rake_dengue_output(tmp_path_factory):
    """Run rake_dengue.main() once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("04_rake_dengue")
    main = _load_main()
    main(
        ssp_scenario=SSP,
        draw=DRAW,
        hold_variable="None",
        dengue_raked_as_read_path=DENGUE_RAKED_AS_READ_PATH,
        modeling_data_path=MODELING_DATA_PATH,
        processed_data_path=PROCESSED_DATA_PATH,
        forecasting_data_read_path=GOLDEN_ROOT,
        forecasting_data_write_path=out_dir,
    )
    return out_dir


# ── Helper ────────────────────────────────────────────────────────────────────

def _check_nc(out_dir, filename, data_vars):
    result = xr.open_dataset(out_dir / filename)
    golden = xr.open_dataset(GOLDEN_ROOT / filename)

    assert dict(result.sizes) == dict(golden.sizes), (
        f"{filename} dim mismatch.\n  Got: {dict(result.sizes)}\n"
        f"  Expected: {dict(golden.sizes)}"
    )
    assert set(result.data_vars) == set(data_vars), (
        f"{filename} data_vars mismatch.\n  Got: {set(result.data_vars)}\n"
        f"  Expected: {set(data_vars)}"
    )

    for loc_id, year_id in SLICE_COORDS:
        for var in data_vars:
            r = result[var].sel(location_id=loc_id, year_id=year_id).values
            g = golden[var].sel(location_id=loc_id, year_id=year_id).values
            np.testing.assert_allclose(
                r, g, rtol=1e-5, atol=1e-8, equal_nan=True,
                err_msg=f"{filename} '{var}' mismatch at location={loc_id}, year={year_id}",
            )

    for var in data_vars:
        r_mean = float(result[var].mean())
        g_mean = float(golden[var].mean())
        np.testing.assert_allclose(
            r_mean, g_mean, rtol=1e-5,
            err_msg=f"{filename} global mean mismatch for '{var}'",
        )

    result.close()
    golden.close()


# ── Test ──────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_rake_dengue(rake_dengue_output):
    _check_nc(
        rake_dengue_output,
        filename=f"raked_dengue_forecast_ssp_scenario_{SSP}_draw_{DRAW}_with_predictions.nc",
        data_vars=["base_log_dengue_inc_rate_pred", "logit_dengue_cfr_pred"],
    )
