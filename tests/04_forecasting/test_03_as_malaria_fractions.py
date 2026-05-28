"""Regression tests for as_malaria_fractions.py.

Strategy: call main() with first_submission inputs (ssp245, Baseline, draw 000).
Compare both output NCs (incidence + mortality) against golden.

75M cells per file — checked via xarray .sel() slices:
  - full age × sex grids for 3 (location, year) combos  → 150 cells per data var
  - global mean of primary value variable

Run with: pytest -m slow --no-cov tests/04_forecasting/test_03_as_malaria_fractions.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# ── Constants ─────────────────────────────────────────────────────────────────

GOLDEN_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/malaria/lsae_1209/first_submission"
)
PROCESSED_DATA_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data"
)
HIERARCHY_READ_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/hierarchy/lsae_1209/current"
)

SSP = "ssp245"
DAH = "Baseline"
DRAW = "000"

# (location_id, year_id) slices — one nonzero early, one nonzero late, one zero
SLICE_COORDS = [
    (25355, 2022),
    (25355, 2080),
    (45007, 2022),
]

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/04_forecasting/as_malaria_fractions.py"
)

# ── Loader ────────────────────────────────────────────────────────────────────

def _load_main():
    spec = importlib.util.spec_from_file_location("as_malaria_fractions", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def malaria_fractions_output(tmp_path_factory):
    """Run as_malaria_fractions.main() once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("03_as_malaria_fractions")
    main = _load_main()
    main(
        ssp_scenario=SSP,
        draw=DRAW,
        dah_scenario=DAH,
        hold_variable="None",
        hierarchy_read_path=HIERARCHY_READ_PATH,
        processed_data_path=PROCESSED_DATA_PATH,
        forecasting_data_read_path=GOLDEN_ROOT,
        forecasting_data_write_path=out_dir,
    )
    return out_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_nc(out_dir, filename, data_vars, primary_var):
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

    r_mean = float(result[primary_var].mean())
    g_mean = float(golden[primary_var].mean())
    np.testing.assert_allclose(
        r_mean, g_mean, rtol=1e-5,
        err_msg=f"{filename} global mean mismatch for '{primary_var}'",
    )

    result.close()
    golden.close()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_malaria_fractions_incidence(malaria_fractions_output):
    _check_nc(
        malaria_fractions_output,
        filename=f"as_malaria_measure_incidence_ssp_scenario_{SSP}_dah_scenario_{DAH}_draw_{DRAW}_with_predictions.nc",
        data_vars=["gbd_location_id", "aa_malaria_inc_count", "malaria_inc_count_pred"],
        primary_var="malaria_inc_count_pred",
    )


@pytest.mark.slow
def test_malaria_fractions_mortality(malaria_fractions_output):
    _check_nc(
        malaria_fractions_output,
        filename=f"as_malaria_measure_mortality_ssp_scenario_{SSP}_dah_scenario_{DAH}_draw_{DRAW}_with_predictions.nc",
        data_vars=["gbd_location_id", "aa_malaria_mort_count", "malaria_mort_count_pred"],
        primary_var="malaria_mort_count_pred",
    )
