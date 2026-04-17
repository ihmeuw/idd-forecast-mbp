"""Regression tests for 03_make_covariate_means.py.

Strategy: call main() with lsae_1209 inputs (hierarchy + population from golden dirs,
DAH and lsae covariates from existing artifact paths, forecasting data from the flat
04-forecasting_data dir), write to tmp_path, then compare covariate_means.nc against
the golden file variable by variable on a 5% sample of locations.

Marked slow because it reads climate, flooding, income, and DAH data for 3 SSP scenarios.
Run with: pytest -m slow tests/02_data_prep/test_03_make_covariate_means.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# ── Constants ─────────────────────────────────────────────────────────────────

GOLDEN_HIERARCHY_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/hierarchy/lsae_1209/current"
)
GOLDEN_POPULATION_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/population/lsae_1209/current"
)
GOLDEN_COV_MEANS_ROOT = Path(
    "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/covariates/covariate_means/lsae_1209/current"
)

# External input paths (not versioned by this pipeline)
FORECASTING_DATA_PATH = Path("/mnt/team/idd/pub/forecast-mbp/04-forecasting_data")
DAH_READ_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/covariates/dah/current"
)
LSAE_INPUT_PATH = Path(
    "/mnt/team/idd/pub/forecast-mbp/02-processed_data/lsae_1209"
)

LSAE_HIERARCHY = "lsae_1209"

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/03_make_covariate_means.py"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_main():
    """Import main() from the numbered script file via importlib."""
    spec = importlib.util.spec_from_file_location("03_make_covariate_means", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cov_means_output(tmp_path_factory):
    """Run script 03 once for the whole module; return the output directory."""
    out_dir = tmp_path_factory.mktemp("03_cov_means")
    main = _load_main()
    main(
        lsae_hierarchy=LSAE_HIERARCHY,
        hierarchy_read_path=GOLDEN_HIERARCHY_ROOT,
        population_read_path=GOLDEN_POPULATION_ROOT,
        cov_means_write_path=out_dir,
        forecasting_data_path=FORECASTING_DATA_PATH,
        dah_read_path=DAH_READ_PATH,
        lsae_input_path=LSAE_INPUT_PATH,
    )
    return out_dir


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_covariate_means_schema(cov_means_output):
    """Output has the same variables and dimension names as golden."""
    result = xr.open_dataset(cov_means_output / "covariate_means.nc")
    golden = xr.open_dataset(GOLDEN_COV_MEANS_ROOT / "covariate_means.nc")
    try:
        assert set(result.data_vars) == set(golden.data_vars), (
            f"Variable mismatch.\n  Got:      {sorted(result.data_vars)}\n"
            f"  Expected: {sorted(golden.data_vars)}"
        )
        assert set(result.dims) == set(golden.dims), (
            f"Dimension mismatch.\n  Got:      {sorted(result.dims)}\n"
            f"  Expected: {sorted(golden.dims)}"
        )
    finally:
        result.close()
        golden.close()


@pytest.mark.slow
def test_covariate_means_dim_sizes(cov_means_output):
    """Output has the same dimension sizes as golden."""
    result = xr.open_dataset(cov_means_output / "covariate_means.nc")
    golden = xr.open_dataset(GOLDEN_COV_MEANS_ROOT / "covariate_means.nc")
    try:
        assert dict(result.sizes) == dict(golden.sizes), (
            f"Dimension size mismatch.\n  Got:      {dict(result.sizes)}\n"
            f"  Expected: {dict(golden.sizes)}"
        )
    finally:
        result.close()
        golden.close()


@pytest.mark.slow
def test_covariate_means_values(cov_means_output):
    """All variables match golden on a 5% random sample of locations (all years and SSPs)."""
    result = xr.open_dataset(cov_means_output / "covariate_means.nc")
    golden = xr.open_dataset(GOLDEN_COV_MEANS_ROOT / "covariate_means.nc")
    try:
        all_locs = golden.location_id.values
        rng = np.random.default_rng(seed=42)
        n_sample = max(1, int(len(all_locs) * 0.05))
        sample_locs = rng.choice(all_locs, size=n_sample, replace=False)

        result_sub = result.sel(location_id=sample_locs)
        golden_sub = golden.sel(location_id=sample_locs)

        for var in golden.data_vars:
            np.testing.assert_allclose(
                result_sub[var].values,
                golden_sub[var].values,
                rtol=1e-4,
                equal_nan=True,
                err_msg=f"Value mismatch in variable '{var}'",
            )
    finally:
        result.close()
        golden.close()
