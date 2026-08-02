"""Smoke + merge-correctness tests for 08b_build_dengue_forecast_inputs.py.

Dengue analog of test_08a. Not a golden regression (08b is new); instead:

  1. Produces a file (no exceptions from current lsae_1285 upstreams).
  2. Schema matches what predict expects (dims, vars) — NO dah_scenario; the two
     climate covariates carry a draw dim.
  3. No NaN in the FORECAST window (08b tolerates past NaN by design, like 08a's
     upstream; the forecast-window guarantee is what 07c enforces + 08b asserts).
  4. Spot-check a few cells against the upstream files 08b reads — catch wrong SSP
     filter, wrong draw column, wrong A0 mapping.

Run with: pytest -m slow --no-cov tests/02_data_prep/test_08b_build_dengue_forecast_inputs.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from idd_forecast_mbp import constants as mbpc

# ── Constants ─────────────────────────────────────────────────────────────────

SSP_SCENARIO = "ssp245"
RCP_SCENARIO = mbpc.ssp_scenarios[SSP_SCENARIO]["rcp_scenario"]
LSAE_HIERARCHY = mbpc.LSAE_HIERARCHY

EXPECTED_VARS = {
    "dengue_suitability",
    "relative_humidity",
    "weighted_1km_urban_threshold_300.0_simple_mean",
    "people_flood_days_per_capita",
    "gdppc_mean",
    "A0_location_id",
}
CLIMATE_DRAW_VARS = {"dengue_suitability", "relative_humidity"}

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/08b_build_dengue_forecast_inputs.py"
)


def _load_main():
    spec = importlib.util.spec_from_file_location(
        "build_08b_dengue_forecast_inputs", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def forecast_inputs_path(tmp_path_factory):
    """Run 08b once for ssp245; return path to the resulting netCDF."""
    out_dir = tmp_path_factory.mktemp("08b_forecast_inputs")
    main = _load_main()
    main(
        ssp_scenarios=[SSP_SCENARIO],
        lsae_hierarchy=LSAE_HIERARCHY,
        output_path=out_dir,
    )
    return out_dir / f"dengue_forecast_inputs_{SSP_SCENARIO}.nc"


@pytest.fixture(scope="module")
def ds(forecast_inputs_path):
    with xr.open_dataset(forecast_inputs_path) as opened:
        return opened.load()


# ── 1. File exists ────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_file_exists(forecast_inputs_path):
    assert forecast_inputs_path.exists(), f"08b did not write {forecast_inputs_path}"
    assert forecast_inputs_path.stat().st_size > 0


# ── 2. Schema ─────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_dims(ds):
    assert set(ds.dims) >= {"location_id", "year_id", "draw"}
    assert "dah_scenario" not in ds.dims, "dengue inputs must NOT carry a DAH dimension"
    assert ds.sizes["draw"] == 100
    assert ds.sizes["year_id"] == len(mbpc.ALL_YEARS)
    assert ds.sizes["location_id"] > 0


@pytest.mark.slow
def test_expected_vars_present(ds):
    missing = EXPECTED_VARS - set(ds.data_vars)
    assert not missing, f"Missing variables: {sorted(missing)}"


@pytest.mark.slow
def test_variable_dims(ds):
    assert ds["dengue_suitability"].dims == ("location_id", "year_id", "draw")
    assert ds["relative_humidity"].dims  == ("location_id", "year_id", "draw")
    assert ds["gdppc_mean"].dims == ("location_id", "year_id")
    assert ds["weighted_1km_urban_threshold_300.0_simple_mean"].dims == ("location_id", "year_id")
    assert ds["people_flood_days_per_capita"].dims == ("location_id", "year_id")
    assert ds["A0_location_id"].dims == ("location_id",)


# ── 3. No NaN in the forecast window ──────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("var", sorted(EXPECTED_VARS - {"A0_location_id"}))
def test_no_nan_forecast_window(ds, var):
    future = ds.sel(year_id=slice(mbpc.FORECAST_YEARS[0], mbpc.FORECAST_YEARS[-1]))
    arr = future[var].values
    n_nan = int(np.isnan(arr).sum())
    assert n_nan == 0, (
        f"{var} has {n_nan} NaN in the forecast window ({mbpc.FORECAST_YEARS[0]}-"
        f"{mbpc.FORECAST_YEARS[-1]}) out of {arr.size}. 07c should have dropped these locs."
    )


# ── 4. Merge correctness — point checks against source files ─────────────────

@pytest.fixture(scope="module")
def hierarchy_df():
    return pd.read_parquet(
        mbpc.HIERARCHY_READ_PATH / f"full_hierarchy_2023_{LSAE_HIERARCHY}.parquet"
    )


@pytest.mark.slow
def test_gdppc_matches_source(ds):
    """Catch ssp/scenario mix-up: gdppc_mean[loc, year] equals the gdppc source value
    for that (loc, year, scenario=rcp_scenario)."""
    loc = int(ds.location_id.values[0])
    year = 2020
    nc_val = float(ds["gdppc_mean"].sel(location_id=loc, year_id=year).item())
    src = pd.read_parquet(
        mbpc.GDPPC_READ_PATH / "gdppc_mean.parquet",
        filters=[("location_id", "==", loc), ("year_id", "==", year),
                 ("scenario", "==", RCP_SCENARIO)],
    )
    assert len(src) == 1
    assert np.isclose(nc_val, float(src["gdppc_mean"].iloc[0]), rtol=1e-6, equal_nan=True)


@pytest.mark.slow
def test_dengue_suitability_draw_index(ds):
    """Catch draw-indexing bugs: dengue_suitability[loc, year, draw=42] equals the
    '042' column of the climate parquet at (loc, year)."""
    loc = int(ds.location_id.values[0])
    year = 2050
    draw_col = "042"
    nc_val = float(ds["dengue_suitability"].sel(location_id=loc, year_id=year, draw=42).item())
    suit_path = mbpc.CLIMATE_AGGREGATES_PATH / LSAE_HIERARCHY / f"dengue_suitability_{SSP_SCENARIO}.parquet"
    src = pd.read_parquet(
        suit_path, columns=[draw_col],
        filters=[("location_id", "==", loc), ("year_id", "==", year)],
    ).reset_index()
    assert len(src) == 1
    assert np.isclose(nc_val, float(src[draw_col].iloc[0]), rtol=1e-5, equal_nan=True)


@pytest.mark.slow
def test_a0_lookup_matches_hierarchy(ds, hierarchy_df):
    """Catch A0 mapping bugs: A0_location_id[loc] equals the hierarchy's A0 for every loc."""
    nc = pd.Series(ds["A0_location_id"].values.astype(int),
                   index=ds.location_id.values.astype(int), name="nc_A0")
    h = hierarchy_df.set_index("location_id")["A0_location_id"].astype(int)
    merged = nc.to_frame().join(h.rename("hier_A0"), how="left")
    mismatch = merged[merged["nc_A0"] != merged["hier_A0"]]
    assert mismatch.empty, f"{len(mismatch)} location(s) disagree with hierarchy A0."
