"""Smoke + merge-correctness tests for 08_build_malaria_forecast_inputs.py.

This is not a strict regression test against historical goldens (08 is a
new script, no prior version to compare against). Instead:

  1. Produces a file (no exceptions from current lsae_1285 upstreams).
  2. Schema matches what R expects (dims, vars, dtypes).
  3. No NaN values anywhere in any data variable.
  4. Spot-check ~5 cells against the same upstream files 08 reads from,
     to catch wiring mistakes — wrong SSP filter, DAH not broadcast,
     DAH-Constant formula bug, wrong draw column, wrong A0 mapping.

Run with: pytest -m slow --no-cov tests/02_data_prep/test_08_build_malaria_forecast_inputs.py
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
SUITABILITY_VARIANT = "mordecai_0_0"
LSAE_HIERARCHY = mbpc.LSAE_HIERARCHY

EXPECTED_VARS = {
    "mal_DAH_total_per_capita",
    "gdppc_mean",
    "weighted_1km_urban_threshold_300.0_simple_mean",
    "people_flood_days_per_capita",
    "malaria_suitability",
    "A0_location_id",
}

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "src/idd_forecast_mbp/02_data_prep/08_build_malaria_forecast_inputs.py"
)


def _load_main():
    spec = importlib.util.spec_from_file_location(
        "build_08_malaria_forecast_inputs", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def forecast_inputs_path(tmp_path_factory):
    """Run 08 once for ssp245; return path to the resulting netCDF."""
    out_dir = tmp_path_factory.mktemp("08_forecast_inputs")
    main = _load_main()
    main(
        ssp_scenarios=[SSP_SCENARIO],
        suitability_variant=SUITABILITY_VARIANT,
        lsae_hierarchy=LSAE_HIERARCHY,
        output_path=out_dir,
    )
    return out_dir / f"malaria_forecast_inputs_{SSP_SCENARIO}.nc"


@pytest.fixture(scope="module")
def ds(forecast_inputs_path):
    with xr.open_dataset(forecast_inputs_path) as opened:
        return opened.load()


# ── 1. File exists ────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_file_exists(forecast_inputs_path):
    assert forecast_inputs_path.exists(), f"08 did not write {forecast_inputs_path}"
    assert forecast_inputs_path.stat().st_size > 0


# ── 2. Schema ─────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_dims(ds):
    assert set(ds.dims) >= {"location_id", "year_id", "draw", "dah_scenario"}
    assert ds.sizes["draw"] == 100
    assert ds.sizes["dah_scenario"] == 2
    assert ds.sizes["year_id"] == len(mbpc.model_years)
    assert ds.sizes["location_id"] > 0


@pytest.mark.slow
def test_dah_scenario_labels(ds):
    assert list(ds.coords["dah_scenario"].values) == ["Baseline", "Constant"]


@pytest.mark.slow
def test_expected_vars_present(ds):
    missing = EXPECTED_VARS - set(ds.data_vars)
    assert not missing, f"Missing variables: {sorted(missing)}"


@pytest.mark.slow
def test_variable_dims(ds):
    # Per registry's DIMS_BY_KIND.
    assert ds["mal_DAH_total_per_capita"].dims == ("location_id", "year_id", "dah_scenario")
    assert ds["gdppc_mean"].dims                 == ("location_id", "year_id")
    assert ds["weighted_1km_urban_threshold_300.0_simple_mean"].dims == ("location_id", "year_id")
    assert ds["people_flood_days_per_capita"].dims == ("location_id", "year_id")
    assert ds["malaria_suitability"].dims        == ("location_id", "year_id", "draw")
    assert ds["A0_location_id"].dims             == ("location_id",)


# ── 3. No NaN anywhere ────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("var", sorted(EXPECTED_VARS - {"A0_location_id"}))
def test_no_nan(ds, var):
    arr = ds[var].values
    n_nan = int(np.isnan(arr).sum())
    assert n_nan == 0, (
        f"{var} has {n_nan} NaN values out of {arr.size}. "
        f"Bobby's rule: all covariates non-NaN for all rows."
    )


# ── 4. Merge correctness — point checks against source files ─────────────────

@pytest.fixture(scope="module")
def hierarchy_df():
    return pd.read_parquet(
        mbpc.HIERARCHY_READ_PATH / f"full_hierarchy_2023_{LSAE_HIERARCHY}.parquet"
    )


@pytest.mark.slow
def test_gdppc_matches_source(ds, hierarchy_df):
    """Catch ssp/scenario mix-up: gdppc_mean[loc, year] should equal the
    gdppc source value for that (loc, year, scenario=rcp_scenario)."""
    loc = int(ds.location_id.values[0])
    year = 2020
    nc_val = float(ds["gdppc_mean"].sel(location_id=loc, year_id=year).item())

    src = pd.read_parquet(
        mbpc.GDPPC_READ_PATH / "gdppc_mean.parquet",
        filters=[
            ("location_id", "==", loc),
            ("year_id", "==", year),
            ("scenario", "==", RCP_SCENARIO),
        ],
    )
    assert len(src) == 1, f"Expected 1 row in gdppc source for (loc={loc}, year={year}, rcp={RCP_SCENARIO}); got {len(src)}"
    src_val = float(src["gdppc_mean"].iloc[0])
    assert np.isclose(nc_val, src_val, rtol=1e-6, equal_nan=True), \
        f"gdppc: nc={nc_val}, source={src_val}"


@pytest.mark.slow
def test_dah_baseline_matches_source(ds, hierarchy_df):
    """Catch DAH broadcast bugs: mal_DAH_total_per_capita[loc, year, dah=Baseline]
    should equal the DAH source value for (A0(loc), year)."""
    loc = int(ds.location_id.values[0])
    year = 2018
    a0 = int(ds["A0_location_id"].sel(location_id=loc).item())

    nc_val = float(
        ds["mal_DAH_total_per_capita"]
        .sel(location_id=loc, year_id=year, dah_scenario="Baseline").item()
    )

    src = pd.read_parquet(
        mbpc.DAH_READ_PATH / "dah_df.parquet",
        filters=[("location_id", "==", a0), ("year_id", "==", year)],
    )
    # Per DECISIONS 2026-05-07: missing rows in source mean DAH = $0.
    src_val = float(src["mal_DAH_total_per_capita"].iloc[0]) if len(src) else 0.0
    assert np.isclose(nc_val, src_val, rtol=1e-6, equal_nan=True), \
        f"DAH-Baseline at (loc={loc}, A0={a0}, year={year}): nc={nc_val}, source={src_val}"


@pytest.mark.slow
def test_dah_constant_formula(ds):
    """Catch DAH-Constant formula bugs: per_capita[loc, year>2023, dah=Constant]
    should equal per_capita[loc, 2023, dah=Baseline] * pop[loc, 2023] / pop[loc, year]."""
    loc = int(ds.location_id.values[0])
    ref_year = 2023
    future_year = 2025

    per_capita_ref  = float(ds["mal_DAH_total_per_capita"].sel(
        location_id=loc, year_id=ref_year, dah_scenario="Baseline").item())

    pop_src = pd.read_parquet(
        mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
        filters=[("location_id", "==", loc), ("year_id", "in", [ref_year, future_year])],
    ).set_index("year_id")["population"]
    pop_ref    = float(pop_src[ref_year])
    pop_future = float(pop_src[future_year])

    expected = per_capita_ref * pop_ref / pop_future if pop_future > 0 else 0.0
    nc_val = float(ds["mal_DAH_total_per_capita"].sel(
        location_id=loc, year_id=future_year, dah_scenario="Constant").item())

    assert np.isclose(nc_val, expected, rtol=1e-5, equal_nan=True), \
        f"DAH-Constant at (loc={loc}, year={future_year}): nc={nc_val}, expected={expected}"


@pytest.mark.slow
def test_suitability_draw_index(ds):
    """Catch draw indexing bugs: malaria_suitability[loc, year, draw=42] should
    equal the '042' column of the suit parquet at (loc, year)."""
    loc = int(ds.location_id.values[0])
    year = 2050
    draw_int = 42
    draw_col = f"{draw_int:03d}"

    nc_val = float(ds["malaria_suitability"].sel(
        location_id=loc, year_id=year, draw=draw_int).item())

    suit_path = mbpc.get_malaria_suitability_path(SUITABILITY_VARIANT, SSP_SCENARIO, LSAE_HIERARCHY)
    src = pd.read_parquet(
        suit_path,
        columns=[draw_col],
        filters=[("location_id", "==", loc), ("year_id", "==", year)],
    ).reset_index()
    assert len(src) == 1, f"Expected 1 row in suitability source; got {len(src)}"
    src_val = float(src[draw_col].iloc[0])
    assert np.isclose(nc_val, src_val, rtol=1e-5, equal_nan=True), \
        f"suitability(draw={draw_col}) at (loc={loc}, year={year}): nc={nc_val}, source={src_val}"


@pytest.mark.slow
def test_a0_lookup_matches_hierarchy(ds, hierarchy_df):
    """Catch A0 mapping bugs: A0_location_id[loc] should equal hierarchy's A0
    for every location."""
    nc = pd.Series(
        ds["A0_location_id"].values.astype(int),
        index=ds.location_id.values.astype(int),
        name="nc_A0",
    )
    h = hierarchy_df.set_index("location_id")["A0_location_id"].astype(int)
    merged = nc.to_frame().join(h.rename("hier_A0"), how="left")
    mismatch = merged[merged["nc_A0"] != merged["hier_A0"]]
    assert mismatch.empty, (
        f"{len(mismatch)} location(s) have A0_location_id in the netCDF that "
        f"disagree with the hierarchy.\nFirst mismatches:\n{mismatch.head()}"
    )
