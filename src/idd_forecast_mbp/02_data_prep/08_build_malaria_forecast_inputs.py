"""Build malaria forecast input netCDFs (one per SSP scenario).

DEPENDENCY ORDER:
  1. Stage 03 model selection (R) must be done first — its winner's
     covariate list is what this script's --covariates flag should match.
  2. 07b_build_malaria_prediction_locations.py must be run first — this
     script reads its kept-location list and naively assumes those locs
     have non-NaN values for every requested covariate.

Output: one netCDF per requested SSP at
    {MAL_FORECAST_INPUTS_WRITE_PATH}/malaria_forecast_inputs_{ssp}.nc
with dims (location_id, year_id, draw, dah_scenario) and the requested
covariate variables. Years span 2000–2100 so R can both rake-to-observed
on past years and predict on future years from a single input file. Past
NaN is tolerated at the sanity-check stage (07b only guarantees coverage
2023+; see its docstring for the upstream-fix TODO).

Transforms (logit, log) are NOT applied here — raw values on disk, R
derives transforms at predict time. This matches the past-inputs
convention.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.array_builders import (
    read_draw_climate, read_shared_covariates, scalar_to_array, wide_to_array,
)
from idd_forecast_mbp.lib.io.covariate_registry import (
    COVARIATE_REGISTRY, DEFAULT_MALARIA_FORECAST_COVARIATES, DIMS_BY_KIND,
)
from idd_forecast_mbp.lib.io.netcdf import write_netcdf
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.processing.dah_scenarios import build_dah_array
from idd_forecast_mbp.lib.versioning import finalize_artifact

DAH_SCENARIOS: tuple[str, ...] = ("Baseline", "Constant")
# Year onward for which 07b guarantees non-NaN coverage. Past-year NaN is
# allowed in the sanity check until 01_map_to_admin_2 historical gap is fixed.
NAN_SANITY_CHECK_START_YEAR: int = 2023


def _resolve_flooding_path(lsae_hierarchy: str, ssp_scenario: str) -> str | None:
    """Probe the two known flooding filenames; return the first that exists."""
    base = Path(f"/mnt/team/rapidresponse/pub/flooding/results/output/{lsae_hierarchy}")
    for fname in [
        f"fldfrc_weightedmin_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
        f"fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
    ]:
        candidate = base / fname
        if candidate.exists():
            return str(candidate)
    return None


def main(
    ssp_scenarios: list[str] | None = None,
    covariates: list[str] | None = None,
    suitability_variant: str = mbpc.MALARIA_SUITABILITY_VARIANT,
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    years: list[int] | None = None,
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    population_read_path: Path = mbpc.POPULATION_READ_PATH,
    dah_read_path: Path = mbpc.DAH_READ_PATH,
    gdppc_read_path: Path = mbpc.GDPPC_READ_PATH,
    ldipc_read_path: Path = mbpc.LDIPC_READ_PATH,
    urban_read_path: Path = mbpc.URBAN_READ_PATH,
    med_consumppc_read_path: Path = mbpc.MED_CONSUMPPC_READ_PATH,
    forecast_locations_read_path: Path = mbpc.MAL_FORECAST_LOCATIONS_READ_PATH,
    output_path: Path = mbpc.MAL_FORECAST_INPUTS_WRITE_PATH,
) -> None:
    ssp_scenarios = ssp_scenarios or list(mbpc.ssp_scenarios.keys())
    covariates    = list(covariates or DEFAULT_MALARIA_FORECAST_COVARIATES)
    years         = list(years or mbpc.model_years)
    output_path   = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    unknown = set(covariates) - set(COVARIATE_REGISTRY)
    if unknown:
        raise ValueError(
            f"Unknown covariate(s): {sorted(unknown)}. "
            f"Add to COVARIATE_REGISTRY before requesting."
        )

    # ── 1. Hierarchy + prediction locations (from 07b) ───────────────────────
    print("Loading hierarchy...")
    hierarchy_df = read_parquet_with_integer_ids(
        Path(hierarchy_read_path) / f"full_hierarchy_2023_{lsae_hierarchy}.parquet"
    )

    print("Loading prediction-location set from 07b artifact...")
    loc_df = read_parquet_with_integer_ids(
        Path(forecast_locations_read_path) / "prediction_location_ids.parquet"
    )
    location_ids = sorted(loc_df["location_id"].astype(int).tolist())
    print(f"  Prediction locations: {len(location_ids)}, years: {len(years)}")

    # ── 2. Population (for DAH-Constant scenario) ─────────────────────────────
    print("Loading population...")
    population_df = read_parquet_with_integer_ids(
        Path(population_read_path) / "aa_2023_full_population_df.parquet",
        filters=[("location_id", "in", location_ids), ("year_id", "in", years)],
    )
    pop_col = "aa_population" if "aa_population" in population_df.columns else "population"
    population_arr = scalar_to_array(population_df, pop_col, location_ids, years)

    # ── 3. DAH array (SSP-independent — built once, reused per SSP) ──────────
    print("Loading DAH and building scenarios...")
    dah_df = read_parquet_with_integer_ids(Path(dah_read_path) / "dah_df.parquet")
    dah_array, dah_names = build_dah_array(
        location_ids=location_ids,
        years=years,
        hierarchy_df=hierarchy_df,
        dah_df=dah_df,
        population=population_arr,
        scenarios=DAH_SCENARIOS,
    )

    # ── 4. Static A0 lookup ──────────────────────────────────────────────────
    a0_lookup = (
        hierarchy_df.set_index("location_id")
        .loc[location_ids, "A0_location_id"]
        .values.astype(np.int32)
    )

    # ── 5. Group requested covariates by kind for dispatch ───────────────────
    by_kind: dict[str, list[str]] = collections.defaultdict(list)
    for cov in covariates:
        by_kind[COVARIATE_REGISTRY[cov].kind].append(cov)

    # ── 6. Per-SSP loop: read covariates, assemble Dataset, write ────────────
    for ssp_scenario in ssp_scenarios:
        print(f"\n=== Building forecast inputs for {ssp_scenario} ===")
        rcp_scenario = mbpc.ssp_scenarios[ssp_scenario]["rcp_scenario"]

        data_vars: dict[str, tuple] = {}

        # 6a. Climate (draw-varying)
        if by_kind["climate_draw"]:
            print("Reading climate (draw-varying)...")
            climate_arrays = read_draw_climate(
                location_ids=location_ids,
                years=years,
                ssp_scenario=ssp_scenario,
                lsae_hierarchy=lsae_hierarchy,
            )
            for name in by_kind["climate_draw"]:
                if name not in climate_arrays:
                    raise KeyError(
                        f"Climate variable {name!r} not returned by read_draw_climate; "
                        "check lib/io/array_builders.py:read_draw_climate vs the registry."
                    )
                data_vars[name] = (DIMS_BY_KIND["climate_draw"], climate_arrays[name])

        # 6b. Suitability (draw-varying, single variant)
        for name in by_kind["suitability"]:
            print(f"Reading {name} ({suitability_variant})...")
            suit_path = mbpc.get_malaria_suitability_path(
                suitability_variant, ssp_scenario, lsae_hierarchy
            )
            data_vars[name] = (
                DIMS_BY_KIND["suitability"],
                wide_to_array(suit_path, location_ids, years),
            )

        # 6c. Shared scalars + flooding (non-draw)
        if by_kind["shared_scalar"] or by_kind["flooding"]:
            print("Reading shared scalars and flooding...")
            flooding_path = _resolve_flooding_path(lsae_hierarchy, ssp_scenario)
            shared_arrays = read_shared_covariates(
                location_ids=location_ids,
                years=years,
                gdppc_read_path=gdppc_read_path,
                ldipc_read_path=ldipc_read_path,
                urban_read_path=urban_read_path,
                flooding_path=flooding_path,
                rcp_scenario=rcp_scenario,
                med_consumppc_read_path=med_consumppc_read_path,
            )
            for name in by_kind["shared_scalar"] + by_kind["flooding"]:
                if name not in shared_arrays:
                    raise KeyError(
                        f"Shared covariate {name!r} not returned by read_shared_covariates. "
                        "Either the source parquet's columns don't match the registry name, "
                        "or read_shared_covariates needs the relevant source path argument."
                    )
                kind = COVARIATE_REGISTRY[name].kind
                data_vars[name] = (DIMS_BY_KIND[kind], shared_arrays[name])

        # 6d. DAH (pre-built above)
        for name in by_kind["dah"]:
            data_vars[name] = (DIMS_BY_KIND["dah"], dah_array)

        # 6e. Static lookups
        for name in by_kind["static_lookup"]:
            if name == "A0_location_id":
                data_vars[name] = (DIMS_BY_KIND["static_lookup"], a0_lookup)
            else:
                raise NotImplementedError(
                    f"Static lookup {name!r} has no builder; only A0_location_id supported."
                )

        # 6f. Assemble Dataset + write
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "location_id":  np.asarray(location_ids, dtype=np.int32),
                "year_id":      np.asarray(years,        dtype=np.int16),
                "draw":         np.arange(len(mbpc.draws), dtype=np.int16),
                "dah_scenario": list(dah_names),
            },
        )

        out_file = output_path / f"malaria_forecast_inputs_{ssp_scenario}.nc"
        print(f"Writing {out_file}...")
        write_netcdf(ds, out_file)
        size_gb = out_file.stat().st_size / 1e9
        print(f"  Done. {size_gb:.2f} GB.")

        _assert_no_forecast_window_nan(ds, ssp_scenario)

    if Path(output_path) == Path(mbpc.MAL_FORECAST_INPUTS_WRITE_PATH):
        finalize_artifact(mbpc._A04_MAL_FORECAST_INPUTS)
    else:
        print(f"output_path overridden to {output_path}; skipping finalize_artifact.")


def _assert_no_forecast_window_nan(ds: xr.Dataset, ssp_scenario: str) -> None:
    """Raise if any data_var has NaN in year >= NAN_SANITY_CHECK_START_YEAR.

    07b is supposed to drop locations whose covariates are NaN in the forecast
    window. If we see NaN here, either 07b wasn't re-run after a source update
    or a covariate was added without a coverage check.
    """
    future = ds.sel(year_id=slice(NAN_SANITY_CHECK_START_YEAR, None))
    offenders = {}
    for name, da in future.data_vars.items():
        if da.dtype.kind != "f":
            continue  # int / static lookups
        n_nan = int(np.isnan(da.values).sum())
        if n_nan > 0:
            offenders[name] = n_nan
    if offenders:
        raise RuntimeError(
            f"[{ssp_scenario}] forecast-window NaN check failed: {offenders}\n"
            f"Re-run 07b_build_malaria_prediction_locations.py to refresh the "
            f"prediction-location set."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build malaria forecast input netCDFs")
    parser.add_argument("--ssp_scenarios", nargs="+", default=None,
                        help="SSP scenarios to build. Defaults to all three.")
    parser.add_argument("--covariates", nargs="+", default=None,
                        help="Covariate names to include. Must appear in "
                             "lib/io/covariate_registry.COVARIATE_REGISTRY. "
                             "Defaults to the current malaria winner's predictor set.")
    parser.add_argument("--suitability_variant", default=mbpc.MALARIA_SUITABILITY_VARIANT,
                        help="Which malaria_suitability variant to load when "
                             "'malaria_suitability' is requested.")
    parser.add_argument("--lsae_hierarchy", default=mbpc.LSAE_HIERARCHY)
    args = parser.parse_args()
    main(
        ssp_scenarios=args.ssp_scenarios,
        covariates=args.covariates,
        suitability_variant=args.suitability_variant,
        lsae_hierarchy=args.lsae_hierarchy,
    )
