"""
Combine per-scenario AS and AA mean files into single analysis datasets.

Reads the mean.nc files produced by create_and_combine_as_and_aa_draws.py for
every cause/measure/scenario combination, stitches in historical data (pre-2022),
adds population and hierarchy metadata, and writes two files:
  - full_aa_ds_{dah_scenario}.nc  (all-age)
  - full_as_ds_{dah_scenario}.nc  (age-specific)

These are the primary analysis datasets used for figures and tables.
"""

import argparse

import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.netcdf import (
    convert_to_xarray,
    ensure_id_coordinates_are_integers,
    read_netcdf_with_integer_ids,
    write_netcdf,
)
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids

# --- ARGUMENT PARSING ---

parser = argparse.ArgumentParser(
    description="Build combined AA and AS mean analysis datasets."
)
parser.add_argument("--run_date", type=str, required=True)
parser.add_argument(
    "--dah_scenarios",
    type=str,
    nargs="+",
    default=["Baseline", "Constant"],
    help="DAH scenarios to process.",
)
args = parser.parse_args()

run_date = args.run_date
dah_scenarios = args.dah_scenarios

# --- PATHS AND CONSTANTS ---

PROCESSED_DATA_PATH = mbpc.MODEL_ROOT / "02-processed_data"
UPLOAD_DATA_PATH = mbpc.MODEL_ROOT / "05-upload_data" / "upload_folders"

cause_map = mbpc.cause_map
measure_map = mbpc.measure_map
ssp_scenarios = mbpc.ssp_scenarios

hierarchy_ds_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.nc"
hierarchy_ds = read_netcdf_with_integer_ids(hierarchy_ds_path, engine="netcdf4")

aa_full_malaria_ds = read_netcdf_with_integer_ids(
    PROCESSED_DATA_PATH / "aa_full_malaria_ds.nc", engine="netcdf4"
)
aa_full_dengue_ds = read_netcdf_with_integer_ids(
    PROCESSED_DATA_PATH / "aa_full_dengue_ds.nc", engine="netcdf4"
)
as_full_malaria_ds = read_netcdf_with_integer_ids(
    PROCESSED_DATA_PATH / "as_2003_full_malaria_ds.nc", engine="netcdf4"
)
as_full_dengue_ds = read_netcdf_with_integer_ids(
    PROCESSED_DATA_PATH / "as_2003_full_dengue_ds.nc", engine="netcdf4"
)

aa_full_population_ds = read_netcdf_with_integer_ids(
    f"{PROCESSED_DATA_PATH}/aa_2023_full_population_ds.nc"
)
as_full_population_ds = read_netcdf_with_integer_ids(
    f"{PROCESSED_DATA_PATH}/as_2023_full_population_ds.nc"
)

past_data = {
    "aa": {"malaria": aa_full_malaria_ds, "dengue": aa_full_dengue_ds},
    "as": {"malaria": as_full_malaria_ds, "dengue": as_full_dengue_ds},
}

future_path_templates = {
    "aa": {
        "malaria": "{UPLOAD_DATA_PATH}/{run_date}/aa_cause_malaria_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}/mean.nc",
        "dengue": "{UPLOAD_DATA_PATH}/{run_date}/aa_cause_dengue_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}/mean.nc",
    },
    "as": {
        "malaria": "{UPLOAD_DATA_PATH}/{run_date}/as_cause_malaria_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}/mean.nc",
        "dengue": "{UPLOAD_DATA_PATH}/{run_date}/as_cause_dengue_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}/mean.nc",
    },
}

chunk_dims = {
    "aa": {"location_id": 1500, "year_id": 79},
    "as": {"location_id": 1500, "year_id": 79},
}

# --- BUILD DATASETS ---

for dah_scenario in dah_scenarios:
    print(f"\n=== dah_scenario={dah_scenario} ===")

    for agg_type in ["aa", "as"]:
        population_ds = aa_full_population_ds if agg_type == "aa" else as_full_population_ds
        datasets = []

        for cause in cause_map:
            past_ds = past_data[agg_type][cause]
            cause_datasets = []

            for ssp_scenario in ssp_scenarios:
                measure_datasets = []

                for measure in measure_map:
                    # Load future mean
                    tmpl = future_path_templates[agg_type][cause]
                    future_path = tmpl.format(
                        UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
                        run_date=run_date,
                        measure=measure,
                        metric="count",
                        ssp_scenario=ssp_scenario,
                        dah_scenario=dah_scenario,
                    )
                    print(f"  Loading {future_path}")
                    future_ds = read_netcdf_with_integer_ids(future_path, engine="netcdf4")
                    future_ds = future_ds.rename({"mean_value": "count"})

                    # Select historical data for this measure (pre-2022)
                    short_name = f"{cause}_{measure_map[measure]['short']}_count"
                    past_df = (
                        past_ds[["location_id", "year_id", short_name]]
                        .to_dataframe()
                        .reset_index()
                    )
                    past_df = past_df[past_df["year_id"] < 2022].copy()
                    past_df = past_df.rename(columns={short_name: "count"})
                    past_measure_ds = convert_to_xarray(
                        past_df,
                        dimensions=["location_id", "year_id"],
                        dimension_dtypes={"location_id": "int32", "year_id": "int16"},
                        variable_dtypes={"count": "float32"},
                        auto_optimize_dtypes=False,
                    )

                    combined_ds = xr.concat(
                        [past_measure_ds, future_ds], dim="year_id"
                    ).sortby("year_id")
                    combined_ds = combined_ds.assign_coords(measure=measure).expand_dims("measure")
                    measure_datasets.append(combined_ds)

                ssp_ds = xr.concat(measure_datasets, dim="measure")
                ssp_ds = ssp_ds.assign_coords(ssp_scenario=ssp_scenario).expand_dims("ssp_scenario")
                cause_datasets.append(ssp_ds)

            cause_ds = xr.concat(cause_datasets, dim="ssp_scenario")
            cause_ds = cause_ds.assign_coords(cause=cause).expand_dims("cause")
            datasets.append(cause_ds)

        full_ds = xr.concat(datasets, dim="cause")
        full_ds = xr.merge([full_ds, population_ds], join="left")
        full_ds["rate"] = full_ds["count"] / full_ds["population"]
        full_ds = xr.merge([full_ds, hierarchy_ds], join="left")
        full_ds = ensure_id_coordinates_are_integers(full_ds)

        out_path = UPLOAD_DATA_PATH / run_date / f"full_{agg_type}_ds_{dah_scenario}.nc"
        n_causes = len(list(cause_map.keys()))
        n_measures = len(list(measure_map.keys()))
        n_scenarios = len(list(ssp_scenarios.keys()))
        chunks = {
            **chunk_dims[agg_type],
            "cause": n_causes,
            "measure": n_measures,
            "ssp_scenario": n_scenarios,
        }
        print(f"  Writing {out_path}")
        write_netcdf(
            full_ds,
            out_path,
            compression=True,
            compression_level=4,
            chunking=True,
            chunk_by_dim=chunks,
            engine="netcdf4",
        )
        print(f"  Done: {agg_type.upper()} {dah_scenario}")
