import pandas as pd
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import write_parquet
from idd_forecast_mbp.lib.versioning import finalize_artifact

# Mapping from income scenario names in the source CSV to the RCP scenario float
# values used by this pipeline's ssp_scenarios dict.
#
# Source file (V3_consumption_forecasting_admin2_scenarios): better, reference, worse
# Pipeline RCP values:                                       2.6,    4.5,       8.5
#   better    = rcp2.6 / ssp126
#   reference = rcp4.5 / ssp245
#   worse     = rcp8.5 / ssp585
#
# NOTE: This mapping is specific to the 2025 income distribution forecasts source.
# If the source file changes its scenario naming convention, verify the mapping
# before re-running this script.
LDIPC_SCENARIO_MAP: dict[str, float] = {
    "better":    2.6,
    "reference": 4.5,
    "worse":     8.5,
}

LDIPC_SOURCE_PATH = (
    "/mnt/share/resource_tracking/forecasting/poverty"
    "/climate_2025_income_distribution_forecasts"
    "/V3_consumption_forecasting_admin2_scenarios"
    "/admin2_ldipc_mean_forecasts_scenarios_2010PPP.csv"
)

mbpc.LDIPC_WRITE_PATH.mkdir(parents=True, exist_ok=True)
ldipc_df_path = mbpc.LDIPC_WRITE_PATH / "ldipc_mean.parquet"

df = pd.read_csv(LDIPC_SOURCE_PATH, usecols=["year_id", "location_id", "scenario", "ldipc_mean"])

unknown = set(df["scenario"].unique()) - set(LDIPC_SCENARIO_MAP)
if unknown:
    raise ValueError(
        f"Unknown scenario values in source file: {unknown}. "
        "Update LDIPC_SCENARIO_MAP before proceeding."
    )

df["scenario"] = df["scenario"].map(LDIPC_SCENARIO_MAP)
df = df[["year_id", "location_id", "scenario", "ldipc_mean"]]
df["location_id"] = df["location_id"].astype("int32")
df["year_id"] = df["year_id"].astype("int16")
df["scenario"] = df["scenario"].astype("float32")
df["ldipc_mean"] = df["ldipc_mean"].astype("float64")

write_parquet(df, ldipc_df_path)
print(f"Wrote {len(df):,} rows to {ldipc_df_path}")
finalize_artifact(mbpc._A02_LDIPC)
