import pandas as pd
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import write_parquet
from idd_forecast_mbp.lib.versioning import finalize_artifact

# Mapping from income scenario names in the source CSV to the RCP scenario float
# values used by this pipeline's ssp_scenarios dict.
#
# Source file (FGH_2026_Feb): better, reference, worse
# Pipeline RCP values:        2.6,    4.5,       8.5
#   better    = rcp2.6 / ssp126
#   reference = rcp4.5 / ssp245
#   worse     = rcp8.5 / ssp585
#
# NOTE: This mapping is specific to FGH_2026_Feb. If the source file changes its
# scenario naming convention, verify the mapping before re-running this script.
GDPPC_SCENARIO_MAP: dict[str, float] = {
    "better":    2.6,
    "reference": 4.5,
    "worse":     8.5,
}

GDPPC_SOURCE_PATH = (
    "/mnt/share/resource_tracking/forecasting/poverty"
    "/climate_2025_income_distribution_forecasts"
    "/V3_consumption_forecasting_admin2_scenarios"
    "/admin2_gdppc_mean_forecasts_scenarios_2010PPP_RUSfix.csv"
)

mbpc.GDPPC_WRITE_PATH.mkdir(parents=True, exist_ok=True)
gdppc_df_path = mbpc.GDPPC_WRITE_PATH / "gdppc_mean.parquet"

df = pd.read_csv(GDPPC_SOURCE_PATH, usecols=["year_id", "location_id", "scenario", "gdppc_mean"])

unknown = set(df["scenario"].unique()) - set(GDPPC_SCENARIO_MAP)
if unknown:
    raise ValueError(
        f"Unknown scenario values in source file: {unknown}. "
        "Update GDPPC_SCENARIO_MAP before proceeding."
    )

df["scenario"] = df["scenario"].map(GDPPC_SCENARIO_MAP)
df = df[["year_id", "location_id", "scenario", "gdppc_mean"]]
df["location_id"] = df["location_id"].astype("int32")
df["year_id"] = df["year_id"].astype("int16")
df["scenario"] = df["scenario"].astype("float32")
df["gdppc_mean"] = df["gdppc_mean"].astype("float64")

write_parquet(df, gdppc_df_path)
print(f"Wrote {len(df):,} rows to {gdppc_df_path}")
finalize_artifact(mbpc._A02_GDPPC)
