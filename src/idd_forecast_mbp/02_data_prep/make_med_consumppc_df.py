import pandas as pd
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import write_parquet
from idd_forecast_mbp.lib.versioning import finalize_artifact

MED_CONSUMPPC_SCENARIO_MAP: dict[str, float] = {
    "better":    2.6,
    "reference": 4.5,
    "worse":     8.5,
}

MED_CONSUMPPC_SOURCE_PATH = (
    "/mnt/share/resource_tracking/forecasting/poverty"
    "/climate_2025_income_distribution_forecasts"
    "/V3_consumption_forecasting_admin2_scenarios"
    "/admin2_median_consumppc_forecasts_scenarios_2021PPP.csv"
)

mbpc.MED_CONSUMPPC_WRITE_PATH.mkdir(parents=True, exist_ok=True)
out_path = mbpc.MED_CONSUMPPC_WRITE_PATH / "med_consumppc_mean.parquet"

df = pd.read_csv(
    MED_CONSUMPPC_SOURCE_PATH,
    usecols=["year_id", "location_id", "scenario", "consumppc"],
)
df = df.rename(columns={"consumppc": "med_consumppc"})

unknown = set(df["scenario"].unique()) - set(MED_CONSUMPPC_SCENARIO_MAP)
if unknown:
    raise ValueError(
        f"Unknown scenario values in source file: {unknown}. "
        "Update MED_CONSUMPPC_SCENARIO_MAP before proceeding."
    )

df["scenario"] = df["scenario"].map(MED_CONSUMPPC_SCENARIO_MAP)
df = df[["year_id", "location_id", "scenario", "med_consumppc"]]
df["location_id"]  = df["location_id"].astype("int32")
df["year_id"]      = df["year_id"].astype("int16")
df["scenario"]     = df["scenario"].astype("float32")
df["med_consumppc"] = df["med_consumppc"].astype("float64")

write_parquet(df, out_path)
print(f"Wrote {len(df):,} rows to {out_path}")
finalize_artifact(mbpc._A02_MED_CONSUMPPC)
