"""Build the gdppc covariate parquet from the FGH income forecast CSV.

Source: FGH climate-2025 income-distribution forecasts, V5 (better/reference/worse),
LSAE admin2, 2010 PPP, years 2000-2100.

GDP VARIES BY CLIMATE SCENARIO (restored 2026-08-03). Each source income scenario is
emitted under its mapped RCP label per GDPPC_SCENARIO_MAP: better->rcp26,
reference->rcp45, worse->rcp85. Confirmed with the data producer that the income
scenarios are meant to be matched to the burden RCP scenarios -- three scenarios, not
all nine combinations -- and that `better` corresponds to RCP2.6.

Note the scenarios separate only slowly: identical in 2023, ~0.2% apart by 2050 and
~1.8% by 2100, with `better` richest. So this changes forecasts by a few percent, not
qualitatively. Between 2026-07-07 and 2026-08-03 the pipeline replicated `reference`
under all three labels, so any forecast from that window has no income variation
across SSPs.

Output: gdppc_mean.parquet under GDPPC_WRITE_PATH; finalizes the _A02_GDPPC
artifact symlink. Scenario is stored as string ("rcp26"/"rcp45"/"rcp85") to
avoid float-representation drift on equality filters.

Importable: `main()` is callable from `00_prep_economic_variables.py`.
Standalone: `python make_gdppc_df.py` still works (no flags).
"""
import pandas as pd
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import write_parquet
from idd_forecast_mbp.lib.versioning import finalize_artifact

# Income scenario in the source CSV applied to ALL climate scenarios while GDP is
# decoupled from the SSP/RCP scenario (see module docstring).
REFERENCE_SCENARIO = "reference"

# Every pipeline RCP label an SSP maps to. Reference GDP is emitted under each so
# downstream per-RCP selection returns the reference trajectory regardless of SSP.
RCP_LABELS = sorted({v["rcp_scenario"] for v in mbpc.ssp_scenarios.values()})

# Full source-scenario -> RCP mapping, kept to document the source labels and to
# restore scenario-varying GDP later. NOT used while GDP is decoupled.
GDPPC_SCENARIO_MAP: dict[str, str] = {
    "better":    "rcp26",
    "reference": "rcp45",
    "worse":     "rcp85",
}

# Source path lives in constants.py so no absolute path is committed in a stage script.
GDPPC_SOURCE_PATH = mbpc.GDPPC_SOURCE_PATH


def main() -> None:
    mbpc.GDPPC_WRITE_PATH.mkdir(parents=True, exist_ok=True)
    gdppc_df_path = mbpc.GDPPC_WRITE_PATH / "gdppc_mean.parquet"

    src = pd.read_csv(
        GDPPC_SOURCE_PATH,
        usecols=["year_id", "location_id", "scenario", "gdppc_mean"],
    )

    missing = set(GDPPC_SCENARIO_MAP) - set(src["scenario"].unique())
    if missing:
        raise ValueError(
            f"Source file is missing income scenario(s) {sorted(missing)} "
            f"(found {sorted(src['scenario'].unique())}). Verify the source before re-running."
        )

    # Drop rows with no GDP value: in V5 these are exactly the zero-population
    # admin2s (zero_pop_masking_flag == 1), where log(gdppc) downstream is undefined.
    n_before = len(src)
    src = src[src["gdppc_mean"].notna()]
    n_dropped = n_before - len(src)

    # Each income scenario under its own RCP label, so a forecast under a given SSP gets
    # the income trajectory that belongs to it. Confirmed with the data producer: the
    # income scenarios are meant to be matched to the burden RCP scenarios (three
    # scenarios, not a 3x3 grid), with `better` pairing to RCP2.6.
    df = pd.concat(
        [
            src[src["scenario"] == source].assign(scenario=label)
            for source, label in GDPPC_SCENARIO_MAP.items()
        ],
        ignore_index=True,
    )

    df = df[["year_id", "location_id", "scenario", "gdppc_mean"]]
    df["location_id"] = df["location_id"].astype("int32")
    df["year_id"]     = df["year_id"].astype("int16")
    df["scenario"]    = df["scenario"].astype("string")
    df["gdppc_mean"]  = df["gdppc_mean"].astype("float64")

    write_parquet(df, gdppc_df_path)
    print(
        f"Wrote {len(df):,} rows to {gdppc_df_path} "
        f"(scenario-varying GDP: {GDPPC_SCENARIO_MAP}; dropped {n_dropped:,} zero-pop/NaN rows)"
    )
    finalize_artifact(mbpc._A02_GDPPC)


if __name__ == "__main__":
    main()
