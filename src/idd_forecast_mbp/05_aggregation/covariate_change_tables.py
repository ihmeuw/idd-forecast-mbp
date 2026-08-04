"""Tables: what changed between the 2025 run and now — by covariate, by SSP, and across SSPs.

Three questions, three tables, because they are genuinely different and conflating them is
how the "+32% DAH" and "+0.81% GDP" figures earlier in this work ended up misleading.

**T1 — levels and change.** For each covariate, each weighting, each SSP and horizon year:
the 2025 value, the current value, and the percent change. This is "what moved".

**T2 — scenario spread within a run.** Within each run separately, how far apart the SSPs
are (RCP8.5 − RCP2.6, RCP4.5 − RCP2.6, as a percent of RCP2.6). This is "how distinct are
the scenarios", which is a property of one run, not a comparison between runs.

**T3 — change in the spread.** T2 for the current run minus T2 for the 2025 run. This is
the "change in change": did the new inputs make the scenarios MORE or LESS distinguishable?
It is the quantity that matters for whether scenario contrasts in the paper got stronger or
weaker, and it cannot be read off T1.

Every table is produced under BOTH population and 2023-burden weighting, because for a
disease pipeline they answer different questions and can disagree in magnitude by 2x or
more (see `.claude/GDP_LEVERAGE_INVESTIGATION.md` — a GDP change measured unweighted was
14x smaller than the burden-weighted truth).

Burden outcomes get the same treatment, from the saved products, so input change and output
change sit side by side.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import click
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.processing.weights import (
    WEIGHT_SCHEMES,
    load_weights,
    weighted_rollup_to_levels,
)

SSPS = ("ssp126", "ssp245", "ssp585")
RCP = {k: v["name"] for k, v in mbpc.ssp_scenarios.items()}

# (label, 2025-run variable, current variable, in_current_model)
COVARIATES = [
    ("GDP per capita", "gdppc_mean", "gdppc_mean", True),
    ("DAH per capita", "dah_pc", "mal_DAH_total_per_capita", True),
    ("Malaria suitability", "malaria_suitability", "malaria_suitability", True),
    ("Flooding per capita", "flooding_pc", "people_flood_days_per_capita", False),
    ("Urbanisation", "urbanization",
     "weighted_1km_urban_threshold_300.0_simple_mean", False),
]

PREV_COV_NC = mbpc.PREVIOUS_COVARIATE_NC


def _global_series(
    da: xr.DataArray, locs: Sequence[int], years: Sequence[int],
    weights: pd.DataFrame, hierarchy: pd.DataFrame, **sel,
) -> dict[int, float]:
    d = da.sel(location_id=list(locs))
    if sel:
        d = d.sel(**sel)
    if "draw" in d.dims:
        d = d.mean("draw")
    df = d.to_dataframe(name="value").reset_index()[["location_id", "year_id", "value"]]
    df = df[df.year_id.isin([int(y) for y in years])]
    agg = weighted_rollup_to_levels(df, weights, hierarchy)
    g = agg[agg.location_id == 1]
    return {int(r.year_id): float(r.value) for r in g.itertuples()}


def covariate_levels(
    years: Sequence[int], weight_kinds: Sequence[str], dah: str = "Baseline"
) -> pd.DataFrame:
    """Long frame: covariate x weighting x ssp x year x run -> global value."""
    hierarchy = load_hierarchy()
    a2 = set(hierarchy.loc[hierarchy.level == 5, "location_id"])
    with xr.open_dataset(PREV_COV_NC) as ds:
        prev_locs = {int(x) for x in ds.location_id.values}
    with xr.open_dataset(
        mbpc.MAL_FORECAST_INPUTS_READ_PATH / f"malaria_forecast_inputs_{SSPS[0]}.nc"
    ) as ds:
        cur_locs = {int(x) for x in ds.location_id.values}
    shared = sorted(a2 & prev_locs & cur_locs)
    click.echo(f"shared admin-2 for the comparison: {len(shared):,}")

    rows = []
    wcache = {wk: load_weights(wk, years) for wk in weight_kinds}
    for label, pvar, cvar, in_model in COVARIATES:
        for ssp in SSPS:
            with xr.open_dataset(PREV_COV_NC) as pds:
                if pvar not in pds.data_vars:
                    continue
                pda = pds[pvar]
                for wk in weight_kinds:
                    pv = _global_series(pda, shared, years, wcache[wk], hierarchy,
                                        ssp_scenario=ssp)
                    for y, v in pv.items():
                        rows.append(dict(covariate=label, in_current_model=in_model,
                                         weight=wk, ssp=ssp, rcp=RCP[ssp], year=y,
                                         run="2025", value=v))
            cur_nc = (mbpc.MAL_FORECAST_INPUTS_READ_PATH
                      / f"malaria_forecast_inputs_{ssp}.nc")
            with xr.open_dataset(cur_nc) as cds:
                if cvar not in cds.data_vars:
                    continue
                cda = cds[cvar]
                sel = {"dah_scenario": dah} if "dah_scenario" in cda.dims else {}
                for wk in weight_kinds:
                    cv = _global_series(cda, shared, years, wcache[wk], hierarchy, **sel)
                    for y, v in cv.items():
                        rows.append(dict(covariate=label, in_current_model=in_model,
                                         weight=wk, ssp=ssp, rcp=RCP[ssp], year=y,
                                         run="current", value=v))
        click.echo(f"  {label}: done")
    return pd.DataFrame(rows)


def burden_levels(
    coupled_dir: Path, decoupled_dir: Path, years: Sequence[int], dah: str = "Baseline"
) -> pd.DataFrame:
    """Global burden from the saved products, both GDP-coupling arms."""
    rows = []
    for run, d in (("2025", decoupled_dir), ("current", coupled_dir)):
        for ssp in SSPS:
            p = d / f"all_age_summary_{ssp}_{dah}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            g = df[(df.level == 0) & (df.year_id.isin([int(y) for y in years]))]
            for r in g.itertuples():
                for meas, col in (("cases", "inc_count_mean"),
                                  ("deaths", "mort_count_mean")):
                    rows.append(dict(covariate=f"BURDEN: {meas}", in_current_model=True,
                                     weight="n/a", ssp=ssp, rcp=RCP[ssp],
                                     year=int(r.year_id), run=run,
                                     value=float(getattr(r, col))))
    return pd.DataFrame(rows)


def build_tables(levels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """T1 change, T2 within-run spread, T3 change in spread."""
    key = ["covariate", "in_current_model", "weight", "ssp", "rcp", "year"]
    wide = levels.pivot_table(index=key, columns="run", values="value").reset_index()
    wide["pct_change"] = 100 * (wide["current"] / wide["2025"] - 1)
    t1 = wide.sort_values(["covariate", "weight", "year", "ssp"])

    # T2: spread within each run, relative to RCP2.6
    piv = levels.pivot_table(
        index=["covariate", "in_current_model", "weight", "year", "run"],
        columns="ssp", values="value",
    ).reset_index()
    for hi in ("ssp245", "ssp585"):
        piv[f"{hi}_vs_126_pct"] = 100 * (piv[hi] / piv["ssp126"] - 1)
    t2 = piv.sort_values(["covariate", "weight", "year", "run"])

    # T3: change in the spread, current minus 2025
    sp = t2.pivot_table(
        index=["covariate", "in_current_model", "weight", "year"], columns="run",
        values=["ssp245_vs_126_pct", "ssp585_vs_126_pct"],
    )
    sp.columns = [f"{a}__{b}" for a, b in sp.columns]
    for hi in ("ssp245", "ssp585"):
        c, p = f"{hi}_vs_126_pct__current", f"{hi}_vs_126_pct__2025"
        if c in sp.columns and p in sp.columns:
            sp[f"{hi}_spread_change_pts"] = sp[c] - sp[p]
    t3 = sp.reset_index().sort_values(["covariate", "weight", "year"])
    return {"T1_levels_and_change": t1, "T2_spread_within_run": t2,
            "T3_change_in_spread": t3}


@click.command()
@click.option("--coupled-dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="product dir of the CURRENT (scenario-varying GDP) run")
@click.option("--decoupled-dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="product dir of the previous-arm run, for the burden rows")
@click.option("--year", "years", multiple=True, default=(2050, 2100),
              show_default=True, type=int)
@click.option("--weight", "weight_kinds", multiple=True,
              type=click.Choice(list(WEIGHT_SCHEMES)),
              default=("population", "mort2023"), show_default=True)
@click.option("--output-dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
def main(coupled_dir, decoupled_dir, years, weight_kinds, output_dir):
    """Build T1/T2/T3 and write them as CSV plus a readable digest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cov = covariate_levels(years, weight_kinds)
    bur = burden_levels(coupled_dir, decoupled_dir, years)
    levels = pd.concat([cov, bur], ignore_index=True)
    levels.to_csv(output_dir / "levels_long.csv", index=False)

    tables = build_tables(levels)
    lines = []
    for name, t in tables.items():
        t.to_csv(output_dir / f"{name}.csv", index=False)
        lines.append(f"\n{'='*100}\n{name}\n{'='*100}\n")
        lines.append(t.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    (output_dir / "tables_digest.txt").write_text("\n".join(lines))
    click.echo(f"\nwrote {len(tables)} tables + levels_long.csv + tables_digest.txt "
               f"to {output_dir}")
    # print the headline: in-model covariates and burden, 2100
    t1 = tables["T1_levels_and_change"]
    head = t1[(t1.year == 2100) & t1.in_current_model]
    click.echo("\n--- 2100, quantities that are in the current model ---")
    click.echo(head[["covariate", "weight", "rcp", "2025", "current", "pct_change"]]
               .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    main()
