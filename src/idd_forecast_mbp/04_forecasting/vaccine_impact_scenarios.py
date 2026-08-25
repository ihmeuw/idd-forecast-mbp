"""
Post-hoc malaria vaccine impact scenarios.

Takes the all-age malaria forecast produced by the
2026_07_31_full_model_selection_results model run, disaggregates it to age/sex
using the frozen as_rr pattern, applies the vaccine cohort protection, and
re-aggregates to no-vaccine / vaccine / difference series per ssp scenario.

Why this is exact rather than an approximation: neither as_rr nor population
carries a draw, so the age sum factors out of the draw dimension --

    vaccine_count(loc, yr, draw)
        = aa_count(loc, yr, draw) x (1 - SUM_as f(loc, yr, as) x protection(loc, yr, as))

-- leaving one draw-free burden-weighted reduction factor per (location, year)
that scales every draw identically. `inc_fraction` pairs with
`effective_protection_case`, `mort_fraction` with `effective_protection_death`.

Protection is reported at admin1 (the coverage geography) and broadcast to each
admin1's admin2 children. That is not an approximation: the protection fractions
carry no population weighting, so a child's value equals its parent's.

Only vaccine-eligible locations are computed; everywhere else the reduction is
identically zero, so vaccine equals no-vaccine by construction.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.processing.vaccine_impact import (
    DAH_SCENARIO,
    MODEL_RUN,
    SSP_SCENARIOS,
    burden_weighted_reduction,
    draw_level,
    eligible_locations,
    scenario_totals,
    summarize,
)


class Clock:
    """Elapsed-time logger. Long steps must report as they run, not at the end:
    a silent multi-minute stage is indistinguishable from a hung one. Passed into
    the lib functions as a plain callable so they carry no reporting policy."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0

    def __call__(self, msg: str) -> None:
        now = time.perf_counter()
        print(f"[{now - self.t0:7.2f}s +{now - self.last:6.2f}s] {msg}", flush=True)
        self.last = now


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ve-variant", choices=rfc.VE_VARIANTS, required=True)
    p.add_argument("--product-scenario", choices=rfc.PRODUCT_SCENARIOS, default="projected")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    clock = Clock()
    variant = args.ve_variant
    args.out_dir.mkdir(parents=True, exist_ok=True)

    products = args.product_scenario
    prot_path = (rfc.MAL_VACCINE_COHORTS_READ_PATH
                 / rfc.MAL_VACCINE_COHORTS_FILENAME_TEMPLATE.format(
                     variant=variant, products=products))
    protection = read_parquet_with_integer_ids(
        prot_path,
        columns=["location_id", "year_id", "age_group_id", "sex_id",
                 "effective_protection_case", "effective_protection_death"],
    )
    clock(f"protection loaded: {len(protection):,} rows")

    ds = xr.open_dataset(rfc.MAL_FORECAST_OUTPUTS_READ_PATH
                         / f"malaria_forecast_{SSP_SCENARIOS[0]}_{DAH_SCENARIO}.nc")
    fc_locs = {int(x) for x in ds.location_id.values}
    years = [int(y) for y in ds.year_id.values]
    ds.close()

    mapping = eligible_locations(rfc.VACCINE_COVERAGE_FILE, fc_locs)
    locs = mapping["location_id"].tolist()
    clock(f"eligible admin2 locations: {len(locs):,} of {len(fc_locs):,}; years {years[0]}-{years[-1]}")

    reduction = burden_weighted_reduction(protection, mapping, years, clock)
    clock(f"reduction factors: {len(reduction):,} location-years; "
          f"peak inc {reduction.r_inc.max():.4f}, peak mort {reduction.r_mort.max():.4f}")

    aa_pop = read_parquet_with_integer_ids(
        rfc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
        columns=["location_id", "year_id", "population"],
        filters=[("location_id", "in", locs), ("year_id", "in", years)],
    )

    totals = pd.concat([scenario_totals(s, reduction, aa_pop, locs, years, clock) for s in SSP_SCENARIOS],
                       ignore_index=True)
    summary = summarize(totals)
    summary["ve_variant"] = variant
    summary["product_scenario"] = products
    summary["model_run"] = MODEL_RUN
    summary["dah_scenario"] = DAH_SCENARIO

    draws = draw_level(totals)
    draws["ve_variant"] = variant
    draws["product_scenario"] = products
    draws_out = args.out_dir / f"vaccine_impact_draws_ve_{variant}_{products}.parquet"
    write_parquet(draws, draws_out)
    clock(f"wrote {len(draws):,} draw-level rows to {draws_out}")

    out = args.out_dir / f"vaccine_impact_summary_ve_{variant}_{products}.parquet"
    write_parquet(summary, out)
    clock(f"wrote {len(summary):,} rows to {out}")
    return summary


if __name__ == "__main__":
    main()
