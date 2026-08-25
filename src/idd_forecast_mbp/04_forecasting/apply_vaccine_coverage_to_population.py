"""
Apply malaria vaccine (RTS,S / R21) cohort coverage to real population.

Produces, for every (location, year, age_group_id, sex_id) in the vaccine-
relevant age range, the fraction and headcount of that age group that has
ever received dose 3, ever received dose 4, and is currently effectively
protected after waning.

All cohort logic lives in
`idd_forecast_mbp.lib.processing.vaccine_cohort_fractions`; this script is
I/O + orchestration only.

Data sources
------------
Population (real, confirmed 2026-08-24):
    constants.POPULATION_READ_PATH / "as_2023_full_population_df.parquet"
    258,438,800 rows; columns [age_group_id, location_id, year_id, sex_id,
    population, aa_population, as_population_fraction]; int64 IDs, float64
    population; year range 2000-2100. An FHS-location-set variant
    (as_2023_fhs_population_df.parquet, 2.6M rows) has the same schema and
    can be passed via --population-file.

Age-group bounds (real, not hardcoded):
    constants.AGE_SPECIFIC_FHS_PATH / "age_metadata.parquet"
    Read for age_group_years_start/end so bin boundaries always match the
    population source's own schema.

Coverage (real, delivered 2026-08-05, wired 2026-08-24):
    constants.VACCINE_COVERAGE_FILE, resolved through the node's `current`
    symlink: 01-raw_data/malaria_vaccine_coverage/current/
    2026_08_05_final_handoff_v2_edu_dtp3_lme.csv
    28,721 rows = 373 admin1 locations (subnat_id) across 37 countries x
    77 years (2024-2100), complete for every location. Products: rtss (71
    locations) and r21 (302). dose_3 <= 0.992, dose_4 <= 0.628, no nulls,
    exactly one vacc_name per location. All 373 locations were confirmed
    present in the population source.

    Override with --coverage-csv to run a different vintage; a new vintage
    should arrive as a new dated dir with `current` repointed.

Sex
---
Vaccine coverage is not sex-specific, so the fractions are computed once per
(location, year, age_group) and broadcast onto both sexes. sex_id is carried
through so the output joins 1:1 onto the age-sex population it came from.

Ages 20+
--------
The program started in 2024, so no one above age 20 can carry a dose. Those
age groups are simply not emitted (fraction is identically 0); join back onto
the full population table downstream if a complete age panel is needed.
"""
import argparse
from pathlib import Path

import pandas as pd

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.lib.data.vaccine_inputs import (
    load_age_groups,
    load_coverage,
    load_population,
)
from idd_forecast_mbp.lib.io.parquet import write_parquet
from idd_forecast_mbp.lib.processing.vaccine_cohort_fractions import (
    AGE_REFERENCE_PHI,
    D3_TRIGGER_AGE,
    D4_TRIGGER_AGE,
    DEFAULT_AGE_REFERENCE,
)
from idd_forecast_mbp.lib.processing.vaccine_coverage import (
    apply_product_scenario,
    backcast_prelag_dose3,
    build_protection_table,
    check_dose4_not_exceeding_dose3,
    compute_fractions,
    zero_prelag_dose4,
)
from idd_forecast_mbp.lib.processing.vaccine_efficacy import load_ve_curve
from idd_forecast_mbp.lib.versioning import finalize_artifact


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Apply malaria vaccine cohort coverage to real age-sex population.",
    )
    p.add_argument(
        "--coverage-csv", type=Path, default=rfc.VACCINE_COVERAGE_FILE,
        help="dose_3/dose_4 coverage CSV (columns: subnat_id, year_id, dose_3, "
             "dose_4, vacc_name, ...). Defaults to the current vintage of the "
             "versioned coverage node (constants.VACCINE_COVERAGE_FILE).",
    )
    ve_src = p.add_mutually_exclusive_group(required=True)
    ve_src.add_argument(
        "--ve-variant", choices=rfc.VE_VARIANTS,
        help="Which VE factorial cell to use, resolved inside the current VE node. "
             "Required (or --ve-file): the four cells encode different curve-shape "
             "assumptions and give different protection, so it is a modelling choice "
             "rather than something to default.",
    )
    ve_src.add_argument(
        "--ve-file", type=Path,
        help="Explicit path to a VE curve CSV, for a vintage outside the current node.",
    )
    p.add_argument(
        "--age-reference", choices=sorted(AGE_REFERENCE_PHI), default=DEFAULT_AGE_REFERENCE,
        help="Where within the reporting year age is evaluated. Decides which "
             "calendar years each dose trigger falls in: start_of_year blends dose 3 "
             "across two years, mid_year blends dose 4 instead. Default preserves "
             "historical behaviour; see DECISIONS.md 2026-08-24.",
    )
    p.add_argument(
        "--product-scenario", choices=rfc.PRODUCT_SCENARIOS, default="projected",
        help="'projected' uses each location's delivered product; 'all_r21' forces R21 "
             "everywhere, isolating product choice from the coverage trajectory.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output parquet path. Defaults to the versioned artifact node "
             "(MAL_VACCINE_COHORTS_WRITE_PATH/<filename>), and in that case the node's "
             "current/ symlink is repointed on success. Passing an explicit path writes "
             "there and leaves current/ alone.",
    )
    p.add_argument(
        "--population-file", type=Path,
        default=rfc.POPULATION_READ_PATH / "as_2023_full_population_df.parquet",
        help="Age-sex population parquet (default: full LSAE hierarchy, current run).",
    )
    prelag = p.add_mutually_exclusive_group()
    prelag.add_argument(
        "--backcast-prelag-dose3", action="store_true",
        help="Recover the pre-series dose_3 implied by each location's early dose_4 "
             "(dose_3(t-2) = dose_4(t) / that location's own dropout ratio). Preferred "
             "over --zero-prelag-dose4: it keeps the vaccinated children instead of "
             "discarding them.",
    )
    prelag.add_argument(
        "--zero-prelag-dose4", action="store_true",
        help="Zero dose_4 whose dose-3 antecedent predates the coverage series (the "
             "2024-2025 early-rollout records). Stopgap that discards real vaccinated "
             "children in those locations; the real fix is pre-2024 dose_3 from the "
             "producers. Logs how many rows it zeroed.",
    )
    p.add_argument(
        "--allow-dose4-exceeding-dose3", action="store_true",
        help="Proceed (with a loud warning) when dose_4 > dose_3 in the output. Off by "
             "default: that violates the coverage contract and indicates an upstream "
             "inconsistency, so it must be an explicit decision, not a silent pass.",
    )
    p.add_argument(
        "--year-start", type=int, default=2024,
        help="First report year (default 2024, the program's first year; everything "
             "earlier is identically zero).",
    )
    p.add_argument(
        "--year-end", type=int, default=None,
        help="Last report year (default: last year in the coverage file).",
    )
    return p.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    variant = args.ve_variant if args.ve_variant else args.ve_file.stem.replace("ve_", "", 1)
    ve_path = args.ve_file if args.ve_file else rfc.vaccine_efficacy_file(args.ve_variant)
    versioned = args.out is None
    out_path = (
        rfc.MAL_VACCINE_COHORTS_WRITE_PATH
        / rfc.MAL_VACCINE_COHORTS_FILENAME_TEMPLATE.format(
            variant=variant,
            products=(args.product_scenario if args.age_reference == DEFAULT_AGE_REFERENCE
                      else f"{args.product_scenario}_{args.age_reference}"))
        if versioned else args.out
    )

    # trigger ages are the cohort model's, passed in so the VE module needs no
    # knowledge of it (and neither imports the other)
    ve = load_ve_curve(str(ve_path), dose3_age=round(D3_TRIGGER_AGE * 12),
                       booster_age=round(D4_TRIGGER_AGE * 12))
    print(f"VE curve:   {ve_path}")
    print(f"            products {list(ve.products())}, "
          f"months 0-{ve.n_months(ve.products()[0]) - 1}")
    tails = ve.nonzero_tails()
    if tails:
        print("  NOTE: VE curve(s) end above zero, so protection is clipped to 0 beyond "
              f"the last month: {tails}. Extend the curves to avoid a discontinuity.")

    coverage_df = load_coverage(args.coverage_csv, ve)
    coverage_df, switched_rows, switched_locs = apply_product_scenario(
        coverage_df, args.product_scenario)
    if switched_rows:
        print(f"--product-scenario {args.product_scenario}: switched {switched_rows} coverage "
              f"row(s) ({switched_locs} location(s)) from RTS,S to R21")
    if args.backcast_prelag_dose3:
        coverage_df, n_added = backcast_prelag_dose3(coverage_df)
        print(f"--backcast-prelag-dose3: recovered {n_added} pre-series dose_3 value(s) "
              "by inverting each location's own dropout ratio")
    if args.zero_prelag_dose4:
        coverage_df, n_zeroed = zero_prelag_dose4(coverage_df)
        print(f"--zero-prelag-dose4: zeroed dose_4 on {n_zeroed} coverage row(s) whose "
              "dose-3 antecedent predates the series (STOPGAP -- discards real "
              "vaccinated children in early-rollout locations)")
    age_groups = load_age_groups()

    year_end = args.year_end if args.year_end is not None else int(coverage_df["year_id"].max())
    years = list(range(args.year_start, year_end + 1))
    location_ids = sorted(int(x) for x in coverage_df["subnat_id"].unique())
    age_group_ids = sorted(int(x) for x in age_groups["age_group_id"])

    print(f"coverage:   {args.coverage_csv}")
    print(f"            {len(location_ids)} location(s), coverage years "
          f"{int(coverage_df.year_id.min())}-{int(coverage_df.year_id.max())}, "
          f"product(s) {sorted(coverage_df.vacc_name.unique())}")
    print(f"population: {args.population_file}")
    print(f"age groups: {age_group_ids}")
    print(f"age reference: {args.age_reference}")
    print(f"report years: {years[0]}-{years[-1]}")

    # A cohort that is age_int in report year t was born in t-age_int-1 and hits
    # its dose-4 trigger in t-age_int+1, so coverage must extend to year_end+1
    # for the youngest cohorts to be looked up rather than silently read as 0.
    cov_max = int(coverage_df["year_id"].max())
    if cov_max < year_end:
        print(f"  NOTE: coverage ends {cov_max} but report years run to {year_end}; "
              "cohorts triggering after the coverage series ends are treated as 0 "
              "coverage (CoverageSeries returns 0.0 for absent years).")

    fractions = compute_fractions(coverage_df, age_groups, years, ve,
                                  age_reference=args.age_reference)

    pop = load_population(args.population_file, location_ids, age_group_ids, years)
    if pop.empty:
        raise ValueError(
            f"no population rows for the requested slice -- check that coverage "
            f"location_ids exist in {args.population_file}"
        )
    missing_locs = sorted(set(location_ids) - set(pop["location_id"].unique()))
    if missing_locs:
        raise ValueError(
            f"{len(missing_locs)} coverage location(s) absent from the population "
            f"source: {missing_locs[:10]}{'...' if len(missing_locs) > 10 else ''}. "
            "Every vaccine-eligible location must have population; fix upstream."
        )

    result = build_protection_table(pop, fractions)

    problem, minor = check_dose4_not_exceeding_dose3(result, coverage_df)
    if minor:
        print(f"\nNOTE: {minor}")
    if problem:
        if not args.allow_dose4_exceeding_dose3:
            raise ValueError(problem)
        print(f"\nWARNING: {problem}")

    write_parquet(result, out_path)
    print(f"\nWrote {len(result):,} rows to {out_path}")
    if versioned:
        finalize_artifact(rfc._A04_MAL_VACCINE_COHORTS)
    return result


if __name__ == "__main__":
    main()
