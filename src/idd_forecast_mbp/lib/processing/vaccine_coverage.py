"""
Coverage-series transforms for the malaria vaccine pipeline.

Pure functions over the delivered dose_3/dose_4 coverage table: validation
against the documented contract, the two treatments for pre-series dose_4, the
product-rollout scenarios, and construction of the per-location CoverageSeries
the cohort model consumes. No file I/O -- readers live in
`lib/data/vaccine_inputs.py`.

The load-bearing fact these functions protect: `dose_4` as delivered is the
UNCONDITIONAL fraction of a birth cohort that completed the booster, so it is
never multiplied by dose_3 again. See DECISIONS.md 2026-08-24.
"""
import pandas as pd

from idd_forecast_mbp.lib.processing.vaccine_cohort_fractions import (
    DEFAULT_AGE_REFERENCE,
    CoverageSeries,
    fraction_for_bin,
)
from idd_forecast_mbp.lib.processing.vaccine_efficacy import VECurve

COVERAGE_REQUIRED_COLUMNS = ["subnat_id", "year_id", "dose_3", "dose_4", "vacc_name"]

# Denominator floor when inferring a location's dose-3 -> dose-4 dropout ratio;
# below this the quotient is numerical noise rather than signal.
MIN_DENOMINATOR_FOR_RATIO = 0.01

# d4_ever may exceed d3_ever by a hair at the very start of a location's ramp:
# the delivered dose_4 is lagged on whole calendar years, while d3_ever blends the
# two calendar years a birth cohort's 6-month trigger straddles. Where dose_3 goes
# 0 -> small, the blend halves it and can land just under dose_4. Observed max in
# the 2026-08-05 vintage: 0.0005 (2 locations). Structural problems are orders of
# magnitude larger -- the pre-lag dose_4 issue was 0.20, and a re-multiplied
# dose_4 would be larger still -- so this admits the artifact without hiding those.
DOSE4_EXCESS_TOLERANCE = 1e-3

ID_DTYPES = {"location_id": "int64", "year_id": "int64",
             "age_group_id": "int64", "sex_id": "int64"}
VALUE_DTYPES = {
    "population": "float64",
    "frac_ever_dose3": "float64",
    "frac_ever_dose4": "float64",
    "effective_protection_case": "float64",
    "effective_protection_death": "float64",
    "n_ever_dose3": "float64",
    "n_ever_dose4": "float64",
    "n_protected_case_equiv": "float64",
    "n_protected_death_equiv": "float64",
}


def validate_coverage_frame(cov: pd.DataFrame, ve: VECurve, source: str = "coverage") -> None:
    """Raise unless the coverage table satisfies the documented contract.

    Checks required columns, that every product present has a VE curve, that
    doses lie in [0, 1], and that no location switches vaccine product over time
    (the coverage lookup assumes one product per location).
    """
    missing = [c for c in COVERAGE_REQUIRED_COLUMNS if c not in cov.columns]
    if missing:
        raise ValueError(f"{source} is missing required column(s): {missing}")

    bad_product = sorted(set(cov["vacc_name"].unique()) - set(ve.products()))
    if bad_product:
        raise ValueError(
            f"vacc_name value(s) {bad_product} have no VE curve "
            f"(curve supplies: {list(ve.products())}). Every product in the coverage "
            "data needs efficacy curves before protection can be computed."
        )
    for col in ("dose_3", "dose_4"):
        out_of_range = cov[(cov[col] < 0) | (cov[col] > 1)]
        if not out_of_range.empty:
            raise ValueError(f"{col} outside [0, 1] for {len(out_of_range)} row(s) in {source}")

    multi = cov.groupby("subnat_id")["vacc_name"].nunique()
    offenders = sorted(multi[multi > 1].index.tolist())
    if offenders:
        raise ValueError(
            f"location(s) {offenders} carry more than one vacc_name over time; the "
            "coverage lookup assumes a single product per location. A location that "
            "switches products mid-series needs explicit handling."
        )


def implied_dropout_ratios(coverage_df: pd.DataFrame) -> pd.Series:
    """Each location's own implied dose-3 -> dose-4 completion ratio.

    dose_4(t) is built as ratio * dose_3(t-2) throughout the delivered series,
    so the ratio is recoverable per location. Taken as the median over years
    where dose_3(t-2) is large enough not to be numerical noise; in the
    2026-08-05 vintage this lands at 0.6332-0.6366 for all 373 locations,
    consistent with the documented 36.7% dropout.
    """
    wide = coverage_df.pivot_table(index="subnat_id", columns="year_id",
                                   values=["dose_3", "dose_4"])
    years = sorted(coverage_df["year_id"].unique())
    per_year = []
    for year in years:
        if (year - 2) in years:
            den = wide[("dose_3", year - 2)]
            per_year.append((wide[("dose_4", year)] / den).where(den > MIN_DENOMINATOR_FOR_RATIO))
    ratios = pd.concat(per_year, axis=1).median(axis=1)
    return ratios.fillna(ratios.median())

def backcast_prelag_dose3(coverage_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Recover the pre-series dose_3 implied by each location's early dose_4.

    dose_4 in year t describes the cohort born in t-2. Where t-2 precedes the
    first year of dose_3 data, the booster is evidence that dose 3 was given in
    t-2 -- the delivered file simply starts too late to contain it. Inverting
    the location's own dropout ratio recovers it: dose_3(t-2) = dose_4(t)/ratio.

    Adds the recovered years as new rows (dose_4 = 0 for them: a cohort born
    before the series would have been boosted before the series too, and the
    file carries no evidence either way). Returns (df, n_rows_added).

    LIMIT: only years whose booster appears in the file can be recovered -- for
    the 2026-08-05 vintage that is 2022 and 2023. Real pilot dose 3 from 2019-2021
    leaves no trace in this file, so cohorts born then still read 0, which
    understates dose-3-ever for children aged 3+ in 2024-2025 in the pilot areas.
    """
    first_dose3_year = int(coverage_df.loc[coverage_df["dose_3"] > 0, "year_id"].min())
    ratios = implied_dropout_ratios(coverage_df)

    prelag = coverage_df[(coverage_df["year_id"] - 2 < first_dose3_year)
                         & (coverage_df["dose_4"] > 0)]
    if prelag.empty:
        return coverage_df, 0

    meta_cols = [c for c in ("country_id", "country", "subnat", "vacc_id", "vacc_name")
                 if c in coverage_df.columns]
    new = prelag[["subnat_id", "year_id", *meta_cols]].copy()
    new["dose_3"] = (prelag["dose_4"] / prelag["subnat_id"].map(ratios)).values
    new["year_id"] = new["year_id"] - 2
    new["dose_4"] = 0.0

    over = new[new["dose_3"] > 1.0]
    if not over.empty:
        raise ValueError(
            f"back-cast implies dose_3 > 1 for {len(over)} row(s) "
            f"(locations {sorted(over['subnat_id'].unique())[:10]}); the implied dropout "
            "ratio is inconsistent with the early dose_4 values -- resolve with the producers."
        )
    # A back-cast year must not collide with a year the file already provides.
    existing = set(zip(coverage_df["subnat_id"], coverage_df["year_id"]))
    collide = [k for k in zip(new["subnat_id"], new["year_id"]) if k in existing]
    if collide:
        raise ValueError(f"back-cast would overwrite delivered rows: {collide[:5]}")

    combined = pd.concat([coverage_df, new], ignore_index=True)
    combined = combined.sort_values(["subnat_id", "year_id"]).reset_index(drop=True)
    return combined, len(new)

def zero_prelag_dose4(coverage_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Zero out dose_4 whose dose-3 antecedent predates the coverage series.

    dose_4 reported in year t describes the cohort born in t-2 (they hit the
    2-year booster trigger in t). If t-2 is earlier than the first year of
    dose_3 data, the file gives the booster without the dose 3 that must have
    preceded it, so d3_ever reads 0 while d4_ever is positive -- violating the
    d4 <= d3 contract and, downstream, resurfacing in older age bins for ~20
    years as those cohorts age.

    This is a STOPGAP, not a fix: it discards real vaccinated children in the
    early-rollout locations rather than recovering them. The real fix is
    pre-2024 dose_3 from the coverage producers. Returns (df, n_rows_zeroed).
    """
    first_dose3_year = int(coverage_df.loc[coverage_df["dose_3"] > 0, "year_id"].min())
    prelag = (coverage_df["year_id"] - 2 < first_dose3_year) & (coverage_df["dose_4"] > 0)
    n = int(prelag.sum())
    coverage_df = coverage_df.copy()
    coverage_df.loc[prelag, "dose_4"] = 0.0
    return coverage_df, n

def apply_product_scenario(coverage_df: pd.DataFrame,
                           scenario: str) -> tuple[pd.DataFrame, int, int]:
    """Override which vaccine product each location uses.

    'projected' keeps the delivered rollout (RTS,S where introduced, R21 elsewhere).
    The delivered coverage is itself a projection to 2100, so it is not 'observed'.
    'all_r21' is a counterfactual in which every eligible location uses R21, whose
    efficacy is materially higher than RTS,S at every age -- so it isolates the
    product choice from the coverage trajectory, which is unchanged.
    """
    if scenario == "projected":
        return coverage_df, 0, 0
    if scenario == "all_r21":
        out = coverage_df.copy()
        switched_rows = int((out["vacc_name"] != "r21").sum())
        switched_locs = int(coverage_df.loc[coverage_df.vacc_name != "r21", "subnat_id"].nunique())
        out["vacc_name"] = "r21"
        if "vacc_id" in out.columns:
            out["vacc_id"] = 2
        return out, switched_rows, switched_locs
    raise ValueError(f"unknown product scenario {scenario!r}")

def build_coverage_series(coverage_df: pd.DataFrame, subnat_id: int,
                          ve: VECurve) -> CoverageSeries:
    """One CoverageSeries per location, keyed off the location's own
    vacc_name (rtss vs r21) so waning parameters are looked up correctly."""
    loc = coverage_df[coverage_df["subnat_id"] == subnat_id].sort_values("year_id")
    return CoverageSeries(
        dose3=dict(zip(loc["year_id"], loc["dose_3"])),
        dose4=dict(zip(loc["year_id"], loc["dose_4"])),
        vacc_name=loc["vacc_name"].iloc[0],
        ve=ve,
    )

def check_dose4_not_exceeding_dose3(result: pd.DataFrame, coverage_df: pd.DataFrame,
                                    tolerance: float = DOSE4_EXCESS_TOLERANCE) -> tuple[str, str]:
    """Check the d4_ever <= d3_ever contract. Returns (fatal_msg, info_msg).

    `fatal_msg` is non-empty only for excesses above `tolerance`; `info_msg`
    reports sub-tolerance excesses so they stay visible rather than silent.

    That invariant is part of the coverage contract: dose_4 is the
    unconditional fraction of a birth cohort that completed the booster, so it
    can never exceed the fraction that got dose 3. Empty string means clean.

    Known root cause in the 2026-08-05 vintage: some locations report dose_4 > 0
    in the first years of the series while dose_3 starts in 2024. A booster at
    age 2 in year t implies a dose 3 in t-2, so those rows describe cohorts
    whose dose-3 coverage predates the file (real early RTS,S rollouts). The
    file supplies the booster consequence without the dose-3 antecedent, so
    d3_ever reads 0 while d4_ever is positive. Those early records keep
    surfacing in progressively older age bins for ~20 years.
    """
    excess = result["frac_ever_dose4"] - result["frac_ever_dose3"]
    minor = result[(excess > 1e-12) & (excess <= tolerance)]
    info = ""
    if not minor.empty:
        info = (f"{len(minor):,} row(s) across {minor['location_id'].nunique()} location(s) have "
                f"dose_4 above dose_3 by <= {tolerance} (max {excess[minor.index].max():.6f}) -- "
                "the known program-start blending artifact, within tolerance.")
    violations = result[excess > tolerance]
    if violations.empty:
        return "", info

    first_years = [int(y) for y in sorted(coverage_df["year_id"].unique())[:2]]
    early_d4 = coverage_df[(coverage_df["year_id"].isin(first_years)) & (coverage_df["dose_4"] > 0)]
    d3_start = int(coverage_df.loc[coverage_df["dose_3"] > 0, "year_id"].min())
    locs = [int(x) for x in sorted(violations["location_id"].unique())]
    return (
        f"dose_4 exceeds dose_3 in {len(violations):,} of {len(result):,} output rows "
        f"({100 * len(violations) / len(result):.3f}%), across {len(locs)} location(s), "
        f"max excess {violations['frac_ever_dose4'].sub(violations['frac_ever_dose3']).max():.4f}. "
        f"{len(early_d4)} coverage row(s) across {early_d4['subnat_id'].nunique()} location(s) "
        f"report dose_4 > 0 in {first_years}, while dose_3 coverage only starts in "
        f"{d3_start} -- i.e. boosters whose dose-3 antecedent predates the coverage file. "
        f"Affected location_ids: {locs[:15]}{'...' if len(locs) > 15 else ''}. "
        "Resolve with the coverage producers (supply pre-2024 dose_3 for the early-rollout "
        "locations), or pass --zero-prelag-dose4 to discard those records, or "
        "--allow-dose4-exceeding-dose3 to proceed knowingly."
    ), info


def compute_fractions(coverage_df: pd.DataFrame, age_groups: pd.DataFrame,
                      years: list[int], ve: VECurve,
                      age_reference: str = DEFAULT_AGE_REFERENCE) -> pd.DataFrame:
    """Fractions on the (location, year, age_group) grid.

    Looped rather than array-vectorized because the grid is small -- one
    entry per location x year x age_group (~9 age groups), independent of
    the population table's size or its sex dimension. The expensive cross
    product (x sex, x every population row) is handled by the merge in
    main(), not here.

    No single_year_weights are passed: no single-year-of-age population
    exists in this pipeline, so fraction_for_bin's documented uniform
    fallback applies for multi-year bins (peak error well under 1% of the
    bin during the fastest ramp years). See that function's docstring.
    """
    rows = []
    for subnat_id in sorted(coverage_df["subnat_id"].unique()):
        series = build_coverage_series(coverage_df, subnat_id, ve)
        for year in years:
            for ag in age_groups.itertuples():
                d3_ever, d4_ever, prot_case, prot_death = fraction_for_bin(
                    series, ag.age_group_years_start, ag.age_group_years_end, year,
                    age_reference=age_reference,
                )
                rows.append((int(subnat_id), int(year), int(ag.age_group_id),
                             d3_ever, d4_ever, prot_case, prot_death))
    return pd.DataFrame(
        rows,
        columns=["location_id", "year_id", "age_group_id",
                 "frac_ever_dose3", "frac_ever_dose4",
                 "effective_protection_case", "effective_protection_death"],
    )


def build_protection_table(pop: pd.DataFrame, fractions: pd.DataFrame) -> pd.DataFrame:
    """Join the (location, year, age_group) fraction grid onto age-sex population
    and produce the output table with headcounts and explicit dtypes.

    The fractions carry no sex dimension -- vaccine coverage is not
    sex-specific -- so they broadcast onto both sexes. The row-count check is a
    real guard: a fraction grid that fails to cover the population slice would
    otherwise silently drop rows rather than fail.
    """
    result = pop.merge(fractions, on=["location_id", "year_id", "age_group_id"], how="inner")
    if len(result) != len(pop):
        raise ValueError(
            f"merge dropped rows ({len(pop)} population -> {len(result)} joined); "
            "the fraction grid should cover every population row in the slice."
        )

    result["n_ever_dose3"] = result["population"] * result["frac_ever_dose3"]
    result["n_ever_dose4"] = result["population"] * result["frac_ever_dose4"]
    result["n_protected_case_equiv"] = result["population"] * result["effective_protection_case"]
    result["n_protected_death_equiv"] = result["population"] * result["effective_protection_death"]

    result = result.astype({**ID_DTYPES, **VALUE_DTYPES})
    return (result[list(ID_DTYPES) + list(VALUE_DTYPES)]
            .sort_values(["location_id", "year_id", "age_group_id", "sex_id"])
            .reset_index(drop=True))


# Age groups below the dose-3 trigger: they carry real death weight but can never
# be covered, so a death-weighted coverage series including them has a structural
# ceiling well below 1. Exposed so a caller can report both framings.
NEVER_COVERABLE_AGE_GROUP_IDS = (2, 3, 388)

COVERAGE_MEASURES = {
    "ever dose 3": "frac_ever_dose3",
    "dose 3 only": "_d3_only",
    "dose 3+4": "frac_ever_dose4",
}


def weighted_coverage(protection: pd.DataFrame, weight_col: str,
                      product_col: str = "vacc_name") -> pd.DataFrame:
    """Weighted-average coverage per (year, product, measure).

    `C(t) = sum_cells w * frac / sum_cells w` -- only the weight changes between
    the population-weighted and death-weighted framings; the fractions are the
    same. Emits one row per (year, product, measure) plus a "both" product row,
    i.e. the 9 series of the coverage panel.

    `protection` must carry `year_id`, `weight_col`, `product_col` and the two
    fraction columns; "dose 3 only" is derived as ever-dose-3 minus dose-3+4.
    """
    d = protection.copy()
    d["_d3_only"] = d["frac_ever_dose3"] - d["frac_ever_dose4"]

    rows = []
    # "either" is the pooled set -- the union of locations receiving either
    # product. No location carries both, so this is a weighted AVERAGE across
    # disjoint sets and necessarily lies between the two product lines.
    for product, sub in list(d.groupby(product_col)) + [("either", d)]:
        for label, col in COVERAGE_MEASURES.items():
            g = sub.groupby("year_id")
            num = g.apply(lambda x: (x[weight_col] * x[col]).sum(), include_groups=False)
            den = g[weight_col].sum()
            value = (num / den).where(den > 0)
            rows.append(pd.DataFrame({
                "year_id": value.index, "product": product,
                "measure": label, "coverage": value.to_numpy(),
            }))
    return pd.concat(rows, ignore_index=True)


# Under-1 age groups. Summed they span exactly one year, so the stock equals one
# annual birth cohort surviving to infancy -- the WUENIC-style "surviving infants"
# denominator. Used ONLY for dose counts.
INFANT_AGE_GROUP_IDS = (2, 3, 388, 389)
DOSE4_LAG_YEARS = 2


def dose_counts(coverage_df: pd.DataFrame, infant_population: pd.DataFrame,
                dose4_lag_years: int = DOSE4_LAG_YEARS) -> pd.DataFrame:
    """Doses administered per (location, year). One multiplication, no cohort tracking.

        doses_3(t) = dose_3(t) x surviving_infants(t)
        doses_4(t) = dose_4(t) x surviving_infants(t - lag)

    This is deliberately NOT routed through the birth-cohort machinery in
    `vaccine_cohort_fractions.py`. That exists for waning-adjusted protection by
    age group over time, which genuinely needs cohort tracking across non-uniform
    bins. A dose count needs none of it: coverage is already the fraction of the
    target cohort reached, so the count is coverage x cohort size.

    Dose 4 targets the cohort that were infants `dose4_lag_years` earlier -- the
    children turning 24 months in year t were the infants of t-2.

    `infant_population` is [location_id, year_id, infants]; callers build it by
    summing INFANT_AGE_GROUP_IDS. Years where the lagged cohort is unavailable
    yield NaN rather than a silent zero.
    """
    cov = coverage_df.rename(columns={"subnat_id": "location_id"})[
        ["location_id", "year_id", "dose_3", "dose_4", "vacc_name"]]

    d3 = infant_population.rename(columns={"infants": "infants_t"})
    d4 = infant_population.copy()
    d4["year_id"] = d4["year_id"] + dose4_lag_years
    d4 = d4.rename(columns={"infants": "infants_t_minus_lag"})

    out = cov.merge(d3, on=["location_id", "year_id"], how="left")
    out = out.merge(d4, on=["location_id", "year_id"], how="left")
    out["doses_dose3"] = out["dose_3"] * out["infants_t"]
    out["doses_dose4"] = out["dose_4"] * out["infants_t_minus_lag"]
    out["doses_total"] = out["doses_dose3"].fillna(0) + out["doses_dose4"].fillna(0)
    return out[["location_id", "year_id", "vacc_name", "dose_3", "dose_4",
                "infants_t", "infants_t_minus_lag",
                "doses_dose3", "doses_dose4", "doses_total"]]
