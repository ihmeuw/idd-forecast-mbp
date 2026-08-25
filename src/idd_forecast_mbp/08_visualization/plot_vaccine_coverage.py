"""
Coverage of the vaccine-receiving population, under three weightings.

Nine series per panel: {ever dose 3, dose 3 only, dose 3+4} x {RTS,S, R21, both}.

Columns are the weighting, which is the whole point of the figure -- the same
fractions, weighted three ways:

  * population        - contemporaneous age-sex population. "What share of
                        children are covered."
  * deaths, 2023      - malaria deaths at the anchor year, held FIXED, so movement
                        over time is coverage changing and not the death
                        distribution shifting. Observed/raked data.
  * deaths, yearly    - deaths in each forecast year. NOTE this is a different
                        SOURCE: the age-specific malaria file stops at 2023, so
                        yearly deaths are the forecast's all-age deaths split by
                        the same as_rr fractions the impact step uses.

Rows are the age domain. Age groups 2/3/388 are below the dose-3 trigger: they
carry real death weight but can never be covered, so including them puts a
structural ceiling on the death-weighted series. Both framings are shown rather
than picking one.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.processing.disaggregation import compute_as_rr, malaria_as_fractions
from idd_forecast_mbp.lib.processing.vaccine_coverage import (
    NEVER_COVERABLE_AGE_GROUP_IDS,
    weighted_coverage,
)
from idd_forecast_mbp.lib.processing.vaccine_impact import ANCHOR_YEAR, eligible_locations
from idd_forecast_mbp.lib.viz.vaccine_impact import _cap

plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 12, "figure.titlesize": 20})

AGES = (2, 3, 388, 389, 238, 34, 6, 7, 8)
MEASURE_STYLE = {"ever dose 3": "-", "dose 3 only": "--", "dose 3+4": ":"}
PRODUCT_COLOR = {"rtss": "#B03A2E", "r21": "#046C9A", "either": "#4D4D4D",
                 "all locations": "#4D4D4D", "none": "#CCCCCC"}
GEOGRAPHIES = ("global", "either", "rtss", "r21")
WEIGHTS = [("population", "Population-weighted"),
           ("deaths_2023", f"Death-weighted ({ANCHOR_YEAR}, fixed)"),
           ("deaths_yearly", "Death-weighted (yearly, forecast)")]


def _geography_locations(geography: str, mapping: pd.DataFrame, fc_locs: set[int],
                         cov: pd.DataFrame) -> list[int]:
    """Which forecast locations form the denominator.

    global -- every forecast location, so places with no vaccine dilute the
              average. This is the "what does the world look like" cut.
    either -- locations under any coverage location (the 37 vaccine countries).
    rtss / r21 -- locations under coverage locations using that product only.
    """
    if geography == "global":
        return sorted(fc_locs)
    if geography == "either":
        return mapping["location_id"].tolist()
    products = cov.drop_duplicates("subnat_id").set_index("subnat_id")["vacc_name"]
    keep = mapping[mapping.coverage_location_id.map(products) == geography]
    if keep.empty:
        raise ValueError(f"no locations use product {geography!r}")
    return keep["location_id"].tolist()


def build_cells(variant: str, products: str, ssp: str,
                geography: str = "either") -> pd.DataFrame:
    """One frame of (location, year, age, sex) with fractions and all three weights."""
    prot = read_parquet_with_integer_ids(
        rfc.MAL_VACCINE_COHORTS_READ_PATH
        / rfc.MAL_VACCINE_COHORTS_FILENAME_TEMPLATE.format(variant=variant, products=products),
        columns=["location_id", "year_id", "age_group_id", "sex_id",
                 "frac_ever_dose3", "frac_ever_dose4"])

    nc = rfc.MAL_FORECAST_OUTPUTS_READ_PATH / f"malaria_forecast_{ssp}_Baseline.nc"
    with xr.open_dataset(nc) as ds:
        fc_locs = {int(x) for x in ds["location_id"].to_numpy()}
        years = [int(y) for y in ds["year_id"].to_numpy()]
        mapping = eligible_locations(rfc.VACCINE_COVERAGE_FILE, fc_locs)
        cov_all = pd.read_csv(rfc.VACCINE_COVERAGE_FILE, usecols=["subnat_id", "vacc_name"])
        locs = _geography_locations(geography, mapping, fc_locs, cov_all)
        lpos = np.searchsorted(ds["location_id"].to_numpy(), locs)
        mort_rate = np.exp(ds["log_malaria_mort_rate_pred"].to_numpy()[lpos]).mean(axis=-1)

    # fractions live at admin1; broadcast to each admin2 child
    prot = prot.merge(mapping, left_on="location_id", right_on="coverage_location_id",
                      suffixes=("_a1", ""))[
        ["location_id", "year_id", "age_group_id", "sex_id",
         "frac_ever_dose3", "frac_ever_dose4"]]

    pop = read_parquet_with_integer_ids(
        rfc.POPULATION_READ_PATH / "as_2023_full_population_df.parquet",
        columns=["location_id", "year_id", "age_group_id", "sex_id", "population"],
        filters=[("location_id", "in", locs), ("age_group_id", "in", list(AGES)),
                 ("year_id", "in", years)])

    as_mal = read_parquet_with_integer_ids(
        rfc.MAL_RAKED_AS_READ_PATH / "as_full_malaria_df.parquet",
        columns=["location_id", "year_id", "age_group_id", "sex_id", "malaria_inc_rate",
                 "aa_malaria_inc_rate", "malaria_mort_rate", "aa_malaria_mort_rate",
                 "malaria_mort_count"],
        filters=[("year_id", "==", ANCHOR_YEAR), ("location_id", "in", locs)])
    fixed = (as_mal[["location_id", "age_group_id", "sex_id", "malaria_mort_count"]]
             .rename(columns={"malaria_mort_count": "deaths_2023"}))

    # yearly deaths: all-age forecast deaths split by the as_rr mortality fractions
    aa_pop = read_parquet_with_integer_ids(
        rfc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
        columns=["location_id", "year_id", "population"],
        filters=[("location_id", "in", locs), ("year_id", "in", years)])
    aa = pd.DataFrame({"location_id": np.repeat(locs, len(years)),
                       "year_id": np.tile(years, len(locs)),
                       "mort_rate": mort_rate.ravel()})
    aa = aa.merge(aa_pop, on=["location_id", "year_id"], how="inner")
    aa["aa_deaths"] = aa["mort_rate"] * aa["population"]

    fracs = malaria_as_fractions(compute_as_rr(as_mal, anchor_year=ANCHOR_YEAR),
                                 pop.rename(columns={"population": "population"}))
    yearly = fracs.merge(aa[["location_id", "year_id", "aa_deaths"]],
                         on=["location_id", "year_id"], how="left")
    yearly["deaths_yearly"] = yearly["mort_fraction"] * yearly["aa_deaths"]

    cells = pop.merge(prot, on=["location_id", "year_id", "age_group_id", "sex_id"],
                      how="left")
    cells = cells.merge(fixed, on=["location_id", "age_group_id", "sex_id"], how="left")
    cells = cells.merge(
        yearly[["location_id", "year_id", "age_group_id", "sex_id", "deaths_yearly"]],
        on=["location_id", "year_id", "age_group_id", "sex_id"], how="left")
    cov = pd.read_csv(rfc.VACCINE_COVERAGE_FILE,
                      usecols=["subnat_id", "vacc_name"]).drop_duplicates()
    cells = cells.merge(mapping, on="location_id", how="left").merge(
        cov, left_on="coverage_location_id", right_on="subnat_id", how="left")
    # outside the vaccine geography there is no product and no coverage
    cells["vacc_name"] = cells["vacc_name"].fillna("none")
    for c in ("frac_ever_dose3", "frac_ever_dose4"):
        cells[c] = cells[c].fillna(0.0)
    for c in ("deaths_2023", "deaths_yearly"):
        cells[c] = cells[c].fillna(0.0)
    return cells.rename(columns={"population": "population"})


def plot(cells: pd.DataFrame, title: str, out: Path, geography: str = "either") -> Path:
    domains = [("All ages 0-20", cells),
               ("Excluding ages that cannot be vaccinated",
                cells[~cells.age_group_id.isin(NEVER_COVERABLE_AGE_GROUP_IDS)])]
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharex=True, sharey="row")
    for row, (domain_label, sub) in enumerate(domains):
        for col, (weight_col, weight_label) in enumerate(WEIGHTS):
            ax = axes[row, col]
            series = weighted_coverage(sub, weight_col)
            pooled_name = {"global": "all locations"}.get(geography, "either")
            series["product"] = series["product"].replace({"either": pooled_name})
            drawable = series[series.groupby("product").coverage.transform("max") > 0]
            for product, style_group in drawable.groupby("product"):
                for measure, g in style_group.groupby("measure"):
                    g = g.sort_values("year_id")
                    ax.plot(g.year_id, g.coverage, color=PRODUCT_COLOR[product],
                            ls=MEASURE_STYLE[measure], lw=2,
                            label=f"{product} - {measure}" if row == 0 and col == 0 else None)
            if row == 0:
                ax.set_title(weight_label, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{domain_label}\n\nfraction covered")
            ax.set_xlabel("Year")
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, ncol=1, fontsize=11, loc="upper left")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ve-variant", choices=rfc.VE_VARIANTS, default="loglinear_severe0")
    p.add_argument("--product-scenario", choices=rfc.PRODUCT_SCENARIOS, default="projected")
    p.add_argument("--ssp", default="ssp245")
    p.add_argument("--geography", choices=GEOGRAPHIES, default="either")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = build_cells(args.ve_variant, args.product_scenario, args.ssp,
                        geography=args.geography)
    print(f"cells: {len(cells):,}  locations: {cells.location_id.nunique():,}")
    labels = {"global": "all locations worldwide",
              "either": "countries receiving either vaccine",
              "rtss": "RTS,S countries only", "r21": "R21 countries only"}
    out = plot(cells,
               f"Vaccine coverage - {labels[args.geography]} "
               f"({cells.location_id.nunique():,} locations) - "
               f"{args.product_scenario}, {args.ssp}",
               args.out_dir
               / f"vaccine_coverage_{args.geography}_{args.product_scenario}.png",
               geography=args.geography)
    print("wrote", out)
    return out


if __name__ == "__main__":
    main()
