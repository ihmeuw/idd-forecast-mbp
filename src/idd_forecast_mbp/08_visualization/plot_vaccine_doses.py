"""
Dose delivery and outcomes averted per dose.

Doses are coverage x the surviving-infant cohort -- one multiplication per
location-year, deliberately not routed through the birth-cohort machinery:

    doses_3(t) = dose_3(t) x surviving_infants(t)
    doses_4(t) = dose_4(t) x surviving_infants(t - 2)

"Surviving infants" is age groups 2+3+388+389 summed, which span exactly one
year and so equal one annual cohort. There is no births file in this pipeline.

Dose counts use the RAW coverage file, NOT the back-cast one: the 2024-25 dose_4
records whose dose-3 antecedent predates the series describe boosters that were
genuinely administered to pilot-era children. They only break the *protection*
calculation, which has no dose 3 to attribute them to.
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.processing.vaccine_coverage import INFANT_AGE_GROUP_IDS, dose_counts
from idd_forecast_mbp.lib.viz.vaccine_impact import _ssp_color, _ssp_label

plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 12, "figure.titlesize": 20})
PRODUCT_COLOR = {"rtss": "#B03A2E", "r21": "#046C9A", "either": "#4D4D4D"}


def build_doses() -> pd.DataFrame:
    cov = pd.read_csv(rfc.VACCINE_COVERAGE_FILE)
    locs = sorted(int(x) for x in cov.subnat_id.unique())
    years = sorted(int(y) for y in cov.year_id.unique())
    pop = read_parquet_with_integer_ids(
        rfc.POPULATION_READ_PATH / "as_2023_full_population_df.parquet",
        columns=["location_id", "year_id", "age_group_id", "population"],
        filters=[("location_id", "in", locs),
                 ("age_group_id", "in", list(INFANT_AGE_GROUP_IDS)),
                 ("year_id", "in", sorted(set(years + [y - 2 for y in years])))])
    infants = (pop.groupby(["location_id", "year_id"], as_index=False)["population"]
               .sum().rename(columns={"population": "infants"}))
    return dose_counts(cov, infants)


def load_averted() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(rfc.MODEL_ROOT / "07-figures" / "20260824_vaccine_impact"
                                  / "vaccine_impact_summary_ve_*.parquet"))):
        rows.append(pd.read_parquet(f))
    return pd.concat(rows, ignore_index=True)


def plot(doses: pd.DataFrame, averted: pd.DataFrame, out: Path) -> Path:
    annual = doses.groupby("year_id")[["doses_dose3", "doses_dose4", "doses_total"]].sum()
    by_product = doses.groupby(["vacc_name", "year_id"]).doses_total.sum().unstack(0)

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    ax = axes[0, 0]
    for col, label, ls in (("doses_dose3", "dose 3", "-"), ("doses_dose4", "dose 4", "--"),
                           ("doses_total", "total", ":")):
        ax.plot(annual.index, annual[col] / 1e6, lw=2, ls=ls, color="#4D4D4D", label=label)
    ax.set_title("Doses delivered per year", fontweight="bold")
    ax.set_ylabel("million doses"); ax.legend(frameon=False)

    ax = axes[0, 1]
    for product in by_product.columns:
        ax.plot(by_product.index, by_product[product] / 1e6, lw=2,
                color=PRODUCT_COLOR.get(product, "#888"), label=product)
    ax.plot(annual.index, annual.doses_total / 1e6, lw=2, color=PRODUCT_COLOR["either"],
            ls="--", label="either")
    ax.set_title("Doses per year by product", fontweight="bold")
    ax.set_ylabel("million doses"); ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(annual.index, annual.doses_total.cumsum() / 1e9, lw=2, color="#4D4D4D")
    ax.set_title("Cumulative doses delivered", fontweight="bold")
    ax.set_ylabel("billion doses")

    # averted per 1,000 doses, per year -- both measures, default variant
    ax = axes[1, 1]
    d = averted[(averted.ve_variant == "loglinear_severe0")
                & (averted.product_scenario == "projected")]
    for measure, ls in (("mortality", "-"), ("incidence", "--")):
        for ssp, g in d[d.measure == measure].groupby("ssp_scenario"):
            g = g.sort_values("year_id").set_index("year_id")
            per_k = (g.averted_mean / annual.doses_total.reindex(g.index)) * 1e3
            ax.plot(per_k.index, per_k, lw=2, ls=ls, color=_ssp_color(ssp),
                    label=f"{_ssp_label(ssp)} - {measure}")
    ax.set_yscale("log")
    ax.set_title("Outcomes averted per 1,000 doses delivered that year", fontweight="bold")
    ax.set_ylabel("averted per 1,000 doses (log)")
    ax.legend(frameon=False, ncol=2, fontsize=11)

    for a in axes.ravel():
        a.set_xlabel("Year"); a.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Malaria vaccine dose delivery and dose efficiency - "
                 "projected rollout, vaccine-receiving locations", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    doses = build_doses()
    print(f"doses: {doses.doses_total.sum()/1e6:,.1f}M over {doses.year_id.nunique()} years")
    out = plot(doses, load_averted(), args.out_dir / "vaccine_doses_and_efficiency.png")
    print("wrote", out)
    return out


if __name__ == "__main__":
    main()
