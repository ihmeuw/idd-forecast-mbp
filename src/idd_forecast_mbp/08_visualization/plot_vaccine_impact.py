"""
Plot malaria vaccine impact scenarios: no-vaccine, vaccine, and the difference.

Consumes the summary written by 04_forecasting/vaccine_impact_scenarios.py.
Per measure and VE variant: one figure with a column per ssp scenario, levels on
the top row (both scenarios with 95% UI) and cases/deaths averted beneath.
Also emits one cross-variant comparison of averted burden.

Totals cover only the vaccine-eligible admin2 locations, so the no-vaccine and
vaccine series are directly comparable; locations outside the coverage geography
are identical between scenarios and excluded from both.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 21,
})
import pandas as pd

from idd_forecast_mbp.lib.viz.vaccine_impact import (
    MEASURE_LABEL,
    NOVACC_COLOR,
    PRODUCT_PRETTY,
    PRODUCT_SHORT,
    SSP_ORDER,
    VACCINE_LABELS,
    VARIANT_PRETTY,
    VARIANT_SHORT,
    _ascii,
    _box_stats,
    _cap,
    _lighten,
    _normalize_vaccine,
    _normalize_variants,
    _pair_labels,
    _ssp_color,
    _ssp_label,
)
from idd_forecast_mbp.lib.io.parquet import write_parquet


def _millions(ax):
    """Scale-suffixed y ticks with precision chosen from the axis SPAN.

    Fixed precision breaks on narrow ranges: a panel spanning 4.5-5.1M rendered
    every tick as "5M". Decimals are therefore derived from how many scale units
    the axis actually covers.
    """
    def fmt(v, _):
        lo, hi = ax.get_ylim()
        span = abs(hi - lo)
        v = v + 0.0 if v != 0 else 0.0          # never render "-0"
        for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
            if max(abs(lo), abs(hi)) >= scale:
                units = span / scale
                if units == 0:                  # constant panel: no range to resolve
                    return f"{v / scale:,.0f}{suffix}"
                decimals = 0 if units >= 10 else (1 if units >= 2 else 2)
                return f"{v / scale:,.{decimals}f}{suffix}"
        return f"{v:,.0f}"

    ax.yaxis.set_major_formatter(fmt)


def _band(ax, x, mean, lo, hi, color, label=None):
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, mean, color=color, lw=2, label=label)


def plot_measure(df: pd.DataFrame, measure: str, tag: str, pretty: str,
                 out_dir: Path) -> Path:
    """Four rows per ssp: annual levels, cumulative levels, annual averted,
    cumulative averted. Cumulative series are cumulated per draw upstream, so
    their intervals widen properly rather than being cumulated quantiles."""
    sub = df[df.measure == measure]
    label = MEASURE_LABEL[measure]
    rows = [
        ("novacc", "vacc", f"Annual {label}"),
        ("novacc_cum", "vacc_cum", f"Cumulative {label}"),
        ("averted", None, f"{label.capitalize()} averted (annual)"),
        ("averted_cum", None, f"Cumulative {label} averted"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), sharex=True, sharey="row")
    for col, ssp in enumerate(SSP_ORDER):
        s = sub[sub.ssp_scenario == ssp].sort_values("year_id")
        color = _ssp_color(ssp)
        for row, (a, b, ylabel) in enumerate(rows):
            ax = axes[row, col]
            if b is not None:
                _band(ax, s.year_id, s[f"{a}_mean"], s[f"{a}_lo"], s[f"{a}_hi"],
                      NOVACC_COLOR, "No vaccine")
                _band(ax, s.year_id, s[f"{b}_mean"], s[f"{b}_lo"], s[f"{b}_hi"],
                      color, "With vaccine")
                if col == 0 and row == 0:
                    ax.legend(frameon=False, fontsize=14, loc="best")
            else:
                _band(ax, s.year_id, s[f"{a}_mean"], s[f"{a}_lo"], s[f"{a}_hi"], color)
                ax.axhline(0, color="black", lw=0.8)
            if row == 0:
                ax.set_title(_ssp_label(ssp), fontweight="bold")
            if row == len(rows) - 1:
                ax.set_xlabel("Year")
            if col == 0:
                ax.set_ylabel(ylabel)
            _millions(ax)
            ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle(f"Malaria vaccine impact — {label} — {pretty}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"vaccine_impact_{measure}_ve_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _boxes(ax, groups, labels, inner_step: float = 0.60,
           group_gap: float = 1.30, width: float = 0.52):
    """Grouped box plots.

    `groups` is one list per x-position group, each holding (values, facecolor)
    pairs. `inner_step` is deliberately much smaller than `group_gap` so the
    within-scenario pair reads as one unit and the eye sees scenario-to-scenario
    separation as the larger distance.
    """
    positions, data, faces, ticks = [], [], [], []
    x = 0.0
    for group, _ in zip(groups, labels):
        start = x
        for values, face in group:
            positions.append(x)
            data.append(values)
            faces.append(face)
            x += inner_step
        ticks.append((start + x - inner_step) / 2)
        x += group_gap
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                    medianprops=dict(color="black", lw=1.4),
                    flierprops=dict(marker=".", ms=4, alpha=0.5))
    for patch, face in zip(bp["boxes"], faces):
        patch.set_facecolor(face)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.9)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, x - group_gap + 0.55)
    return bp


def _mark_if_identical(ax, values, note: str = "identical — no difference") -> bool:
    """A difference panel that is exactly zero everywhere is a real result, not a
    broken plot. Give it a readable axis and say so on the panel."""
    if len(values) == 0 or float(abs(values).max()) > 0.0:
        return False
    ax.set_ylim(-1, 1)
    ax.set_yticks([0])
    # sit clear of the zero line, otherwise the text reads as struck through
    ax.text(0.5, 0.68, note, transform=ax.transAxes, ha="center", va="center",
            fontsize=16, color="#555555", style="italic",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    return True


def plot_overview(norm_summary: pd.DataFrame, norm_draws: pd.DataFrame, through: int,
                  labels: dict, suptitle: str, out: Path,
                  figure_name: str) -> tuple:
    """2 x 4 overview: cases on top, deaths beneath.

    C1 no-vaccine, C2 with-vaccine, C3 annual averted -- each with all three
    scenarios overlaid. C4 is split: paired cumulative totals per scenario
    (no-vaccine beside vaccine) and cumulative averted per scenario, both as box
    plots over the 100 draws.

    C1 and C2 share y-limits within a row -- they are the same quantity and the
    comparison is the point. C3 and the box columns carry their own scales,
    since averted burden is roughly a tenth of the level and forcing a shared
    axis would flatten it to a line.
    """
    stats_rows: list[dict] = []
    fig = plt.figure(figsize=(26, 11))
    # explicit margins rather than bbox_inches="tight", which shifted artists
    # off the canvas
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.75],
                          hspace=0.32, wspace=0.42,
                          left=0.045, right=0.965, top=0.895, bottom=0.075)

    for row, measure in enumerate(("incidence", "mortality")):
        s = norm_summary[norm_summary.measure == measure]
        d = norm_draws[norm_draws.measure == measure]
        label = MEASURE_LABEL[measure]

        ax_a = fig.add_subplot(gs[row, 0])
        ax_b = fig.add_subplot(gs[row, 1], sharey=ax_a)
        ax_d = fig.add_subplot(gs[row, 2])
        # column 4 is one region split into two panels that share an edge:
        # left keeps its y-axis on the left, right moves its y-axis to the right
        pair_gs = gs[row, 3].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.0)
        ax_cum = fig.add_subplot(pair_gs[0, 0])
        ax_dif = fig.add_subplot(pair_gs[0, 1])
        ax_dif.yaxis.tick_right()
        ax_dif.yaxis.set_label_position("right")

        for ssp in SSP_ORDER:
            ss = s[s.ssp_scenario == ssp].sort_values("year_id")
            color = _ssp_color(ssp)
            for ax, pre in ((ax_a, "a"), (ax_b, "b"), (ax_d, "d")):
                _band(ax, ss.year_id, ss[f"{pre}_mean"], ss[f"{pre}_lo"], ss[f"{pre}_hi"],
                      color, _ssp_label(ssp))

        for ax, title in ((ax_a, labels["a"]), (ax_b, labels["b"]), (ax_d, labels["d"])):
            if row == 0:
                ax.set_title(title, fontsize=16, fontweight="bold")
            ax.set_xlabel("Year")
            _millions(ax)
            ax.grid(alpha=0.25, lw=0.5)
        ax_d.axhline(0, color="black", lw=0.8)
        _mark_if_identical(ax_d, s["d_mean"].to_numpy())
        ax_a.set_ylabel(labels["y_level"].format(label=label))
        ax_d.set_ylabel(_cap(labels["y_diff"].format(label=label)))
        if row == 0:
            ax_a.legend(frameon=False, fontsize=14)

        paired, diffs, ticklabels = [], [], []
        for ssp in SSP_ORDER:
            dd = d[d.ssp_scenario == ssp]
            color = _ssp_color(ssp)
            for role, col in (("a", "a_cum"), ("b", "b_cum"), ("d", "d_cum")):
                stats_rows.append({
                    "figure": _ascii(figure_name),
                    "figure_file": out.stem,
                    "measure": measure,
                    "ssp_scenario": _ssp_label(ssp),
                    "box": labels[role] if role != "d" else labels["d"],
                    "quantity": f"cumulative through {through}",
                    **_box_stats(dd[col].to_numpy()),
                })
            paired.append([(dd.a_cum.to_numpy(), _lighten(color)),
                           (dd.b_cum.to_numpy(), color)])
            diffs.append([(dd.d_cum.to_numpy(), color)])
            ticklabels.append(_ssp_label(ssp))

        _boxes(ax_cum, paired, ticklabels)
        _boxes(ax_dif, diffs, ticklabels, width=0.46)
        _mark_if_identical(ax_dif, d["d_cum"].to_numpy())
        if row == 0:
            ax_cum.set_title(labels["box"].format(through=through),
                             fontsize=16, fontweight="bold")
            ax_dif.set_title(labels["box_diff"], fontsize=16, fontweight="bold")
            ax_cum.legend(handles=[Patch(facecolor="#BBBBBB", edgecolor="black",
                                         label=labels["arm_a"]),
                                   Patch(facecolor="#555555", edgecolor="black",
                                         label=labels["arm_b"])],
                          frameon=False, fontsize=13, loc="upper left")
        ax_cum.set_ylabel(labels["y_cum"].format(label=label))
        ax_dif.set_ylabel(labels["y_cum_diff"].format(label=label))
        for ax in (ax_cum, ax_dif):
            _millions(ax)
            ax.grid(alpha=0.25, lw=0.5, axis="y")

    fig.suptitle(suptitle, fontweight="bold", y=0.975)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, pd.DataFrame(stats_rows)


# Panel titles and legends use SHORT names. The long forms above are for the
# the suptitle only -- using them as column titles made adjacent titles
# collide, which is what this split fixes.


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--summary-dir", type=Path, required=True,
                   help="Directory holding vaccine_impact_summary_ve_*.parquet")
    return p.parse_args(argv)


def main(argv=None) -> list[Path]:
    args = parse_args(argv)
    files = sorted(args.summary_dir.glob("vaccine_impact_summary_ve_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no summary parquet in {args.summary_dir}")

    runs = {}
    for f in files:
        df = pd.read_parquet(f)
        variant = df.ve_variant.iloc[0]
        products = df.product_scenario.iloc[0] if "product_scenario" in df else "projected"
        dpath = args.summary_dir / f"vaccine_impact_draws_ve_{variant}_{products}.parquet"
        runs[(variant, products)] = {
            "summary": df,
            "draws": pd.read_parquet(dpath) if dpath.exists() else None,
        }

    written, tables = [], []
    for (variant, products), run in sorted(runs.items()):
        df, draws = run["summary"], run["draws"]
        tag = f"{variant}_{products}"
        pretty = (f"{PRODUCT_PRETTY.get(products, products)}  ·  "
                  f"{VARIANT_PRETTY.get(variant, variant)}")
        for measure in ("incidence", "mortality"):
            written.append(plot_measure(df, measure, tag, pretty, args.summary_dir))
        if draws is None:
            print(f"  (no draw-level file for {tag}; skipping overview)")
            continue
        ns, nd, through = _normalize_vaccine(df, draws)
        path, stats = plot_overview(
            ns, nd, through, VACCINE_LABELS,
            f"Malaria vaccine impact — {pretty}",
            args.summary_dir / f"vaccine_impact_overview_ve_{tag}.png",
            f"overview: {pretty}")
        written.append(path)
        stats["ve_variant"], stats["product_scenario"] = variant, products
        tables.append(stats)

    def pair(a_key, b_key, labels, title, stem):
        a, b = runs.get(a_key), runs.get(b_key)
        if not a or not b or a["draws"] is None or b["draws"] is None:
            return
        ns, nd, through = _normalize_variants(a["draws"], b["draws"])
        path, stats = plot_overview(ns, nd, through, labels, title,
                                    args.summary_dir / stem, title)
        written.append(path)
        stats["ve_variant"] = f'{a_key[0]} vs {b_key[0]}'
        stats["product_scenario"] = f'{a_key[1]} vs {b_key[1]}'
        tables.append(stats)

    V0, VS = "loglinear_severe0", "loglinear_severeSmooth"
    for products in ("projected", "all_r21"):
        pair((V0, products), (VS, products),
             _pair_labels(VARIANT_SHORT[V0], VARIANT_SHORT[VS],
                          arm_a="Ends at 24 mo", arm_b="Decays past 24 mo"),
             f"VE curve comparison — {PRODUCT_PRETTY[products]}",
             f"vaccine_impact_overview_vecompare_{products}.png")
    for variant in (V0, VS):
        pair((variant, "projected"), (variant, "all_r21"),
             _pair_labels(PRODUCT_SHORT["projected"], PRODUCT_SHORT["all_r21"],
                          arm_a="Projected", arm_b="R21 everywhere"),
             f"Product rollout comparison — {VARIANT_PRETTY[variant]}",
             f"vaccine_impact_overview_productcompare_{variant}.png")

    if tables:
        cols = ["figure", "figure_file", "ve_variant", "product_scenario",
                "measure", "ssp_scenario",
                "box", "quantity", "n_draws", "mean", "lower_95", "upper_95",
                "min", "whisker_lo", "q1", "median", "q3", "whisker_hi", "max", "n_fliers"]
        table = pd.concat(tables, ignore_index=True)[cols]
        tp = args.summary_dir / "vaccine_impact_box_table.parquet"
        write_parquet(table, tp)
        # CSV alongside purely for reading off by eye; the parquet is canonical
        table.to_csv(args.summary_dir / "vaccine_impact_box_table.csv",
                     index=False, encoding="utf-8-sig")
        written += [tp, args.summary_dir / "vaccine_impact_box_table.csv"]
        print(f"box table: {len(table)} rows")

    for w in written:
        print("wrote", w)

    # Renaming outputs leaves the old files behind, and a stale PNG next to a
    # current one is indistinguishable by eye -- so say which figures in this
    # directory this run did not produce.
    produced = {w.name for w in written}
    stale = sorted(f.name for f in args.summary_dir.glob("*.png")
                   if f.name not in produced)
    if stale:
        print(f"\nWARNING: {len(stale)} PNG(s) in {args.summary_dir} were NOT written "
              "by this run and may be stale leftovers from an earlier naming scheme:")
        for s in stale:
            print(f"    {s}")
    return written


if __name__ == "__main__":
    main()
