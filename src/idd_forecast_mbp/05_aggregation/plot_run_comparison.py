"""Global and super-region comparison figures: current forecast run vs a previous run.

Three figure types, all reading SAVED products rather than re-deriving aggregation:

1. ``timeseries``  — per (measure, location): two panels, count left and rate right.
   Observed in black through the anchor year, then the three SSP scenarios forward.
   Current run solid, previous run dashed.
2. ``differences`` — per measure, global: two stacked panels. Annual scenario
   differences on top with the band between them shaded, cumulative scenario
   differences below. The differences are WITHIN a run; the previous run appears as
   its own dashed set.
3. ``bars``        — per measure, global: counts at two horizon years, previous run
   against current run, grouped by scenario, with 95% intervals.

Count and rate never share an axis — they are separate panels, because a dual y-scale
makes two incomparable units look comparable.

The previous run is stored as admin-2-through-global draw-level COUNTS (one netCDF per
cause/measure/metric/scenario arm), so its rates are derived here using the same
population artifact the current products used. That is worth remembering when reading
a rate comparison: the previous run never asserted a rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import click
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from idd_forecast_mbp import constants as mbpc  # noqa: E402
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy  # noqa: E402
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids  # noqa: E402
from idd_forecast_mbp.lib.processing.weights import (  # noqa: E402
    WEIGHT_SCHEMES,
    load_weights,
    weighted_rollup_to_levels,
)

# Colours and display labels come from the project scenario map — never invented here.
# The dict keys are SSP (ssp126/245/585) because that is what the file paths use, but
# every label a reader sees is the RCP name, per mbpc.ssp_scenarios[...]["name"].
# Checked for colourblind separation: minimum OKLab dE 25.0 for normal vision and 23.0
# under simulated protanopia/deuteranopia/tritanopia.
SCENARIO_COLOR = {k: v["color"] for k, v in mbpc.ssp_scenarios.items()}
SCENARIO_LABEL = {k: v["name"] for k, v in mbpc.ssp_scenarios.items()}
OBSERVED_COLOR = "#1A1A1A"
GRID_KW = {"color": "#CCCCCC", "linewidth": 0.6, "linestyle": ":", "alpha": 0.9}

# Our short measure key -> the previous run's directory token.
OLD_MEASURE_TOKEN = {"inc": "incidence", "mort": "mortality"}
MEASURE_LABEL = {"inc": "Incidence", "mort": "Mortality"}
COUNT_NOUN = {"inc": "Cases", "mort": "Deaths"}

SCALES = [(1e9, "billions"), (1e6, "millions"), (1e3, "thousands"), (1.0, "")]


def annotate_direction(
    ax,
    above: str = "averted",
    below: str = "incurred",
    *,
    x: float = 0.015,
) -> None:
    """Label both sides of the zero line so the sign convention reads off the figure.

    This is what lets every difference figure keep ONE subtraction order. A fixed
    subtraction cannot be positive for every sensitivity — population growth adds
    burden while income growth removes it — so instead of flipping the arithmetic per
    panel (which destroys comparability between panels) the axis says what each side
    of zero means. Call AFTER plotting, since it reads the settled y-limits.
    """
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if span <= 0:
        return
    tr = ax.get_yaxis_transform()  # x in axes fraction, y in data units
    common = {
        "transform": tr, "rotation": 90, "ha": "left", "va": "center",
        "fontsize": 8, "color": "#666666",
    }
    # Only label a side that actually occupies a usable slice of the panel. Without
    # this, an axis whose data is entirely positive still has a sliver below zero from
    # matplotlib's margin, and both labels get drawn on top of each other.
    floor = 0.18 * span
    if y1 > floor:
        ax.text(x, y1 * 0.5, f"↑ {above}", **common)
    if y0 < -floor:
        ax.text(x, y0 * 0.5, f"↓ {below}", **common)


def pick_scale(values: np.ndarray) -> tuple[float, str]:
    """Choose a human-readable unit scale from the largest magnitude present."""
    peak = float(np.nanmax(np.abs(values))) if values.size else 0.0
    for divisor, name in SCALES:
        if peak >= divisor:
            return divisor, name
    return 1.0, ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_current(
    product_dir: Path, ssps: Sequence[str], dah: str, levels: Sequence[int]
) -> pd.DataFrame:
    """Long frame from the saved all-age summary products."""
    frames = []
    for ssp in ssps:
        path = product_dir / f"all_age_summary_{ssp}_{dah}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"current product missing: {path}")
        df = pd.read_parquet(path)
        df = df[df.level.isin(list(levels))]
        for measure in ("inc", "mort"):
            if f"{measure}_count_mean" not in df.columns:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "location_id": df.location_id.to_numpy(),
                        "year_id": df.year_id.to_numpy(),
                        "level": df.level.to_numpy(),
                        "ssp": ssp,
                        "measure": measure,
                        "run": "current",
                        "population": df.population.to_numpy(),
                        "count_mean": df[f"{measure}_count_mean"].to_numpy(),
                        "count_lower": df[f"{measure}_count_lower"].to_numpy(),
                        "count_upper": df[f"{measure}_count_upper"].to_numpy(),
                        "rate_mean": df[f"{measure}_rate_mean"].to_numpy(),
                        "rate_lower": df[f"{measure}_rate_lower"].to_numpy(),
                        "rate_upper": df[f"{measure}_rate_upper"].to_numpy(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def load_previous(
    old_root: Path,
    ssps: Sequence[str],
    dah: str,
    location_ids: Sequence[int],
    population: pd.DataFrame,
    cause: str = "malaria",
    hold: str | None = None,
) -> pd.DataFrame:
    """Long frame from the previous run's draw-level count netCDFs.

    Only the requested locations are read, so the draw arrays stay small. Rates are
    derived from the population artifact because the previous run stored counts only.

    ``hold`` selects one of the previous run's SENSITIVITY arms rather than its
    baseline, by appending ``_hold_<hold>`` after the dah token — the naming the 2025
    chain used. Valid values on that run: ``gdppc``, ``population``, ``as_structure``,
    ``DAH``, ``suitability``, ``flood``. This is what makes a like-for-like
    previous-vs-current comparison of the SAME sensitivity possible.
    """
    want = [int(x) for x in location_ids]
    frames = []
    for ssp in ssps:
        for measure, token in OLD_MEASURE_TOKEN.items():
            arm = (
                f"aa_cause_{cause}_measure_{token}_metric_count"
                f"_ssp_scenario_{ssp}_dah_scenario_{dah}"
                f"{f'_hold_{hold}' if hold else ''}"
            )
            path = old_root / arm / "draws.nc"
            if not path.exists():
                click.echo(f"    previous run: SKIP missing {arm}")
                continue
            with xr.open_dataset(path) as ds:
                present = {int(x) for x in ds.location_id.values}   # once, not per location
                have = [i for i in want if i in present]
                if not have:
                    continue
                sub = ds["val"].sel(location_id=have)
                mean = sub.mean("draw_id")
                qs = sub.quantile([0.025, 0.975], dim="draw_id")
                long = (
                    mean.to_dataframe(name="count_mean")
                    .reset_index()
                    .merge(
                        qs.sel(quantile=0.025)
                        .to_dataframe(name="count_lower")
                        .reset_index()
                        .drop(columns="quantile"),
                        on=["location_id", "year_id"],
                    )
                    .merge(
                        qs.sel(quantile=0.975)
                        .to_dataframe(name="count_upper")
                        .reset_index()
                        .drop(columns="quantile"),
                        on=["location_id", "year_id"],
                    )
                )
            long["ssp"] = ssp
            long["measure"] = measure
            long["run"] = "previous"
            frames.append(long)

    if not frames:
        raise FileNotFoundError(f"no previous-run arms found under {old_root}")

    out = pd.concat(frames, ignore_index=True)
    out = out.merge(population, on=["location_id", "year_id"], how="left")
    for stat in ("mean", "lower", "upper"):
        out[f"rate_{stat}"] = np.where(
            out.population > 0, out[f"count_{stat}"] / out.population, np.nan
        )
    return out


def load_population_admin2(years: Sequence[int]) -> pd.DataFrame:
    """Admin-2 population, the weight for aggregating a driver to reporting levels."""
    pop = read_parquet_with_integer_ids(
        mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
        columns=["location_id", "year_id", "population"],
    )
    return pop[pop.year_id.isin([int(y) for y in years])].reset_index(drop=True)


def load_observed(location_ids: Sequence[int], first_year: int, last_year: int) -> pd.DataFrame:
    """Observed raked counts and rates for the historical panel."""
    path = mbpc.MAL_RAKED_AA_READ_PATH / "aa_full_malaria_df.parquet"
    obs = read_parquet_with_integer_ids(
        path,
        columns=[
            "location_id",
            "year_id",
            "malaria_inc_count",
            "malaria_inc_rate",
            "malaria_mort_count",
            "malaria_mort_rate",
        ],
    )
    obs = obs[
        obs.location_id.isin([int(x) for x in location_ids])
        & obs.year_id.between(first_year, last_year)
    ]
    frames = []
    for measure in ("inc", "mort"):
        frames.append(
            pd.DataFrame(
                {
                    "location_id": obs.location_id.to_numpy(),
                    "year_id": obs.year_id.to_numpy(),
                    "measure": measure,
                    "count_mean": obs[f"malaria_{measure}_count"].to_numpy(),
                    "rate_mean": obs[f"malaria_{measure}_rate"].to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["location_id", "year_id"])


# ---------------------------------------------------------------------------
# Figure 1 — timeseries, count and rate side by side
# ---------------------------------------------------------------------------

def figure_timeseries(
    location_id: int,
    location_name: str,
    measure: str,
    observed: pd.DataFrame,
    current: pd.DataFrame,
    previous: pd.DataFrame,
    ssps: Sequence[str],
    out_path: Path,
    main_label: str = "current",
    comp_label: str = "previous",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    obs = observed.query("location_id == @location_id and measure == @measure")
    cur = current.query("location_id == @location_id and measure == @measure")
    prv = previous.query("location_id == @location_id and measure == @measure")

    for ax, metric in zip(axes, ("count", "rate"), strict=True):
        col = f"{metric}_mean"
        pool = np.concatenate(
            [s[col].to_numpy() for s in (obs, cur, prv) if not s.empty]
        )
        divisor, unit = pick_scale(pool) if metric == "count" else (1.0, "")

        if not obs.empty:
            o = obs.sort_values("year_id")
            ax.plot(
                o.year_id,
                o[col] / divisor,
                color=OBSERVED_COLOR,
                linewidth=2.0,
                label="Observed",
                zorder=5,
            )

        for ssp in ssps:
            c = cur[cur.ssp == ssp].sort_values("year_id")
            if not c.empty:
                ax.plot(
                    c.year_id,
                    c[col] / divisor,
                    color=SCENARIO_COLOR[ssp],
                    linewidth=2.0,
                    label=f"{SCENARIO_LABEL[ssp]} ({main_label})",
                    zorder=4,
                )
            p = prv[prv.ssp == ssp].sort_values("year_id")
            if not p.empty:
                ax.plot(
                    p.year_id,
                    p[col] / divisor,
                    color=SCENARIO_COLOR[ssp],
                    linewidth=1.6,
                    linestyle=(0, (5, 2)),
                    label=f"{SCENARIO_LABEL[ssp]} ({comp_label})",
                    zorder=3,
                )

        if metric == "count":
            noun = COUNT_NOUN[measure]
            ax.set_ylabel(f"{noun}" + (f" (in {unit})" if unit else ""))
            ax.set_title("Count", fontsize=11)
        else:
            ax.set_ylabel(f"{MEASURE_LABEL[measure]} rate (per person per year)")
            ax.set_title("Rate", fontsize=11)
        ax.set_xlabel("Year")
        ax.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(f"{MEASURE_LABEL[measure]} — {location_name}", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — annual and cumulative scenario differences
# ---------------------------------------------------------------------------

# Each entry is (high scenario, low scenario). Colour is carried by the pairing.
ANNUAL_PAIRS = [("ssp585", "ssp126"), ("ssp245", "ssp126")]
CUMULATIVE_PAIRS = [("ssp585", "ssp126"), ("ssp585", "ssp245"), ("ssp245", "ssp126")]
PAIR_COLOR = {
    ("ssp585", "ssp126"): SCENARIO_COLOR["ssp585"],
    ("ssp585", "ssp245"): SCENARIO_COLOR["ssp245"],
    ("ssp245", "ssp126"): SCENARIO_COLOR["ssp126"],
}


def _wide(df: pd.DataFrame, location_id: int, measure: str) -> pd.DataFrame:
    """year_id index, one column per scenario, of mean counts."""
    sub = df.query("location_id == @location_id and measure == @measure")
    return sub.pivot_table(index="year_id", columns="ssp", values="count_mean")


def figure_differences(
    location_id: int,
    location_name: str,
    measure: str,
    current: pd.DataFrame,
    previous: pd.DataFrame,
    start_year: int,
    out_path: Path,
    comp_label: str = "previous",
) -> None:
    cur = _wide(current, location_id, measure)
    prv = _wide(previous, location_id, measure)
    cur = cur[cur.index >= start_year]
    prv = prv[prv.index >= start_year]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # --- top: annual differences, band shaded between the two contrasts ---
    ax = axes[0]
    annual = {}
    for hi, lo in ANNUAL_PAIRS:
        for label, frame in (("current", cur), ("previous", prv)):
            if hi in frame.columns and lo in frame.columns:
                annual[(hi, lo, label)] = frame[hi] - frame[lo]
    pool = np.concatenate([s.to_numpy() for s in annual.values()]) if annual else np.array([0.0])
    divisor, unit = pick_scale(pool)

    cur_annual = [k for k in annual if k[2] == "current"]
    if len(cur_annual) == 2:
        a = annual[cur_annual[0]] / divisor
        b = annual[cur_annual[1]] / divisor
        common = a.index.intersection(b.index)
        ax.fill_between(
            common,
            a.loc[common],
            b.loc[common],
            color=SCENARIO_COLOR["ssp585"],
            alpha=0.12,
            zorder=1,
        )
    for (hi, lo, label), series in annual.items():
        ax.plot(
            series.index,
            series / divisor,
            color=PAIR_COLOR[(hi, lo)],
            linewidth=2.0 if label == "current" else 1.6,
            linestyle="-" if label == "current" else (0, (5, 2)),
            label=f"{SCENARIO_LABEL[hi]} − {SCENARIO_LABEL[lo]}"
            + ("" if label == "current" else f" ({comp_label})"),
            zorder=3,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)
    noun = COUNT_NOUN[measure]
    ax.set_ylabel(f"{noun}" + (f"\n(in {unit})" if unit else ""))
    ax.legend(frameon=False, loc="best", fontsize=9)

    # --- bottom: cumulative differences ---
    ax = axes[1]
    cumulative = {}
    for hi, lo in CUMULATIVE_PAIRS:
        for label, frame in (("current", cur), ("previous", prv)):
            if hi in frame.columns and lo in frame.columns:
                # AMENABLE, not a plain cumulative difference: the burden the lower
                # scenario avoids, accumulated forward as a positive quantity. Hence
                # (lo - hi), the negation of the annual panel above.
                # Higher forcing MINUS lower forcing, the same order as the annual
                # panel above. For malaria this goes negative, because RCP8.5 yields
                # fewer deaths than RCP2.6 — that is the intended headline, not a bug,
                # and the two-sided axis annotation says so.
                cumulative[(hi, lo, label)] = (frame[hi] - frame[lo]).cumsum()
    pool = (
        np.concatenate([s.to_numpy() for s in cumulative.values()])
        if cumulative
        else np.array([0.0])
    )
    divisor, unit = pick_scale(pool)
    for (hi, lo, label), series in cumulative.items():
        ax.plot(
            series.index,
            series / divisor,
            color=PAIR_COLOR[(hi, lo)],
            linewidth=2.0 if label == "current" else 1.6,
            linestyle="-" if label == "current" else (0, (5, 2)),
            label=f"{SCENARIO_LABEL[hi]} − {SCENARIO_LABEL[lo]}"
            + ("" if label == "current" else f" ({comp_label})"),
            zorder=3,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_ylabel(
        f"Cumulative Δ {noun.lower()}" + (f"\n(in {unit})" if unit else ""), labelpad=26
    )
    ax.set_xlabel("Year")
    ax.legend(frameon=False, loc="best", fontsize=9, ncol=2)

    for ax in axes:
        ax.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # No two-sided annotation here: with four to six series plus a legend the
        # rotated side text collides with everything. The subtraction order is stated
        # in the series labels themselves (RCP8.5 - RCP2.6).

    fig.suptitle(
        f"{MEASURE_LABEL[measure]} — scenario differences within run, {location_name}\n"
        f"cumulative from {start_year}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — horizon-year bars, previous vs current
# ---------------------------------------------------------------------------

def _draw_bars(  # noqa: PLR0913
    ax,
    location_id: int,
    location_name: str,
    measure: str,
    current: pd.DataFrame,
    previous: pd.DataFrame,
    ssps: Sequence[str],
    horizon_years: Sequence[int],
    anchor_year: int = 2023,
    main_label: str = "current",
    comp_label: str = "previous",
    overlay: dict[str, pd.DataFrame] | None = None,
    overlay_label: str = "2025 run",
    annotate_values: bool = False,
) -> None:
    """Draw the cumulative bar chart onto an existing Axes.

    ``annotate_values`` prints mean and 95% interval above each bar. With twelve bars the
    text only fits rotated, and the y-limit is expanded to make room, so this is offered as
    a SECOND version of each bar figure rather than replacing the clean one.

    Extracted from figure_bars so the standalone figure and the
    timeseries-over-bars composite share one implementation rather than drifting.
    """
    # `overlay` maps the same run keys to the PREVIOUS vintage of each series, drawn as
    # an interval on top of the corresponding bar so the shift between runs is visible
    # without doubling the number of bars. Magenta, not red: red is too close to the
    # RCP8.5 bar colour to read on that group.
    hmax = max(int(y) for y in horizon_years)
    runs = [("previous", previous), ("current", current)]
    OVERLAY_COLOR = "#C51B8A"

    def cumulative(frame: pd.DataFrame, ssp: str, year: int) -> pd.Series | None:
        """Total over anchor_year..year, not the single-year value.

        The interval is the sum of the per-year bounds. That is exact only under
        perfect rank correlation of draws across years -- which the anchored shift
        makes very nearly true, since a draw keeps its identity along its whole
        trajectory -- but it is an approximation, and a slightly wide one.
        """
        # Boolean mask, not .query(): inside a nested function the @-names pandas looks
        # for are closure variables, not locals of the frame it inspects.
        m = (
            (frame.location_id == location_id)
            & (frame.measure == measure)
            & (frame.ssp == ssp)
            & (frame.year_id >= anchor_year)
            & (frame.year_id <= year)
        )
        sel = frame.loc[m]
        if sel.empty:
            return None
        return sel[["count_mean", "count_lower", "count_upper"]].sum()
    pool = []
    for _, frame in runs:
        m = (
            (frame.location_id == location_id)
            & (frame.measure == measure)
            & (frame.year_id >= anchor_year)
            & (frame.year_id <= hmax)
        )
        sub = frame.loc[m]
        if sub.empty:
            continue
        pool.append(
            sub.groupby(["ssp"], observed=True).count_upper.sum().to_numpy()
        )
    divisor, unit = pick_scale(np.concatenate(pool) if pool else np.array([0.0]))

    # Nested spacing so the eye groups by scenario first, then by year: the two
    # horizon-year clusters sit close together inside a scenario, and the scenario
    # groups are separated by real whitespace.
    width = 0.34
    year_gap = 0.80
    scenario_span = (len(horizon_years) - 1) * year_gap
    scenario_gap = scenario_span + 1.35

    year_centres, year_labels, group_spans = [], [], []
    value_labels: list[tuple[float, float, str]] = []

    for i, ssp in enumerate(ssps):
        base = i * scenario_gap
        for j, year in enumerate(horizon_years):
            centre = base + j * year_gap
            for offset, (run_name, frame) in zip((-0.5, 0.5), runs, strict=True):
                r = cumulative(frame, ssp, year)
                if r is None:
                    continue
                x = centre + offset * width * 1.06
                mean = r.count_mean / divisor
                lo = (r.count_mean - r.count_lower) / divisor
                hi = (r.count_upper - r.count_mean) / divisor
                ax.bar(
                    x,
                    mean,
                    width=width,
                    color=SCENARIO_COLOR[ssp],
                    alpha=1.0 if run_name == "current" else 0.45,
                    hatch="" if run_name == "current" else "///",
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=3,
                )
                if annotate_values:
                    value_labels.append((
                        x, mean + max(hi, 0),
                        f"{mean:,.3g}\n[{r.count_lower / divisor:,.3g}, "
                        f"{r.count_upper / divisor:,.3g}]",
                    ))
                ax.errorbar(
                    x,
                    mean,
                    yerr=[[max(lo, 0)], [max(hi, 0)]],
                    fmt="none",
                    ecolor="#444444",
                    elinewidth=1.2,
                    capsize=3,
                    zorder=4,
                )
                if overlay is not None and run_name in overlay:
                    pr = cumulative(overlay[run_name], ssp, year)
                    if pr is not None:
                        pm = pr.count_mean / divisor
                        ax.errorbar(
                            x - width * 0.30,
                            pm,
                            yerr=[
                                [max((pr.count_mean - pr.count_lower) / divisor, 0)],
                                [max((pr.count_upper - pr.count_mean) / divisor, 0)],
                            ],
                            fmt="_",
                            markersize=11,
                            markeredgewidth=1.6,
                            color=OVERLAY_COLOR,
                            ecolor=OVERLAY_COLOR,
                            elinewidth=1.6,
                            capsize=4,
                            zorder=6,
                        )
            year_centres.append(centre)
            year_labels.append(f"to {year}")
        group_spans.append((base, base + scenario_span))

    ax.set_xticks(year_centres)
    ax.set_xticklabels(year_labels, fontsize=9)
    ax.tick_params(axis="x", length=0)
    # Scenario name centred beneath its pair of year clusters.
    for (left, right), ssp in zip(group_spans, ssps, strict=True):
        ax.text(
            (left + right) / 2,
            -0.075,
            SCENARIO_LABEL[ssp],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
        )
    ax.set_xlim(group_spans[0][0] - 0.75, group_spans[-1][1] + 0.75)
    noun = COUNT_NOUN[measure]
    ax.set_ylabel(f"Cumulative {noun.lower()} from {anchor_year}"
                  + (f" (in {unit})" if unit else ""))
    if value_labels:
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 * 1.42)           # headroom for the rotated labels
        pad = 0.012 * (y1 - y0)
        for xx, top, txt in value_labels:
            ax.text(xx, top + pad, txt, rotation=90, ha="center", va="bottom",
                    fontsize=6.5, color="#333333", linespacing=1.05)

    ax.grid(True, axis="y", **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend_marks = [
        mpl.patches.Patch(facecolor="#777777", alpha=0.45, hatch="///", edgecolor="white",
                          label=comp_label[:1].upper() + comp_label[1:]),
        mpl.patches.Patch(facecolor="#777777", edgecolor="white",
                          label=main_label[:1].upper() + main_label[1:]),
    ]
    if overlay:
        legend_marks.append(
            mpl.lines.Line2D([], [], color=OVERLAY_COLOR, marker="_", markersize=11,
                             linewidth=1.6, label=f"{overlay_label} (mean + 95% UI)")
        )
    ax.legend(handles=legend_marks, frameon=False, loc="upper left", fontsize=9)


# ---------------------------------------------------------------------------
# Figure 4 — sensitivity against baseline, one column per scenario
# ---------------------------------------------------------------------------



def figure_bars(  # noqa: PLR0913
    location_id: int,
    location_name: str,
    measure: str,
    current: pd.DataFrame,
    previous: pd.DataFrame,
    ssps: Sequence[str],
    horizon_years: Sequence[int],
    out_path: Path,
    anchor_year: int = 2023,
    main_label: str = "current",
    comp_label: str = "previous",
    overlay: dict[str, pd.DataFrame] | None = None,
    overlay_label: str = "2025 run",
    annotate_values: bool = False,
) -> None:
    """Standalone cumulative bar figure."""
    h = 7.4 if annotate_values else 5.5
    fig, ax = plt.subplots(figsize=(2.7 * len(ssps) + 2.2, h))
    _draw_bars(ax, location_id, location_name, measure, current, previous,
               ssps, horizon_years, anchor_year, main_label, comp_label,
               overlay, overlay_label, annotate_values)
    fig.suptitle(
        f"Cumulative {MEASURE_LABEL[measure].lower()} — {location_name}, "
        f"{comp_label} vs {main_label}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def figure_sensitivity_difference(
    location_id: int,
    location_name: str,
    measure: str,
    sensitivity: pd.DataFrame,
    baseline: pd.DataFrame,
    ssps: Sequence[str],
    start_year: int,
    out_path: Path,
    sensitivity_label: str = "sensitivity",
) -> None:
    """Annual (top) and cumulative (bottom) baseline-minus-sensitivity, per scenario.

    Same spirit as the scenario-difference figure, but the contrast is between two
    RUNS at a fixed scenario rather than between two scenarios within one run. Sign
    follows the amenable convention used in the cumulative panel elsewhere: baseline
    minus sensitivity, so a sensitivity that REDUCES burden reads positive.
    """
    sens = _wide(sensitivity, location_id, measure)
    base = _wide(baseline, location_id, measure)
    sens = sens[sens.index >= start_year]
    base = base[base.index >= start_year]

    annual, cumulative = {}, {}
    for ssp in ssps:
        if ssp in sens.columns and ssp in base.columns:
            common = sens.index.intersection(base.index)
            # counterfactual MINUS baseline: positive => the 2023-fixed world has
            # MORE burden, i.e. the projected trend AVERTED it; negative => incurred.
            diff = sens.loc[common, ssp] - base.loc[common, ssp]
            annual[ssp] = diff
            cumulative[ssp] = diff.cumsum()
    if not annual:
        raise ValueError(f"no overlapping scenarios for location {location_id}")

    a_div, a_unit = pick_scale(np.concatenate([s.to_numpy() for s in annual.values()]))
    c_div, c_unit = pick_scale(np.concatenate([s.to_numpy() for s in cumulative.values()]))

    ncol = len(ssps)
    fig, axes = plt.subplots(
        2, ncol, figsize=(4.6 * ncol, 7.4), sharex=True, sharey="row", squeeze=False
    )
    noun = COUNT_NOUN[measure]

    for j, ssp in enumerate(ssps):
        colour = SCENARIO_COLOR[ssp]
        for row, (series_map, divisor) in enumerate(
            ((annual, a_div), (cumulative, c_div))
        ):
            ax = axes[row][j]
            if ssp in series_map:
                s = series_map[ssp]
                ax.plot(s.index, s / divisor, color=colour, linewidth=2.0, zorder=3)
                ax.fill_between(
                    s.index, 0, s / divisor, color=colour, alpha=0.12, zorder=1
                )
            ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)
            ax.grid(True, **GRID_KW)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if row == 0:
                ax.set_title(SCENARIO_LABEL[ssp], fontsize=12)
            if row == 1:
                ax.set_xlabel("Year")

    axes[0][0].set_ylabel(
        f"Annual Δ {noun.lower()}" + (f" (in {a_unit})" if a_unit else ""), labelpad=26
    )
    axes[1][0].set_ylabel(
        f"Cumulative Δ {noun.lower()}" + (f" (in {c_unit})" if c_unit else ""), labelpad=26
    )
    for row in (0, 1):
        annotate_direction(
            axes[row][0],
            above="averted",
            below="incurred",
        )
    fig.suptitle(
        f"{MEASURE_LABEL[measure]} — effect of {sensitivity_label}, {location_name}"
        f"\ncounterfactual minus baseline: ↑ = trend averted burden, "
        f"↓ = trend incurred it · cumulative from {start_year}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2b — scenario differences with the level trajectories above them
# ---------------------------------------------------------------------------

def figure_differences_stack(  # noqa: PLR0913
    location_id: int,
    location_name: str,
    measure: str,
    current: pd.DataFrame,
    previous: pd.DataFrame,
    start_year: int,
    out_path: Path,
    comp_label: str = "previous",
) -> None:
    """Three rows: the scenario levels, then their annual and cumulative differences.

    Same content as figure_differences with the inputs shown above it, so the reader can
    see which trajectories the contrast is built from. Both difference rows are
    higher-forcing minus lower-forcing.
    """
    cur = _wide(current, location_id, measure)
    prv = _wide(previous, location_id, measure)
    cur = cur[cur.index >= start_year]
    prv = prv[prv.index >= start_year]
    noun = COUNT_NOUN[measure]

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    lvl_div, lvl_unit = pick_scale(
        np.concatenate([cur[c].to_numpy() for c in cur.columns] or [np.array([0.0])]))
    ax = axes[0]
    for ssp in cur.columns:
        ax.plot(cur.index, cur[ssp] / lvl_div, color=SCENARIO_COLOR[ssp], linewidth=2.0,
                label=SCENARIO_LABEL[ssp], zorder=4)
        if ssp in prv.columns:
            ax.plot(prv.index, prv[ssp] / lvl_div, color=SCENARIO_COLOR[ssp],
                    linewidth=1.6, linestyle=(0, (5, 2)), zorder=3)
    ax.set_ylabel(f"{noun}" + (f"\n(in {lvl_unit})" if lvl_unit else ""))
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="best")

    for row, pairs, cumulate in ((1, ANNUAL_PAIRS, False), (2, CUMULATIVE_PAIRS, True)):
        ax = axes[row]
        series = {}
        for hi, lo in pairs:
            for lab, frame in (("current", cur), ("previous", prv)):
                if hi in frame.columns and lo in frame.columns:
                    d = frame[hi] - frame[lo]
                    series[(hi, lo, lab)] = d.cumsum() if cumulate else d
        if not series:
            continue
        div, unit = pick_scale(np.concatenate([v.to_numpy() for v in series.values()]))
        for (hi, lo, lab), v in series.items():
            ax.plot(v.index, v / div, color=PAIR_COLOR[(hi, lo)],
                    linewidth=2.0 if lab == "current" else 1.6,
                    linestyle="-" if lab == "current" else (0, (5, 2)),
                    label=f"{SCENARIO_LABEL[hi]} − {SCENARIO_LABEL[lo]}"
                          + ("" if lab == "current" else f" ({comp_label})"),
                    zorder=3)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)
        pre = "Cumulative " if cumulate else ""
        ax.set_ylabel(f"{pre}Δ {noun.lower()}" + (f"\n(in {unit})" if unit else ""))
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")

    for ax in axes:
        ax.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[2].set_xlabel("Year")
    fig.suptitle(
        f"{MEASURE_LABEL[measure]} — {location_name}: levels, then scenario differences",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Covariate arms — the driver trajectory under baseline vs counterfactual
# ---------------------------------------------------------------------------

def _weighted_to_levels(
    values: pd.DataFrame, pop: pd.DataFrame, hierarchy: pd.DataFrame, *,
    total: bool = False,
) -> pd.DataFrame:
    """Delegate to lib/processing/weights so there is one implementation."""
    w = pop.rename(columns={"population": "weight"}) if "population" in pop.columns else pop
    return weighted_rollup_to_levels(values, w, hierarchy, total=total)


def load_covariate_arms(  # noqa: PLR0913
    ssps: Sequence[str],
    admin2: Sequence[int],
    pop: pd.DataFrame,
    hierarchy: pd.DataFrame,
    var: str,
    *,
    dah_base: str = "Baseline",
    dah_alt: str | None = None,
    hold_year: int | None = None,
    total: bool = False,
) -> dict[str, pd.DataFrame]:
    """The driver under BOTH arms, population-weighted to global and super-region.

    Two ways a counterfactual arm is formed, matching how the sensitivity was actually
    produced: ``dah_alt`` selects a different ``dah_scenario`` slice of the same input
    netCDF (the DAH sensitivity is a real input dimension), while ``hold_year`` freezes
    the trajectory at that year (the GDP hold is a rocket-side freeze). Exactly one of
    the two applies.
    """
    keep = [int(x) for x in admin2]
    out = {}
    for ssp in ssps:
        path = mbpc.MAL_FORECAST_INPUTS_READ_PATH / f"malaria_forecast_inputs_{ssp}.nc"
        with xr.open_dataset(path) as ds:
            if var not in ds.data_vars:
                return {}
            present = {int(x) for x in ds.location_id.values}   # once, not per location
            have = [i for i in keep if i in present]
            da = ds[var].sel(location_id=have)
            if "draw" in da.dims:
                da = da.mean("draw")
            base = da.sel(dah_scenario=dah_base) if "dah_scenario" in da.dims else da
            base = base.to_dataframe(name="value").reset_index()[
                ["location_id", "year_id", "value"]]
            if dah_alt is not None:
                alt = da.sel(dah_scenario=dah_alt).to_dataframe(name="value") \
                        .reset_index()[["location_id", "year_id", "value"]]
            elif hold_year is not None:
                ref = base[base.year_id == int(hold_year)][["location_id", "value"]] \
                        .rename(columns={"value": "held"})
                alt = base.merge(ref, on="location_id", how="left")
                alt["value"] = np.where(alt.year_id > int(hold_year), alt.held, alt.value)
                alt = alt[["location_id", "year_id", "value"]]
            else:
                return {}
        b = _weighted_to_levels(base, pop, hierarchy, total=total).rename(
            columns={"value": "baseline"})
        a = _weighted_to_levels(alt, pop, hierarchy, total=total).rename(
            columns={"value": "counterfactual"})
        out[ssp] = b.merge(a, on=["location_id", "year_id"])
    return out


# ---------------------------------------------------------------------------
# Figure 4c — 4 rows: the driver, the burden levels, then annual and cumulative Δ
# ---------------------------------------------------------------------------

def figure_sensitivity_full(  # noqa: PLR0913
    location_id: int,
    location_name: str,
    measure: str,
    sensitivity: pd.DataFrame,
    baseline: pd.DataFrame,
    cov_arms: dict[str, pd.DataFrame],
    ssps: Sequence[str],
    start_year: int,
    out_path: Path,
    sensitivity_label: str = "sensitivity",
    cov_label: str = "driver",
) -> None:
    """The whole causal chain in one figure, one column per scenario.

    Row 1 is the DRIVER itself under both arms — what the sensitivity actually changed.
    Row 2 is the resulting burden levels. Rows 3 and 4 are the annual and cumulative
    difference. Reading top to bottom you see the intervention, its effect on the
    trajectory, and the effect's size, without having to trust any of the steps.

    Not available for the age/sex-structure hold: that intervention is a reweighting
    across age groups, so there is no single covariate trajectory to draw.
    """
    sens = _wide(sensitivity, location_id, measure)
    base = _wide(baseline, location_id, measure)
    sens, base = sens[sens.index >= start_year], base[base.index >= start_year]
    annual, cumul = {}, {}
    for ssp in ssps:
        if ssp in sens.columns and ssp in base.columns:
            common = sens.index.intersection(base.index)
            d = sens.loc[common, ssp] - base.loc[common, ssp]
            annual[ssp], cumul[ssp] = d, d.cumsum()
    if not annual:
        raise ValueError(f"no overlapping scenarios for location {location_id}")

    noun = COUNT_NOUN[measure]
    lvl_div, lvl_unit = pick_scale(np.concatenate(
        [sens[c].to_numpy() for c in sens.columns] + [base[c].to_numpy() for c in base.columns]))
    a_div, a_unit = pick_scale(np.concatenate([v.to_numpy() for v in annual.values()]))
    c_div, c_unit = pick_scale(np.concatenate([v.to_numpy() for v in cumul.values()]))

    ncol = len(ssps)
    fig, axes = plt.subplots(4, ncol, figsize=(4.6 * ncol, 13.0),
                             sharex=True, sharey="row", squeeze=False)
    for j, ssp in enumerate(ssps):
        colour = SCENARIO_COLOR[ssp]
        ax = axes[0][j]
        cov = cov_arms.get(ssp)
        if cov is not None:
            c = cov[(cov.location_id == location_id) & (cov.year_id >= start_year)] \
                    .sort_values("year_id")
            if not c.empty:
                ax.plot(c.year_id, c.baseline, color=colour, linewidth=1.6,
                        linestyle=(0, (5, 2)), label="baseline", zorder=3)
                ax.plot(c.year_id, c.counterfactual, color=colour, linewidth=2.0,
                        label=sensitivity_label, zorder=4)
        ax.set_title(SCENARIO_LABEL[ssp], fontsize=12)

        ax = axes[1][j]
        if ssp in base.columns:
            ax.plot(base.index, base[ssp] / lvl_div, color=colour, linewidth=1.6,
                    linestyle=(0, (5, 2)), zorder=3)
        if ssp in sens.columns:
            ax.plot(sens.index, sens[ssp] / lvl_div, color=colour, linewidth=2.0, zorder=4)

        for row, (series, div) in ((2, (annual, a_div)), (3, (cumul, c_div))):
            ax = axes[row][j]
            if ssp in series:
                v = series[ssp]
                ax.plot(v.index, v / div, color=colour, linewidth=2.0, zorder=3)
                ax.fill_between(v.index, 0, v / div, color=colour, alpha=0.12, zorder=1)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)

        for row in range(4):
            a = axes[row][j]
            a.grid(True, **GRID_KW); a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        axes[3][j].set_xlabel("Year")

    axes[0][0].set_ylabel(cov_label)
    axes[1][0].set_ylabel(f"{noun}" + (f" (in {lvl_unit})" if lvl_unit else ""))
    axes[2][0].set_ylabel(f"Annual Δ {noun.lower()}"
                          + (f" (in {a_unit})" if a_unit else ""), labelpad=26)
    axes[3][0].set_ylabel(f"Cumulative Δ {noun.lower()}"
                          + (f" (in {c_unit})" if c_unit else ""), labelpad=26)
    axes[0][0].legend(frameon=False, fontsize=9, loc="best")
    for row in (2, 3):
        annotate_direction(axes[row][0])
    fig.suptitle(
        f"{MEASURE_LABEL[measure]} — {sensitivity_label}, {location_name}"
        f"\ndriver (top), burden, then counterfactual minus baseline: "
        f"↑ = averted, ↓ = incurred · cumulative from {start_year}",
        fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4b — 3x3: the two level trajectories, then their annual and cumulative diff
# ---------------------------------------------------------------------------

def figure_sensitivity_stack(  # noqa: PLR0913
    location_id: int,
    location_name: str,
    measure: str,
    sensitivity: pd.DataFrame,
    baseline: pd.DataFrame,
    ssps: Sequence[str],
    start_year: int,
    out_path: Path,
    sensitivity_label: str = "sensitivity",
) -> None:
    """Same content as figure_sensitivity_difference, with the inputs shown above it.

    Row 1 is the two LEVEL trajectories the difference is built from — baseline dashed,
    counterfactual solid — so the reader can see where the difference comes from instead
    of taking it on faith. Row 2 is the annual difference, row 3 the cumulative. Kept
    alongside the 2x3 version rather than replacing it.
    """
    sens = _wide(sensitivity, location_id, measure)
    base = _wide(baseline, location_id, measure)
    sens = sens[sens.index >= start_year]
    base = base[base.index >= start_year]

    annual, cumulative = {}, {}
    for ssp in ssps:
        if ssp in sens.columns and ssp in base.columns:
            common = sens.index.intersection(base.index)
            d = sens.loc[common, ssp] - base.loc[common, ssp]
            annual[ssp] = d
            cumulative[ssp] = d.cumsum()
    if not annual:
        raise ValueError(f"no overlapping scenarios for location {location_id}")

    lvl_div, lvl_unit = pick_scale(
        np.concatenate([sens[c].to_numpy() for c in sens.columns]
                       + [base[c].to_numpy() for c in base.columns]))
    a_div, a_unit = pick_scale(np.concatenate([v.to_numpy() for v in annual.values()]))
    c_div, c_unit = pick_scale(
        np.concatenate([v.to_numpy() for v in cumulative.values()]))

    ncol = len(ssps)
    fig, axes = plt.subplots(3, ncol, figsize=(4.6 * ncol, 10.4),
                             sharex=True, sharey="row", squeeze=False)
    noun = COUNT_NOUN[measure]

    for j, ssp in enumerate(ssps):
        colour = SCENARIO_COLOR[ssp]
        ax = axes[0][j]
        if ssp in base.columns:
            ax.plot(base.index, base[ssp] / lvl_div, color=colour, linewidth=1.6,
                    linestyle=(0, (5, 2)), label="baseline", zorder=3)
        if ssp in sens.columns:
            ax.plot(sens.index, sens[ssp] / lvl_div, color=colour, linewidth=2.0,
                    label=sensitivity_label, zorder=4)
        ax.set_title(SCENARIO_LABEL[ssp], fontsize=12)

        for row, (series, divisor) in ((1, (annual, a_div)), (2, (cumulative, c_div))):
            ax = axes[row][j]
            if ssp in series:
                v = series[ssp]
                ax.plot(v.index, v / divisor, color=colour, linewidth=2.0, zorder=3)
                ax.fill_between(v.index, 0, v / divisor, color=colour, alpha=0.12,
                                zorder=1)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)

        for row in (0, 1, 2):
            a = axes[row][j]
            a.grid(True, **GRID_KW)
            a.set_axisbelow(True)
            for spine in ("top", "right"):
                a.spines[spine].set_visible(False)
        axes[2][j].set_xlabel("Year")

    axes[0][0].set_ylabel(f"{noun}" + (f" (in {lvl_unit})" if lvl_unit else ""))
    axes[1][0].set_ylabel(f"Annual Δ {noun.lower()}"
                          + (f" (in {a_unit})" if a_unit else ""), labelpad=26)
    axes[2][0].set_ylabel(f"Cumulative Δ {noun.lower()}"
                          + (f" (in {c_unit})" if c_unit else ""), labelpad=26)
    axes[0][0].legend(frameon=False, fontsize=9, loc="best")
    for row in (1, 2):
        annotate_direction(axes[row][0])
    fig.suptitle(
        f"{MEASURE_LABEL[measure]} — effect of {sensitivity_label}, {location_name}"
        f"\nlevels (top), then counterfactual minus baseline: ↑ = trend averted burden, "
        f"↓ = trend incurred it · cumulative from {start_year}",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — the sensitivity EFFECT, this run against the previous run
# ---------------------------------------------------------------------------

def figure_effect_comparison(  # noqa: PLR0913
    location_id: int,
    location_name: str,
    measure: str,
    cur_sens: pd.DataFrame,
    cur_base: pd.DataFrame,
    prev_sens: pd.DataFrame,
    prev_base: pd.DataFrame,
    ssps: Sequence[str],
    start_year: int,
    out_path: Path,
    sensitivity_label: str = "sensitivity",
) -> None:
    """The sensitivity's EFFECT in each run, overlaid: annual (top), cumulative (bottom).

    One column per scenario. Solid = this run's effect, dashed = the previous run's
    effect, so the question answered is "does the sensitivity do the same thing it did
    last time", independent of the two runs sitting at different baseline levels.

    DIRECTION, stated once and used for both rows: ``sensitivity - baseline``. Positive
    means the sensitivity ADDS burden. This is the opposite of the ``baseline -
    sensitivity`` orientation used by figure_sensitivity_difference; that one answers
    "how much does the sensitivity avert", this one answers "how much does it cost".
    """
    def effect(sens: pd.DataFrame, base: pd.DataFrame) -> dict[str, pd.Series]:
        s = _wide(sens, location_id, measure)
        b = _wide(base, location_id, measure)
        s = s[s.index >= start_year]
        b = b[b.index >= start_year]
        out = {}
        for ssp in ssps:
            if ssp in s.columns and ssp in b.columns:
                common = s.index.intersection(b.index)
                # baseline - counterfactual, the ONE subtraction order used by every
                # difference figure. Sign carries the meaning; annotate_direction says
                # what each side of zero means.
                out[ssp] = s.loc[common, ssp] - b.loc[common, ssp]
        return out

    cur_eff = effect(cur_sens, cur_base)
    prev_eff = effect(prev_sens, prev_base)
    if not cur_eff:
        raise ValueError(f"no overlapping scenarios for location {location_id}")

    pool_a = np.concatenate(
        [s.to_numpy() for s in (*cur_eff.values(), *prev_eff.values())]
    )
    pool_c = np.concatenate(
        [s.cumsum().to_numpy() for s in (*cur_eff.values(), *prev_eff.values())]
    )
    a_div, a_unit = pick_scale(pool_a)
    c_div, c_unit = pick_scale(pool_c)

    ncol = len(ssps)
    fig, axes = plt.subplots(
        2, ncol, figsize=(4.6 * ncol, 7.6), sharex=True, sharey="row", squeeze=False
    )
    noun = COUNT_NOUN[measure]

    for j, ssp in enumerate(ssps):
        colour = SCENARIO_COLOR[ssp]
        for row in (0, 1):
            ax = axes[row][j]
            for eff, style, lw, lab in (
                (cur_eff, "-", 2.0, "current"),
                (prev_eff, (0, (5, 2)), 1.6, "previous"),
            ):
                if ssp not in eff:
                    continue
                s = eff[ssp] if row == 0 else eff[ssp].cumsum()
                ax.plot(
                    s.index, s / (a_div if row == 0 else c_div),
                    color=colour, linewidth=lw, linestyle=style,
                    label=lab if j == 0 else None, zorder=3,
                )
            ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)
            ax.grid(True, **GRID_KW)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if row == 0:
                ax.set_title(SCENARIO_LABEL[ssp], fontsize=12)
            else:
                ax.set_xlabel("Year")

    axes[0][0].set_ylabel(
        f"Annual Δ {noun.lower()}" + (f" (in {a_unit})" if a_unit else ""), labelpad=26
    )
    axes[1][0].set_ylabel(
        f"Cumulative Δ {noun.lower()}" + (f" (in {c_unit})" if c_unit else ""), labelpad=26
    )
    for row in (0, 1):
        annotate_direction(
            axes[row][0],
            above="averted",
            below="incurred",
        )
    axes[0][0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(
        f"{MEASURE_LABEL[measure]} — effect of {sensitivity_label}, this run vs previous"
        f"\n{location_name} · counterfactual minus baseline · cumulative from {start_year}",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5b — effect vs previous, with the four burden trajectories on top
# ---------------------------------------------------------------------------

def figure_effect_comparison_stack(  # noqa: PLR0913
    location_id: int,
    location_name: str,
    measure: str,
    cur_sens: pd.DataFrame,
    cur_base: pd.DataFrame,
    prev_sens: pd.DataFrame,
    prev_base: pd.DataFrame,
    ssps: Sequence[str],
    start_year: int,
    out_path: Path,
    sensitivity_label: str = "sensitivity",
) -> None:
    """Three rows: the four burden trajectories, then each run's effect, annual and cumulative.

    Row 1 shows all four series the comparison rests on — this run's baseline and
    sensitivity, and the previous run's — so the reader can see that the two runs sit at
    different levels before being asked to compare their differences. Rows 2 and 3 are
    the effects, solid for this run and dashed for the previous, on the shared
    counterfactual-minus-baseline convention.
    """
    def wide(df):
        w = _wide(df, location_id, measure)
        return w[w.index >= start_year]
    cs, cb, ps, pb = (wide(x) for x in (cur_sens, cur_base, prev_sens, prev_base))

    def effect(s, b):
        out = {}
        for ssp in ssps:
            if ssp in s.columns and ssp in b.columns:
                common = s.index.intersection(b.index)
                out[ssp] = s.loc[common, ssp] - b.loc[common, ssp]
        return out
    cur_eff, prev_eff = effect(cs, cb), effect(ps, pb)
    if not cur_eff:
        raise ValueError(f"no overlapping scenarios for location {location_id}")

    noun = COUNT_NOUN[measure]
    lvl_div, lvl_unit = pick_scale(np.concatenate(
        [f[c].to_numpy() for f in (cs, cb, ps, pb) for c in f.columns] or [np.array([0.0])]))
    a_div, a_unit = pick_scale(np.concatenate(
        [v.to_numpy() for v in (*cur_eff.values(), *prev_eff.values())]))
    c_div, c_unit = pick_scale(np.concatenate(
        [v.cumsum().to_numpy() for v in (*cur_eff.values(), *prev_eff.values())]))

    ncol = len(ssps)
    fig, axes = plt.subplots(3, ncol, figsize=(4.6 * ncol, 10.6),
                             sharex=True, sharey="row", squeeze=False)
    for j, ssp in enumerate(ssps):
        colour = SCENARIO_COLOR[ssp]
        ax = axes[0][j]
        for frame, style, lw, lab in (
            (cb, (0, (5, 2)), 1.4, "baseline (current)"),
            (cs, "-", 2.0, f"{sensitivity_label} (current)"),
            (pb, (0, (1, 1.6)), 1.4, "baseline (2025)"),
            (ps, (0, (3, 1, 1, 1)), 1.6, f"{sensitivity_label} (2025)"),
        ):
            if ssp in frame.columns:
                ax.plot(frame.index, frame[ssp] / lvl_div, color=colour, linewidth=lw,
                        linestyle=style, label=lab if j == 0 else None, zorder=3)
        ax.set_title(SCENARIO_LABEL[ssp], fontsize=12)

        for row, (div, cumulate) in ((1, (a_div, False)), (2, (c_div, True))):
            ax = axes[row][j]
            for eff, style, lw in ((cur_eff, "-", 2.0), (prev_eff, (0, (5, 2)), 1.6)):
                if ssp not in eff:
                    continue
                v = eff[ssp].cumsum() if cumulate else eff[ssp]
                ax.plot(v.index, v / div, color=colour, linewidth=lw, linestyle=style,
                        zorder=3)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)

        for row in range(3):
            a = axes[row][j]
            a.grid(True, **GRID_KW); a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        axes[2][j].set_xlabel("Year")

    axes[0][0].set_ylabel(f"{noun}" + (f" (in {lvl_unit})" if lvl_unit else ""))
    axes[1][0].set_ylabel(f"Annual Δ {noun.lower()}"
                          + (f" (in {a_unit})" if a_unit else ""), labelpad=26)
    axes[2][0].set_ylabel(f"Cumulative Δ {noun.lower()}"
                          + (f" (in {c_unit})" if c_unit else ""), labelpad=26)
    axes[0][0].legend(frameon=False, fontsize=8, loc="best")
    for row in (1, 2):
        annotate_direction(axes[row][0])
    fig.suptitle(
        f"{MEASURE_LABEL[measure]} — effect of {sensitivity_label}, this run vs 2025"
        f"\n{location_name} · levels (top), then counterfactual minus baseline: "
        f"↑ = averted, ↓ = incurred · cumulative from {start_year}",
        fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — timeseries over bars, one composite per (measure, location)
# ---------------------------------------------------------------------------

def figure_timeseries_over_bars(  # noqa: PLR0913
    location_id: int,
    location_name: str,
    measure: str,
    observed: pd.DataFrame,
    current: pd.DataFrame,
    previous: pd.DataFrame,
    ssps: Sequence[str],
    horizon_years: Sequence[int],
    out_path: Path,
    anchor_year: int = 2023,
    main_label: str = "current",
    comp_label: str = "previous",
    overlay: dict[str, pd.DataFrame] | None = None,
    overlay_label: str = "2025 run",
) -> None:
    """The trajectory and the horizon totals in one figure.

    Top row is the two timeseries panels (count, rate); bottom row is the cumulative bar
    chart spanning the full width. Composing them means the reader sees the shape and the
    headline numbers without holding two files side by side — and the bars are cumulative
    totals of the very curves plotted above them, which is only obvious when they share a
    figure.

    Rendered by writing each component to a temporary figure and compositing would lose
    vector quality, so instead this re-draws both panels directly. Any change to
    figure_timeseries or figure_bars must be mirrored here — the alternative, refactoring
    both to accept an Axes, is the better fix once the jobmon version lands.
    """
    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.32, wspace=0.22)

    obs = observed.query("location_id == @location_id and measure == @measure")
    cur = current.query("location_id == @location_id and measure == @measure")
    prv = previous.query("location_id == @location_id and measure == @measure")

    for col, metric in enumerate(("count", "rate")):
        ax = fig.add_subplot(gs[0, col])
        colname = f"{metric}_mean"
        pool = np.concatenate(
            [s[colname].to_numpy() for s in (obs, cur, prv) if not s.empty]
            or [np.array([0.0])])
        divisor, unit = pick_scale(pool) if metric == "count" else (1.0, "")
        if not obs.empty:
            o = obs.sort_values("year_id")
            ax.plot(o.year_id, o[colname] / divisor, color=OBSERVED_COLOR,
                    linewidth=2.0, label="Observed", zorder=5)
        for ssp in ssps:
            c = cur[cur.ssp == ssp].sort_values("year_id")
            if not c.empty:
                ax.plot(c.year_id, c[colname] / divisor, color=SCENARIO_COLOR[ssp],
                        linewidth=2.0, label=f"{SCENARIO_LABEL[ssp]} ({main_label})",
                        zorder=4)
            q = prv[prv.ssp == ssp].sort_values("year_id")
            if not q.empty:
                ax.plot(q.year_id, q[colname] / divisor, color=SCENARIO_COLOR[ssp],
                        linewidth=1.6, linestyle=(0, (5, 2)),
                        label=f"{SCENARIO_LABEL[ssp]} ({comp_label})", zorder=3)
        noun = COUNT_NOUN[measure]
        if metric == "count":
            ax.set_ylabel(f"{noun}" + (f" (in {unit})" if unit else ""))
            ax.set_title("Count", fontsize=11)
        else:
            ax.set_ylabel(f"{MEASURE_LABEL[measure]} rate (per person per year)")
            ax.set_title("Rate", fontsize=11)
        ax.set_xlabel("Year")
        ax.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if col == 0:
            handles, labels = ax.get_legend_handles_labels()

    axb = fig.add_subplot(gs[1, :])
    _draw_bars(axb, location_id, location_name, measure, current, previous, ssps,
               horizon_years, anchor_year, main_label, comp_label, overlay, overlay_label)

    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{MEASURE_LABEL[measure]} — {location_name}", fontsize=14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--product-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="saved all-age product dir for the CURRENT run",
)
@click.option(
    "--previous-run-dir",
    default=str(mbpc.FIRST_SUBMISSION_RUN_PATH),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="previous run's upload_folders arm directory (2025_08_28 = first submission). "
         "Ignored when --reference-product-dir is given.",
)
@click.option(
    "--reference-product-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="use a SAVED PRODUCT dir as the dashed comparison series instead of the "
         "previous upload run. This is the sensitivity mode: point --product-dir at "
         "the sensitivity and this at the baseline, so the sensitivity is solid.",
)
@click.option(
    "--comparison-label", default=None,
    help="legend/title word for the dashed series; defaults to 'previous' for the "
         "upload run and 'baseline' for a reference product dir",
)
@click.option(
    "--sensitivity-label", default="sensitivity",
    help="name of the held quantity, used in the difference figure's title "
         "(e.g. 'GDP held at 2023')",
)
@click.option(
    "--covariate-var", default=None,
    help="input netCDF variable to draw as the DRIVER row of the 4-row sensitivity "
         "figure, e.g. mal_DAH_total_per_capita or gdppc_mean. Omit to skip that figure.",
)
@click.option(
    "--covariate-dah-alt", default=None,
    help="counterfactual arm = this dah_scenario slice (the DAH sensitivity).",
)
@click.option(
    "--covariate-hold-year", default=None, type=int,
    help="counterfactual arm = the driver frozen at this year (the GDP-style holds).",
)
@click.option("--covariate-label", default=None, help="y-label for the driver row")
@click.option(
    "--driver-weight", "driver_weights", multiple=True,
    type=click.Choice(list(WEIGHT_SCHEMES)), default=("population", "mort2023"),
    show_default=True,
    help="weight scheme(s) for aggregating the DRIVER row. One figure per scheme, tagged "
         "in the filename, so burden-weighted sits beside population-weighted. A "
         "population-weighted covariate dilutes changes concentrated where the disease is.",
)
@click.option(
    "--covariate-total/--no-covariate-total", default=False, show_default=True,
    help="ALSO emit a driver row showing the TOTAL (per-capita x population, summed) "
         "alongside the per-capita version. For the DAH `Constant` scenario the total "
         "is flat by construction, which is the cleanest demonstration of what it does.",
)
@click.option(
    "--previous-hold",
    type=click.Choice(["gdppc", "population", "as_structure", "DAH",
                       "suitability", "flood"]),
    default=None,
    help="compare against the PREVIOUS run's matching sensitivity arm instead of its "
         "baseline. Use with --product-dir pointing at our equivalent hold, to see "
         "previous-vs-current for the same experiment.",
)
@click.option("--ssp-scenario", "ssps", multiple=True,
              default=("ssp126", "ssp245", "ssp585"), show_default=True)
@click.option("--dah-scenario", default="Baseline", show_default=True,
              help="DAH scenario of the MAIN (solid) series")
@click.option("--reference-dah-scenario", default=None,
              help="DAH scenario of the dashed reference series; defaults to "
                   "--dah-scenario. Set this to compare Constant against Baseline "
                   "out of a single product dir.")
@click.option("--anchor-year", default=2023, show_default=True, type=int)
@click.option("--observed-start", default=2000, show_default=True, type=int)
@click.option("--horizon-year", "horizon_years", multiple=True, default=(2050, 2100),
              show_default=True, type=int)
@click.option("--output-dir", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="defaults to <product-dir>/figures")
def main(  # noqa: PLR0913
    product_dir: Path,
    previous_run_dir: Path,
    reference_product_dir: Path | None,
    comparison_label: str | None,
    sensitivity_label: str,
    covariate_var: str | None,
    covariate_dah_alt: str | None,
    covariate_hold_year: int | None,
    covariate_label: str | None,
    covariate_total: bool,
    driver_weights: tuple[str, ...],
    previous_hold: str | None,
    ssps: tuple[str, ...],
    dah_scenario: str,
    reference_dah_scenario: str | None,
    anchor_year: int,
    observed_start: int,
    horizon_years: tuple[int, ...],
    output_dir: Path | None,
) -> None:
    """Build the global and super-region comparison figures."""
    out_dir = output_dir or (product_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dah = reference_dah_scenario or dah_scenario
    # Sensitivity mode = the dashed series is another SAVED PRODUCT (the baseline)
    # rather than the 2025_08_28 upload run.
    sensitivity_mode = reference_product_dir is not None or ref_dah != dah_scenario
    # Order matters: whenever a reference PRODUCT is named, the dashed series IS that
    # product (this run's baseline), regardless of whether a previous hold arm was also
    # requested for the effect-comparison figure.
    if comparison_label:
        comp_label = comparison_label
    elif sensitivity_mode:
        comp_label = "baseline"
    elif previous_hold:
        comp_label = f"previous, {previous_hold} held"
    else:
        comp_label = "previous"
    # The solid series is the sensitivity whenever we are comparing against a baseline
    # product; against the previous run it is simply the current run.
    main_label = sensitivity_label if sensitivity_mode else "current"

    hierarchy = load_hierarchy()
    names = dict(zip(hierarchy.location_id, hierarchy.location_name, strict=True))
    targets = hierarchy.loc[hierarchy.level.isin([0, 1]), "location_id"].tolist()

    click.echo(f"main (solid) : {product_dir}  dah={dah_scenario}")
    click.echo(
        f"ref  (dashed): "
        f"{reference_product_dir or (product_dir if sensitivity_mode else previous_run_dir)}"
        f"  dah={ref_dah}  label={comp_label}"
    )
    click.echo(f"figures      : {out_dir}")

    current = load_current(product_dir, ssps, dah_scenario, levels=[0, 1])
    have_current = sorted(set(current.location_id) & set(targets))
    missing = sorted(set(targets) - set(have_current))
    if missing:
        click.echo(
            f"  note: {len(missing)} level-0/1 locations absent from the main "
            f"product (no forecast): {[names.get(m, m) for m in missing]}"
        )

    population = current[["location_id", "year_id", "population"]].drop_duplicates()

    if sensitivity_mode:
        reference = load_current(
            reference_product_dir or product_dir, ssps, ref_dah, levels=[0, 1]
        )
    else:
        reference = load_previous(
            previous_run_dir, ssps, dah_scenario, have_current, population,
            hold=previous_hold,
        )
    previous = reference
    observed = load_observed(have_current, observed_start, anchor_year)

    # Effect-comparison mode needs all FOUR series: this run's sensitivity and
    # baseline, plus the previous run's sensitivity and baseline. Available only when a
    # current baseline product AND a previous hold arm are both named.
    effect_mode = reference_product_dir is not None and previous_hold is not None
    if effect_mode:
        click.echo(
            f"  effect mode: comparing the {previous_hold} effect in this run against "
            f"the previous run's own {previous_hold} effect"
        )
        prev_sens = load_previous(
            previous_run_dir, ssps, dah_scenario, have_current, population,
            hold=previous_hold,
        )
        prev_base = load_previous(
            previous_run_dir, ssps, dah_scenario, have_current, population, hold=None,
        )

    cov_arms: dict[str, pd.DataFrame] = {}
    cov_by_weight: dict = {}
    if covariate_var and (covariate_dah_alt or covariate_hold_year):
        a2 = hierarchy.loc[hierarchy.level == 5, "location_id"].tolist()
        yrs = sorted(set(current.year_id))
        cov_by_weight = {}
        for wk in driver_weights:
            w = load_weights(wk, yrs, anchor_year=anchor_year)
            cov_by_weight[wk] = (
                load_covariate_arms(
                    ssps, a2, w, hierarchy, covariate_var,
                    dah_base=ref_dah, dah_alt=covariate_dah_alt,
                    hold_year=covariate_hold_year,
                ),
                load_covariate_arms(
                    ssps, a2, w, hierarchy, covariate_var,
                    dah_base=ref_dah, dah_alt=covariate_dah_alt,
                    hold_year=covariate_hold_year, total=True,
                ) if covariate_total else {},
            )
        cov_arms = cov_by_weight[driver_weights[0]][0]
        click.echo(f"  driver row: {covariate_var} "
                   f"({'dah=' + covariate_dah_alt if covariate_dah_alt else 'held at ' + str(covariate_hold_year)})"
                   f" -> {len(cov_arms)} scenario(s)")

    n = 0
    for measure in ("inc", "mort"):
        for loc in have_current:
            name = names.get(loc, str(loc))
            figure_timeseries(
                loc, name, measure, observed, current, previous, ssps,
                out_dir / f"timeseries_{measure}_{loc}.png",
                main_label=main_label, comp_label=comp_label,
            )
            n += 1
        # Bars for global AND every super-region; the difference panels stay global.
        for loc in have_current:
            figure_bars(
                loc, names.get(loc, str(loc)), measure, current, previous, ssps,
                horizon_years, out_dir / f"bars_{measure}_{loc}.png",
                anchor_year=anchor_year,
                main_label=main_label, comp_label=comp_label,
                # In effect mode the previous vintage of BOTH series is available, so
                # each bar carries its own old-run interval: the comparison bar gets the
                # old baseline, the main bar gets the old sensitivity.
                overlay=(
                    {"previous": prev_base, "current": prev_sens}
                    if effect_mode else None
                ),
            )
            figure_bars(
                loc, names.get(loc, str(loc)), measure, current, previous, ssps,
                horizon_years, out_dir / f"bars_values_{measure}_{loc}.png",
                anchor_year=anchor_year,
                main_label=main_label, comp_label=comp_label,
                overlay=(
                    {"previous": prev_base, "current": prev_sens}
                    if effect_mode else None
                ),
                annotate_values=True,
            )
            figure_timeseries_over_bars(
                loc, names.get(loc, str(loc)), measure, observed, current, previous,
                ssps, horizon_years,
                out_dir / f"ts_bars_{measure}_{loc}.png",
                anchor_year=anchor_year,
                main_label=main_label, comp_label=comp_label,
                overlay=(
                    {"previous": prev_base, "current": prev_sens}
                    if effect_mode else None
                ),
            )
            n += 3
        globe = have_current[0] if 1 not in have_current else 1
        figure_differences_stack(
            globe, names.get(globe, str(globe)), measure, current, previous,
            anchor_year, out_dir / f"differences_stack_{measure}_{globe}.png",
            comp_label=comp_label,
        )
        n += 1
        figure_differences(
            globe, names.get(globe, str(globe)), measure, current, previous,
            anchor_year, out_dir / f"differences_{measure}_{globe}.png",
            comp_label=comp_label,
        )
        n += 1
        # Sensitivity-vs-baseline difference, 2 rows x one column per scenario.
        # Only meaningful when the dashed series IS a baseline of the same vintage.
        if sensitivity_mode:
            for loc in have_current:
                figure_sensitivity_difference(
                    loc, names.get(loc, str(loc)), measure, current, reference, ssps,
                    anchor_year,
                    out_dir / f"sens_diff_{measure}_{loc}.png",
                    sensitivity_label=sensitivity_label,
                )
                figure_sensitivity_stack(
                    loc, names.get(loc, str(loc)), measure, current, reference, ssps,
                    anchor_year,
                    out_dir / f"sens_stack_{measure}_{loc}.png",
                    sensitivity_label=sensitivity_label,
                )
                n += 2
                for wk, (arms, arms_total) in cov_by_weight.items():
                    base_lbl = covariate_label or covariate_var or "driver"
                    if arms:
                        figure_sensitivity_full(
                            loc, names.get(loc, str(loc)), measure, current, reference,
                            arms, ssps, anchor_year,
                            out_dir / f"sens_full_{measure}_{loc}__{wk}.png",
                            sensitivity_label=sensitivity_label,
                            cov_label=f"{base_lbl}\n({WEIGHT_SCHEMES[wk]})",
                        )
                        n += 1
                    if arms_total:
                        figure_sensitivity_full(
                            loc, names.get(loc, str(loc)), measure, current, reference,
                            arms_total, ssps, anchor_year,
                            out_dir / f"sens_full_total_{measure}_{loc}__{wk}.png",
                            sensitivity_label=sensitivity_label,
                            cov_label=f"{base_lbl} — TOTAL\n({WEIGHT_SCHEMES[wk]})",
                        )
                        n += 1
        if effect_mode:
            for loc in have_current:
                figure_effect_comparison(
                    loc, names.get(loc, str(loc)), measure,
                    current, reference, prev_sens, prev_base, ssps, anchor_year,
                    out_dir / f"effect_vs_previous_{measure}_{loc}.png",
                    sensitivity_label=sensitivity_label,
                )
                figure_effect_comparison_stack(
                    loc, names.get(loc, str(loc)), measure,
                    current, reference, prev_sens, prev_base, ssps, anchor_year,
                    out_dir / f"effect_stack_{measure}_{loc}.png",
                    sensitivity_label=sensitivity_label,
                )
                n += 2

    click.echo(f"wrote {n} figures to {out_dir}")


if __name__ == "__main__":
    main()
