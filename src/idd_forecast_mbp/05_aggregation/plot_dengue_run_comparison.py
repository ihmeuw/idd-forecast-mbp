"""Global and super-region dengue figures: each decay arm against the previous run.

One figure type only, deliberately. The dengue forecast carries a ``decay`` axis with
no malaria analogue — how fast the fitted year effect is switched off — and at 2100 the
decay choice moves global incidence by ~15x while the SSP scenario moves it by ~7%.
Until decay is settled, the within-run scenario-difference figures in
``plot_run_comparison.py`` would be measuring the smaller axis, so they are not ported.

Layout, per location: one row per decay, one column per (measure, metric). Every row is
a self-contained comparison of that decay against the SAME dashed previous-run
reference, which is what makes the rows readable without also comparing decays to each
other.

Three series per panel:

- **observed**, black, through the anchor year.
- **current**, solid, one colour per SSP scenario.
- **previous**, dashed, same colours — the 2025 first-submission upload run.

Count and rate never share an axis; they are separate columns, because a dual y-scale
makes two incomparable units look comparable.

Each location is written twice, once per y-axis scale, as ``*_natural.png`` and
``*_log.png``. They answer different questions and neither substitutes for the other:
natural space shows the absolute size of the gap between the runs, log space shows
whether the growth *rates* differ — an 8x gap that is constant in log space is a level
disagreement, one that widens is a trend disagreement.

Two things about the numbers that a reader has to know or the figures mislead:

1. The current run's anchor is a **2014-2023 mean baseline**, not a point anchor to
   2023, so its 2023 is not observed 2023 and is not meant to be. 2023 was a record
   dengue year; the decade mean deliberately does not chase it. Observed is drawn in
   every panel so that gap is visible rather than inferred.
2. The previous run stored draw-level COUNTS only, so its rates are derived here from
   the same population artifact the current products used — verified to be each level's
   own full population, not a sum over modelled children. The previous run never
   asserted a rate.

The current run covers only the 305 FHS-most-detailed locations with a positive
observed reference-cell count at the anchor year. Those 305 carry 100% of observed
dengue burden, so aggregate totals lose nothing, but super-region 31 has exactly zero
dengue and is absent from the products entirely; it is skipped with a note rather than
drawn empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.first_submission import (
    FIRST_SUBMISSION_RUN,
    load_comparison,
)
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids

if TYPE_CHECKING:
    from collections.abc import Sequence

# Colours and labels come from the project scenario map — never invented here.
SCENARIO_COLOR = {k: v["color"] for k, v in mbpc.ssp_scenarios.items()}
SCENARIO_LABEL = {k: v["name"] for k, v in mbpc.ssp_scenarios.items()}
OBSERVED_COLOR = "#1A1A1A"
GRID_KW = {"color": "#CCCCCC", "linewidth": 0.6, "linestyle": ":", "alpha": 0.9}

MEASURE_LABEL = {"inc": "Incidence", "mort": "Mortality"}
COUNT_NOUN = {"inc": "Cases", "mort": "Deaths"}

#: Column order across the figure. (measure, metric) -> nothing else varies.
PANEL_COLUMNS = (("inc", "count"), ("inc", "rate"), ("mort", "count"), ("mort", "rate"))

#: Decay arms in the order they should be stacked: slowest switch-off last, so the
#: undecayed arm — which is 15x the others and would otherwise dominate a shared axis —
#: sits at the bottom where its scale is obviously its own.
DECAY_ORDER = ("linear_2100", "logistic_k8", "logistic_k20", "no_decay")

DECAY_LABEL = {
    "linear_2100": "linear decay to 0 by 2100",
    "logistic_k8": "logistic decay, mid 2060, k=8",
    "logistic_k20": "logistic decay, mid 2060, k=20",
    "no_decay": "no decay (year effect held)",
}

SCALES = [(1e9, "billions"), (1e6, "millions"), (1e3, "thousands"), (1.0, "")]


def pick_scale(values: np.ndarray) -> tuple[float, str]:
    """Choose a human-readable unit scale from the largest magnitude present."""
    finite = values[np.isfinite(values)] if values.size else values
    peak = float(np.nanmax(np.abs(finite))) if finite.size else 0.0
    for divisor, name in SCALES:
        if peak >= divisor:
            return divisor, name
    return 1.0, ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def current_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide forecast summary to one row per (location, year, ssp, decay, measure).

    Separated from the read so the reshape is testable without the 43 MB product. The
    input is the wide ``dengue_{measure}_{metric}_{stat}`` schema written by the
    forecast driver; the output is the long schema every figure function consumes.
    """
    frames = []
    for measure in ("inc", "mort"):
        need = [f"dengue_{measure}_{m}_{s}"
                for m in ("count", "rate") for s in ("mean", "lower", "upper")]
        missing = [c for c in need if c not in df.columns]
        if missing:
            msg = f"forecast summary missing columns: {missing}"
            raise KeyError(msg)
        piece = pd.DataFrame({
            "location_id": df["location_id"].to_numpy(),
            "year_id": df["year_id"].to_numpy(),
            "ssp": df["ssp_scenario"].to_numpy(),
            "decay": df["decay"].to_numpy(),
            "measure": measure,
            "run": "current",
        })
        for metric in ("count", "rate"):
            for stat in ("mean", "lower", "upper"):
                piece[f"{metric}_{stat}"] = df[f"dengue_{measure}_{metric}_{stat}"].to_numpy()
        frames.append(piece)
    return pd.concat(frames, ignore_index=True)


def load_current(summary_path: Path, levels: Sequence[int]) -> pd.DataFrame:
    """Long frame from the saved forecast summary, restricted to ``levels``.

    The summary carries no ``level`` column — it is keyed by location only — so the
    level filter comes from the hierarchy rather than the product.
    """
    if not summary_path.exists():
        msg = f"current product missing: {summary_path}"
        raise FileNotFoundError(msg)
    wide = pd.read_parquet(summary_path)
    hierarchy = load_hierarchy()
    keep = set(hierarchy.loc[hierarchy.level.isin(list(levels)), "location_id"])
    return current_to_long(wide[wide["location_id"].isin(keep)])


def load_population(location_ids: Sequence[int], years: Sequence[int]) -> pd.DataFrame:
    """Population at the requested locations and years.

    The artifact already stores every hierarchy level, including global and
    super-region, so this is a filter rather than an aggregation — and it is each
    level's OWN full population, which is the correct rate denominator and the one the
    current products used.
    """
    pop = read_parquet_with_integer_ids(
        mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
        columns=["location_id", "year_id", "population"],
    )
    return pop[
        pop.location_id.isin([int(x) for x in location_ids])
        & pop.year_id.isin([int(y) for y in years])
    ].reset_index(drop=True)


def load_previous(
    ssps: Sequence[str],
    location_ids: Sequence[int],
    years: Sequence[int],
    population: pd.DataFrame,
    run_date: str = FIRST_SUBMISSION_RUN,
) -> pd.DataFrame:
    """Long frame from the previous run's draw-level count netCDFs.

    Delegates the arm naming and the draw collapse to
    ``lib/data/first_submission.py`` — the dengue arms carry no ``dah_scenario`` token,
    so the malaria reader in ``plot_run_comparison.py`` cannot name them. Rates are
    derived here because that run stored counts only.
    """
    out = load_comparison(
        ssps, locations=location_ids, years=years, measures=("inc", "mort"),
        cause="dengue", run_date=run_date,
    )
    out = out.rename(columns={"ssp_scenario": "ssp"})
    out = out.merge(population, on=["location_id", "year_id"], how="left")
    for stat in ("mean", "lower", "upper"):
        out[f"rate_{stat}"] = np.where(
            out.population > 0, out[f"count_{stat}"] / out.population, np.nan,
        )
    out["run"] = "previous"
    return out


#: Coefficient name carrying the mortality -> incidence pathway in a mort_then_inc
#: structure. Incidence responds to time twice when this is present: through its own
#: year slope and through predicted mortality's.
MEDIATOR_TERM = "log_dengue_mort_rate"


def format_year_slopes(coefs: pd.DataFrame, super_region_id: int | None) -> str:
    """One line naming this super-region's fitted year slopes, for the figure subtitle.

    The slopes are the first-order forecast driver, and every decay arm is a
    transformation of exactly these numbers, so a reader cannot interpret the rows
    without them. Returns an empty string when there is nothing to report — global has
    no single slope, and an absent coefficient file is not an error.

    Slopes are per year in MODEL space (log rate), which is why a value near 0.01 is
    ~1% per year and compounds to a factor of e^(0.01*77) over the horizon.
    """
    if coefs is None or coefs.empty or super_region_id is None:
        return ""
    rows = coefs[coefs.super_region_id == super_region_id]
    if rows.empty:
        return ""
    parts = []
    for outcome in ("mortality", "incidence"):
        hit = rows[rows.outcome == outcome]
        if not hit.empty:
            r = hit.iloc[0]
            parts.append(f"{outcome} {float(r.coef):+.5f}/yr (p={float(r.p_coef):.1g})")
    if not parts:
        return ""
    line = "fitted year slope, log space: " + ", ".join(parts)
    mediator = coefs[coefs.term == MEDIATOR_TERM]
    if not mediator.empty:
        line += (f"; incidence also responds via {float(mediator.iloc[0].coef):+.3f} "
                 f"x log mortality rate")
    return line


def load_coefficients(path: Path | None) -> pd.DataFrame:
    """Linear coefficients saved by the fit, or an empty frame when not supplied."""
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["outcome", "term", "coef", "p_coef",
                                     "super_region_id"])
    return pd.read_parquet(path)


def load_observed(
    location_ids: Sequence[int], first_year: int, last_year: int,
) -> pd.DataFrame:
    """Observed raked all-age dengue counts and rates for the historical panel."""
    obs = read_parquet_with_integer_ids(
        mbpc.DEN_RAKED_AA_READ_PATH / "aa_full_dengue_df.parquet",
        columns=["location_id", "year_id", "dengue_inc_count", "dengue_inc_rate",
                 "dengue_mort_count", "dengue_mort_rate"],
    )
    obs = obs[
        obs.location_id.isin([int(x) for x in location_ids])
        & obs.year_id.between(first_year, last_year)
    ]
    frames = [
        pd.DataFrame({
            "location_id": obs.location_id.to_numpy(),
            "year_id": obs.year_id.to_numpy(),
            "measure": measure,
            "count_mean": obs[f"dengue_{measure}_count"].to_numpy(),
            "rate_mean": obs[f"dengue_{measure}_rate"].to_numpy(),
        })
        for measure in ("inc", "mort")
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["location_id", "year_id"])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _panel(  # noqa: PLR0913
    ax,
    location_id: int,
    measure: str,
    metric: str,
    decay: str,
    current: pd.DataFrame,
    previous: pd.DataFrame,
    observed: pd.DataFrame,
    ssps: Sequence[str],
    *,
    ribbons: bool,
    log_y: bool,
) -> None:
    """One (decay, measure, metric) cell: observed, then current solid vs previous dashed."""
    cur = current[
        (current.location_id == location_id)
        & (current.measure == measure)
        & (current.decay == decay)
    ]
    prev = previous[(previous.location_id == location_id) & (previous.measure == measure)]
    obs = observed[(observed.location_id == location_id) & (observed.measure == measure)]

    pool = np.concatenate([
        cur[f"{metric}_mean"].to_numpy(dtype=float),
        prev[f"{metric}_mean"].to_numpy(dtype=float),
        obs[f"{metric}_mean"].to_numpy(dtype=float),
    ]) if len(cur) or len(prev) or len(obs) else np.array([])
    # In log space the unit rescale is a constant offset that buys nothing and makes the
    # tick labels ("millions", ticks at 10^1) contradict each other, so keep raw units
    # and let the powers of ten carry the magnitude.
    divisor, unit = (1.0, "") if log_y else pick_scale(pool)

    if len(obs):
        o = obs.sort_values("year_id")
        ax.plot(o.year_id, o[f"{metric}_mean"] / divisor, color=OBSERVED_COLOR,
                linewidth=1.6, label="observed", zorder=5)

    for ssp in ssps:
        color = SCENARIO_COLOR.get(ssp, "#888888")
        c = cur[cur.ssp == ssp].sort_values("year_id")
        if len(c):
            ax.plot(c.year_id, c[f"{metric}_mean"] / divisor, color=color,
                    linewidth=1.5, label=f"current, {SCENARIO_LABEL.get(ssp, ssp)}")
            if ribbons:
                ax.fill_between(c.year_id, c[f"{metric}_lower"] / divisor,
                                c[f"{metric}_upper"] / divisor, color=color, alpha=0.12,
                                linewidth=0)
        p = prev[prev.ssp == ssp].sort_values("year_id")
        if len(p):
            ax.plot(p.year_id, p[f"{metric}_mean"] / divisor, color=color,
                    linewidth=1.3, linestyle="--",
                    label=f"previous, {SCENARIO_LABEL.get(ssp, ssp)}")

    if log_y:
        ax.set_yscale("log")
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
    noun = COUNT_NOUN[measure] if metric == "count" else f"{MEASURE_LABEL[measure]} rate"
    ylab = f"{noun} ({unit})" if unit else noun
    ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=8)


def figure_timeseries(  # noqa: PLR0913
    location_id: int,
    location_name: str,
    current: pd.DataFrame,
    previous: pd.DataFrame,
    observed: pd.DataFrame,
    decays: Sequence[str],
    ssps: Sequence[str],
    out_path: Path,
    *,
    ribbons: bool = False,
    log_y: bool = False,
    slope_line: str = "",
) -> None:
    """One figure per location: rows = decay, columns = (measure, metric).

    Each row is autoscaled independently. That is deliberate: the undecayed arm reaches
    ~15x the decayed ones by 2100, so a shared y-axis would compress the three decayed
    rows into a flat line. Rows are comparisons against the dashed previous run, not
    against each other.
    """
    n_rows, n_cols = len(decays), len(PANEL_COLUMNS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.1 * n_cols, 2.7 * n_rows), squeeze=False,
    )
    for r, decay in enumerate(decays):
        for c, (measure, metric) in enumerate(PANEL_COLUMNS):
            ax = axes[r][c]
            _panel(ax, location_id, measure, metric, decay, current, previous, observed,
                   ssps, ribbons=ribbons, log_y=log_y)
            if r == 0:
                ax.set_title(
                    f"{MEASURE_LABEL[measure]} {metric}", fontsize=10, pad=8,
                )
            if r == n_rows - 1:
                ax.set_xlabel("Year", fontsize=8)
        # Row label on the left, outside the leftmost axis, so it names the decay arm
        # without stealing panel width.
        axes[r][0].annotate(
            DECAY_LABEL.get(decay, decay),
            xy=(-0.34, 0.5), xycoords="axes fraction", rotation=90,
            ha="center", va="center", fontsize=9, fontweight="bold",
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    subtitle = ("current anchored to the 2014-2023 mean, so its 2023 is not "
                "observed 2023")
    if slope_line:
        subtitle += f"\n{slope_line}"
    fig.suptitle(
        f"Dengue — {location_name}: current run by decay arm vs "
        f"{FIRST_SUBMISSION_RUN} run ({'log' if log_y else 'natural'} scale)\n"
        f"{subtitle}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.02, 0.05, 1.0, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--summary-path", type=click.Path(path_type=Path), required=True,
              help="Forecast summary parquet written by the dengue forecast driver.")
@click.option("--previous-run", default=FIRST_SUBMISSION_RUN, show_default=True,
              help="Upload-folder run date to use as the dashed reference.")
@click.option("--ssp", "ssps", multiple=True,
              default=("ssp126", "ssp245", "ssp585"), show_default=True)
@click.option("--decay", "decays", multiple=True, default=DECAY_ORDER, show_default=True,
              help="Decay arms to stack as rows, in the order given.")
@click.option("--observed-start", default=2000, show_default=True, type=int)
@click.option("--anchor-year", default=2023, show_default=True, type=int)
@click.option("--ribbons/--no-ribbons", default=False, show_default=True,
              help="Shade the current run's 95% interval.")
@click.option("--scale", "scales", multiple=True,
              type=click.Choice(["natural", "log"]),
              default=("natural", "log"), show_default=True,
              help="Y-axis scale(s). Both are written by default, one file each.")
@click.option("--coefficients-path", type=click.Path(path_type=Path), default=None,
              help="Linear coefficients parquet from the fit; adds the fitted "
                   "year-by-super-region slope to each super-region figure.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Defaults to <summary parent>/figures/run_comparison.")
def main(  # noqa: PLR0913
    summary_path: Path,
    previous_run: str,
    ssps: tuple[str, ...],
    decays: tuple[str, ...],
    observed_start: int,
    anchor_year: int,
    ribbons: bool,
    scales: tuple[str, ...],
    coefficients_path: Path | None,
    output_dir: Path | None,
) -> None:
    """Build the global and super-region dengue decay-vs-previous-run figures."""
    out_dir = output_dir or (summary_path.parent / "figures" / "run_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = load_hierarchy()
    names = dict(zip(hierarchy.location_id, hierarchy.location_name, strict=True))
    levels = dict(zip(hierarchy.location_id, hierarchy.level, strict=True))
    targets = hierarchy.loc[hierarchy.level.isin([0, 1]), "location_id"].tolist()
    coefs = load_coefficients(coefficients_path)

    current = load_current(summary_path, levels=[0, 1])
    have = sorted(set(current.location_id) & set(targets))
    absent = sorted(set(targets) - set(have))
    if absent:
        # Expected for super-region 31, which has exactly zero observed dengue and so
        # contributes no fit locations. Reported rather than silently dropped.
        click.echo(f"  note: {len(absent)} level-0/1 locations absent from the current "
                   f"product: {[names.get(i, i) for i in absent]}")

    present_decays = [d for d in decays if d in set(current.decay)]
    if not present_decays:
        msg = f"none of {list(decays)} present in {summary_path}"
        raise click.ClickException(msg)
    missing_decays = [d for d in decays if d not in set(current.decay)]
    if missing_decays:
        click.echo(f"  note: decay arms not in product, skipped: {missing_decays}")

    years = sorted(current.year_id.unique().tolist())
    population = load_population(have, years)
    previous = load_previous(ssps, have, years, population, run_date=previous_run)
    observed = load_observed(have, observed_start, anchor_year)

    click.echo(f"current : {summary_path}")
    click.echo(f"previous: {previous_run}")
    click.echo(f"figures : {out_dir}")

    for location_id in have:
        name = names.get(location_id, str(location_id))
        # A level-1 location IS its own super-region, so it keys the masked year term
        # directly. Global spans all six slopes and gets no single number.
        sr_id = location_id if levels.get(location_id) == 1 else None
        slope_line = format_year_slopes(coefs, sr_id)
        for scale in scales:
            out_path = out_dir / f"timeseries_{location_id}_{scale}.png"
            figure_timeseries(
                location_id, name, current, previous, observed, present_decays, ssps,
                out_path, ribbons=ribbons, log_y=(scale == "log"),
                slope_line=slope_line,
            )
            click.echo(f"  wrote {out_path.name}  ({name})")


if __name__ == "__main__":
    main()
