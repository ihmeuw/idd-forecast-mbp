"""Compare the forecast COVARIATES between this run and the previous run.

Step 1 of asking why the two runs disagree: before attributing anything to the model,
find out which inputs actually moved. This reads the previous run's covariate netCDF and
the current stage-08a forecast inputs, aggregates both to super-region and global, and
plots them together.

Two choices worth knowing when reading the output:

* **The same population weights are used for both runs.** A covariate like GDP per capita
  or DAH per capita only aggregates through a population-weighted mean, and if each run
  brought its own weights the difference would mix "the covariate moved" with "the
  weights moved". Holding the weights fixed makes the plotted difference purely the
  covariate. The weights come from the current population artifact.
* **Only locations present in BOTH runs are used.** The previous run is on the
  ``lsae_1209`` hierarchy and the current one on ``lsae_1285``, with different admin-2
  counts, so the intersection is taken and the dropped count is reported. Comparing at
  super-region and global keeps this from mattering much, since the IDs are stable there.

The previous run's suitability carries no draw dimension while the current one does, so
the current one is averaged over draws first.
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

SCENARIO_COLOR = {k: v["color"] for k, v in mbpc.ssp_scenarios.items()}
SCENARIO_LABEL = {k: v["name"] for k, v in mbpc.ssp_scenarios.items()}
GRID_KW = {"color": "#CCCCCC", "linewidth": 0.6, "linestyle": ":", "alpha": 0.9}

# (label, previous-run variable, current-run variable). Names differ between runs; the
# quantity is the same in each pair.
COVARIATES: list[tuple[str, str, str]] = [
    ("GDP per capita", "gdppc_mean", "gdppc_mean"),
    ("DAH per capita", "dah_pc", "mal_DAH_total_per_capita"),
    ("Malaria suitability", "malaria_suitability", "malaria_suitability"),
    ("Flooding per capita", "flooding_pc", "people_flood_days_per_capita"),
    ("Urbanisation", "urbanization",
     "weighted_1km_urban_threshold_300.0_simple_mean"),
]


# Weight schemes, following the house pattern in
# notebooks/09_figures/TS_weighted_suitability.ipynb (cell 13), which weights covariates
# by population, mortality and incidence and shows burden-weighted beside
# population-weighted rather than picking one.
WEIGHTS = {
    "population": "year-varying population",
    "mort2023":   "2023 malaria deaths (fixed)",
    "inc2023":    "2023 malaria cases (fixed)",
}


def load_weights(kind: str, years: Sequence[int], anchor_year: int = 2023) -> pd.DataFrame:
    """Admin-2 aggregation weights as ``[location_id, year_id, weight]``.

    ``population`` uses each year's own population — the right weight for "what does the
    average person experience". The burden weights are FIXED at ``anchor_year`` and
    broadcast across years, which answers a different and, for this pipeline, more
    relevant question: what does the covariate look like where the disease actually is?
    A population-weighted global mean puts most of its weight on places with no malaria,
    so it systematically dilutes any change concentrated in endemic areas.

    Burden weighting also makes non-endemic locations weight ZERO rather than small,
    which is what makes it a sharper instrument than population for this model.
    """
    yrs = [int(y) for y in years]
    if kind == "population":
        pop = read_parquet_with_integer_ids(
            mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
            columns=["location_id", "year_id", "population"],
        )
        out = pop[pop.year_id.isin(yrs)].rename(columns={"population": "weight"})
        return out.reset_index(drop=True)

    col = {"mort2023": "malaria_mort_count", "inc2023": "malaria_inc_count"}[kind]
    obs = read_parquet_with_integer_ids(
        mbpc.MAL_RAKED_AA_READ_PATH / "aa_full_malaria_df.parquet",
        columns=["location_id", "year_id", col],
    )
    ref = obs.loc[obs.year_id == int(anchor_year), ["location_id", col]].rename(
        columns={col: "weight"}
    )
    ref = ref[ref.weight > 0]
    return (
        pd.MultiIndex.from_product(
            [ref.location_id, yrs], names=["location_id", "year_id"]
        )
        .to_frame(index=False)
        .merge(ref, on="location_id")
    )


def population_weights(years: Sequence[int]) -> pd.DataFrame:
    """Back-compat shim: population weights named ``population``."""
    return load_weights("population", years).rename(columns={"weight": "population"})


def weighted_rollup(
    values: pd.DataFrame, pop: pd.DataFrame, hierarchy: pd.DataFrame
) -> pd.DataFrame:
    """Population-weighted mean of ``value`` at global (0) and super-region (1).

    A per-capita or intensive covariate has no meaningful sum, so it aggregates as a
    weighted mean. Weighting by population rather than by area is deliberate: what
    matters for burden is the covariate the exposed people actually experience.
    """
    wcol = "weight" if "weight" in pop.columns else "population"
    df = values.merge(pop, on=["location_id", "year_id"], how="inner")
    df = df.merge(hierarchy[["location_id", "super_region_id"]], on="location_id",
                  how="inner")
    df["wv"] = df.value * df[wcol]

    sr = df.groupby(["super_region_id", "year_id"], observed=True)[["wv", wcol]].sum()
    g = df.groupby("year_id", observed=True)[["wv", wcol]].sum()
    return pd.concat(
        [
            (sr.wv / sr[wcol]).reset_index(name="value")
            .rename(columns={"super_region_id": "location_id"}),
            (g.wv / g[wcol]).reset_index(name="value").assign(location_id=1),
        ],
        ignore_index=True,
    )


def load_previous_covariate(
    path: Path, var: str, ssp: str, locations: Sequence[int]
) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        if var not in ds.data_vars:
            return pd.DataFrame(columns=["location_id", "year_id", "value"])
        # Build the membership set ONCE. Inlining it into the comprehension's condition
        # rebuilds it per candidate location -- O(n^2) and the dominant cost of this script.
        have = {int(x) for x in ds.location_id.values}
        keep = [i for i in locations if i in have]
        da = ds[var].sel(location_id=keep, ssp_scenario=ssp)
        return da.to_dataframe(name="value").reset_index()[
            ["location_id", "year_id", "value"]
        ]


def load_current_covariate(
    path: Path, var: str, dah: str, locations: Sequence[int]
) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        if var not in ds.data_vars:
            return pd.DataFrame(columns=["location_id", "year_id", "value"])
        have = {int(x) for x in ds.location_id.values}   # once, not per location
        keep = [i for i in locations if i in have]
        da = ds[var].sel(location_id=keep)
        if "dah_scenario" in da.dims:
            da = da.sel(dah_scenario=dah)
        if "draw" in da.dims:      # previous run has no draws; average to match
            da = da.mean("draw")
        return da.to_dataframe(name="value").reset_index()[
            ["location_id", "year_id", "value"]
        ]


def figure_weight_pair(
    label: str,
    by_weight: dict[str, tuple[dict, dict]],
    weight_order: list[str],
    out_path: Path,
    location_id: int = 1,
) -> None:
    """Global series under each weighting, side by side — one panel per weight scheme.

    The point of putting them adjacent is that the two answer different questions and can
    disagree in level by a factor of two. Population weighting spreads over every admin-2
    unit including the ~75% with no malaria; burden weighting restricts to where the
    disease is. Neither is a correction of the other, so neither is shown alone.
    """
    def _invariant(d: dict) -> bool:
        ser = [x[x.location_id == location_id].sort_values("year_id").value.to_numpy()
               for x in d.values()]
        ser = [x for x in ser if x.size]
        if len(ser) < 2:
            return True
        return all(x.shape == ser[0].shape and np.allclose(x, ser[0], rtol=1e-9)
                   for x in ser[1:])

    n = len(weight_order)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.6), squeeze=False)
    for j, wk in enumerate(weight_order):
        ax = axes[0][j]
        prev, curr = by_weight[wk]
        flat = _invariant(curr) and _invariant(prev)
        if flat:
            k = next(iter(curr))
            c = curr[k][curr[k].location_id == location_id].sort_values("year_id")
            q = prev[k][prev[k].location_id == location_id].sort_values("year_id")
            if not c.empty:
                ax.plot(c.year_id, c.value, color="black", linewidth=2.0, label="current")
            if not q.empty:
                ax.plot(q.year_id, q.value, color="black", linewidth=1.6,
                        linestyle=(0, (5, 2)), label="2025 run")
        else:
            for ssp in curr:
                c = curr[ssp][curr[ssp].location_id == location_id].sort_values("year_id")
                q = prev[ssp][prev[ssp].location_id == location_id].sort_values("year_id")
                if not c.empty:
                    ax.plot(c.year_id, c.value, color=SCENARIO_COLOR[ssp], linewidth=2.0,
                            label=f"{SCENARIO_LABEL[ssp]} (current)")
                if not q.empty:
                    ax.plot(q.year_id, q.value, color=SCENARIO_COLOR[ssp], linewidth=1.6,
                            linestyle=(0, (5, 2)),
                            label=f"{SCENARIO_LABEL[ssp]} (2025 run)")
        ax.set_title(WEIGHTS[wk], fontsize=11)
        ax.set_xlabel("Year")
        ax.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel(label)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 3), frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{label} — global, current run vs 2025 run", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_covariate(
    label: str,
    wk: str,
    prev: dict[str, pd.DataFrame],
    curr: dict[str, pd.DataFrame],
    panels: Sequence[int],
    names: dict[int, str],
    out_path: Path,
) -> None:
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             squeeze=False, sharex=True)
    # A covariate that does not vary by scenario must not be drawn as three coloured
    # lines -- that implies variation the data does not have. Detect invariance instead of
    # hard-coding which covariates are flat, so this cannot regress silently.
    def _invariant(d: dict[str, pd.DataFrame], loc: int) -> bool:
        ser = [x[x.location_id == loc].sort_values("year_id").value.to_numpy()
               for x in d.values()]
        ser = [x for x in ser if x.size]
        if len(ser) < 2:
            return True
        ref = ser[0]
        return all(x.shape == ref.shape and np.allclose(x, ref, rtol=1e-9, atol=0)
                   for x in ser[1:])

    for idx, loc in enumerate(panels):
        ax = axes[idx // ncol][idx % ncol]
        flat = _invariant(curr, loc) and _invariant(prev, loc)
        if flat:
            ssp0 = next(iter(curr))
            c = curr[ssp0][curr[ssp0].location_id == loc].sort_values("year_id")
            p_ = prev[ssp0][prev[ssp0].location_id == loc].sort_values("year_id")
            if not c.empty:
                ax.plot(c.year_id, c.value, color="black", linewidth=2.0,
                        label="current", zorder=4)
            if not p_.empty:
                ax.plot(p_.year_id, p_.value, color="black", linewidth=1.6,
                        linestyle=(0, (5, 2)), label="2025 run", zorder=3)
        else:
            for ssp in curr:
                c = curr[ssp][curr[ssp].location_id == loc].sort_values("year_id")
                p_ = prev[ssp][prev[ssp].location_id == loc].sort_values("year_id")
                if not c.empty:
                    ax.plot(c.year_id, c.value, color=SCENARIO_COLOR[ssp], linewidth=2.0,
                            label=f"{SCENARIO_LABEL[ssp]} (current)", zorder=4)
                if not p_.empty:
                    ax.plot(p_.year_id, p_.value, color=SCENARIO_COLOR[ssp], linewidth=1.6,
                            linestyle=(0, (5, 2)),
                            label=f"{SCENARIO_LABEL[ssp]} (2025 run)", zorder=3)
        ax.set_title(names.get(loc, str(loc))[:46], fontsize=10)
        ax.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for idx in range(len(panels), nrow * ncol):
        axes[idx // ncol][idx % ncol].set_visible(False)
    for j in range(ncol):
        axes[nrow - 1][j].set_xlabel("Year")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"{label} — {WEIGHTS[wk]}, current run vs 2025 run",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option("--previous-cov-nc",
              default=str(mbpc.PREVIOUS_COVARIATE_NC),
              show_default=True, type=click.Path(exists=True, path_type=Path))
@click.option("--ssp-scenario", "ssps", multiple=True,
              default=("ssp126", "ssp245", "ssp585"), show_default=True)
@click.option("--dah-scenario", default="Baseline", show_default=True)
@click.option("--first-year", default=2000, show_default=True, type=int)
@click.option("--last-year", default=2100, show_default=True, type=int)
@click.option("--weight", "weight_kinds", multiple=True,
              type=click.Choice(list(WEIGHTS)), default=("population", "mort2023"),
              show_default=True,
              help="aggregation weight scheme(s). One figure set per scheme, tagged in "
                   "the filename, so burden-weighted sits beside population-weighted "
                   "rather than replacing it.")
@click.option("--output-dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
def main(previous_cov_nc, ssps, dah_scenario, first_year, last_year,
         weight_kinds, output_dir):
    """Plot each forecast covariate, this run against the previous run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(first_year, last_year + 1))
    hierarchy = load_hierarchy()
    names = dict(zip(hierarchy.location_id, hierarchy.location_name, strict=True))
    a2 = hierarchy.loc[hierarchy.level == 5, "location_id"].tolist()

    with xr.open_dataset(previous_cov_nc) as ds:
        prev_locs = set(int(x) for x in ds.location_id.values)
    cur_nc = mbpc.MAL_FORECAST_INPUTS_READ_PATH / f"malaria_forecast_inputs_{ssps[0]}.nc"
    with xr.open_dataset(cur_nc) as ds:
        cur_locs = set(int(x) for x in ds.location_id.values)
    shared = sorted(set(a2) & prev_locs & cur_locs)
    click.echo(
        f"admin-2 locations: previous {len(prev_locs & set(a2)):,}, "
        f"current {len(cur_locs & set(a2)):,}, shared {len(shared):,} "
        f"(dropped {len(((prev_locs | cur_locs) & set(a2)) - set(shared)):,})"
    )

    panels = [1] + sorted(
        hierarchy.loc[hierarchy.level == 1, "location_id"].tolist()
    )

    summary_rows = []
    pair_store: dict[str, dict[str, tuple[dict, dict]]] = {}
    # Weights are cheap and reused, so load them once up front.
    wcache = {}
    for wk in weight_kinds:
        wcache[wk] = load_weights(wk, years)
        click.echo(f"weight {wk} ({WEIGHTS[wk]}): "
                   f"{wcache[wk].location_id.nunique():,} admin-2 with nonzero weight")

    # Covariate is the OUTER loop and weighting the inner one, so each netCDF is read and
    # draw-averaged exactly once and then rolled up under every weighting. With the weight
    # loop outside, all of that I/O and the draw mean happened once per scheme.
    for label, prev_var, cur_var in COVARIATES:
        loaded = {}
        for ssp in ssps:
            pv = load_previous_covariate(previous_cov_nc, prev_var, ssp, shared)
            cv = load_current_covariate(
                mbpc.MAL_FORECAST_INPUTS_READ_PATH / f"malaria_forecast_inputs_{ssp}.nc",
                cur_var, dah_scenario, shared)
            if pv.empty or cv.empty:
                click.echo(f"  {label}: SKIP (missing in one run)")
                loaded = None
                break
            loaded[ssp] = (pv[pv.year_id.isin(years)], cv[cv.year_id.isin(years)])
        if not loaded:
            continue
        click.echo(f"  {label}: loaded {len(ssps)} ssp x 2 runs")

        for wk in weight_kinds:
            pop = wcache[wk]
            prev_agg, cur_agg = {}, {}
            for ssp in ssps:
                pv, cv = loaded[ssp]
                prev_agg[ssp] = weighted_rollup(pv, pop, hierarchy)
                cur_agg[ssp] = weighted_rollup(cv, pop, hierarchy)
            pair_store.setdefault(cur_var, {})[wk] = (prev_agg, cur_agg)
            pair_store[cur_var]["__label__"] = label
            figure_covariate(label, wk, prev_agg, cur_agg, panels, names,
                             output_dir / f"cov_{cur_var}__{wk}.png")
            for ssp in ssps:
                for yr in (2050, 2100):
                    pa, ca = prev_agg[ssp], cur_agg[ssp]
                    p = pa[(pa.location_id == 1) & (pa.year_id == yr)]
                    c = ca[(ca.location_id == 1) & (ca.year_id == yr)]
                    if p.empty or c.empty:
                        continue
                    pv_, cv_ = float(p.value.iloc[0]), float(c.value.iloc[0])
                    summary_rows.append({
                        "weight": wk, "covariate": label, "ssp": ssp, "year": yr,
                        "previous": pv_, "current": cv_,
                        "pct_change": 100 * (cv_ / pv_ - 1) if pv_ else np.nan,
                    })
            click.echo(f"    wrote cov_{cur_var}__{wk}.png")
        del loaded

    for cur_var, store in pair_store.items():
        lbl = store.pop("__label__", cur_var)
        order = [w for w in weight_kinds if w in store]
        if len(order) >= 2:
            figure_weight_pair(lbl, store, order,
                               output_dir / f"cov_{cur_var}__weightpair.png")
            click.echo(f"  wrote cov_{cur_var}__weightpair.png")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(output_dir / "covariate_shift_global.csv", index=False)
        click.echo("\nGLOBAL population-weighted covariate shift, current vs 2025 run:")
        click.echo(summary.to_string(index=False))


if __name__ == "__main__":
    main()
