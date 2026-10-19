"""Tables behind the RCP4.5 scenario-gap page, written as one JSON file.

Reads what earlier steps produced and adds burden-weighted summaries of the scenario inputs:
- global and super-region all-age counts and rates by SSP: this run's paper products (Baseline, no_vaccine,
  gdppc_hold2023, and wind_hold2023 for a model with wind) and the 2025 first-submission run (Baseline and its
  GDP-held and suitability-held arms, upload_folders/2025_08_28);
- the Shapley parts from ``decompose_scenario_gap.py`` (global and super-region by year; regions and countries
  in 2100 and summed over 2024-2100);
- each SSP's inputs (GDP per capita, suitability, wind) and their logit-PfPR contributions since 2023, weighted by
  each admin-2's 2023 observed incidence count;
- the fitted term curves the gap passes through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
import pandas as pd
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.processing.all_age_products import load_population
from idd_forecast_mbp.lib.processing.scenario_gap import (
    ADMIN2_LEVEL,
    MEASURES,
    PRODUCTS,
    FloatArray,
    by_group,
    load_run,
    read_factors,
    roll_up,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

SSPS = ("ssp126", "ssp245", "ssp585")
PREVIOUS_RUN = "2025_08_28"
PREVIOUS_ARMS = {
    "Baseline": "",
    "gdppc_hold": "_hold_gdppc",
    "suitability_hold": "_hold_suitability",
}
PREVIOUS_MEASURE = {"inc": "incidence", "mort": "mortality"}
LINK_PARTS = ("dah", "gdp_pfpr", "suitability", "wind", "log_gdp")


def ranked_parts(
    agg: pd.DataFrame,
    *,
    level: int,
    years: Sequence[int],
    part_cols: Sequence[str],
    n: int,
) -> pd.DataFrame:
    """Locations at ``level`` ranked by |gap| summed over ``years``: gap, each part, and A and B counts."""
    sub = agg[(agg.level == level) & agg.year_id.isin(list(years))]
    cols = ["count_a", "count_b", *part_cols]
    tot = sub.groupby(["location_id", "location_name"], as_index=False)[cols].sum(
        min_count=1
    )
    tot["gap"] = tot.count_a - tot.count_b
    return (
        tot.reindex(tot.gap.abs().sort_values(ascending=False).index)
        .head(n)
        .reset_index(drop=True)
    )


def _series(frame: pd.DataFrame, cols: Sequence[str]) -> dict[Any, Any]:
    return {c: [None if pd.isna(v) else float(v) for v in frame[c]] for c in cols}


def read_products(
    spec: str, arm: str, locs: Sequence[int]
) -> pd.DataFrame:  # pragma: no cover - I/O
    frames = []
    for ssp in SSPS:
        df = pd.read_parquet(
            PRODUCTS / f"{spec}__{arm}/current/all_age_summary_{ssp}.parquet"
        )
        frames.append(df[df.location_id.isin(list(locs))].assign(ssp=ssp))
    return pd.concat(frames, ignore_index=True)


def read_previous(
    arm_suffix: str, locs: Sequence[int]
) -> pd.DataFrame:  # pragma: no cover - I/O
    root = mbpc.previous_upload_path(PREVIOUS_RUN)
    rows = []
    for ssp in SSPS:
        for m, token in PREVIOUS_MEASURE.items():
            name = (
                f"aa_cause_malaria_measure_{token}_metric_count_ssp_scenario_{ssp}"
                f"_dah_scenario_Baseline{arm_suffix}"
            )
            with xr.open_dataset(root / name / "draws.nc") as ds:
                have = [x for x in locs if x in set(ds.location_id.to_numpy().tolist())]
                mean = (
                    ds["val"]
                    .sel(location_id=have)
                    .mean("draw_id")
                    .to_dataframe(name="count")
                )
            rows.append(mean.reset_index().assign(ssp=ssp, measure=m))
    return pd.concat(rows, ignore_index=True)


def scenario_block(products: pd.DataFrame, locs: Sequence[int]) -> dict[Any, Any]:
    """{location_id: {ssp: {year, inc_count, mort_count, inc_rate, mort_rate}}} from a paper product frame."""
    cols = [f"{m}_{k}_mean" for m in MEASURES for k in ("count", "rate")]
    out: dict[Any, Any] = {}
    for loc in locs:
        out[int(loc)] = {}
        for ssp in SSPS:
            f = products[
                (products.location_id == loc) & (products.ssp == ssp)
            ].sort_values("year_id")
            out[int(loc)][ssp] = {
                "year": f.year_id.astype(int).tolist(),
                **_series(f, cols),
            }
    return out


def previous_block(prev: pd.DataFrame, locs: Sequence[int]) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for loc in locs:
        out[int(loc)] = {}
        for ssp in SSPS:
            f = prev[(prev.location_id == loc) & (prev.ssp == ssp)]
            wide = f.pivot_table(
                index="year_id", columns="measure", values="count"
            ).sort_index()
            out[int(loc)][ssp] = {
                "year": wide.index.astype(int).tolist(),
                **{f"{m}_count": _series(wide, [m])[m] for m in MEASURES if m in wide},
            }
    return out


def gap_block(
    agg: pd.DataFrame, part_cols: Sequence[str], locs: Sequence[int]
) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for (pair, m), f in agg[
        agg.location_id.isin(list(locs)) & agg.level.isin([0, 1])
    ].groupby(["pair", "measure"]):
        out.setdefault(pair, {})[m] = {
            int(loc): {
                "year": g.year_id.astype(int).tolist(),
                **_series(g, ["count_a", "count_b", *part_cols]),
            }
            for loc, g in f.sort_values("year_id").groupby("location_id")
        }
    return out


def ranked_block(
    agg: pd.DataFrame, part_cols: Sequence[str], population: pd.DataFrame | None = None
) -> dict[Any, Any]:
    """Regions and countries ranked by gap, in 2100 and summed over 2024-2100.

    With ``population`` (``location_id, year_id, population``), each record also carries the population summed over
    the same years, so a rate is the record's count over it (for the sum, the mean annual rate).
    """
    out: dict[Any, Any] = {}
    for (pair, m), f in agg.groupby(["pair", "measure"]):
        entry = {}
        for label, years in (("y2100", [2100]), ("cumulative", range(2024, 2101))):
            pop_sum = (
                population[population.year_id.isin(list(years))]
                .groupby("location_id")
                .population.sum()
                if population is not None
                else None
            )
            for name, level, n in (("regions", 2, 21), ("countries", 3, 15)):
                r = ranked_parts(f, level=level, years=years, part_cols=part_cols, n=n)
                r = r.assign(location_id=r.location_id.astype(int))
                if pop_sum is not None:
                    r["population"] = r.location_id.map(pop_sum).astype(float)
                entry[f"{name}_{label}"] = r.to_dict("records")
        out.setdefault(pair, {})[m] = entry
    return out


def population_block(population: pd.DataFrame, locs: Sequence[int]) -> dict[Any, Any]:
    """{location_id: {year, population}} for ``locs``, years ascending."""
    out: dict[Any, Any] = {}
    for loc in locs:
        f = population[population.location_id == loc].sort_values("year_id")
        out[int(loc)] = {
            "year": f.year_id.astype(int).tolist(),
            **_series(f, ["population"]),
        }
    return out


def variant_block(
    counts: pd.DataFrame, population: pd.DataFrame, checks: dict[str, Any]
) -> dict[Any, Any]:
    """{checks, population: {loc: {year, population}}, counts: {source: {variant: {measure: {loc: {ssp: {year, count}}}}}}}."""
    block: dict[Any, Any] = {
        "checks": checks,
        "population": population_block(
            population, sorted(population.location_id.unique().tolist())
        ),
        "counts": {},
    }
    for (src, var, m, loc, ssp), f in counts.groupby(
        ["source", "variant", "measure", "location_id", "ssp"]
    ):
        g = f.sort_values("year_id")
        leaf = (
            block["counts"]
            .setdefault(src, {})
            .setdefault(var, {})
            .setdefault(m, {})
            .setdefault(int(loc), {})
        )
        leaf[ssp] = {"year": g.year_id.astype(int).tolist(), **_series(g, ["count"])}
    return block


def trajectory_block(traj: pd.DataFrame) -> dict[Any, Any]:
    """{covariate: {source: {ssp: {loc: {year, value}}}}} from the input trajectories."""
    out: dict[Any, Any] = {}
    for (cov, src, ssp, loc), f in traj.groupby(
        ["covariate", "source", "ssp", "location_id"]
    ):
        g = f.sort_values("year_id")
        out.setdefault(cov, {}).setdefault(src, {}).setdefault(ssp, {})[int(loc)] = {
            "year": g.year_id.astype(int).tolist(),
            **_series(g, ["value"]),
        }
    return out


def draw_mean_parts(parts: dict[str, FloatArray], n_loc: int) -> dict[str, FloatArray]:
    """Each rebuild piece (``Rebuild.year_parts`` for one SSP) as a per-location draw mean.

    The (loc, 1) pieces do not vary by draw; they broadcast against the (loc, draw) suitability piece.
    """
    n_draw = parts["suitability"].shape[1]
    return {
        k: np.nanmean(np.broadcast_to(parts[k], (n_loc, n_draw)), axis=1)
        for k in LINK_PARTS
    }


def input_block(  # pragma: no cover - needs the full rebuild inputs
    spec: str,
    *,
    forecast_label: str,
    term_root: Path,
    weights_by_loc: pd.Series,
    super_region: pd.Series,
) -> dict[Any, Any]:
    """Burden-weighted inputs and logit-PfPR contributions since 2023, per SSP, global and super-region."""
    rebuild, _, locs, years, _ = load_run(
        spec,
        forecast_label=forecast_label,
        term_root=term_root,
        ssps=list(SSPS),
        dah="Baseline",
        rake_year=2023,
    )
    n = locs.size
    w = weights_by_loc.reindex(locs).to_numpy(dtype=float)
    groups = super_region.reindex(locs).to_numpy()
    fyears = list(range(rebuild.ry, years.size))
    # rake year: no change
    links: dict[str, dict[str, list[FloatArray]]] = {
        ssp: {k: [np.zeros(n, dtype=np.float64)] for k in LINK_PARTS} for ssp in SSPS
    }
    for t in fyears[1:]:
        parts = rebuild.year_parts(t)
        for ssp in SSPS:
            for k, v in draw_mean_parts(parts[ssp], n).items():
                links[ssp][k].append(v)
    out: dict[Any, Any] = {"year": [int(years[t]) for t in fyears], "ssp": {}}
    for ssp in SSPS:
        x = rebuild.inputs[ssp]
        series = {
            "gdppc": x["gdp"][:, fyears],
            "suitability_days": np.nanmean(x["suit"][:, fyears], axis=2),
            **(
                {"wind_speed": np.nanmean(x["wind"][:, fyears], axis=2)}
                if rebuild.wind
                else {}
            ),
            **{f"link_{k}": np.column_stack(v) for k, v in links[ssp].items()},
        }
        for m in MEASURES:
            series[f"link_gdp_{m}"] = (
                rebuild.slopes[(m, "log_gdppc_mean")] * series["link_log_gdp"]
            )
        out["ssp"][ssp] = {
            k: {
                str(g): [float(v) if np.isfinite(v) else None for v in arr]
                for g, arr in by_group(s, w, groups).items()
            }
            for k, s in series.items()
        }
    return out


def term_block(term_dir: Path) -> dict[Any, Any]:  # pragma: no cover - parquet I/O
    grids = pd.read_parquet(term_dir / "term_grids.parquet")
    slopes = pd.read_parquet(term_dir / "linear_terms.parquet")
    out = {
        "slopes": {
            f"{r.model}:{r.covariate}": float(r.slope) for r in slopes.itertuples()
        }
    }
    keep = {
        ("pfpr", "gdppc_mean"): (100.0, 150000.0),
        ("pfpr", "mal_DAH_total_per_capita"): (0.0, 25.0),
        ("inc", "logit_malaria_pfpr"): (-12.0, 4.0),
        ("mort", "logit_malaria_pfpr"): (-12.0, 4.0),
    }
    for (m, c), (lo, hi) in keep.items():
        g = grids[(grids.model == m) & (grids.covariate == c) & grids.x.between(lo, hi)]
        g = g.iloc[np.linspace(0, len(g) - 1, 300).astype(int)]
        out[f"{m}:{c}"] = {
            "x": g.x.round(6).tolist(),
            "y": (g.y - g.y.iloc[0]).round(6).tolist(),
        }
    return out


def apply_factors(
    admin2: pd.DataFrame, factors: pd.DataFrame, value_cols: Sequence[str]
) -> pd.DataFrame:
    """Scale each admin-2 row's counts and parts by its measure's vaccine factor (Baseline / no_vaccine).

    ``factors`` is ``[location_id, year_id, measure, factor]``. The factor is the same in every SSP, so the
    Shapley parts of the no-vaccine gap times the factor are the parts of the Baseline gap. A row without a
    factor raises.
    """
    out = admin2.merge(factors, on=["location_id", "year_id", "measure"], how="left")
    if (
        out.factor.isna().any()
        and not out.loc[out.factor.isna(), value_cols[0]].isna().all()
    ):
        msg = "admin-2 rows with a forecast but no vaccine factor"
        raise ValueError(msg)
    for c in value_cols:
        out[c] = out[c] * out.factor
    return out.drop(columns="factor")


def factor_spread(frames: dict[str, pd.DataFrame]) -> float:
    """Max |factor difference| between any SSP's vaccine factors and the first SSP's, over shared rows."""
    keys = ["location_id", "year_id", "measure"]
    ref_name, *others = list(frames)
    ref = frames[ref_name].set_index(keys).factor
    worst = 0.0
    for name in others:
        diff = (frames[name].set_index(keys).factor - ref).abs()
        worst = max(worst, float(diff.max()))
    return worst


def held_block(agg: pd.DataFrame, locs: Sequence[int]) -> dict[Any, Any]:
    """{held: {measure: {location_id: {ssp: {year, count}}}}} from rolled-up held-input counts."""
    out: dict[Any, Any] = {}
    sub = agg[agg.location_id.isin(list(locs)) & agg.level.isin([0, 1])]
    for (arm, m, loc), grp in sub.groupby(["arm", "measure", "location_id"]):
        ssp, held = arm.split(":")
        g = grp.sort_values("year_id")
        out.setdefault(held, {}).setdefault(m, {}).setdefault(int(loc), {})[ssp] = {
            "year": g.year_id.astype(int).tolist(),
            **_series(g, ["count"]),
        }
    return out


def held_check(agg: pd.DataFrame, products: pd.DataFrame, held: str) -> float:
    """Max relative difference between rolled-up global held counts and a hold arm's global product counts."""
    glob = products[products.location_id == 1].set_index(["ssp", "year_id"])
    sub = agg[(agg.level == 0) & agg.arm.str.endswith(f":{held}")]
    worst = 0.0
    for (arm, m), f in sub.groupby(["arm", "measure"]):
        ref = (
            glob.loc[arm.split(":")[0]][f"{m}_count_mean"].reindex(f.year_id).to_numpy()
        )
        worst = max(worst, float(np.nanmax(np.abs(f["count"].to_numpy() / ref - 1))))
    return worst


def render_page(template: str, data: dict[Any, Any]) -> str:
    """The page template with ``__PAGE_DATA__`` replaced by the JSON (``</`` escaped so it cannot close the script)."""
    if template.count("__PAGE_DATA__") != 1:
        msg = "template must contain __PAGE_DATA__ exactly once"
        raise ValueError(msg)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return template.replace("__PAGE_DATA__", payload)


def total_check(agg: pd.DataFrame, products: pd.DataFrame) -> float:
    """Max relative difference between the rolled-up global A and B counts and the product's global counts."""
    glob = products[products.location_id == 1].set_index(["ssp", "year_id"])
    worst = 0.0
    for (pair, m), f in agg[agg.level == 0].groupby(["pair", "measure"]):
        a, b = pair.split(":")
        for ssp, col in ((a, "count_a"), (b, "count_b")):
            ref = glob.loc[ssp][f"{m}_count_mean"].reindex(f.year_id).to_numpy()
            worst = max(worst, float(np.nanmax(np.abs(f[col].to_numpy() / ref - 1))))
    return worst


@click.command()
@click.option(
    "--gap-root",
    type=click.Path(path_type=Path),
    required=True,
    help="folder with <spec>/gap_parts_admin2.parquet and term_grids/<spec>",
)
@click.option("--out-file", type=click.Path(path_type=Path), required=True)
@click.option(
    "--html-out",
    type=click.Path(path_type=Path),
    default=None,
    help="also render scenario_gap_page.html with the data into this file",
)
@click.option(
    "--spec",
    "specs",
    multiple=True,
    default=("spec7426_best", "spec7427_alternate"),
    show_default=True,
)
def main(
    gap_root: Path, out_file: Path, html_out: Path | None, specs: tuple[str, ...]
) -> None:  # pragma: no cover - orchestration
    """Write the page's JSON."""
    hier = load_hierarchy()
    sr = hier[hier.level == 1][["location_id", "location_name"]]
    locs = [1, *sr.location_id.astype(int).tolist()]
    names = {
        1: "Global",
        **dict(zip(sr.location_id.astype(int), sr.location_name, strict=True)),
    }
    a2 = hier[hier.level == ADMIN2_LEVEL].set_index("location_id")
    obs = (
        read_parquet_with_integer_ids(
            mbpc.MAL_RAKED_AA_READ_PATH / "aa_full_malaria_df.parquet",
            columns=["location_id", "year_id", "malaria_inc_count"],
            filters=[("year_id", "==", 2023)],
        )
        .set_index("location_id")
        .malaria_inc_count
    )
    data: dict[Any, Any] = {"names": {str(k): v for k, v in names.items()}, "specs": {}}
    population = load_population(list(range(2000, 2101)))
    data["population"] = population_block(population, locs)
    prev = {arm: read_previous(sfx, locs) for arm, sfx in PREVIOUS_ARMS.items()}
    data["previous"] = {arm: previous_block(p, locs) for arm, p in prev.items()}
    for spec in specs:
        run = json.loads((gap_root / spec / "run.json").read_text())
        part_cols = [f"part_{f}" for f in run["factors"]]
        value_cols = ["count_a", "count_b", *part_cols]
        admin2 = pd.read_parquet(gap_root / spec / "gap_parts_admin2.parquet")
        keys = (
            hier.set_index("location_id")
            .loc[
                admin2.location_id.unique(),
                ["A0_location_id", "region_id", "super_region_id"],
            ]
            .reset_index()
        )
        factors = read_factors(spec, range(2024, 2101))
        spread = factor_spread(
            {ssp: read_factors(spec, range(2024, 2101), ssp) for ssp in SSPS}
        )
        agg = roll_up(apply_factors(admin2, factors, value_cols), keys, value_cols)
        agg["location_name"] = agg.location_id.map(
            hier.set_index("location_id").location_name
        ).fillna("Global")
        arms = [
            "Baseline",
            "no_vaccine",
            "gdppc_hold2023",
            *(["wind_hold2023"] if "wind" in run["factors"] else []),
        ]
        products = {arm: read_products(spec, arm, locs) for arm in arms}
        nv_agg = pd.read_parquet(gap_root / spec / "gap_parts_aggregated.parquet")
        held = pd.read_parquet(gap_root / spec / "held_admin2.parquet")
        held_agg = roll_up(
            apply_factors(held, factors, ["count"]),
            keys,
            ["count"],
            by=("arm", "measure", "year_id"),
        )
        data["specs"][spec] = {
            "run": run,
            "checks": {
                "vaccine_factor_max_abs_diff_across_ssps": spread,
                "baseline_total_max_rel_diff": total_check(agg, products["Baseline"]),
                "no_vaccine_total_max_rel_diff": total_check(
                    nv_agg, products["no_vaccine"]
                ),
                "gdp_held_vs_gdppc_hold2023_max_rel_diff": held_check(
                    held_agg, products["gdppc_hold2023"], "gdp_held"
                ),
                **(
                    {
                        "wind_held_vs_wind_hold2023_max_rel_diff": held_check(
                            held_agg, products["wind_hold2023"], "wind_held"
                        )
                    }
                    if "wind_hold2023" in products
                    else {}
                ),
            },
            "scenarios": {arm: scenario_block(p, locs) for arm, p in products.items()},
            "gap": gap_block(agg, part_cols, locs),
            "gap_no_vaccine": gap_block(nv_agg, part_cols, [1]),
            "ranked": ranked_block(agg, part_cols, population),
            "held": held_block(held_agg, locs),
            "inputs": input_block(
                spec,
                forecast_label=run["forecast_label"],
                term_root=gap_root / "term_grids",
                weights_by_loc=obs,
                super_region=a2.super_region_id,
            ),
            "terms": term_block(gap_root / "term_grids" / spec),
        }
        vdir = gap_root / spec / "variants"
        if (vdir / "variant_counts.parquet").is_file():
            data["specs"][spec]["variants"] = variant_block(
                pd.read_parquet(vdir / "variant_counts.parquet"),
                pd.read_parquet(vdir / "variant_population.parquet"),
                json.loads((vdir / "variant_checks.json").read_text()),
            )
            data["specs"][spec]["input_trajectories"] = trajectory_block(
                pd.read_parquet(vdir / "input_trajectories.parquet")
            )
        sdir = gap_root / spec / "swaps"
        if (sdir / "swap_counts.parquet").is_file():
            data["specs"][spec]["swaps"] = variant_block(
                pd.read_parquet(sdir / "swap_counts.parquet"),
                pd.read_parquet(sdir / "swap_population.parquet"),
                json.loads((sdir / "swap_checks.json").read_text()),
            )
        click.echo(f"  {spec} done; checks {data['specs'][spec]['checks']}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(data, separators=(",", ":")))
    click.echo(f"wrote {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")
    if html_out is not None:
        template = Path(__file__).with_name("scenario_gap_page.html").read_text()
        html_out.write_text(render_page(template, data))
        click.echo(f"wrote {html_out} ({html_out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
