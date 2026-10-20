"""Both models on every mix of input sets for the covariates they share (this run's or the 1st submission's).

Models: this run's selected model (models node ``spec7426_best``) and the 1st submission's (models node label
``first_submission`` = 2025_07_08). Covariates in both: GDP per capita, DAH per capita, suitability. The 1st
submission's model also has flood days per capita (not in this run's model), which stays at its own input throughout.
A mix is coded per covariate in the order GDP, DAH, suitability: ``T`` this run's input, ``F`` the 1st submission's
(``TTT`` is this run's inputs, ``FFF`` the 1st submission's).

Each model keeps its own anchor: this run's model its 2023 observation; the 1st submission's model the 2022
observation in its input files. A swapped covariate enters through its change since that model's rake year. Units:
admin-2s present in both hierarchies (1285 and 1209), with complete inputs in both input sets and the 1st
submission's rake year 2022. Counts are draw means times this run's all-age population and this run's Baseline vaccine
factor for both models, so the two models share denominators. Rows are summed to global and super-region.

Checks: this run's model on its own inputs vs its saved forecast on the same units; the 1st submission's model on its
own inputs vs its saved per-draw predictions (log rates) for ``--check-draws`` draws.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.processing.scenario_gap import (
    FIRST_SUBMISSION_INPUTS,
    INPUT_FACTORS,
    MEASURES,
    OLD_ANCHORS,
    FloatArray,
    IntArray,
    Rebuild,
    complete_rows,
    group_rows,
    load_run,
    load_terms,
    read_factors,
    read_old_anchors,
    read_old_inputs,
    scatter_to_grid,
    subset_rebuild,
)

SSPS = ("ssp126", "ssp245", "ssp585")
OLD_RAKE_YEAR = 2022
RAKE_YEAR = 2023
SHARED = {"gdp": "gdp", "dah": "dah", "suitability": "suit"}  # covariate -> input key
OLD_PREDICTIONS = (
    mbpc.MODEL_ROOT / "04-forecasting_data/malaria/lsae_1209/first_submission"
    "/malaria_forecast_ssp_scenario_{ssp}_dah_scenario_Baseline_draw_{draw:03d}_with_predictions.nc"
)
OWN = {"current": "TTT", "first_submission": "FFF"}  # each model's own inputs
OLD_PRED_VARS = {
    "inc": "log_aa_malaria_inc_rate_pred",
    "mort": "log_aa_malaria_mort_rate_pred",
}


def input_combinations() -> list[str]:
    """Every choice of input set per shared covariate, in ``SHARED`` order (GDP, DAH, suitability): ``T`` this run's,
    ``F`` the 1st submission's. ``TTT`` is all this run's inputs, ``FFF`` all the 1st submission's."""
    return ["".join(c) for c in product("TF", repeat=len(SHARED))]


def mix_inputs(
    this: dict[str, dict[str, FloatArray]],
    first: dict[str, dict[str, FloatArray]],
    code: str,
    base: dict[str, dict[str, FloatArray]],
) -> dict[str, dict[str, FloatArray]]:
    """Per SSP, each shared covariate from this run (``T``) or the 1st submission (``F``) as ``code`` says; any other
    input in ``base`` (the 1st submission model's flood) stays ``base``'s."""
    keys = list(SHARED.values())
    return {
        s: {
            **base[s],
            **{
                k: (this[s][k] if c == "T" else first[s][k])
                for k, c in zip(keys, code, strict=True)
            },
        }
        for s in base
    }


def read_old_predictions(  # pragma: no cover - netCDF I/O
    locs: IntArray, years: IntArray, draws: list[int]
) -> dict[str, dict[str, FloatArray]]:
    """The 1st submission's saved all-age log rates: ``{ssp: {measure: (loc, year, draw)}}``."""
    out: dict[str, dict[str, FloatArray]] = {}
    for ssp in SSPS:
        arrs: dict[str, FloatArray] = {
            m: np.full((locs.size, years.size, len(draws)), np.nan, dtype=np.float64)
            for m in MEASURES
        }
        for j, d in enumerate(draws):
            with xr.open_dataset(str(OLD_PREDICTIONS).format(ssp=ssp, draw=d)) as ds:
                sub = ds.sel(location_id=locs, year_id=years)
                for m, var in OLD_PRED_VARS.items():
                    arrs[m][:, :, j] = (
                        sub[var].transpose("location_id", "year_id").to_numpy()
                    )
        out[ssp] = arrs
    return out


def run_year(  # noqa: PLR0913 - every model's rebuilds for one year, plus the count frame
    rebuilds: dict[str, dict[str, Rebuild]],
    t: int,
    *,
    year: int,
    population: FloatArray,
    masks: dict[str, npt.NDArray[np.bool_]],
    factor: dict[str, FloatArray],
    groups: FloatArray,
) -> tuple[list[dict[str, object]], dict[str, dict[str, FloatArray]]]:
    """Global and super-region counts of every model x variant x SSP x measure in year index ``t``.

    Also returns each model's own-input log rates (loc x draw) per SSP and measure, for the checks.
    """
    rows: list[dict[str, object]] = []
    own_rates: dict[str, dict[str, FloatArray]] = {}
    for model, variants in rebuilds.items():
        for name, rb in variants.items():
            parts = rb.year_parts(t)
            for ssp in SSPS:
                for m in MEASURES:
                    lr = rb.log_rate(parts, m, dict.fromkeys(INPUT_FACTORS, ssp))
                    if name == OWN[model]:
                        own_rates.setdefault(model, {})[f"{ssp}:{m}"] = lr
                    c = np.exp(lr) * population[:, t][:, None]
                    c[masks[f"{ssp}:{m}"][:, t]] = np.nan
                    rows += group_rows(
                        c.mean(axis=1) * factor[m][:, t],
                        groups,
                        year=year,
                        source=model,
                        variant=name,
                        ssp=ssp,
                        measure=m,
                    )
    return rows, own_rates


@click.command()
@click.option("--spec", default="spec7426_best", show_default=True)
@click.option("--forecast-label", default="spec7426_best__base", show_default=True)
@click.option(
    "--term-root",
    type=click.Path(path_type=Path),
    required=True,
    help="holds <spec>/ and first_submission/ term grids",
)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--n-draws", default=100, show_default=True)
@click.option(
    "--check-draws",
    default=10,
    show_default=True,
    help="1st-submission draws checked against its saved predictions",
)
def main(  # noqa: PLR0913, PLR0915, PLR0917 - click passes one argument per option; one run with its checks
    spec: str,
    forecast_label: str,
    term_root: Path,
    out_dir: Path,
    n_draws: int,
    check_draws: int,
) -> None:  # pragma: no cover - orchestration over tested pieces
    """Write swap_counts.parquet, swap_population.parquet and swap_checks.json."""
    cur_full, forecasts, locs, years, population = load_run(
        spec,
        forecast_label=forecast_label,
        term_root=term_root,
        ssps=list(SSPS),
        dah="Baseline",
        rake_year=RAKE_YEAR,
    )
    old_ids = pq.read_table(
        str(FIRST_SUBMISSION_INPUTS).format(ssp="ssp245", draw=0),
        columns=["location_id"],
    ).column(0)
    rows0 = np.flatnonzero(np.isin(locs, np.unique(old_ids.to_numpy())))
    old_inputs = read_old_inputs(
        locs[rows0], years, n_draws, ssps=SSPS, with_flood=True
    )
    anchors = read_old_anchors(locs[rows0])
    ory = int(np.flatnonzero(years == OLD_RAKE_YEAR)[0])
    keep = (
        complete_rows(old_inputs, first_year_index=ory)
        & (anchors.year_to_rake_to.to_numpy() == OLD_RAKE_YEAR)
        & np.isfinite(anchors[OLD_ANCHORS["pfpr"]].to_numpy(float))
    )
    rows = rows0[keep]
    old_inputs = {s: {k: v[keep] for k, v in x.items()} for s, x in old_inputs.items()}
    anchors = anchors[keep]
    cur = subset_rebuild(cur_full, rows)
    grids_old, slopes_old = load_terms(term_root / "first_submission")
    old = Rebuild(
        grids=grids_old,
        slopes=slopes_old,
        inputs=old_inputs,
        l_obs=anchors[OLD_ANCHORS["pfpr"]].to_numpy(float)[:, None],
        obs_log={m: anchors[OLD_ANCHORS[m]].to_numpy(float)[:, None] for m in MEASURES},
        ry=ory,
    )
    rebuilds = {
        model: {
            code: Rebuild(
                grids=rb.grids,
                slopes=rb.slopes,
                inputs=mix_inputs(cur.inputs, old_inputs, code, rb.inputs),
                l_obs=rb.l_obs,
                obs_log=rb.obs_log,
                ry=rb.ry,
            )
            for code in input_combinations()
        }
        for model, rb in (("current", cur), ("first_submission", old))
    }
    # one unit set for both models: this run's outcome masks plus any 1st-submission anchor that is not finite
    masks = {
        f"{s}:{m}": np.isnan(forecasts[s][m][rows])
        | (~np.isfinite(old.obs_log[m]))[:, :, None]
        for s in SSPS
        for m in MEASURES
    }  # (loc, year, draw)
    m_locs = locs[rows]
    pop = population[rows]
    hier = load_hierarchy().set_index("location_id")
    groups = hier.loc[m_locs, "super_region_id"].to_numpy(float)
    fac_long = read_factors(spec, [int(y) for y in years if y > RAKE_YEAR])
    factor = {}
    for m in MEASURES:
        f = fac_long[(fac_long.measure == m) & fac_long.location_id.isin(m_locs)]
        factor[m] = scatter_to_grid(
            f.location_id.to_numpy(np.int64),
            f.year_id.to_numpy(np.int64),
            f.factor.to_numpy(float),
            m_locs,
            years,
        )
    check_years = years[years > OLD_RAKE_YEAR]
    old_pred = read_old_predictions(m_locs, check_years, list(range(check_draws)))
    worst = {
        "current_own_vs_forecast": 0.0,
        "first_submission_own_vs_saved_log_rate": 0.0,
    }
    out_rows: list[dict[str, object]] = []
    pop_rows: list[dict[str, object]] = []
    for t in range(cur.ry + 1, years.size):
        year = int(years[t])
        rows_t, own = run_year(
            rebuilds,
            t,
            year=year,
            population=pop,
            masks=masks,
            factor=factor,
            groups=groups,
        )
        out_rows += rows_t
        for g in (1, *np.unique(groups).astype(int).tolist()):
            sel = np.ones(groups.size, bool) if g == 1 else groups == g
            pop_rows.append(
                {
                    "location_id": g,
                    "year_id": year,
                    "population": float(pop[sel, t].sum()),
                }
            )
        ci = int(np.flatnonzero(check_years == year)[0])
        for s in SSPS:
            for m in MEASURES:
                key = f"{s}:{m}"
                saved = forecasts[s][m][rows, t]
                ok = ~masks[key][:, t, :]
                cnt = (np.exp(own["current"][key]) * pop[:, t][:, None])[ok].sum()
                ref = (np.exp(saved) * pop[:, t][:, None])[ok].sum()
                worst["current_own_vs_forecast"] = max(
                    worst["current_own_vs_forecast"], float(abs(cnt / ref - 1))
                )
                diff = np.abs(
                    own["first_submission"][key][:, :check_draws]
                    - old_pred[s][m][:, ci, :]
                )
                fin = np.isfinite(diff)
                if fin.any():
                    worst["first_submission_own_vs_saved_log_rate"] = max(
                        worst["first_submission_own_vs_saved_log_rate"],
                        float(diff[fin].max()),
                    )
        click.echo(f"  {year} done")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(out_dir / "swap_counts.parquet", index=False)
    pd.DataFrame(pop_rows).to_parquet(out_dir / "swap_population.parquet", index=False)
    checks = {
        **worst,
        "n_admin2_used": int(rows.size),
        "n_admin2_forecast": int(locs.size),
        "check_draws": check_draws,
        "slopes_first_submission": {f"{k[0]}:{k[1]}": v for k, v in slopes_old.items()},
    }
    (out_dir / "swap_checks.json").write_text(json.dumps(checks, indent=2))
    click.echo(f"checks {checks}")


if __name__ == "__main__":
    main()
