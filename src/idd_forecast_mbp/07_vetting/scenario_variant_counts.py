"""Counts under input variants: this run's model on this run's inputs and on the 2025 first-submission inputs.

Variants, per SSP:
- hold factorial: every subset of {GDP per capita, DAH per capita, suitability} held at its 2023 value (its change
  since 2023 zeroed), on both input sets;
- common GDP (this run's inputs): every scenario given one SSP's GDP, everything else its own.

The 2025 inputs (upload of 2025-08-28; forecast inputs ``lsae_1209/first_submission``) enter only through their change
since 2023: the anchor stays this run's 2023 observation, and the model is this run's. Both input sets are
restricted to the admin-2 units present in both hierarchies (1285 and 1209) so they are summed over the same units.
Counts are draw means times the Baseline vaccine factor (SSP-invariant), summed to global and super-region; the
summed population is written alongside for rates. Also writes each input's trajectory (2000-2100) per input set and
SSP, weighted by each admin-2's 2023 observed incidence count, for the input-comparison panels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.processing.scenario_gap import (
    INPUT_FACTORS,
    MEASURES,
    FloatArray,
    IntArray,
    Rebuild,
    align,
    by_group,
    hold_pieces,
    load_run,
    read_factors,
    read_forecast,
    subset_rebuild,
)

OLD_INPUTS = (
    mbpc.MODEL_ROOT / "04-forecasting_data/malaria/lsae_1209/first_submission"
    "/malaria_forecast_ssp_scenario_{ssp}_dah_scenario_Baseline_draw_{draw:03d}.parquet"
)
OLD_COLUMNS = {
    "gdp": "gdppc_mean",
    "dah": "mal_DAH_total_per_capita",
    "suit": "malaria_suitability",
}
HOLD_INPUTS = ("gdp", "dah", "suitability")
SSPS = ("ssp126", "ssp245", "ssp585")
RAKE_YEAR = 2023
CHECKS = {
    "hold:none": "current_hold_none_vs_forecast",
    "hold:gdp": "current_hold_gdp_vs_gdppc_hold2023",
}


def hold_variants() -> list[tuple[str, tuple[str, ...]]]:
    """Every subset of ``HOLD_INPUTS`` as ``("hold:<a>+<b>", (a, b))``; the empty subset is ``hold:none``."""
    return [
        (f"hold:{'+'.join(held) if held else 'none'}", held)
        for r in range(len(HOLD_INPUTS) + 1)
        for held in combinations(HOLD_INPUTS, r)
    ]


def scatter_to_grid(
    loc_ids: IntArray,
    year_ids: IntArray,
    values: FloatArray,
    locs: IntArray,
    years: IntArray,
) -> FloatArray:
    """Long ``(location, year, value)`` rows as a ``locs x years`` array; a cell with no row is NaN."""
    out = np.full((locs.size, years.size), np.nan)
    out[align(locs, loc_ids), align(years, year_ids)] = values
    return out


def group_rows(  # noqa: PLR0913 - the labels of one row set plus the values and groups
    values: FloatArray,
    groups: FloatArray,
    *,
    year: int,
    source: str,
    variant: str,
    ssp: str,
    measure: str,
) -> list[dict[str, object]]:
    """Sums of ``values`` per super-region and globally (location 1), as output rows; an all-NaN group is NaN."""
    rows = []
    for g in (1, *np.unique(groups).astype(int).tolist()):
        sel = np.ones(values.size, bool) if g == 1 else groups == g
        v = values[sel]
        total = float(np.nansum(v)) if np.isfinite(v).any() else np.nan
        rows.append(
            {
                "source": source,
                "variant": variant,
                "ssp": ssp,
                "measure": measure,
                "location_id": g,
                "year_id": year,
                "count": total,
            }
        )
    return rows


def read_old_inputs(  # pragma: no cover - parquet I/O
    locs: IntArray, years: IntArray, n_draws: int
) -> dict[str, dict[str, FloatArray]]:
    """The 2025 run's GDP and DAH per capita (draw 0; they do not vary by draw) and suitability by draw."""
    out = {}
    for ssp in SSPS:
        suit = np.full((locs.size, years.size, n_draws), np.nan)
        x: dict[str, FloatArray] = {}
        for d in range(n_draws):
            cols = ["location_id", "year_id", OLD_COLUMNS["suit"]]
            if d == 0:
                cols += [OLD_COLUMNS["gdp"], OLD_COLUMNS["dah"]]
            t = pq.read_table(
                str(OLD_INPUTS).format(ssp=ssp, draw=d),
                columns=cols,
                filters=[
                    ("location_id", "in", locs.tolist()),
                    ("year_id", "in", years.tolist()),
                ],
            ).to_pandas()
            li, yi = t.location_id.to_numpy(np.int64), t.year_id.to_numpy(np.int64)
            suit[:, :, d] = scatter_to_grid(
                li, yi, t[OLD_COLUMNS["suit"]].to_numpy(float), locs, years
            )
            if d == 0:
                for k in ("gdp", "dah"):
                    x[k] = scatter_to_grid(
                        li, yi, t[OLD_COLUMNS[k]].to_numpy(float), locs, years
                    )
        x["suit"] = suit
        out[ssp] = x
        click.echo(f"  read 2025 inputs {ssp}")
    return out


def complete_rows(
    inputs: dict[str, dict[str, FloatArray]], first_year_index: int
) -> npt.NDArray[np.bool_]:
    """Rows (admin-2 units) whose inputs are finite in every SSP, input and draw from ``first_year_index`` on."""
    ok = None
    for x in inputs.values():
        for v in x.values():
            tail = v[:, first_year_index:]
            fin = np.isfinite(tail).reshape(tail.shape[0], -1).all(axis=1)
            ok = fin if ok is None else ok & fin
    if ok is None:
        msg = "no inputs"
        raise ValueError(msg)
    return np.asarray(ok, dtype=bool)


def counts(  # noqa: PLR0913 - the rebuild, one year's pieces, the mix and the count frame
    rebuild: Rebuild,
    parts: dict[str, dict[str, FloatArray]],
    *,
    measure: str,
    src: dict[str, str],
    population: FloatArray,
    mask: npt.NDArray[np.bool_],
    factor: FloatArray,
) -> FloatArray:
    """Draw-mean count per admin-2 for one input mix, times the vaccine factor; masked outcomes stay NaN."""
    c = np.exp(rebuild.log_rate(parts, measure, src)) * population[:, None]
    c[mask] = np.nan
    out: FloatArray = c.mean(axis=1) * factor
    return out


@dataclass
class VariantRun:
    """Both input sets on the shared admin-2 units, with everything each year's variants need."""

    cur: Rebuild
    old: Rebuild
    forecasts: dict[
        str, dict[str, FloatArray]
    ]  # saved Baseline forecast, shared rows only
    held_forecasts: dict[
        str, dict[str, FloatArray]
    ]  # saved gdppc_hold2023 forecast, shared rows only
    years: IntArray
    population: FloatArray
    groups: FloatArray
    factor: dict[str, FloatArray]
    worst: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(CHECKS.values(), 0.0)
    )

    def _count(  # noqa: PLR0913 - the rebuild, its pieces and the cell (measure, ssp, mix, year)
        self,
        rb: Rebuild,
        parts: dict[str, dict[str, FloatArray]],
        *,
        m: str,
        ssp: str,
        src: dict[str, str],
        t: int,
    ) -> FloatArray:
        return counts(
            rb,
            parts,
            measure=m,
            src=src,
            population=self.population[:, t],
            mask=np.isnan(self.forecasts[ssp][m][:, t]),
            factor=self.factor[m][:, t],
        )

    def _check(self, name: str, ssp: str, m: str, t: int, c: FloatArray) -> None:
        ref_lr = (self.forecasts if name == "hold:none" else self.held_forecasts)[ssp][
            m
        ][:, t]
        ref = (np.exp(ref_lr) * self.population[:, t][:, None]).mean(
            axis=1
        ) * self.factor[m][:, t]
        self.worst[CHECKS[name]] = max(
            self.worst[CHECKS[name]], float(abs(np.nansum(c) / np.nansum(ref) - 1))
        )

    def year_rows(self, t: int) -> list[dict[str, object]]:
        """Every variant's global and super-region counts in year index ``t``; records the checks as it goes."""
        year, rows = int(self.years[t]), []
        for source, rb in (("current", self.cur), ("2025", self.old)):
            parts = rb.year_parts(t)
            for name, held in hold_variants():
                for ssp in SSPS:
                    one = {"x": hold_pieces(parts[ssp], held)}
                    for m in MEASURES:
                        c = self._count(
                            rb,
                            one,
                            m=m,
                            ssp=ssp,
                            src=dict.fromkeys(INPUT_FACTORS, "x"),
                            t=t,
                        )
                        rows += group_rows(
                            c,
                            self.groups,
                            year=year,
                            source=source,
                            variant=name,
                            ssp=ssp,
                            measure=m,
                        )
                        if source == "current" and name in CHECKS:
                            self._check(name, ssp, m, t, c)
            if source == "current":
                for ref_ssp in SSPS:
                    for ssp in SSPS:
                        src = {"gdp": ref_ssp, "suitability": ssp, "wind": ssp}
                        for m in MEASURES:
                            c = self._count(rb, parts, m=m, ssp=ssp, src=src, t=t)
                            rows += group_rows(
                                c,
                                self.groups,
                                year=year,
                                source=source,
                                variant=f"gdp_from:{ref_ssp}",
                                ssp=ssp,
                                measure=m,
                            )
        return rows

    def population_rows(self, t: int) -> list[dict[str, object]]:
        out = []
        for g in (1, *np.unique(self.groups).astype(int).tolist()):
            sel = np.ones(self.groups.size, bool) if g == 1 else self.groups == g
            out.append(
                {
                    "location_id": g,
                    "year_id": int(self.years[t]),
                    "population": float(self.population[sel, t].sum()),
                }
            )
        return out

    def trajectories(self, weights: FloatArray) -> list[dict[str, object]]:
        """Each input per input set, SSP, super-region and year, weighted by ``weights`` (one per admin-2)."""
        out: list[dict[str, object]] = []
        for source, rb in (("current", self.cur), ("2025", self.old)):
            for ssp in SSPS:
                x = rb.inputs[ssp]
                for cov, arr in (
                    ("gdppc", x["gdp"]),
                    ("dah_pc", x["dah"]),
                    ("suitability_days", np.nanmean(x["suit"], axis=2)),
                ):
                    for g, series in by_group(arr, weights, self.groups).items():
                        out += [
                            {
                                "source": source,
                                "covariate": cov,
                                "ssp": ssp,
                                "location_id": int(g),
                                "year_id": int(y),
                                "value": float(v),
                            }
                            for y, v in zip(self.years, series, strict=True)
                        ]
        return out


def prepare(  # pragma: no cover - reads every input
    spec: str,
    *,
    forecast_label: str,
    gdp_hold_label: str,
    term_root: Path,
    n_draws: int,
) -> tuple[VariantRun, IntArray, dict[str, int]]:
    """Load this run, the 2025 inputs on the shared admin-2 units, the vaccine factors and the hold-arm forecast."""
    rebuild, forecasts, locs, years, population = load_run(
        spec,
        forecast_label=forecast_label,
        term_root=term_root,
        ssps=list(SSPS),
        dah="Baseline",
        rake_year=RAKE_YEAR,
    )
    if rebuild.wind:
        msg = "the 2025 inputs carry no wind speed; run this for a model without wind"
        raise click.ClickException(msg)
    old_ids = (
        pq.read_table(
            str(OLD_INPUTS).format(ssp=SSPS[0], draw=0), columns=["location_id"]
        )
        .column(0)
        .to_numpy()
    )
    rows = np.flatnonzero(np.isin(locs, np.unique(old_ids)))
    old_inputs = read_old_inputs(locs[rows], years, n_draws)
    # units whose 2025 inputs are complete over the forecast years; NaN inputs would drop out of the sums silently
    keep = complete_rows(old_inputs, first_year_index=rebuild.ry)
    old_inputs = {s: {k: v[keep] for k, v in x.items()} for s, x in old_inputs.items()}
    counts_info = {
        "n_admin2_forecast": int(locs.size),
        "n_admin2_in_2025_inputs": int(rows.size),
        "n_admin2_incomplete_2025_inputs": int((~keep).sum()),
    }
    rows = rows[keep]
    m_locs = locs[rows]
    cur = subset_rebuild(rebuild, rows)
    old = Rebuild(
        grids=cur.grids,
        slopes=cur.slopes,
        inputs=old_inputs,
        l_obs=cur.l_obs,
        obs_log=cur.obs_log,
        ry=cur.ry,
    )
    fac_long = read_factors(spec, [int(y) for y in years[cur.ry + 1 :]])
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
    held = {ssp: read_forecast(gdp_hold_label, ssp, "Baseline")[2] for ssp in SSPS}
    run = VariantRun(
        cur=cur,
        old=old,
        forecasts={
            s: {m: v[rows] for m, v in fc.items()} for s, fc in forecasts.items()
        },
        held_forecasts={
            s: {m: v[rows] for m, v in fc.items()} for s, fc in held.items()
        },
        years=years,
        population=population[rows],
        groups=load_hierarchy()
        .set_index("location_id")
        .loc[m_locs, "super_region_id"]
        .to_numpy(float),
        factor=factor,
    )
    return run, m_locs, counts_info


@click.command()
@click.option("--spec", default="spec7426_best", show_default=True)
@click.option("--forecast-label", default="spec7426_best__base", show_default=True)
@click.option(
    "--gdp-hold-label", default="spec7426_best__gdppc_hold2023", show_default=True
)
@click.option("--term-root", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--n-draws", default=100, show_default=True)
def main(  # noqa: PLR0913, PLR0917 - click passes one argument per option
    spec: str,
    forecast_label: str,
    gdp_hold_label: str,
    term_root: Path,
    out_dir: Path,
    n_draws: int,
) -> None:  # pragma: no cover - orchestration over tested pieces
    """Write variant_counts.parquet, variant_population.parquet, input_trajectories.parquet and variant_checks.json."""
    run, m_locs, info = prepare(
        spec,
        forecast_label=forecast_label,
        gdp_hold_label=gdp_hold_label,
        term_root=term_root,
        n_draws=n_draws,
    )
    rows: list[dict[str, object]] = []
    pops: list[dict[str, object]] = []
    for t in range(run.cur.ry + 1, run.years.size):
        rows += run.year_rows(t)
        pops += run.population_rows(t)
        click.echo(f"  {run.years[t]} done")
    weights = (
        read_parquet_with_integer_ids(
            mbpc.MAL_RAKED_AA_READ_PATH / "aa_full_malaria_df.parquet",
            columns=["location_id", "year_id", "malaria_inc_count"],
            filters=[("year_id", "==", RAKE_YEAR)],
        )
        .set_index("location_id")
        .malaria_inc_count.reindex(m_locs)
        .to_numpy(float)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / "variant_counts.parquet", index=False)
    pd.DataFrame(pops).to_parquet(out_dir / "variant_population.parquet", index=False)
    pd.DataFrame(run.trajectories(weights)).to_parquet(
        out_dir / "input_trajectories.parquet", index=False
    )
    checks = {**run.worst, **info, "n_admin2_used": int(m_locs.size)}
    (out_dir / "variant_checks.json").write_text(json.dumps(checks, indent=2))
    click.echo(f"checks {checks}")


if __name__ == "__main__":
    main()
