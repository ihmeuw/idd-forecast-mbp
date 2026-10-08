"""Split the all-age count gap between two SSP forecasts into the inputs that differ between them.

For one model (term grids from ``export_malaria_term_grids.R`` plus its Baseline-DAH forecast), every admin-2,
draw and year after the rake year: rebuild the anchored log rates for each mix of GDP / suitability (/ wind for a
model with wind) taken from scenario A or B (``scenario_gap.log_rate_from_terms``), check that the all-A and all-B
mixes reproduce the saved forecasts, and write the Shapley parts of ``count_A - count_B`` (draw means) per
admin-2 and year, plus roll-ups to country, region, super-region and global.

Counts here are forecast rate x population with no vaccine protection. Population and DAH per capita are the same
in every SSP (checked at run time), so they contribute nothing to the gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.processing.scenario_gap import (
    LOG_RATE_VARS,
    ROLL_UP,
    check_rebuild,
    load_run,
    pair_parts,
    roll_up,
)


@click.command()
@click.option(
    "--spec",
    required=True,
    help="model key, e.g. spec7426_best (names the term-grid folder)",
)
@click.option(
    "--forecast-label",
    required=True,
    help="forecast_outputs label, e.g. spec7426_best__base",
)
@click.option(
    "--term-root",
    type=click.Path(path_type=Path),
    required=True,
    help="folder holding <spec>/term_grids.parquet",
)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--pair",
    "pairs",
    multiple=True,
    default=("ssp245:ssp126", "ssp245:ssp585", "ssp585:ssp126"),
    show_default=True,
    help="A:B; the gap is A minus B",
)
@click.option("--dah-scenario", default="Baseline", show_default=True)
@click.option("--rake-year", default=2023, show_default=True)
@click.option(
    "--last-year",
    default=2100,
    show_default=True,
    help="stop after this year (smoke runs)",
)
@click.option(
    "--check-tol",
    default=1e-4,
    show_default=True,
    help="max |log-rate| difference allowed between the rebuilt pure mixes and the saved forecast",
)
def main(  # noqa: PLR0913, PLR0917 - click passes one argument per option
    spec: str,
    forecast_label: str,
    term_root: Path,
    out_dir: Path,
    pairs: tuple[str, ...],
    dah_scenario: str,
    rake_year: int,
    last_year: int,
    check_tol: float,
) -> None:  # pragma: no cover - orchestration over the tested pieces
    """Shapley split of each SSP pair's all-age count gap into GDP / suitability (/ wind)."""
    ssps = sorted({s for p in pairs for s in p.split(":")})
    rebuild, forecasts, locs, years, population = load_run(
        spec,
        forecast_label=forecast_label,
        term_root=term_root,
        ssps=ssps,
        dah=dah_scenario,
        rake_year=rake_year,
    )
    factors = ["gdp", "suitability", *(["wind"] if rebuild.wind else [])]
    rows, checks = [], []
    for t in range(rebuild.ry + 1, int(np.flatnonzero(years == last_year)[0]) + 1):
        parts = rebuild.year_parts(t)
        checks += [
            {**c, "year_id": int(years[t])}
            for c in check_rebuild(rebuild, parts, forecasts, t)
        ]
        for pair in pairs:
            a, b = pair.split(":")
            for m in LOG_RATE_VARS:
                cols = pair_parts(
                    rebuild,
                    parts,
                    measure=m,
                    a=a,
                    b=b,
                    population=population[:, t],
                    mask=np.isnan(forecasts[a][m][:, t]),
                    factors=factors,
                )
                rows.append(
                    pd.DataFrame(
                        {
                            "pair": pair,
                            "measure": m,
                            "location_id": locs,
                            "year_id": int(years[t]),
                            **cols,
                        }
                    )
                )
        click.echo(f"  {years[t]} done")

    check = pd.DataFrame(checks)
    worst = float(check.max_abs_log_diff.max())
    admin2 = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    admin2.to_parquet(out_dir / "gap_parts_admin2.parquet", index=False)
    check.to_parquet(out_dir / "rebuild_check.parquet", index=False)
    hier = load_hierarchy().set_index("location_id")
    keys = hier.loc[locs, [c for c, _ in ROLL_UP]].reset_index()
    agg = roll_up(admin2, keys, ["count_a", "count_b", *[f"part_{f}" for f in factors]])
    agg["location_name"] = agg.location_id.map(hier.location_name).fillna("Global")
    agg.to_parquet(out_dir / "gap_parts_aggregated.parquet", index=False)
    run = {
        "spec": spec,
        "forecast_label": forecast_label,
        "pairs": list(pairs),
        "factors": factors,
        "dah_scenario": dah_scenario,
        "rake_year": rake_year,
        "rebuild_max_abs_log_diff": worst,
        "check_tol": check_tol,
        "n_admin2": int(locs.size),
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2))
    click.echo(f"rebuild max |log diff| {worst:.3g} (tol {check_tol})")
    if worst > check_tol:
        msg = "rebuilt pure mixes do not reproduce the saved forecast; parts not trustworthy"
        raise click.ClickException(msg)


if __name__ == "__main__":
    main()
