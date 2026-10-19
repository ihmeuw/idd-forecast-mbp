"""All-age counts with one scenario input held at its rake-year value, rebuilt from the term grids.

Mirrors the hold arms, so this run can be set beside the 2025 first-submission run input by input: GDP held (as in
this run's gdppc_hold2023 forecast arm and the 2025 run's hold_gdppc) and suitability held (as in the 2025 run's
hold_suitability; this run has no suitability-held forecast). Each SSP keeps its other inputs. Draw-mean counts per
admin-2 and year with no vaccine protection (forecast rate x population), as ``decompose_scenario_gap.py``.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

from idd_forecast_mbp.lib.processing.scenario_gap import (
    INPUT_FACTORS,
    LOG_RATE_VARS,
    hold_pieces,
    load_run,
)


@click.command()
@click.option("--spec", required=True)
@click.option("--forecast-label", required=True)
@click.option("--term-root", type=click.Path(path_type=Path), required=True)
@click.option("--out-file", type=click.Path(path_type=Path), required=True)
@click.option(
    "--held", "helds", multiple=True, default=("gdp", "suitability"), show_default=True
)
def main(
    spec: str,
    forecast_label: str,
    term_root: Path,
    out_file: Path,
    helds: tuple[str, ...],
) -> None:  # pragma: no cover - orchestration over tested pieces
    """Write ``[arm, measure, location_id, year_id, count]`` with arm = ``<ssp>:<held>_held``."""
    ssps = ["ssp126", "ssp245", "ssp585"]
    rebuild, forecasts, locs, years, population = load_run(
        spec,
        forecast_label=forecast_label,
        term_root=term_root,
        ssps=ssps,
        dah="Baseline",
        rake_year=2023,
    )
    rows = []
    for t in range(rebuild.ry + 1, years.size):
        parts = rebuild.year_parts(t)
        for ssp in ssps:
            for held in helds:
                one = {"held": hold_pieces(parts[ssp], (held,))}
                for m in LOG_RATE_VARS:
                    lr = rebuild.log_rate(one, m, dict.fromkeys(INPUT_FACTORS, "held"))
                    c = np.exp(lr) * population[:, t][:, None]
                    c[np.isnan(forecasts[ssp][m][:, t])] = np.nan
                    rows.append(
                        pd.DataFrame(
                            {
                                "arm": f"{ssp}:{held}_held",
                                "measure": m,
                                "location_id": locs,
                                "year_id": int(years[t]),
                                "count": c.mean(axis=1),
                            }
                        )
                    )
        click.echo(f"  {years[t]} done")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(out_file, index=False)
    click.echo(f"wrote {out_file}")


if __name__ == "__main__":
    main()
