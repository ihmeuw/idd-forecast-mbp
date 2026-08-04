"""Create a malaria model-selection run dir + spec_table.parquet from the typed spec design.

Replaces OLD_build_malaria_neighborhood_specs.r. Enumerates the
``idd_tools.model_selection`` ModelSpace defined in ``malaria_spec_design`` and writes
``spec_table.parquet`` (spec_index, n_smooths, n_scams, formula_text) into a fresh dated run
dir. There is deliberately NO ``neighborhood_specs.rds`` — the R worker fits ``formula_text``
from ``spec_table`` (the single source of truth).

    python build_malaria_spec_design.py            # -> <OUTPUT_ROOT>/<YYYYMMDD>_efs/spec_table.parquet
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import click
import pandas as pd

from idd_forecast_mbp.select.malaria_spec_design import (
    build_universe,
    formula_text,
    n_scams,
    n_smooths,
)

# Same output root the R builder used (build_malaria_neighborhood_specs.r hardcoded it too).
OUTPUT_ROOT = Path("/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/scam_prelim/lsae_1285")


def spec_table_frame() -> pd.DataFrame:
    """Enumerate the universe -> the exact spec_table the orchestrator + worker consume."""
    configs = build_universe().configs
    rows = [
        {
            "spec_index": i + 1,
            "n_smooths": n_smooths(c),
            "n_scams": n_scams(c),
            "formula_text": formula_text(c),
        }
        for i, c in enumerate(configs)
    ]
    return pd.DataFrame(rows).astype(
        {"spec_index": "int32", "n_smooths": "int32", "n_scams": "int32", "formula_text": "string"}
    )


@click.command()
@click.option("--output-root", default=str(OUTPUT_ROOT), show_default=True)
@click.option("--tag", default="efs", show_default=True, help="run-dir suffix: YYYYMMDD_<tag>")
def main(output_root: str, tag: str) -> None:
    root = Path(output_root)
    run_dir = root / f"{date.today():%Y%m%d}_{tag}"
    v = 2
    while run_dir.exists():
        run_dir = root / f"{date.today():%Y%m%d}_{tag}_v{v}"
        v += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    frame = spec_table_frame()
    frame.to_parquet(run_dir / "spec_table.parquet", index=False)
    click.echo(f"wrote {len(frame)} specs -> {run_dir}/spec_table.parquet")
    click.echo(f"n_smooths dist: {frame.n_smooths.value_counts().sort_index().to_dict()}")
    click.echo(str(run_dir))


if __name__ == "__main__":
    main()
