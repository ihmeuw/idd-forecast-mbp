"""Create a malaria model-selection run dir + spec_table.parquet from the typed spec design.

Replaces OLD_build_malaria_neighborhood_specs.r. Enumerates the
``idd_tools.model_selection`` ModelSpace defined in ``malaria_spec_design`` and writes
``spec_table.parquet`` (spec_index, n_smooths, n_scams, formula_text) plus the groups it
enumerated (``spec_design.json``, read back by rank / gate / report) into a fresh dated run
dir. There is deliberately NO ``neighborhood_specs.rds`` — the R worker fits ``formula_text``
from ``spec_table`` (the single source of truth).

    python build_malaria_spec_design.py            # -> <OUTPUT_ROOT>/<YYYYMMDD>_efs/spec_table.parquet
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.select.malaria_spec_design import (
    CURRENT_DESIGN,
    SpecDesign,
    write_design,
)

# The selection node (constants.MAL_SELECTION_NODE): one dated run dir per spec build.
OUTPUT_ROOT = mbpc.MAL_SELECTION_NODE


def spec_table_frame(design: SpecDesign = CURRENT_DESIGN) -> pd.DataFrame:
    """Enumerate the universe -> the exact spec_table the orchestrator + worker consume."""
    configs = design.universe.configs
    rows = [
        {
            "spec_index": i + 1,
            "n_smooths": design.n_smooths(c),
            "n_scams": design.n_scams(c),
            "formula_text": design.formula_text(c),
        }
        for i, c in enumerate(configs)
    ]
    return pd.DataFrame(rows).astype(
        {
            "spec_index": "int32",
            "n_smooths": "int32",
            "n_scams": "int32",
            "formula_text": "string",
        }
    )


@click.command()
@click.option("--output-root", default=str(OUTPUT_ROOT), show_default=True)
@click.option(
    "--tag", default="efs", show_default=True, help="run-dir suffix: YYYYMMDD_<tag>"
)
def main(output_root: str, tag: str) -> None:
    root = Path(output_root)
    run_dir = root / f"{datetime.now().astimezone():%Y%m%d}_{tag}"
    v = 2
    while run_dir.exists():
        run_dir = root / f"{datetime.now().astimezone():%Y%m%d}_{tag}_v{v}"
        v += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    frame = spec_table_frame()
    frame.to_parquet(run_dir / "spec_table.parquet", index=False)
    write_design(CURRENT_DESIGN, run_dir)
    click.echo(f"wrote {len(frame)} specs -> {run_dir}/spec_table.parquet")
    click.echo(
        f"n_smooths dist: {frame.n_smooths.value_counts().sort_index().to_dict()}"
    )
    click.echo(str(run_dir))


if __name__ == "__main__":
    main()
