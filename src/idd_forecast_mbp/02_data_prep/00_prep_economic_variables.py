"""Orchestrator for the four economic-variable artifact builders.

Each of {gdppc, ldipc, med_consumppc, dah} has its own `make_*_df.py` helper
in this directory. This script picks any subset and runs them in dependency
order. Pass `--variables` to select which (default: all four).

  gdppc          — FGH GDP per capita (RCP-string scenarios).
  ldipc          — log disposable income per capita (RCP-string scenarios).
  med_consumppc  — median consumption per capita (RCP-string scenarios).
  dah            — development assistance for health. **Depends on
                   _A02_HIERARCHY + _A02_POPULATION being current** (consumes
                   both); the other three have no pipeline-internal deps.

Examples:
  python 00_prep_economic_variables.py                              # all four
  python 00_prep_economic_variables.py --variables gdppc dah        # subset
  python 00_prep_economic_variables.py --variables med_consumppc    # one
"""
from __future__ import annotations

import argparse

# Each helper exposes `main()`; importing this module does not run it.
from idd_forecast_mbp.lib.processing import locations as _locations  # noqa: F401  (smoke-import for early-fail)

# Import the four helpers. The make_*_df modules are siblings in this dir.
# Importing by relative path so the orchestrator works whether invoked as
# `python 00_prep_economic_variables.py` or imported as a module.
import importlib.util
from pathlib import Path

_HERE = Path(__file__).parent

def _load(stem: str):
    spec = importlib.util.spec_from_file_location(stem, _HERE / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Order matters: dah depends on hierarchy + population, so it runs last so
# the other three (which have no pipeline-internal deps) get refreshed first.
_BUILDERS: dict[str, str] = {
    "gdppc":         "make_gdppc_df",
    "ldipc":         "make_ldipc_df",
    "med_consumppc": "make_med_consumppc_df",
    "dah":           "make_dah_df",
}
ALL_VARIABLES = list(_BUILDERS)


def main(variables: list[str] | None = None) -> None:
    targets = variables if variables else ALL_VARIABLES
    unknown = set(targets) - set(_BUILDERS)
    if unknown:
        raise ValueError(
            f"Unknown variable(s): {sorted(unknown)}. "
            f"Known: {ALL_VARIABLES}"
        )
    # Preserve declared order regardless of CLI argument order
    ordered = [v for v in ALL_VARIABLES if v in targets]
    for v in ordered:
        print(f"\n=== {v} → {_BUILDERS[v]}.py ===")
        _load(_BUILDERS[v]).main()
    print(f"\nDone. Rebuilt: {ordered}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild any subset of the four economic-variable artifacts."
    )
    parser.add_argument(
        "--variables", "-v",
        nargs="+",
        choices=ALL_VARIABLES,
        default=None,
        help=f"Which variables to rebuild. Default: all of {ALL_VARIABLES}.",
    )
    args = parser.parse_args()
    main(variables=args.variables)
