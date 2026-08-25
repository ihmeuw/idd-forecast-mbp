"""
Run the whole malaria vaccine chain for one assumption set.

The point of this script is that "run it again under different assumptions" is
one command, and that the assumptions are named rather than implied. Every stage
is invoked in-process so a failure stops the chain instead of leaving a
half-updated set of outputs.

Chain:
    1. build VE curves from VE_ANCHORS.yaml        (optional; --skip-curves)
    2. coverage -> cohort protection by age group
    3. protection -> scenario totals + draw-level output
    4. figures

The assumption set is (ve_variant, product_scenario, age_reference, prelag) and
is echoed at the start of every run. Outputs are keyed so runs under different
assumptions coexist rather than overwrite.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.lib.processing.vaccine_cohort_fractions import (
    AGE_REFERENCE_PHI,
    DEFAULT_AGE_REFERENCE,
)

STAGE_DIR = Path(__file__).parent
VIZ_DIR = STAGE_DIR.parent / "08_visualization"
PRELAG = {"backcast": "--backcast-prelag-dose3",
          "zero": "--zero-prelag-dose4",
          "none": None}


def _load(path: Path):
    """Import a numbered stage script by path (the dirs are not packages)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(label: str, path: Path, argv: list[str]) -> None:
    print(f"\n{'=' * 78}\n{label}\n  {path.name} {' '.join(argv)}\n{'=' * 78}", flush=True)
    started = time.perf_counter()
    _load(path).main(argv)
    print(f"-- {label} done in {time.perf_counter() - started:.1f}s", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ve-variant", choices=rfc.VE_VARIANTS, default="loglinear_severe0")
    p.add_argument("--product-scenario", choices=rfc.PRODUCT_SCENARIOS, default="projected")
    p.add_argument("--age-reference", choices=sorted(AGE_REFERENCE_PHI),
                   default=DEFAULT_AGE_REFERENCE)
    p.add_argument("--prelag", choices=sorted(PRELAG), default="backcast",
                   help="How to treat dose_4 whose dose-3 antecedent predates the "
                        "coverage series. 'backcast' recovers it, 'zero' discards it.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where impact tables and figures are written.")
    p.add_argument("--skip-curves", action="store_true",
                   help="Reuse the VE curves already in the processed node.")
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("assumption set")
    for k in ("ve_variant", "product_scenario", "age_reference", "prelag"):
        print(f"    {k:18s} {getattr(args, k)}")
    print(f"    {'out_dir':18s} {args.out_dir}")

    if not args.skip_curves:
        _run("1/4  build VE curves",
             STAGE_DIR.parent / "02_data_prep" / "09_build_vaccine_efficacy_curves.py", [])

    cohort_argv = ["--ve-variant", args.ve_variant,
                   "--product-scenario", args.product_scenario,
                   "--age-reference", args.age_reference]
    if PRELAG[args.prelag]:
        cohort_argv.append(PRELAG[args.prelag])
    _run("2/4  coverage -> cohort protection",
         STAGE_DIR / "apply_vaccine_coverage_to_population.py", cohort_argv)

    _run("3/4  protection -> scenario totals",
         STAGE_DIR / "vaccine_impact_scenarios.py",
         ["--ve-variant", args.ve_variant,
          "--product-scenario", args.product_scenario,
          "--out-dir", str(args.out_dir)])

    if not args.skip_figures:
        _run("4/4  figures: impact", VIZ_DIR / "plot_vaccine_impact.py",
             ["--summary-dir", str(args.out_dir)])
        _run("4/4  figures: doses", VIZ_DIR / "plot_vaccine_doses.py",
             ["--out-dir", str(args.out_dir)])
        for geography in ("either", "global"):
            _run(f"4/4  figures: coverage ({geography})", VIZ_DIR / "plot_vaccine_coverage.py",
                 ["--geography", geography, "--product-scenario", args.product_scenario,
                  "--out-dir", str(args.out_dir)])

    print(f"\nPIPELINE COMPLETE in {time.perf_counter() - started:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
