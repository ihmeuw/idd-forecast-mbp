"""
Build monthly malaria vaccine-efficacy curves from VE_ANCHORS.yaml.

Construction from raw anchors is a data-prep step, so this sits in 02_data_prep/
with the other NN_build_*.py scripts rather than 00_pull_raw_data/, which holds
external fetchers only.

Raw -> processed boundary:
  * anchors are RAW      : src/idd_forecast_mbp/VE_ANCHORS.yaml, never written here
  * curves are PROCESSED : 02-processed_data/malaria_vaccine_efficacy/<RUN_DATE>/
                           with a `current` symlink via finish_stage

Every cell is validated against the loader contract before anything is written --
an invalid curve is a hard error, not a warning, because downstream protection
numbers would silently inherit the fault.
"""
import argparse

import yaml

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.lib.processing import vaccine_efficacy as ve
from idd_forecast_mbp.lib.versioning import finish_stage, write_path

EXPECTED_PRODUCTS = ("rtss", "r21")


def build_and_write(expected_products=EXPECTED_PRODUCTS) -> list:
    anchors = yaml.safe_load(rfc.VE_ANCHORS_PATH.read_text())
    schedule = anchors["schedule"]
    run_dir = write_path(rfc._A02_VE_CURVES)  # the node's working slot
    run_dir.mkdir(parents=True, exist_ok=True)

    cells = ve.build_all_cells(anchors)
    written = []
    for name, df in cells.items():
        ve.validate_ve_frame(
            df, expected_products=expected_products,
            dose3_age=schedule["dose3_age_months"],
            booster_age=schedule["booster_age_months"],
        )
        out = run_dir / f"ve_{name}.csv"
        df.to_csv(out, index=False)
        written.append(out)
        print(f"  {out.name}: {len(df):,} rows, products {sorted(df.vaccine.unique())}")

    print(f"default cell (from VE_ANCHORS.yaml): {anchors['build']['default_cell']}")
    finish_stage(rfc._A02_VE_CURVES)
    return written


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    return p.parse_args(argv)


def main(argv=None) -> list:
    args = parse_args(argv)
    written = build_and_write()
    print(f"wrote {len(written)} VE curve(s)")
    return written


if __name__ == "__main__":
    main()
