"""Verify a combined draw-dimensioned netCDF reproduces the per-draw originals EXACTLY.

Safety gate before deleting the per-draw files. The combine is a pure float32
copy (no transform), so for any draw d the combined file's [..., draw=d] slab
MUST be bit-identical to draw d's original prediction. For a sample of draws we
assert exactly that (and that the shared coords match). Any mismatch => a
transpose / draw-indexing bug => do NOT delete the originals.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

SRC_DIMS = ("location_id", "year_id", "age_group_id", "sex_id")
PRED_VAR = {"incidence": "{cause}_inc_count_pred", "mortality": "{cause}_mort_count_pred"}


def _orig_path(input_dir: Path, cause: str, measure: str, ssp: str, hold: str, d: int) -> Path:
    return Path(input_dir) / (
        f"as_{cause}_measure_{measure}_ssp_scenario_{ssp}"
        f"_draw_{d:03d}_with_predictions{hold}.nc"
    )


def verify(combined_path: Path, input_dir: Path, cause: str, measure: str,
           ssp: str, hold_suffix: str, check_draws: list[int]) -> bool:
    pred = PRED_VAR[measure].format(cause=cause)
    cds = xr.open_dataset(combined_path)

    # coord sanity vs the first checked draw's original
    with xr.open_dataset(_orig_path(input_dir, cause, measure, ssp, hold_suffix, check_draws[0])) as o0:
        for dim in SRC_DIMS:
            if not np.array_equal(cds[dim].values, o0[dim].values):
                raise ValueError(f"coord {dim!r} differs between combined and original")

    all_ok = True
    for d in check_draws:
        with xr.open_dataset(_orig_path(input_dir, cause, measure, ssp, hold_suffix, d)) as od:
            a = od[pred].transpose(*SRC_DIMS).values
        b = cds[pred].sel(draw=d).transpose(*SRC_DIMS).values
        same = (a == b) | (np.isnan(a) & np.isnan(b))     # exact, NaN-aware
        n_bad = int((~same).sum())
        print(f"  draw {d:03d}: exact={n_bad == 0}  mismatched_cells={n_bad}/{a.size}")
        all_ok = all_ok and (n_bad == 0)

    cds.close()
    print("\nVALUE-EQUALITY:",
          "PASS — combined == originals (safe to delete those originals)" if all_ok
          else "FAIL — do NOT delete originals")
    return all_ok


if __name__ == "__main__":
    DENGUE_DIR = "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/dengue/lsae_1209/20250811"
    p = argparse.ArgumentParser(description="Verify a combined draw-dim file == its per-draw originals.")
    p.add_argument("--combined", required=True)
    p.add_argument("--cause", default="dengue")
    p.add_argument("--measure", default="incidence", choices=["incidence", "mortality"])
    p.add_argument("--ssp", default="ssp126")
    p.add_argument("--hold", default="_hold_urban")
    p.add_argument("--input_dir", default=DENGUE_DIR)
    p.add_argument("--check_draws", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = p.parse_args()
    ok = verify(Path(a.combined), Path(a.input_dir), a.cause, a.measure,
                a.ssp, a.hold, a.check_draws)
    raise SystemExit(0 if ok else 1)
