"""Combine per-draw age-sex forecast netCDFs into one draw-dimensioned file.

EXPERIMENT / TEST script — the point is to exercise reading + writing the
age-sex-draw-specific files. For now it does ONE (cause, measure, ssp, hold)
combine, incidence only, and writes the result into the SAME folder as the
inputs. It NEVER deletes or modifies the per-draw originals.

Memory model (the thing we're testing):
  The combined array (~45 GB at 100 draws) is built ON DISK, one draw at a
  time — never held whole in RAM. We create the full (loc, year, age, sex,
  draw) variable up front (metadata only), then loop draws: read one draw
  (~0.45 GB float32), write it into var[..., i], release it. Peak RAM ≈ one
  draw. The variable is chunked with draw=1 so each slab write hits its own
  HDF5 chunks (no read-modify-write).

  `population` is dropped — it's invariant across draws and read elsewhere
  when needed; it does not belong in these files.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import netCDF4 as nc4
import numpy as np
import xarray as xr

# Prediction variable name by measure (extend when we generalize past incidence).
PRED_VAR = {
    "incidence": "{cause}_inc_count_pred",
    "mortality": "{cause}_mort_count_pred",
}
# Source dim order (verified from the 20250811 dengue files).
SRC_DIMS = ("location_id", "year_id", "age_group_id", "sex_id")
LOC_CHUNK = 2000  # locations per HDF5 chunk; tune in the experiment


def _draw_path(input_dir: Path, cause: str, measure: str, ssp: str,
               hold_suffix: str, draw: int) -> Path:
    return input_dir / (
        f"as_{cause}_measure_{measure}_ssp_scenario_{ssp}"
        f"_draw_{draw:03d}_with_predictions{hold_suffix}.nc"
    )


def combine_as_draws(
    cause: str,
    measure: str,
    ssp_scenario: str,
    hold_suffix: str,
    input_dir: Path,
    output_dir: Path | None = None,
    n_draws: int = 100,
    complevel: int = 4,
) -> Path:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    draws = list(range(n_draws))
    pred_var = PRED_VAR[measure].format(cause=cause)

    # ── 1. resolve inputs; require ALL draws present (no silent partial combine) ──
    paths = {d: _draw_path(input_dir, cause, measure, ssp_scenario, hold_suffix, d) for d in draws}
    missing = [d for d, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} draw file(s) missing for {measure}/{ssp_scenario}{hold_suffix}: "
            f"first few = {missing[:5]}"
        )

    # ── 2. reference grid from draw 0 (metadata only) ──
    with xr.open_dataset(paths[draws[0]]) as ds0:
        if pred_var not in ds0.data_vars:
            raise KeyError(f"{pred_var!r} not in {paths[draws[0]].name}; vars={list(ds0.data_vars)}")
        ref = {d: ds0[d].values for d in SRC_DIMS}
    shape4 = tuple(len(ref[d]) for d in SRC_DIMS)
    print(f"Grid: { {d: len(ref[d]) for d in SRC_DIMS} }  x draw={n_draws}")
    print(f"Per-draw slab: {np.prod(shape4) * 4 / 1e9:.2f} GB (float32); "
          f"combined on disk ≈ {np.prod(shape4) * 4 * n_draws / 1e9:.1f} GB uncompressed")

    # ── 3. create the output file structure ON DISK (empty; ~zero RAM) ──
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (
        f"as_{cause}_measure_{measure}_ssp_scenario_{ssp_scenario}"
        f"_with_predictions{hold_suffix}.nc"
    )
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {out_path}; remove it first.")
    fd, tmp_name = tempfile.mkstemp(suffix=".nc", dir=str(output_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        out = nc4.Dataset(tmp_path, "w", format="NETCDF4")
        # dims + coord vars (coords are small)
        coord_dtype = {"location_id": "i4", "year_id": "i2", "age_group_id": "i2", "sex_id": "i1"}
        for d in SRC_DIMS:
            out.createDimension(d, len(ref[d]))
            cv = out.createVariable(d, coord_dtype[d], (d,))
            cv[:] = ref[d].astype(coord_dtype[d])
        out.createDimension("draw", n_draws)
        dv = out.createVariable("draw", "i2", ("draw",))
        dv[:] = np.asarray(draws, dtype="i2")

        # data var: (loc, year, age, sex, draw) float32, draw chunked to 1 for per-draw writes
        chunks = (min(LOC_CHUNK, shape4[0]), shape4[1], shape4[2], shape4[3], 1)
        var = out.createVariable(
            pred_var, "f4", (*SRC_DIMS, "draw"),
            zlib=True, complevel=complevel, shuffle=True, chunksizes=chunks, fill_value=np.nan,
        )

        # ── 4. per-draw incremental write (peak RAM ≈ one draw) ──
        for i, d in enumerate(draws):
            with xr.open_dataset(paths[d]) as dsd:
                for dim in SRC_DIMS:                              # square-data check vs draw 0
                    if not np.array_equal(dsd[dim].values, ref[dim]):
                        raise ValueError(f"draw {d}: coord {dim!r} differs from draw 0 — not square.")
                arr = dsd[pred_var].transpose(*SRC_DIMS).values.astype("f4")  # ~0.45 GB
            var[:, :, :, :, i] = arr                              # write this draw's slab to disk
            del arr
            if i % 10 == 0:
                print(f"  wrote draw {d:03d} ({i + 1}/{n_draws})")
        out.close()

        # ── 5. metadata-only validation (no full read-back) ──
        with nc4.Dataset(tmp_path) as chk:
            got = tuple(chk.variables[pred_var].shape)
            want = (*shape4, n_draws)
            if got != want:
                raise ValueError(f"shape mismatch after write: {got} vs {want}")

        os.replace(tmp_path, out_path)        # atomic; both are new files I created
        os.chmod(out_path, 0o775)
        size_gb = out_path.stat().st_size / 1e9
        print(f"\nWrote {out_path}  ({size_gb:.2f} GB compressed)")
        return out_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink()                 # clean the temp on any failure


if __name__ == "__main__":
    DENGUE_DIR = "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/dengue/lsae_1209/20250811"
    DEFAULT_HOLDS = ["", "_hold_gdppc", "_hold_suitability", "_hold_urban"]  # "" = base/no-hold
    p = argparse.ArgumentParser(description="Combine per-draw age-sex netCDFs into one draw-dim file per (ssp, hold).")
    p.add_argument("--cause", default="dengue")
    p.add_argument("--measure", default="incidence", choices=["incidence", "mortality"])
    p.add_argument("--ssps", nargs="+", default=["ssp126", "ssp245", "ssp585"])
    p.add_argument("--holds", nargs="+", default=DEFAULT_HOLDS,
                   help="hold suffixes; pass 'base' or 'none' for the no-hold run ('').")
    p.add_argument("--input_dir", default=DENGUE_DIR)
    p.add_argument("--output_dir", default=None, help="defaults to --input_dir (same folder)")
    p.add_argument("--n_draws", type=int, default=100)
    p.add_argument("--complevel", type=int, default=4, help="zlib level: lower = faster write, larger file")
    a = p.parse_args()

    holds  = ["" if h in ("base", "none") else h for h in a.holds]
    combos = [(s, h) for s in a.ssps for h in holds]
    print(f"Running {len(combos)} combo(s): ssps={a.ssps} x holds={[h or '(base)' for h in holds]}")
    for s, h in combos:
        print(f"\n=== ssp={s} hold={h or '(base)'} ===")
        combine_as_draws(a.cause, a.measure, s, h,
                         Path(a.input_dir), a.output_dir, a.n_draws, a.complevel)
