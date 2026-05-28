"""
Output comparison utility for stage migration testing.

Usage:
    python compare_outputs.py --ref_dir <production_dir> --test_dir <test_dir> [--rtol 1e-5]

Compares every .parquet and .nc file in test_dir against the matching file
in ref_dir. Reports PASS / FAIL / MISSING for each file.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def compare_parquet(ref_path: Path, test_path: Path, rtol: float = 1e-5) -> tuple[bool, str]:
    ref = pd.read_parquet(ref_path)
    test = pd.read_parquet(test_path)

    if ref.shape != test.shape:
        return False, f"shape mismatch: ref={ref.shape} test={test.shape}"

    if set(ref.columns) != set(test.columns):
        extra = set(test.columns) - set(ref.columns)
        missing = set(ref.columns) - set(test.columns)
        return False, f"column mismatch: extra={extra} missing={missing}"

    # Reorder test columns to match ref
    test = test[ref.columns]

    mismatches = []
    for col in ref.columns:
        if pd.api.types.is_float_dtype(ref[col]):
            if not np.allclose(ref[col].fillna(0), test[col].fillna(0), rtol=rtol, equal_nan=True):
                max_diff = (ref[col] - test[col]).abs().max()
                mismatches.append(f"{col}: max_abs_diff={max_diff:.3e}")
        else:
            if not ref[col].equals(test[col]):
                n_diff = (ref[col] != test[col]).sum()
                mismatches.append(f"{col}: {n_diff} values differ")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, "identical"


def compare_netcdf(ref_path: Path, test_path: Path, rtol: float = 1e-5) -> tuple[bool, str]:
    ref = xr.open_dataset(ref_path)
    test = xr.open_dataset(test_path)

    ref_vars = set(ref.data_vars)
    test_vars = set(test.data_vars)
    if ref_vars != test_vars:
        return False, f"variable mismatch: ref={ref_vars} test={test_vars}"

    mismatches = []
    for var in ref.data_vars:
        rv = ref[var].values
        tv = test[var].values
        if rv.shape != tv.shape:
            mismatches.append(f"{var}: shape mismatch ref={rv.shape} test={tv.shape}")
            continue
        if not np.issubdtype(rv.dtype, np.number):
            if not np.array_equal(rv, tv):
                n_diff = int(np.sum(rv != tv))
                mismatches.append(f"{var}: {n_diff} string values differ")
            continue
        rv_f, tv_f = rv.astype(float), tv.astype(float)
        if not np.allclose(np.nan_to_num(rv_f), np.nan_to_num(tv_f), rtol=rtol):
            max_diff = np.nanmax(np.abs(rv_f - tv_f))
            mismatches.append(f"{var}: max_abs_diff={max_diff:.3e}")

    ref.close()
    test.close()

    if mismatches:
        return False, "; ".join(mismatches)
    return True, "identical"


def run_comparison(ref_dir: Path, test_dir: Path, rtol: float = 1e-5) -> bool:
    test_files = sorted(list(test_dir.glob("**/*.parquet")) + list(test_dir.glob("**/*.nc")))

    if not test_files:
        print(f"No output files found in {test_dir}")
        return False

    all_passed = True
    results = []

    for test_path in test_files:
        rel = test_path.relative_to(test_dir)
        ref_path = ref_dir / rel

        if not ref_path.exists():
            results.append(("MISSING_REF", str(rel), "no reference file"))
            all_passed = False
            continue

        suffix = test_path.suffix
        try:
            if suffix == ".parquet":
                passed, msg = compare_parquet(ref_path, test_path, rtol)
            elif suffix == ".nc":
                passed, msg = compare_netcdf(ref_path, test_path, rtol)
            else:
                results.append(("SKIP", str(rel), "unknown format"))
                continue
        except Exception as e:
            results.append(("ERROR", str(rel), str(e)))
            all_passed = False
            continue

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        results.append((status, str(rel), msg))

    # Print results
    width = max(len(r[1]) for r in results) + 2
    print(f"\n{'Status':<8} {'File':<{width}} Detail")
    print("-" * (8 + width + 40))
    for status, name, detail in results:
        marker = "✅" if status == "PASS" else ("⚠️ " if status == "SKIP" else "❌")
        print(f"{marker} {status:<6} {name:<{width}} {detail}")

    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_fail = sum(1 for r in results if r[0] == "FAIL")
    n_err  = sum(1 for r in results if r[0] in ("ERROR", "MISSING_REF"))
    print(f"\nSummary: {n_pass} passed, {n_fail} failed, {n_err} errors/missing")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", required=True, type=Path)
    parser.add_argument("--test_dir", required=True, type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    ok = run_comparison(args.ref_dir, args.test_dir, args.rtol)
    sys.exit(0 if ok else 1)
