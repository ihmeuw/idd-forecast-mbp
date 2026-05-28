"""Download Malaria Atlas Project (MAP) GeoTIFF rasters via WCS.

For each Pf/Pv covariate, queries the MAP WCS endpoint for the list of years
available in the given release, then downloads one GeoTIFF per year and saves
it to a release-versioned subdirectory under the rapidresponse processed-data
tree.

Output layout:
    {RR_PATH}/data/02-processed-data/{subdir}/{release}/
        {release}_Global_{stem}_{year}.tif

Idempotent: skips files that already exist on disk above MIN_VALID_SIZE.
Atomic writes via tmp → rename, so an interrupted run never leaves a partial
file at the canonical path.

Usage:
    python -m idd_forecast_mbp.00_pull_raw_data.pull_map_rasters
    python -m idd_forecast_mbp.00_pull_raw_data.pull_map_rasters --release 202406
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

from idd_forecast_mbp import constants as mbpc


GEOSERVER = "https://data.malariaatlas.org/geoserver/Malaria/ows"
BASE_PATH = mbpc.RR_PATH / "data" / "02-processed-data"
MIN_VALID_SIZE = 100 * 1024  # global rasters are MB-scale; smaller = broken

# (coverage_short, output_subdir, filename_stem)
COVARIATES: list[tuple[str, str, str]] = [
    ("Pf_Parasite_Rate",    "malaria-pfpr",                 "Pf_Parasite_Rate"),
    ("Pf_Incidence_Count",  "malaria-pf-incidence-count",   "Pf_Incidence_Count"),
    ("Pf_Incidence_Rate",   "malaria-pf-incidence-rate",    "Pf_Incidence_Rate"),
    ("Pf_Mortality_Count",  "malaria-pf-mortality-count",   "Pf_Mortality_Count"),
    ("Pf_Mortality_Rate",   "malaria-pf-mortality-rate",    "Pf_Mortality_Rate"),
    ("Pv_Incidence_Count",  "malaria-pv-incidence-count",   "Pv_Incidence_Count"),
    ("Pv_Incidence_Rate",   "malaria-pv-incidence-rate",    "Pv_Incidence_Rate"),
]


def describe_coverage_years(coverage_id: str) -> list[int]:
    """Return sorted unique integer years exposed by DescribeCoverage."""
    url = (
        f"{GEOSERVER}?service=WCS&version=2.0.1&request=DescribeCoverage"
        f"&coverageid={coverage_id}"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        xml_text = resp.read().decode("utf-8")
    years = re.findall(r"<gml:timePosition>(\d{4})-", xml_text)
    if not years:
        raise RuntimeError(f"no <gml:timePosition> entries for {coverage_id}")
    return sorted({int(y) for y in years})


def download_coverage(coverage_id: str, year: int, out_path: Path) -> int:
    """Download one (coverage, year) GeoTIFF via WCS GetCoverage. Returns bytes written."""
    time_param = quote(f'"{year}-01-01T00:00:00.000Z"', safe="")
    url = (
        f"{GEOSERVER}?service=WCS&version=2.0.1&request=GetCoverage"
        f"&coverageid={coverage_id}"
        f"&format=image/tiff"
        f"&subset=time({time_param})"
    )
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    bytes_written = 0
    with urllib.request.urlopen(url, timeout=600) as resp:
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype not in {"image/tiff", "application/octet-stream"}:
            raise RuntimeError(
                f"unexpected Content-Type {ctype!r} for {coverage_id} {year}"
            )
        with tmp_path.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                bytes_written += len(chunk)
    if bytes_written < MIN_VALID_SIZE:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"download for {coverage_id} {year} too small ({bytes_written} bytes)"
        )
    tmp_path.replace(out_path)
    return bytes_written


def main(release: str) -> None:
    print(f"Release: {release}")
    print(f"Base path: {BASE_PATH}")
    if not BASE_PATH.is_dir():
        raise SystemExit(f"base path does not exist: {BASE_PATH}")

    failures: list[tuple[str, int, str]] = []
    for cov_short, subdir, stem in COVARIATES:
        coverage_id = f"Malaria__{release}_Global_{cov_short}"
        out_dir = BASE_PATH / subdir / release
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {coverage_id} → {out_dir} ===")
        try:
            years = describe_coverage_years(coverage_id)
        except Exception as e:
            print(f"  ! DescribeCoverage failed: {e}")
            failures.append((coverage_id, -1, str(e)))
            continue
        print(f"  Available years: {min(years)}-{max(years)} ({len(years)} total)")
        for year in years:
            out_path = out_dir / f"{release}_Global_{stem}_{year}.tif"
            if out_path.exists() and out_path.stat().st_size >= MIN_VALID_SIZE:
                print(f"  {year}: skip (exists, {out_path.stat().st_size >> 20} MB)")
                continue
            print(f"  {year}: downloading...", end=" ", flush=True)
            try:
                size = download_coverage(coverage_id, year, out_path)
                print(f"OK ({size >> 20} MB)")
            except Exception as e:
                print(f"FAILED: {e}")
                failures.append((coverage_id, year, str(e)))
            time.sleep(0.5)  # be nice to the server

    print("\n=== Summary ===")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for cov, yr, msg in failures:
            print(f"  - {cov} {yr if yr > 0 else '(describe)'}: {msg}")
        raise SystemExit(1)
    print("All downloads complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        default="202508",
        help="MAP release tag (e.g. 202508, 202406, 202206). Default: 202508",
    )
    args = parser.parse_args()
    main(args.release)
