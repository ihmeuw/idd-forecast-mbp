"""Finalize a malaria model-selection run into one stat-complete artifact.

Joins the per-cell ``select_summary_*.parquet`` files a run produced into a single
``selection_summary.parquet`` — ONE row per spec, carrying:
  - spec metadata: spec_index, n_smooths, formula_text
  - every in-sample metric  (all ``is_*`` columns from the IS cell)
  - every out-of-sample metric per window (all ``oos_*`` / ``cv_*`` columns from each
    temporal cell, prefixed ``<window>__``)
  - fit provenance (optimizer, scam_version, ...)

This is deliberately NOT a selection step: it runs no τ-prune, no Borda/TOPSIS/
dominance ranking, and declares no finalists. It is the complete, un-collapsed stat
table the (supervised, interactive) selection + ensemble-membership judgement is made
from. Nothing is dropped, so an ensemble over the top-X specs can be weighted on any
retained statistic later.
"""
from __future__ import annotations

import glob
from pathlib import Path

import click
import pandas as pd


def build_summary(run_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(run_dir / "select_summary_*_n*_bin*.parquet")))
    if not files:
        raise FileNotFoundError(f"no select_summary_*_n*_bin*.parquet under {run_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # cell = task_id with the trailing _n<ns>_bin<idx> stripped (cell names have no '_n')
    df["cell"] = df["task_id"].map(lambda t: t.rsplit("_n", 1)[0])

    spec = pd.read_parquet(run_dir / "spec_table.parquet")[
        ["spec_index", "n_smooths", "formula_text"]
    ]

    # --- in-sample block: all is_* columns + provenance, one row per spec ---
    is_rows = df[df.cell == "IS"].copy()
    prov = [c for c in ("optimizer", "maxit_setting", "scam_version", "r_version")
            if c in is_rows.columns]
    is_cols = ["spec_index"] + prov + [c for c in is_rows.columns if c.startswith("is_")]
    out = spec.merge(is_rows[is_cols], on="spec_index", how="left")

    # --- per-window OOS block: all oos_/cv_ columns, prefixed by window name ---
    oos_cols = [c for c in df.columns if c.startswith("oos_") or c.startswith("cv_")]
    windows = sorted(df.loc[df.cell != "IS", "cell"].unique())
    for w in windows:
        wdf = df.loc[df.cell == w, ["spec_index"] + oos_cols].rename(
            columns={c: f"{w}__{c}" for c in oos_cols}
        )
        out = out.merge(wdf, on="spec_index", how="left")

    # coverage: how many windows produced an oos_pfpr_r for this spec
    win_r = [f"{w}__oos_pfpr_r" for w in windows if f"{w}__oos_pfpr_r" in out.columns]
    out["n_windows_ok"] = out[win_r].notna().sum(axis=1)
    out.attrs["windows"] = windows
    return out


@click.command()
@click.option("--run-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="the selection run dir (holds select_summary_*.parquet + spec_table.parquet)")
@click.option("--out-name", default="selection_summary.parquet", show_default=True)
def main(run_dir: str, out_name: str) -> None:
    run_dir = Path(run_dir)
    out = build_summary(run_dir)
    path = run_dir / out_name
    out.to_parquet(path, index=False)

    windows = out.attrs["windows"]
    click.echo(f"wrote {path}")
    click.echo(f"  {out.shape[0]} specs x {out.shape[1]} cols  |  {len(windows)} windows: {windows}")
    click.echo(f"  n_windows_ok distribution:\n{out['n_windows_ok'].value_counts().sort_index().to_string()}")
    incomplete = int((out['n_windows_ok'] < len(windows)).sum())
    click.echo(f"  specs missing >=1 window: {incomplete} of {len(out)}")


if __name__ == "__main__":
    main()
