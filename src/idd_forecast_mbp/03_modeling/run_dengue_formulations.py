"""Fit dengue formulations on the past and score them at every hierarchy level.

In-sample only. The past covariates are single-realization, so this produces a
point estimate rather than mean/lower/upper — uncertainty enters with the
forecast, where draws come from the 08b climate covariates.

What it does, per formulation: fit the outcomes its structure declares, anchor
each to observed, convert to counts, roll up to every ancestor in count space,
divide by each level's own population, and score predicted against observed.

    python 03_modeling/run_dengue_formulations.py --output-dir <dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from idd_forecast_mbp.lib.data.dengue_inputs import load_dengue_inputs  # noqa: E402
from idd_forecast_mbp.lib.io.parquet import write_parquet  # noqa: E402
from idd_forecast_mbp.lib.modeling.dengue_pipeline import run_formulation  # noqa: E402
from idd_forecast_mbp.lib.processing.dengue_products import (  # noqa: E402
    MEASURES,
    build_hierarchy_products,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dengue_formulation_set import build_formulations  # noqa: E402

LEVEL_NAME = {0: "global", 1: "super_region", 2: "region", 3: "country", 4: "admin_1"}


def score(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Pearson r plus the OLS fit of observed on predicted (want slope 1, intercept 0)."""
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    keep = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[keep], pred[keep]
    if obs.size < 2 or np.ptp(pred) == 0:
        return {"n": int(obs.size), "r": np.nan, "slope": np.nan, "intercept": np.nan}
    slope, intercept = np.polyfit(pred, obs, 1)
    return {
        "n": int(obs.size),
        "r": float(np.corrcoef(obs, pred)[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def score_by_level(
    products: pd.DataFrame, observed: pd.DataFrame, hierarchy: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (level, measure), comparing predicted to observed rates."""
    levels = hierarchy.set_index("location_id")["level"]
    merged = products.merge(
        observed[["location_id", "year_id", "dengue_inc_rate", "dengue_mort_rate"]]
        .rename(columns={"dengue_inc_rate": "obs_inc_rate",
                         "dengue_mort_rate": "obs_mort_rate"}),
        on=["location_id", "year_id"], how="inner",
    )
    merged["level"] = merged["location_id"].map(levels)

    rows = []
    for level, block in merged.groupby("level"):
        for measure in MEASURES:
            rate_col = MEASURES[measure][0]
            rows.append({
                "level": int(level),
                "level_name": LEVEL_NAME.get(int(level), str(level)),
                "measure": measure,
                **score(block[f"obs_{measure}_rate"], block[rate_col]),
            })
    return pd.DataFrame(rows)


@click.command()
@click.option("--output-dir", type=click.Path(path_type=Path), required=True,
              help="Directory for the per-formulation products and the summary.")
@click.option("--grain", default="fhs", type=click.Choice(["fhs", "lsae"]),
              help="Location grain to fit and predict at.")
@click.option("--anchor-year", default=2023, type=int,
              help="Year the observed age/sex structure is taken from.")
@click.option("--formulation", "only", multiple=True,
              help="Run only these formulation ids (repeatable). Default: all.")
def main(output_dir: Path, grain: str, anchor_year: int, only: tuple[str, ...]) -> None:
    """Fit every formulation and write its products plus a scoring summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading dengue inputs at grain={grain} (anchor {anchor_year})...")
    inputs = load_dengue_inputs(grain=grain, anchor_year=anchor_year)
    click.echo(
        f"  fit frame: {len(inputs.fit_frame):,} rows, "
        f"{inputs.fit_frame['location_id'].nunique():,} locations, "
        f"years {int(inputs.fit_frame.year_id.min())}-"
        f"{int(inputs.fit_frame.year_id.max())}"
    )

    formulations = [f for f in build_formulations() if not only or f.id in only]
    if not formulations:
        msg = f"no formulation matched {only}"
        raise click.ClickException(msg)

    summaries = []
    for formulation in formulations:
        click.echo(f"\n=== {formulation.id} ({formulation.structure}) ===")
        run = run_formulation(inputs, formulation)
        for outcome, result in run.fits.items():
            click.echo(f"  fit {outcome}: {result.n_fit_rows:,} rows, "
                       f"{result.n_unanchored:,} cells without an anchor")

        # run.rates is per (location, year, age, sex) because the anchor is per
        # cell, so each cell must be costed at its OWN population before the
        # cells are summed — see age_sex_rates_to_all_age_counts.
        age_sex_population = inputs.fit_frame[
            ["location_id", "year_id", "age_group_id", "sex_id", "population"]
        ]
        products = build_hierarchy_products(
            run.rates, inputs.population, inputs.hierarchy,
            age_sex_population=age_sex_population,
        )
        write_parquet(products, output_dir / f"{formulation.id}_products.parquet")

        summary = score_by_level(products, inputs.observed_all_age, inputs.hierarchy)
        summary.insert(0, "formulation", formulation.id)
        summaries.append(summary)
        for _, row in summary[summary.level <= 1].iterrows():
            click.echo(f"  {row.level_name:>13s} {row.measure:>4s}: "
                       f"r={row.r:6.3f} slope={row.slope:6.3f} n={row.n}")

    combined = pd.concat(summaries, ignore_index=True)
    write_parquet(combined, output_dir / "formulation_summary.parquet")
    click.echo(f"\nWrote {len(formulations)} product files + summary to {output_dir}")


if __name__ == "__main__":
    main()
