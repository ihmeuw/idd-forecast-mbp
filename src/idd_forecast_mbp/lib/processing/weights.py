"""Aggregation weights for covariates, and the weighted roll-up to reporting levels.

A covariate is intensive — a per-capita amount, a suitability index, a temperature — so it
has no meaningful sum and must aggregate as a weighted mean. *Which* weight you choose
changes the question being answered, and for a disease pipeline the default is usually the
wrong one:

``population``
    What does the average person experience? Spreads weight over every admin-2 unit,
    including the ~80% with no malaria at all, so it systematically dilutes any change
    concentrated in endemic areas.
``mort2023`` / ``inc2023``
    What does the covariate look like where the disease actually is? Weights are fixed at
    the anchor year and non-endemic locations weigh exactly ZERO, which makes these a much
    sharper instrument for anything that feeds a burden model.

Both are worth showing side by side rather than picking one — the house pattern in
``notebooks/09_figures/TS_weighted_suitability.ipynb`` (cell 13) weights by population,
mortality and incidence and plots burden-weighted beside population-weighted. This module
is that arithmetic, lifted out so the stage-05 scripts share one implementation instead of
each carrying a copy (stage dirs like ``05_aggregation`` are not importable packages, so a
shared lib module is the only way to avoid duplication).

Caveat that no weighting fixes: a weighted mean is still a mean. Where a model's response
to a covariate is non-linear — malaria PfPR takes GDP through a monotone spline on the raw
dollar scale with a logit response — no single aggregate captures its leverage. See
``.claude/GDP_LEVERAGE_INVESTIGATION.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids

#: Weight scheme -> human description, for figure titles and CLI help.
WEIGHT_SCHEMES: dict[str, str] = {
    "population": "population-weighted",
    "mort2023": "2023 malaria-death weighted",
    "inc2023": "2023 malaria-case weighted",
}

_BURDEN_COL = {"mort2023": "malaria_mort_count", "inc2023": "malaria_inc_count"}


def load_weights(
    kind: str,
    years: Sequence[int],
    *,
    anchor_year: int = 2023,
    cause: str = "malaria",
) -> pd.DataFrame:
    """Admin-2 weights as ``[location_id, year_id, weight]``.

    ``population`` varies by year; the burden schemes are fixed at ``anchor_year`` and
    broadcast, so they describe where the disease is *now* rather than tracking a forecast
    that the weighting is meant to be independent of. Locations with zero anchor-year
    burden are dropped, so they contribute no weight at all.
    """
    if kind not in WEIGHT_SCHEMES:
        raise ValueError(f"unknown weight scheme {kind!r}; valid: {list(WEIGHT_SCHEMES)}")
    yrs = [int(y) for y in years]

    if kind == "population":
        pop = read_parquet_with_integer_ids(
            mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet",
            columns=["location_id", "year_id", "population"],
        )
        out = pop[pop.year_id.isin(yrs)].rename(columns={"population": "weight"})
        if out.empty:
            raise ValueError(f"no population rows for years {yrs[:3]}...")
        return out.reset_index(drop=True)

    col = _BURDEN_COL[kind]
    read_path = (
        mbpc.MAL_RAKED_AA_READ_PATH if cause == "malaria" else mbpc.DEN_RAKED_AA_READ_PATH
    )
    obs = read_parquet_with_integer_ids(
        read_path / f"aa_full_{cause}_df.parquet",
        columns=["location_id", "year_id", col],
    )
    ref = obs.loc[obs.year_id == int(anchor_year), ["location_id", col]].rename(
        columns={col: "weight"}
    )
    ref = ref[ref.weight > 0]
    if ref.empty:
        raise ValueError(f"no nonzero {col} at {anchor_year} for cause {cause!r}")
    return (
        pd.MultiIndex.from_product(
            [ref.location_id, yrs], names=["location_id", "year_id"]
        )
        .to_frame(index=False)
        .merge(ref, on="location_id")
    )


def weighted_rollup_to_levels(
    values: pd.DataFrame,
    weights: pd.DataFrame,
    hierarchy: pd.DataFrame,
    *,
    total: bool = False,
) -> pd.DataFrame:
    """Aggregate a per-location ``value`` to global (1) and super-region.

    ``total=False`` returns the weighted MEAN — correct for an intensive covariate.
    ``total=True`` treats ``value`` as a per-capita amount and returns ``value x weight``
    SUMMED, which recovers an extensive total. A total has no meaningful average, so the
    two modes are not interchangeable.

    Only locations present in both ``values`` and ``weights`` contribute, so a burden
    scheme silently restricts to endemic locations — which is the point, but it means the
    denominator differs between schemes and the two results are not directly subtractable.
    """
    df = values.merge(weights, on=["location_id", "year_id"], how="inner").merge(
        hierarchy[["location_id", "super_region_id"]], on="location_id", how="inner"
    )
    if df.empty:
        return pd.DataFrame(columns=["location_id", "year_id", "value"])
    df["wv"] = df.value * df.weight

    sr = df.groupby(["super_region_id", "year_id"], observed=True)[["wv", "weight"]].sum()
    g = df.groupby("year_id", observed=True)[["wv", "weight"]].sum()
    if total:
        parts = [
            sr.wv.reset_index(name="value").rename(
                columns={"super_region_id": "location_id"}
            ),
            g.wv.reset_index(name="value").assign(location_id=1),
        ]
    else:
        parts = [
            (sr.wv / sr.weight).reset_index(name="value").rename(
                columns={"super_region_id": "location_id"}
            ),
            (g.wv / g.weight).reset_index(name="value").assign(location_id=1),
        ]
    return pd.concat(parts, ignore_index=True)
