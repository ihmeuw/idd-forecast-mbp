"""Draw-free per-cell weights that turn one malaria forecast into a result arm.

An arm (DECISIONS 2026-10-07) is Baseline with exactly one change. Changes to a covariate
(DAH Constant, a covariate held at 2023) live in the forecast run itself. The other three need
no re-forecast and act on admin-2 age/sex counts (DECISIONS 2026-07-31, 2026-08-04, 2026-08-25):

- vaccine protection ``p`` (none, projected product mix, or R21 everywhere),
- the population hold (the whole population frame replaced by its hold-year value),
- the age-structure hold (each age/sex group's count times ``as_ref_frac / aa_ref_frac``).

The forecast's own age/sex split is ``count[l,y,a,s,d] = aa_count[l,y,d] * f[l,y,a,s]`` with ``f``
draw-free (``malaria_as_fractions``). So every arm is

    arm_count[l,y,a,s,d] = aa_count[l,y,d] * w[l,y,a,s],     w = f * m * (1 - p)

with ``m`` the structure multiplier (1 unless the age-structure hold is on). Because ``w`` carries
no draw, the all-age arm count is exactly ``aa_count * sum_as(w)``: the all-age products use that
scalar (``all_age_factors``) and the admin-2 age/sex recreation uses ``w`` cell-wise
(``arm_age_sex_counts``). The two agree by construction, which is what lets a recreated 2,A,X
roll up to the saved summaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

KEYS_AS = ["location_id", "year_id", "age_group_id", "sex_id"]
KEYS_AA = ["location_id", "year_id"]
PROTECTION_COLS = {
    "inc": "effective_protection_case",
    "mort": "effective_protection_death",
}
MEASURES = ("inc", "mort")


def hold_population(
    pop: pd.DataFrame, hold_year: int, *, keys: list[str]
) -> pd.DataFrame:
    """Every year's ``population`` replaced by its ``hold_year`` value, per ``keys`` minus year.

    The population hold is accounting: one frame supplies the rate-to-count multiplier, the
    age/sex split and every denominator, so freezing it is the whole intervention.
    """
    by = [k for k in keys if k != "year_id"]
    ref = pop.loc[pop.year_id == int(hold_year), [*by, "population"]]
    if ref.empty:
        msg = f"population has no rows for hold year {hold_year}"
        raise ValueError(msg)
    out = pop.drop(columns="population").merge(ref, on=by, how="left")
    missing = int(out.population.isna().sum())
    if missing:
        msg = f"{missing} rows have no {hold_year} population to hold at"
        raise ValueError(msg)
    return out


def structure_multiplier(as_pop: pd.DataFrame, hold_year: int) -> pd.DataFrame:
    """``m = share[ref] / share[y]`` per age/sex cell, where share is the cell's population share.

    ``count * (as_pop[ref]/as_pop[y]) / (aa_pop[ref]/aa_pop[y])`` (DECISIONS 2026-07-31) reduces to
    this: total population cancels, so each group is reweighted by the change in its share.
    A cell with no population in year y gets 0 (it carries no count to reweight).
    """
    share = as_pop[[*KEYS_AS, "population"]].copy()
    total = share.groupby(KEYS_AA, observed=True)["population"].transform("sum")
    share["share"] = np.where(total > 0, share.population / total, 0.0)
    ref = share.loc[
        share.year_id == int(hold_year),
        ["location_id", "age_group_id", "sex_id", "share"],
    ]
    if ref.empty:
        msg = f"age/sex population has no rows for hold year {hold_year}"
        raise ValueError(msg)
    out = share.merge(
        ref.rename(columns={"share": "share_ref"}),
        on=["location_id", "age_group_id", "sex_id"],
        how="left",
    )
    out["share_ref"] = out.share_ref.fillna(0.0)
    out["multiplier"] = np.where(out.share > 0, out.share_ref / out.share, 0.0)
    return out[[*KEYS_AS, "multiplier"]]


def admin2_protection(protection: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Broadcast admin-1 protection to each admin-2 child (exact: protection is a rate).

    ``mapping`` is ``[location_id (admin-2), coverage_location_id (admin-1)]``.
    """
    prot = protection.rename(columns={"location_id": "coverage_location_id"}).merge(
        mapping, on="coverage_location_id", how="inner"
    )
    return prot[[*KEYS_AS, *PROTECTION_COLS.values()]].reset_index(drop=True)


def cell_weights(
    fractions: pd.DataFrame,
    *,
    protection: pd.DataFrame | None = None,
    multiplier: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """``w = f * m * (1 - p)`` per (location, year, age, sex) and measure.

    ``fractions`` is ``malaria_as_fractions`` output. Cells without a protection row get p = 0;
    cells without a multiplier row get m = 1. Protection outside [0, 1] refuses.
    """
    w = fractions[[*KEYS_AS, "inc_fraction", "mort_fraction"]].copy()
    if multiplier is not None:
        w = w.merge(multiplier[[*KEYS_AS, "multiplier"]], on=KEYS_AS, how="left")
        w["multiplier"] = w.multiplier.fillna(1.0)
    else:
        w["multiplier"] = 1.0
    prot_cols = list(PROTECTION_COLS.values())
    if protection is not None:
        w = w.merge(protection[[*KEYS_AS, *prot_cols]], on=KEYS_AS, how="left")
        w[prot_cols] = w[prot_cols].fillna(0.0)
        bad = (w[prot_cols] < 0).any(axis=1) | (w[prot_cols] > 1).any(axis=1)
        if bad.any():
            msg = f"protection outside [0,1] on {int(bad.sum())} cell(s)"
            raise ValueError(msg)
    else:
        for c in prot_cols:
            w[c] = 0.0
    for m in MEASURES:
        w[f"{m}_weight"] = (
            w[f"{m}_fraction"] * w["multiplier"] * (1.0 - w[PROTECTION_COLS[m]])
        )
    return w[[*KEYS_AS, "inc_weight", "mort_weight"]]


def all_age_factors(weights: pd.DataFrame) -> pd.DataFrame:
    """``sum_as(w)`` per (location, year): the exact all-age multiplier of the arm."""
    return (
        weights.groupby(KEYS_AA, observed=True)[["inc_weight", "mort_weight"]]
        .sum()
        .rename(columns={"inc_weight": "inc_factor", "mort_weight": "mort_factor"})
        .reset_index()
    )


def arm_age_sex_counts(aa_counts: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Admin-2 age/sex arm counts per draw: ``aa_count * w`` cell-wise.

    ``aa_counts`` is ``[location_id, year_id, draw, inc_count, mort_count]`` (all-age, any number
    of draws). Returns ``[location_id, year_id, age_group_id, sex_id, draw, inc_count, mort_count]``.
    Locations without weights (non-endemic, no age/sex pattern) contribute no rows.
    """
    out = aa_counts.merge(weights, on=KEYS_AA, how="inner")
    for m in MEASURES:
        out[f"{m}_count"] = out[f"{m}_count"] * out[f"{m}_weight"]
    return out[[*KEYS_AS, "draw", "inc_count", "mort_count"]]
