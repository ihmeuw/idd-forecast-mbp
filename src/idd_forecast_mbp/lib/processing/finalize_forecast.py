"""Pure-compute cores for finalizing a malaria forecast run into saved products.

A *run* (a registered regression + covariates + sensitivity assumptions, or an
ensemble run) produces, per (SSP, DAH), admin-2 all-age draw-level rate predictions
(``exp(log_*_rate_pred)``). These cores turn those into the finished products:

  finalize_all_age        — all-age counts + rates over the full hierarchy, in COUNT
                            space, with the FULL-population level-row denominator.
  finalize_age_sex_draws  — age/sex draws at a requested node set (per-draw
                            disaggregate at admin-2 → aggregate up preserving age/sex).
  finalize_age_sex_summary— age/sex mean + UI over the full hierarchy, using the
                            admin-2 monotone-scaling shortcut for the leaf level.

Invariants enforced here (the load-bearing correctness rules):

- **Count-space aggregation.** Everything is summed in count space up the hierarchy;
  rates are derived last.
- **Full-population level-row denominator.** An aggregate rate is count ÷ that level's
  OWN population row (from the population artifact), NEVER ÷ a summed endemic subset.
- **True zeros.** A product covering the complete hierarchy zero-fills non-endemic
  locations (a location we don't forecast is a true zero, not a gap).

These functions are pure (DataFrames in, DataFrame out) and run-agnostic — the stage
CLI reads the netCDFs / population artifacts and calls them. The IO, versioning, and
jobmon orchestration live in ``05_aggregation/finalize_malaria_forecast_run.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from idd_forecast_mbp.lib.processing.aggregation import roll_up_hierarchy
from idd_forecast_mbp.lib.processing.disaggregation import (
    disaggregate_malaria_draws,
    malaria_as_fractions,
)

_KEYS_AA = ["location_id", "year_id", "draw"]
_KEYS_AS = ["location_id", "year_id", "age_group_id", "sex_id"]
_PRED_COL = {"inc": "malaria_inc_count_pred", "mort": "malaria_mort_count_pred"}


def _rate_to_count(
    aa_rate_draws: pd.DataFrame,
    admin2_pop: pd.DataFrame,
    measures: Sequence[str],
) -> pd.DataFrame:
    """admin-2 rate × admin-2 population → admin-2 count, per measure.

    Fails loudly if any admin-2 (loc, year) has no population — a missing denominator
    at the leaf is an upstream/vintage bug, not a benign skip.
    """
    df = aa_rate_draws.merge(admin2_pop[["location_id", "year_id", "population"]],
                             on=["location_id", "year_id"], how="left")
    if df["population"].isna().any():
        bad = df.loc[df["population"].isna(), "location_id"].unique()[:5]
        msg = f"admin-2 rows missing population (stale pop vintage?), e.g. {list(bad)}"
        raise ValueError(msg)
    for m in measures:
        df[f"{m}_count"] = df[f"{m}_rate"] * df["population"]
    return df


def finalize_all_age(
    aa_rate_draws: pd.DataFrame,
    admin2_pop: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    full_pop: pd.DataFrame,
    *,
    measures: Sequence[str] = ("inc", "mort"),
    start_level: int = 5,
) -> pd.DataFrame:
    """All-age hierarchy product: rate → count → roll up (count space) → full-pop rate.

    Parameters
    ----------
    aa_rate_draws:
        Admin-2 (``start_level``) all-age rate draws:
        ``[location_id, year_id, draw, {m}_rate ...]`` for each ``m`` in ``measures``.
        Typically only the endemic admin-2 the run forecasts.
    admin2_pop:
        Admin-2 all-age population ``[location_id, year_id, population]`` — the numerator
        weight that turns a rate into a count at the leaf.
    hierarchy_df:
        Full hierarchy (``location_id``, ``parent_id``, ``level``).
    full_pop:
        All-age population at EVERY level ``[location_id, year_id, population]`` (the
        population artifact's own rows). Supplies each aggregate level's denominator.
    measures:
        Which measures to build (default incidence + mortality).

    Returns
    -------
    ``[location_id, year_id, draw, {m}_count, {m}_rate ...]`` over every hierarchy node
    reached by rolling the input up (endemic leaves + their ancestors). Rate at each
    node = its rolled-up count ÷ its OWN ``full_pop`` row (draw-independent denominator,
    broadcast across draws); 0 where that population is 0. Call ``square_to_locations``
    afterward to zero-fill to the complete node set of a given product.
    """
    counts = _rate_to_count(aa_rate_draws, admin2_pop, measures)

    rolled: pd.DataFrame | None = None
    for m in measures:
        r = roll_up_hierarchy(
            counts[["location_id", "year_id", "draw", f"{m}_count"]],
            hierarchy_df, f"{m}_count",
            extra_group_cols=["draw"], start_level=start_level,
        )
        rolled = r if rolled is None else rolled.merge(r, on=_KEYS_AA, how="outer")

    assert rolled is not None  # measures is non-empty
    out = rolled.merge(full_pop.rename(columns={"population": "_pop"})
                       [["location_id", "year_id", "_pop"]],
                       on=["location_id", "year_id"], how="left")
    for m in measures:
        out[f"{m}_count"] = out[f"{m}_count"].fillna(0.0)
        out[f"{m}_rate"] = np.where(out["_pop"] > 0, out[f"{m}_count"] / out["_pop"], 0.0)
    return out.drop(columns=["_pop"])


def square_to_locations(
    df: pd.DataFrame,
    location_ids: Sequence[int],
    value_cols: Sequence[str],
    *,
    other_key_cols: Sequence[str] = ("year_id", "draw"),
) -> pd.DataFrame:
    """Zero-fill ``df`` to the complete ``location_ids`` × (existing other-key combos) grid.

    Guarantees every location in ``location_ids`` has a row for every ``other_key_cols``
    combination present in ``df`` (e.g. every year × draw), with ``value_cols`` filled to
    0 where ``df`` had no row — a non-endemic / dropped location is a TRUE ZERO, not a
    gap. Locations already in ``df`` keep their values.
    """
    other = df[list(other_key_cols)].drop_duplicates()
    locs = pd.DataFrame({"location_id": sorted({int(x) for x in location_ids})})
    grid = locs.merge(other, how="cross")
    out = grid.merge(df, on=["location_id", *other_key_cols], how="left")
    out[list(value_cols)] = out[list(value_cols)].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Age/sex cores
# ---------------------------------------------------------------------------

def _rollup_age_sex_one_draw(
    as_admin2: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    measures: Sequence[str],
    start_level: int,
) -> pd.DataFrame:
    """Roll one draw's admin-2 age/sex counts up the hierarchy into one merged frame:
    ``[location_id, year_id, age_group_id, sex_id, draw, {m}_count ...]``."""
    merged: pd.DataFrame | None = None
    for m in measures:
        col = _PRED_COL[m]
        r = roll_up_hierarchy(
            as_admin2[[*_KEYS_AS, "draw", col]],
            hierarchy_df, col,
            preserve_age_sex=True, extra_group_cols=["draw"], start_level=start_level,
        ).rename(columns={col: f"{m}_count"})
        merged = r if merged is None else merged.merge(r, on=[*_KEYS_AS, "draw"])
    assert merged is not None
    return merged


def finalize_age_sex_draws(
    aa_count_draws: pd.DataFrame,
    rr: pd.DataFrame,
    as_pop: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    node_ids: Sequence[int],
    *,
    measures: Sequence[str] = ("inc", "mort"),
    start_level: int = 5,
) -> pd.DataFrame:
    """Age/sex count + rate DRAWS at a requested node set (e.g. the FHS hierarchy).

    Streams **per draw** to bound memory: for each draw, disaggregate the admin-2
    all-age counts to age/sex (``disaggregate_malaria_draws``), roll up preserving
    age/sex, keep only ``node_ids``, then concatenate. The admin-2 age/sex frame — the
    memory blow-up — is transient to one draw and discarded before the next.

    Parameters
    ----------
    aa_count_draws:
        Admin-2 all-age count draws
        ``[location_id, year_id, draw, aa_malaria_inc_count, aa_malaria_mort_count]``.
    rr, as_pop:
        Static age/sex rr and year-varying age/sex population (see
        ``disaggregate_malaria_draws``). ``as_pop`` also supplies the rate denominator.
    node_ids:
        Locations to keep (e.g. the 513 ``in_fhs_hierarchy`` nodes).

    Returns
    -------
    ``[location_id, year_id, age_group_id, sex_id, draw, {m}_count, {m}_rate ...]`` at
    ``node_ids``. Rate = count ÷ that node's own age/sex ``as_pop`` row.
    """
    node_set = {int(x) for x in node_ids}
    parts = []
    for d in sorted(aa_count_draws["draw"].unique()):
        as_d = disaggregate_malaria_draws(aa_count_draws[aa_count_draws["draw"] == d], rr, as_pop)
        rolled = _rollup_age_sex_one_draw(as_d, hierarchy_df, measures, start_level)
        parts.append(rolled[rolled["location_id"].isin(node_set)])
    out = pd.concat(parts, ignore_index=True)
    out = out.merge(as_pop.rename(columns={"population": "_pop"})[[*_KEYS_AS, "_pop"]],
                    on=_KEYS_AS, how="left")
    for m in measures:
        out[f"{m}_rate"] = np.where(out["_pop"] > 0, out[f"{m}_count"] / out["_pop"], 0.0)
    return out.drop(columns=["_pop"])


def finalize_age_sex_summary(
    aa_count_draws: pd.DataFrame,
    rr: pd.DataFrame,
    as_pop: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    *,
    measures: Sequence[str] = ("inc", "mort"),
    quantiles: tuple[float, float] = (0.025, 0.975),
    leaf_level: int = 5,
    aa_inc_col: str = "aa_malaria_inc_count",
    aa_mort_col: str = "aa_malaria_mort_count",
) -> pd.DataFrame:
    """Full-hierarchy age/sex mean + lower/upper UI, split by the shortcut boundary.

    - **Leaf (admin-2, ``leaf_level``):** ``as = f · aa`` with ``f ≥ 0`` draw-independent,
      so `mean`/quantiles commute with the scaling — take the all-age admin-2 draw
      mean/quantiles once and multiply by ``f``. This dodges ever materializing the
      admin-2 age/sex draw array (the ~74 GB term).
    - **Aggregate levels (``< leaf_level``):** ``as(L) = Σ f·aa`` is a sum, so quantiles
      don't factor — disaggregate + roll up **per draw**, hold the aggregate-level draws,
      and take mean/quantiles over the draw axis.

    Returns ``[location_id, year_id, age_group_id, sex_id, {m}_mean, {m}_lo, {m}_hi ...]``
    over every node reached (leaf + aggregates). Non-endemic locations are added as
    true zeros by a later ``square_to_locations`` at the product's node set.
    """
    lo, hi = quantiles
    aa_cols = {"inc": aa_inc_col, "mort": aa_mort_col}

    # ---- leaf level: monotone-scaling shortcut ----
    frac = malaria_as_fractions(rr, as_pop)
    frac = frac.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
    frac = frac[frac["level"] == leaf_level].drop(columns="level")

    agg_spec: dict[str, tuple] = {}
    for m in measures:
        c = aa_cols[m]
        agg_spec[f"{m}_mean"] = (c, "mean")
        agg_spec[f"{m}_lo"] = (c, lambda s: s.quantile(lo))
        agg_spec[f"{m}_hi"] = (c, lambda s: s.quantile(hi))
    aa_stats = aa_count_draws.groupby(["location_id", "year_id"]).agg(**agg_spec).reset_index()

    leaf = frac.merge(aa_stats, on=["location_id", "year_id"], how="inner")
    for m in measures:
        for stat in ("mean", "lo", "hi"):
            leaf[f"{m}_{stat}"] = leaf[f"{m}_fraction"] * leaf[f"{m}_{stat}"]
    summary_cols = [f"{m}_{s}" for m in measures for s in ("mean", "lo", "hi")]
    leaf = leaf[[*_KEYS_AS, *summary_cols]]

    # ---- aggregate levels: per-draw disaggregate + roll up, then quantiles over draws ----
    per_draw_cols: dict[str, dict[int, pd.Series]] = {m: {} for m in measures}
    for d in sorted(aa_count_draws["draw"].unique()):
        as_d = disaggregate_malaria_draws(aa_count_draws[aa_count_draws["draw"] == d], rr, as_pop)
        rolled = _rollup_age_sex_one_draw(as_d, hierarchy_df, measures, leaf_level)
        rolled = rolled.merge(hierarchy_df[["location_id", "level"]], on="location_id", how="left")
        rolled = rolled[rolled["level"] < leaf_level].set_index(_KEYS_AS)
        for m in measures:
            per_draw_cols[m][int(d)] = rolled[f"{m}_count"]

    agg_summary = None
    for m in measures:
        wide = pd.DataFrame(per_draw_cols[m]).fillna(0.0)  # rows=(loc,year,age,sex), cols=draws
        block = pd.DataFrame({
            f"{m}_mean": wide.mean(axis=1),
            f"{m}_lo": wide.quantile(lo, axis=1),
            f"{m}_hi": wide.quantile(hi, axis=1),
        })
        agg_summary = block if agg_summary is None else agg_summary.join(block)
    agg_summary = agg_summary.reset_index() if agg_summary is not None else pd.DataFrame(columns=[*_KEYS_AS, *summary_cols])

    return pd.concat([leaf, agg_summary], ignore_index=True)
