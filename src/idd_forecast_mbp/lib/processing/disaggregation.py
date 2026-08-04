"""
Age-sex disaggregation functions for the idd-forecast-mbp pipeline.

Two structurally distinct methods are preserved — they cannot be consolidated:

  disaggregate_age_sex_malaria:
    Normalized RR-fraction method. Distributes all-age counts using
    population-weighted relative-risk fractions. Canonical source:
    04_forecasting/as_malaria_fractions.py

  disaggregate_age_sex_dengue:
    Log-rate + CFR method. Computes age-sex-specific counts directly from
    a log-rate prediction and a CFR without an all-age input.
    Canonical source: 04_forecasting/as_dengue_shifts.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Malaria — normalized RR-fraction method
# ---------------------------------------------------------------------------

def _as_fraction_columns(
    df: pd.DataFrame,
    rr_inc_col: str,
    rr_mort_col: str,
    population_col: str,
    zero_age_group_id: int,
    group_keys: tuple[str, ...] = ('location_id', 'year_id'),
) -> pd.DataFrame:
    """Add 'inc_fraction'/'mort_fraction' columns: population-weighted rr normalized
    within each group over the RETAINED ages.

    Single implementation of the zero-then-normalize rule shared by
    disaggregate_age_sex_malaria (per-draw frame) and disaggregate_malaria_draws
    (draw-independent fraction table). ``zero_age_group_id`` is dropped from the
    denominator so retained-age fractions sum to 1 and the all-age total is preserved;
    a group with no retained-age burden gets fraction 0, not 0/0 = NaN.
    """
    keys = list(group_keys)
    w_inc = (df[rr_inc_col] * df[population_col]).where(df['age_group_id'] != zero_age_group_id, 0.0)
    w_mort = (df[rr_mort_col] * df[population_col]).where(df['age_group_id'] != zero_age_group_id, 0.0)
    s_inc = w_inc.groupby([df[k] for k in keys]).transform('sum')
    s_mort = w_mort.groupby([df[k] for k in keys]).transform('sum')
    df['inc_fraction'] = np.where(s_inc > 0, w_inc / s_inc, 0.0)
    df['mort_fraction'] = np.where(s_mort > 0, w_mort / s_mort, 0.0)
    return df


def disaggregate_age_sex_malaria(
    forecast_df: pd.DataFrame,
    inc_count_col: str = 'aa_malaria_inc_count',
    mort_count_col: str = 'aa_malaria_mort_count',
    rr_inc_col: str = 'rr_inc_as',
    rr_mort_col: str = 'rr_mort_as',
    population_col: str = 'population',
    zero_age_group_id: int = 2,
) -> pd.DataFrame:
    """Disaggregate all-age malaria counts to age-sex strata using RR fractions.

    For each (location_id, year_id) group:
      1. Compute population-weighted RR: rr_pop = rr * population.
      2. Zero rr_pop at age_group_id == zero_age_group_id BEFORE summing, so that
         age group is excluded from the normalization denominator (neonates carry no
         malaria burden).
      3. Normalize: fraction = rr_pop / sum(rr_pop) over the *retained* ages.
      4. Multiply: as_count = fraction * aa_count.

    Order matters: age zero_age_group_id is dropped from the denominator, NOT zeroed
    after normalizing. Normalize-then-zero silently undercounts — the retained ages
    would sum to (1 - fraction_age2) * aa_count, losing the all-age total — whenever
    that age has nonzero rr (e.g. a swapped-in external age/sex pattern). Zero-then-
    normalize keeps the retained-age fractions summing to 1 and preserves the total.

    Groups whose retained-age rr_pop sum is 0 (no burden anywhere in the group) get a
    fraction of 0 rather than 0/0 = NaN, so a zero all-age count yields zero everywhere.

    Parameters
    ----------
    forecast_df:
        DataFrame with all-age count columns, age/sex-specific RR columns,
        and population. Must have 'location_id', 'year_id', 'age_group_id'.
    inc_count_col:
        All-age incidence count column name.
    mort_count_col:
        All-age mortality count column name.
    rr_inc_col:
        Age-sex relative risk column for incidence.
    rr_mort_col:
        Age-sex relative risk column for mortality.
    population_col:
        Population column name.
    zero_age_group_id:
        Age group ID excluded from the normalization. Default 2 (early neonatal).

    Returns
    -------
    forecast_df with 'malaria_inc_count_pred' and 'malaria_mort_count_pred' added.
    Intermediate columns (rr_*_pop, sum_rr_*_pop, *_fraction) are dropped.

    # Extracted from: 04_forecasting/as_malaria_fractions.py:236-255
    """
    df = forecast_df.copy()
    df = _as_fraction_columns(df, rr_inc_col, rr_mort_col, population_col, zero_age_group_id)
    df['malaria_inc_count_pred'] = df['inc_fraction'] * df[inc_count_col]
    df['malaria_mort_count_pred'] = df['mort_fraction'] * df[mort_count_col]
    df = df.drop(columns=['inc_fraction', 'mort_fraction'])
    return df


def compute_as_rr(
    as_df: pd.DataFrame,
    anchor_year: int = 2023,
    inc_rate_col: str = 'malaria_inc_rate',
    aa_inc_rate_col: str = 'aa_malaria_inc_rate',
    mort_rate_col: str = 'malaria_mort_rate',
    aa_mort_rate_col: str = 'aa_malaria_mort_rate',
) -> pd.DataFrame:
    """Age/sex relative-risk pattern rr = as_rate / aa_rate, frozen at one anchor year.

    ``anchor_year`` is a PARAMETER (default 2023 = the rake / last-observed year), not
    hard-coded — consistent with the rr source being swappable. rr is the observation-
    limited "borrowed" malaria age/sex pattern held fixed across the forecast;
    disaggregate_malaria_draws then ages the population under this fixed rr.

    Parameters
    ----------
    as_df:
        Observed age/sex malaria frame with 'location_id', 'year_id', 'age_group_id',
        'sex_id' and the four rate columns below (e.g. our raked-AS df).
    anchor_year:
        Year whose observed age/sex pattern to freeze.

    Returns
    -------
    [location_id, age_group_id, sex_id, rr_inc_as, rr_mort_as] — year dropped (the
    pattern is static). rr is 0 where the all-age rate is 0 (guarded, no 0/0).
    """
    a = as_df[as_df['year_id'] == anchor_year]
    if a.empty:
        raise ValueError(f"no rows at anchor_year={anchor_year} in the age/sex rate frame")
    out = a[['location_id', 'age_group_id', 'sex_id']].copy()
    out['rr_inc_as'] = np.where(a[aa_inc_rate_col] > 0, a[inc_rate_col] / a[aa_inc_rate_col], 0.0)
    out['rr_mort_as'] = np.where(a[aa_mort_rate_col] > 0, a[mort_rate_col] / a[aa_mort_rate_col], 0.0)
    return out.reset_index(drop=True)


def malaria_as_fractions(
    rr: pd.DataFrame,
    as_population: pd.DataFrame,
    *,
    rr_inc_col: str = 'rr_inc_as',
    rr_mort_col: str = 'rr_mort_as',
    population_col: str = 'population',
    zero_age_group_id: int = 2,
) -> pd.DataFrame:
    """The `(location_id, year_id, age_group_id, sex_id)` disaggregation fraction table.

    `f = rr·pop / Σ(rr·pop)`, normalized within each `(location, year)` over the
    RETAINED ages (`zero_age_group_id` excluded from the denominator). Draw-independent
    (neither rr nor pop carries a draw), which is exactly what lets the summary shortcut
    scale all-age draw statistics by `f` at the admin-2 leaf.

    Returns `[location_id, year_id, age_group_id, sex_id, inc_fraction, mort_fraction]`.
    Locations absent from `rr` contribute fraction 0.
    """
    frac = as_population.merge(rr, on=['location_id', 'age_group_id', 'sex_id'], how='left')
    frac[[rr_inc_col, rr_mort_col]] = frac[[rr_inc_col, rr_mort_col]].fillna(0.0)
    frac = _as_fraction_columns(frac, rr_inc_col, rr_mort_col, population_col, zero_age_group_id)
    return frac[['location_id', 'year_id', 'age_group_id', 'sex_id',
                 'inc_fraction', 'mort_fraction']]


def disaggregate_malaria_draws(
    aa_draws: pd.DataFrame,
    rr: pd.DataFrame,
    as_population: pd.DataFrame,
    aa_inc_count_col: str = 'aa_malaria_inc_count',
    aa_mort_count_col: str = 'aa_malaria_mort_count',
    rr_inc_col: str = 'rr_inc_as',
    rr_mort_col: str = 'rr_mort_as',
    population_col: str = 'population',
    zero_age_group_id: int = 2,
) -> pd.DataFrame:
    """Disaggregate all-age malaria draws to age/sex draws: (all-age draws + static rr +
    year-varying age/sex population) → age/sex draws.

    The reusable primitive behind both the saved ``as_full_summary`` product and the
    on-demand admin-2 age/sex reconstruction — neither reimplements the math.

    The disaggregation fraction ``f = rr·pop / Σ(rr·pop)`` carries NO draw dimension
    (``rr`` is the anchor-year malaria age/sex pattern; ``pop`` is demographic), so the
    ``(location_id, year_id, age_group_id, sex_id)`` fraction table is built ONCE and
    applied to every draw. Holding ``rr`` fixed while ``pop`` varies by year ages the
    population under a fixed epidemiological pattern.

    Memory: the ``aa_draws`` → age/sex merge multiplies row count by the age/sex grid
    (~50). The full standard product (all endemic admin-2 × all draws) is far too large
    to hold at once, so the driver feeds this per-draw and aggregates immediately; the
    on-demand path feeds a small location subset and takes all draws in one call.

    Parameters
    ----------
    aa_draws:
        [location_id, year_id, draw, aa_inc_count_col, aa_mort_count_col] — all-age
        counts per draw at the locations being disaggregated.
    rr:
        [location_id, age_group_id, sex_id, rr_inc_col, rr_mort_col] — static (anchor
        year) age/sex relative-risk pattern. Locations absent here contribute rr 0.
    as_population:
        [location_id, year_id, age_group_id, sex_id, population_col] — year-varying
        age/sex population (the demographic weights).

    Returns
    -------
    [location_id, year_id, age_group_id, sex_id, draw,
     malaria_inc_count_pred, malaria_mort_count_pred].
    """
    frac = malaria_as_fractions(
        rr, as_population,
        rr_inc_col=rr_inc_col, rr_mort_col=rr_mort_col,
        population_col=population_col, zero_age_group_id=zero_age_group_id,
    )

    out = aa_draws.merge(frac, on=['location_id', 'year_id'], how='inner')
    out['malaria_inc_count_pred'] = out['inc_fraction'] * out[aa_inc_count_col]
    out['malaria_mort_count_pred'] = out['mort_fraction'] * out[aa_mort_count_col]
    return out[['location_id', 'year_id', 'age_group_id', 'sex_id', 'draw',
                'malaria_inc_count_pred', 'malaria_mort_count_pred']]


# ---------------------------------------------------------------------------
# Dengue — log-rate + CFR method
# ---------------------------------------------------------------------------

def disaggregate_age_sex_dengue(
    forecast_df: pd.DataFrame,
    log_inc_rate_col: str = 'base_log_dengue_inc_rate_pred',
    cfr_col: str = 'dengue_cfr_pred',
    rr_inc_col: str = 'rr_inc_as',
    population_col: str = 'population',
) -> pd.DataFrame:
    """Compute age-sex dengue counts from a log-rate prediction and (optionally) CFR.

    Counts are computed directly — no all-age input required:
      inc_count = population * exp(log_inc_rate) * rr_inc_as
      mort_count = inc_count * cfr   (only when cfr_col is present)

    Mortality is derived only when ``cfr_col`` is a column of ``forecast_df``; on an
    incidence-only frame (no CFR), only ``dengue_inc_count_pred`` is added. This lets the
    same helper serve the production inc+mort disaggregation and an incidence-only pipeline.

    Parameters
    ----------
    forecast_df:
        DataFrame with log-rate, age-sex RR, and population columns (CFR optional).
    log_inc_rate_col:
        Log incidence rate column (base prediction, before age-sex shift).
    cfr_col:
        Case fatality rate column (already raked / post-logit-shift). If absent from
        ``forecast_df``, mortality is skipped.
    rr_inc_col:
        Age-sex relative risk for incidence.
    population_col:
        Population column name.

    Returns
    -------
    forecast_df with 'dengue_inc_count_pred' added, plus 'dengue_mort_count_pred' when
    ``cfr_col`` is present.

    # Extracted from: 04_forecasting/as_dengue_shifts.py:202-203
    """
    df = forecast_df.copy()

    df['dengue_inc_count_pred'] = (
        df[population_col] * np.exp(df[log_inc_rate_col]) * df[rr_inc_col]
    )
    if cfr_col in df.columns:
        df['dengue_mort_count_pred'] = df['dengue_inc_count_pred'] * df[cfr_col]

    return df
