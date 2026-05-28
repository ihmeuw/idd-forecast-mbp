"""Array-shaped DAH scenario builder for the forecast-input netCDF.

Produces a (n_loc, n_year, n_scenario) float32 array of
mal_DAH_total_per_capita with one slice per requested scenario. Mirrors the
arithmetic in `lib/processing/scenarios.generate_dah_scenarios` (which returns
DataFrames) — both will be kept in sync until the legacy DataFrame helper is
retired with the per-draw forecast scripts.

Scenarios:
  Baseline   — per-capita DAH from source, broadcast A0 → A2 via hierarchy.
               Missing (A0, year) rows fill with 0.0 per DECISIONS 2026-05-07.
  Constant   — DAH_total held at reference_year value; per-capita recomputed
               against actual population so the funding amount stays fixed
               as population changes.
  Increasing — per-capita multiplied by [1.2, 1.4, 1.6, 1.8, 2.0] starting
               at modification_start_year, then held at 2.0× thereafter.
  Decreasing — per-capita multiplied by [0.8, 0.6, 0.4, 0.2, 0.0] starting
               at modification_start_year, then held at 0.0 thereafter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INCREASING_RAMP: tuple[float, ...] = (1.2, 1.4, 1.6, 1.8, 2.0)
DECREASING_RAMP: tuple[float, ...] = (0.8, 0.6, 0.4, 0.2, 0.0)
SUPPORTED_SCENARIOS: frozenset[str] = frozenset(
    {"Baseline", "Constant", "Increasing", "Decreasing"}
)


def _baseline_array(
    location_ids: list[int],
    years: list[int],
    hierarchy_df: pd.DataFrame,
    dah_df: pd.DataFrame,
) -> np.ndarray:
    """Broadcast A0-level mal_DAH_total_per_capita to (n_loc, n_year)."""
    loc_to_a0 = hierarchy_df.set_index('location_id')['A0_location_id'].to_dict()
    grid = pd.MultiIndex.from_product(
        [location_ids, years], names=['location_id', 'year_id']
    ).to_frame(index=False)
    grid['A0_location_id'] = grid['location_id'].map(loc_to_a0)

    dah_renamed = dah_df.rename(columns={'location_id': 'A0_location_id'})
    merged = grid.merge(
        dah_renamed[['A0_location_id', 'year_id', 'mal_DAH_total_per_capita']],
        on=['A0_location_id', 'year_id'], how='left',
    )
    merged['mal_DAH_total_per_capita'] = merged['mal_DAH_total_per_capita'].fillna(0.0)

    return (
        merged.set_index(['location_id', 'year_id'])['mal_DAH_total_per_capita']
        .values.reshape(len(location_ids), len(years))
        .astype(np.float32)
    )


def _apply_ramp(
    baseline: np.ndarray,
    years: list[int],
    ramp: tuple[float, ...],
    modification_start_year: int,
) -> np.ndarray:
    """Multiply baseline by `ramp` factors starting at modification_start_year,
    then hold at the terminal factor for years past the ramp."""
    out = baseline.copy()
    for i, factor in enumerate(ramp):
        y = modification_start_year + i
        if y in years:
            j = years.index(y)
            out[:, j] = baseline[:, j] * factor
    max_year = modification_start_year + len(ramp) - 1
    terminal = ramp[-1]
    for j, y in enumerate(years):
        if y > max_year:
            out[:, j] = baseline[:, j] * terminal
    return out


def _constant_array(
    baseline: np.ndarray,
    population: np.ndarray,
    years: list[int],
    reference_year: int,
) -> np.ndarray:
    """Hold DAH_total constant at reference_year value; recompute per_capita
    against the actual population in each subsequent year.

    For year <= reference_year: per_capita unchanged from baseline.
    For year >  reference_year: per_capita[y] = per_capita[ref] * pop[ref] / pop[y].
    """
    if reference_year not in years:
        raise ValueError(
            f"reference_year={reference_year} not in years; cannot build Constant scenario."
        )
    ref_idx = years.index(reference_year)
    per_capita_ref = baseline[:, ref_idx]
    pop_ref        = population[:, ref_idx]
    dah_total_ref  = per_capita_ref * pop_ref     # (n_loc,)

    out = baseline.copy()
    for j, y in enumerate(years):
        if y > reference_year:
            pop_y = population[:, j]
            with np.errstate(divide='ignore', invalid='ignore'):
                per_capita_y = np.where(pop_y > 0, dah_total_ref / pop_y, 0.0)
            out[:, j] = per_capita_y.astype(np.float32)
    return out


def build_dah_array(
    location_ids: list[int],
    years: list[int],
    hierarchy_df: pd.DataFrame,
    dah_df: pd.DataFrame,
    population: np.ndarray,
    scenarios: tuple[str, ...] = ("Baseline", "Constant"),
    reference_year: int = 2023,
    modification_start_year: int = 2026,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build (n_loc, n_year, n_scenario) mal_DAH_total_per_capita array.

    Parameters
    ----------
    location_ids:
        Prediction location IDs (A2 / level-5).
    years:
        Output years (must include reference_year if "Constant" is requested).
    hierarchy_df:
        Full hierarchy DataFrame with `location_id` and `A0_location_id` columns.
    dah_df:
        Raw DAH source (one row per A0 location_id × year_id) with column
        `mal_DAH_total_per_capita`.
    population:
        (n_loc, n_year) population at A2; only used by the Constant scenario.
    scenarios:
        Scenario names to include, in order. Each must be in SUPPORTED_SCENARIOS.
    reference_year, modification_start_year:
        See module docstring.

    Returns
    -------
    (array of shape (n_loc, n_year, len(scenarios)), tuple of scenario names)
    """
    unknown = set(scenarios) - SUPPORTED_SCENARIOS
    if unknown:
        raise ValueError(
            f"Unknown DAH scenario(s): {sorted(unknown)}. "
            f"Supported: {sorted(SUPPORTED_SCENARIOS)}."
        )
    if population.shape != (len(location_ids), len(years)):
        raise ValueError(
            f"population shape {population.shape} does not match "
            f"(n_loc={len(location_ids)}, n_year={len(years)})."
        )

    baseline = _baseline_array(location_ids, years, hierarchy_df, dah_df)

    slices: list[np.ndarray] = []
    for scen in scenarios:
        if scen == "Baseline":
            slices.append(baseline.copy())
        elif scen == "Constant":
            slices.append(_constant_array(baseline, population, years, reference_year))
        elif scen == "Increasing":
            slices.append(_apply_ramp(baseline, years, INCREASING_RAMP, modification_start_year))
        elif scen == "Decreasing":
            slices.append(_apply_ramp(baseline, years, DECREASING_RAMP, modification_start_year))

    return np.stack(slices, axis=-1).astype(np.float32), tuple(scenarios)
