"""Load + clean the malaria past-inputs parquet for pyGAM model selection.

Reproduces the ``load_past_data()`` steps used by the scam path so pyGAM trains
on the same rows + transforms:

- ``03_modeling/select_malaria_models_rocket.r`` — the *selection* row filter
  (``malaria_inc_count >= 1 & malaria_pfpr >= 1e-4``), used for PfPR
  spec selection (Phases 1-2).
- ``03_modeling/02_fit_final_malaria_models.r`` — the *final*-fit rows (drop NA
  on pfpr/gdppc/DAH/inc_rate/mort_rate, no modeled-count filter) plus the
  ``log_malaria_{inc,mort}_rate`` responses, used for the downstream inc/mort
  chain (Phase 3).

Transforms match the scam load exactly: suitability / days-over-30C / relative
humidity are turned to fractions, clipped to ``[1e-3, 1-1e-3]``, then logit'd;
DAH / GDP / ldipc / med_consumppc are logged. ``logit_malaria_pfpr`` is already
in the parquet (0.999-scaled at build time) and is used as-is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idd_forecast_mbp import constants as mbpc

DEFAULT_SUIT_VARIANT = "mordecai_0_0"
_FRAC_CLIP = (1e-3, 1.0 - 1e-3)  # matches scam load_past_data pmin/pmax clip
_LOG_COVS = ("mal_DAH_total_per_capita", "gdppc_mean", "ldipc_mean", "med_consumppc")
_ROW_FILTERS = ("modeled", "final", "none")


def default_parquet_path() -> Path:
    """The current malaria past-inputs parquet, resolved via constants (no hardcoded path)."""
    return Path(mbpc.MAL_PAST_INPUTS_READ_PATH) / "malaria_past_inputs.parquet"


def _logit_fraction(x, denom: float) -> np.ndarray:
    """``x/denom`` clipped to (0,1) then logit — the scam covariate transform."""
    frac = np.clip(np.asarray(x, dtype=float) / denom, *_FRAC_CLIP)
    return np.log(frac / (1.0 - frac))


def apply_transforms(df: pd.DataFrame, suit_variant: str = DEFAULT_SUIT_VARIANT) -> pd.DataFrame:
    """Add the modeled columns the scam load builds (mutates a copy, returns it).

    Split out from I/O so it is unit-testable on a synthetic frame.
    """
    df = df.copy()
    # Incidence COUNT = rate x population (parquet stores rate + pop, not the count);
    # required by the 'modeled' row filter.
    df["malaria_inc_count"] = df["malaria_inc_rate"] * df["population"]

    suit_col = f"malaria_suitability_{suit_variant}"
    if suit_col not in df.columns:
        raise KeyError(f"suitability variant column {suit_col!r} not in the parquet")
    df["malaria_suit"] = df[suit_col].astype(float)  # scam renames the variant to malaria_suit
    df["logit_malaria_suitability"] = _logit_fraction(df["malaria_suit"], 365)
    df["logit_do30"] = _logit_fraction(df["days_over_30C"], 365)
    df["logit_relative_humidity"] = _logit_fraction(df["relative_humidity"], 100)

    for cov in _LOG_COVS:
        df[f"log_{cov}"] = np.log(df[cov].astype(float))
    return df


def apply_row_filter(df: pd.DataFrame, row_filter: str) -> pd.DataFrame:
    """Drop rows to match the chosen scam regime; add log-rate responses for 'final'.

    - ``modeled``: NA-drop pfpr/gdppc/DAH, then ``inc_count >= 1 & pfpr >= 1e-4``
      (the selection worker's row set).
    - ``final``  : NA-drop pfpr/gdppc/DAH/inc_rate/mort_rate; add
      ``log_malaria_{inc,mort}_rate`` (log of a zero rate stays -inf and is
      dropped per-response at fit time, not here).
    - ``none``   : NA-drop pfpr/gdppc/DAH only.
    """
    if row_filter not in _ROW_FILTERS:
        raise ValueError(f"row_filter must be one of {_ROW_FILTERS}, got {row_filter!r}")

    for v in ("malaria_pfpr", "gdppc_mean", "mal_DAH_total_per_capita"):
        df = df[df[v].notna()]

    if row_filter == "modeled":
        df = df[(df["malaria_inc_count"] >= 1) & (df["malaria_pfpr"] >= 1e-4)]
    elif row_filter == "final":
        for v in ("malaria_inc_rate", "malaria_mort_rate"):
            df = df[df[v].notna()]
        df = df.copy()
        df["log_malaria_inc_rate"] = np.log(df["malaria_inc_rate"].astype(float))
        df["log_malaria_mort_rate"] = np.log(df["malaria_mort_rate"].astype(float))

    df = df.reset_index(drop=True)
    # pyGAM's factor term f() needs CONTIGUOUS 0..K-1 codes. Raw location_ids
    # (10..522, with gaps) make it silently under-fit the country effect (a real IS
    # fit went r_sq 0.73 -> 0.84 once encoded, and the smooths stop absorbing the
    # country-level signal). Encode on the FINAL (filtered) rows so codes are
    # gap-free; A0_location_id is kept so codes map back to countries.
    df["A0_af"] = df["A0_location_id"].astype("category").cat.codes.astype("int64")
    return df


def load_modeling_data(
    parquet_path: str | Path | None = None,
    *,
    suit_variant: str = DEFAULT_SUIT_VARIANT,
    row_filter: str = "modeled",
) -> pd.DataFrame:
    """Read the past-inputs parquet and return the cleaned modeling frame.

    Parameters
    ----------
    parquet_path : path or None
        Defaults to :func:`default_parquet_path` (resolved via constants).
    suit_variant : str
        Which ``malaria_suitability_<variant>`` column becomes ``malaria_suit``.
    row_filter : {'modeled', 'final', 'none'}
        See :func:`apply_row_filter`. ``modeled`` = PfPR selection (Phases 1-2);
        ``final`` = downstream inc/mort chain (Phase 3).
    """
    parquet_path = Path(parquet_path) if parquet_path is not None else default_parquet_path()
    df = pd.read_parquet(parquet_path)
    df = apply_transforms(df, suit_variant=suit_variant)
    return apply_row_filter(df, row_filter)


_LAG_TRANSFORMS = (None, "log", "logit")


def add_a0_lag(frame: pd.DataFrame, var: str = "malaria_pfpr", lag_years: int = 10,
               *, transform: str | None = None) -> pd.DataFrame:
    """Add a lagged admin-0 (population-weighted) covariate for ``var``.

    For each row (admin-2 location, year Y), the value is the country's
    population-weighted mean of ``var`` at year ``Y - lag_years``, computed over
    the admin-2s present in ``frame``. Rows whose lag year is absent (e.g.
    ``Y - lag_years`` before the data starts) get NaN — those are dropped at fit
    time by the model frame's na.omit, so the effective window shrinks to years
    >= min_year + lag_years.

    Adds ``lag{lag_years}_a0_{short}`` where ``short`` is ``var`` with any leading
    ``malaria_`` stripped (``malaria_pfpr`` -> ``lag10_a0_pfpr``,
    ``malaria_mort_rate`` -> ``lag10_a0_mort_rate``). ``transform`` optionally adds
    a transformed sibling column:

    - ``"logit"`` -> ``logit_lag{...}`` (clip to [1e-3, 1-1e-3]; for prevalences
      like PfPR that span the full (0,1) range).
    - ``"log"``   -> ``log_lag{...}`` (plain ``np.log``, matching the module's
      ``log_malaria_mort_rate`` response; for small rates like mort_rate where the
      logit clip floor would swallow the signal). An exact-zero country mean gives
      -inf (essentially never for a pop-weighted average over endemic locs).

    The population-weighted average is NaN-safe: only rows with a non-NaN ``var``
    contribute to both numerator and denominator, so a partly-missing column (e.g.
    ``malaria_mort_rate`` in the 'modeled' frame, which only guarantees pfpr/gdppc/
    DAH) is handled correctly. Compute this on the FULL frame before any row-subset
    so the country average uses all its locs.
    """
    if transform not in _LAG_TRANSFORMS:
        raise ValueError(f"transform must be one of {_LAG_TRANSFORMS}, got {transform!r}")

    key = ["A0_location_id", "year_id"]
    valid = frame[frame[var].notna() & frame["population"].notna()]
    num = (valid[var] * valid["population"]).groupby([valid[k] for k in key]).sum()
    den = valid.groupby(key)["population"].sum()
    a0_avg = num / den

    short = var[len("malaria_"):] if var.startswith("malaria_") else var
    col = f"lag{lag_years}_a0_{short}"
    lag = a0_avg.rename(col).reset_index()
    lag["year_id"] = lag["year_id"] + lag_years  # a country's year-T value aligns to target year T+lag
    out = frame.merge(lag, on=key, how="left")

    if transform == "logit":
        frac = np.clip(out[col].to_numpy(dtype=float), 1e-3, 1.0 - 1e-3)
        out[f"logit_{col}"] = np.log(frac / (1.0 - frac))
    elif transform == "log":
        out[f"log_{col}"] = np.log(out[col].to_numpy(dtype=float))
    return out


def add_a0_pfpr_reference(frame: pd.DataFrame, ref_year: int = 2023, *,
                          logit: bool = False, population_weighted: bool = True) -> pd.DataFrame:
    """Add a time-invariant admin-0 reference-year PfPR covariate.

    Every row of a country gets that country's PfPR at ``ref_year`` — the mean of
    admin-2 ``malaria_pfpr`` (population-weighted, or simple mean if
    ``population_weighted=False``) — broadcast across ALL years. It's a single
    continuous country-level proxy for the fixed effect, available for every year
    (no window shrink, unlike the lag). Countries with no ``ref_year`` data → NaN.

    The average is over the admin-2s in ``frame``: for the 'modeled' frame that's
    only endemic locations, so it is NOT a true whole-country average — fine as a
    covariate/ranking, but pass a full (unfiltered) frame for the true value.

    Adds ``a0_pfpr_{ref_year}`` (+ ``logit_a0_pfpr_{ref_year}`` if ``logit``).
    """
    ref = frame[frame["year_id"] == ref_year]
    if population_weighted:
        a0 = ((ref["malaria_pfpr"] * ref["population"]).groupby(ref["A0_location_id"]).sum()
              / ref.groupby("A0_location_id")["population"].sum())
    else:
        a0 = ref.groupby("A0_location_id")["malaria_pfpr"].mean()

    col = f"a0_pfpr_{ref_year}"
    ref_tbl = a0.rename(col).reset_index()
    out = frame.merge(ref_tbl, on="A0_location_id", how="left")  # broadcast to every year
    if logit:
        frac = np.clip(out[col].to_numpy(dtype=float), 1e-3, 1.0 - 1e-3)
        out[f"logit_{col}"] = np.log(frac / (1.0 - frac))
    return out
