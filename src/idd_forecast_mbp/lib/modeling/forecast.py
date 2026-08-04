"""Compounded rolling out-of-sample forecast for pyGAM PfPR models.

Extends ``fit.fit_predict`` / ``fit.Temporal``. Where ``fit.Temporal`` feeds the
model the *observed* lagged covariate in test years (a generalization check),
this rolls the forecast forward one year at a time and feeds the model's *own*
predictions back into the lag covariate — the production semantics, where the
lagged input at a forecast year is itself a prior forecast and error compounds.

Only a lag covariate whose source **is the response being predicted** is rolled
(:attr:`LagCovariate.roll`); everything else — base covariates, a fixed effect,
or an *exogenous* lag of another variable — is an ordinary observed feature.
That is a deliberate consequence of the PfPR-AR architecture (the admin-0 PfPR
lag is the only covariate mechanically a function of the thing being predicted,
because inc/mort are ~invertible transforms of PfPR and were dropped in favour of
PfPR-AR), **not** an unhandled case. The harness requires exactly one rolled lag,
so the recursion depth is defined against a single, unambiguous chain.

Built entirely on the split-agnostic primitives :func:`fit.fit_gam` /
:func:`fit.predict_gam`, so ``fit.Temporal`` / ``fit.fit_predict`` are untouched.

Per-prediction stamps
---------------------
- ``recursion_depth`` : counted during the roll (0 = the lag reached back into
  observed data; +1 each time a predicted value feeds the lag). Counted from the
  source year's stamped depth, not a closed-form formula, so it is robust to the
  (validated-against) irregular-year case.
- ``seed_year``       : the observed year that ultimately grounds the chain.
- ``is_seeded_from_observed`` : ``recursion_depth == 0``.
The injected lag value used for each row is also attached (column ``lag.column``)
for transparency and for the depth>=1 correctness test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pygam import LinearGAM

from idd_forecast_mbp.lib.modeling import data as data_mod
from idd_forecast_mbp.lib.modeling import fit

_KEYS = ["A0_location_id", "year_id"]


@dataclass(frozen=True)
class LagCovariate:
    """Description of a lagged admin-0 covariate for the roll-forward.

    ``column`` is the design-matrix column the spec references (e.g.
    ``"logit_lag5_a0_pfpr"``); ``source_var`` / ``lag_years`` / ``transform`` are
    exactly the arguments to :func:`data.add_a0_lag` that build it. ``roll`` is
    True only when ``source_var`` is the response being predicted.
    """
    column: str
    source_var: str
    lag_years: int
    transform: str | None = None
    roll: bool = True


@dataclass
class RollingForecastResult:
    predictions: pd.DataFrame   # per (loc, year) + recursion_depth, seed_year, is_seeded_from_observed, lag col
    gam: LinearGAM
    train_end: int
    lags: tuple[LagCovariate, ...]
    meta: dict = field(default_factory=dict)


def _a0_lag_lookup(src: pd.DataFrame, lag: LagCovariate) -> pd.DataFrame:
    """Unique ``(A0_location_id, year_id, lag.column)`` table from a per-admin-2
    ``src`` frame (columns: the keys + ``population`` + ``lag.source_var``).

    Reuses :func:`data.add_a0_lag` (single source of truth for the pop-weighted
    aggregate + shift + transform), then dedupes to one row per country-year so it
    can be broadcast back onto the admin-2 rows without exploding the merge.
    """
    tab = data_mod.add_a0_lag(src, var=lag.source_var, lag_years=lag.lag_years,
                              transform=lag.transform)
    return tab[_KEYS + [lag.column]].drop_duplicates(_KEYS)


def _validate_consecutive(years: list[int], train_end: int) -> None:
    """forecast_years must be consecutive integers starting at ``train_end + 1``.

    Contiguity alone isn't enough — a step>1 (e.g. every other year) or a start
    later than train_end+1 orphans a lag chain, so check the exact sequence.
    """
    if not years:
        raise ValueError("forecast_years is empty")
    expected = list(range(train_end + 1, train_end + 1 + len(years)))
    if years != expected:
        raise ValueError(
            "forecast_years must be consecutive integers starting at "
            f"train_end+1 ({train_end + 1}..{train_end + len(years)}); got "
            f"{years!r}. A gap, step>1, or a later start orphans a lag chain.")


def rolling_forecast(
    data: pd.DataFrame,
    spec,
    lags: "LagCovariate | Sequence[LagCovariate]",
    *,
    train_end: int,
    forecast_years: Iterable[int] | None = None,
    train_lo: int | None = None,
    outcome: fit.Outcome = fit.PFPR,
    factor_lam=None,
    restrict_to_seen_countries: bool = False,
) -> RollingForecastResult:
    """Roll a compounded forecast forward from a model fit on ``years <= train_end``.

    ``data`` must carry ``location_id, year_id, A0_location_id, population``, the
    outcome response + natural columns, ``A0_af``, and the spec's base-covariate
    columns (as ``data.load_modeling_data`` provides). Exactly one lag in ``lags``
    must have ``roll=True`` and its ``source_var`` must equal
    ``outcome.natural_col``. ``restrict_to_seen_countries`` is for the equivalence
    check only (a real forecast predicts all rows); leave it False otherwise.
    """
    lag_list = (lags,) if isinstance(lags, LagCovariate) else tuple(lags)
    roll_lags = [l for l in lag_list if l.roll]
    if len(roll_lags) != 1:
        raise ValueError(
            f"exactly one rolled lag is required (got {len(roll_lags)}); the "
            "recursion depth is defined against a single chain. Non-response "
            "(exogenous) lags must have roll=False.")
    (roll_lag,) = roll_lags
    if roll_lag.source_var != outcome.natural_col:
        raise ValueError(
            f"rolled lag {roll_lag.column!r} has source_var={roll_lag.source_var!r} "
            f"but the outcome's natural column is {outcome.natural_col!r}; only a "
            "lag OF the response can be rolled.")

    data = data.copy()
    L = roll_lag.lag_years

    # Working natural-response series: OBSERVED for <= train_end, filled with
    # predictions afterwards. Blanking > train_end once guarantees an observed
    # value at a forecast year can never re-enter the lag.
    work = data[["location_id", "year_id", "A0_location_id", "population",
                 outcome.natural_col]].copy()
    work = work.rename(columns={outcome.natural_col: "_yhat_nat"})
    work.loc[work["year_id"] > train_end, "_yhat_nat"] = np.nan

    # Observed rolled-lag column for the FIT. Building from the blanked `work`
    # yields the correct observed lag for every training year (their sources are
    # all <= train_end); forecast-year values are overwritten in the loop.
    src = work.rename(columns={"_yhat_nat": roll_lag.source_var})[
        ["A0_location_id", "year_id", "population", roll_lag.source_var]]
    data = data.drop(columns=[roll_lag.column], errors="ignore").merge(
        _a0_lag_lookup(src, roll_lag), on=_KEYS, how="left")

    train = data[data["year_id"] <= train_end]
    if train_lo is not None:
        train = train[train["year_id"] >= train_lo]
    gam = fit.fit_gam(train, spec, outcome, factor_lam=factor_lam)
    seen = set(train[fit.A0_COL].unique()) if restrict_to_seen_countries else None

    if forecast_years is None:
        forecast_years = range(train_end + 1, int(data["year_id"].max()) + 1)
    forecast_years = sorted(int(y) for y in forecast_years)
    _validate_consecutive(forecast_years, train_end)

    depth_of: dict[int, int] = {}
    seed_of: dict[int, int] = {}
    n_dropped: dict[int, int] = {}
    out = []
    for t in forecast_years:
        # (1) rebuild the rolled lag from the current working response and take year t
        src_t = work.rename(columns={"_yhat_nat": roll_lag.source_var})[
            ["A0_location_id", "year_id", "population", roll_lag.source_var]]
        lookup = _a0_lag_lookup(src_t, roll_lag)
        col_t = lookup[lookup["year_id"] == t]

        # (2) predict frame for year t = observed features(t) + injected rolled lag
        frame_t = data[data["year_id"] == t].drop(columns=[roll_lag.column], errors="ignore")
        frame_t = frame_t.merge(col_t, on=_KEYS, how="left")
        if seen is not None:
            frame_t = frame_t[frame_t[fit.A0_COL].isin(seen)]

        preds_t = fit.predict_gam(gam, frame_t, spec, outcome, fold=t)
        preds_t = preds_t.merge(frame_t[["location_id", roll_lag.column]],
                                on="location_id", how="left")  # attach the injected lag value

        # (3) depth/seed, counted from the source year's stamped state
        s = t - L
        if s <= train_end:
            depth, seed = 0, s
        else:
            depth, seed = depth_of[s] + 1, seed_of[s]
        depth_of[t], seed_of[t] = depth, seed
        preds_t["recursion_depth"] = depth
        preds_t["seed_year"] = seed
        preds_t["is_seeded_from_observed"] = (depth == 0)

        # (4) feed predictions back into the working response (natural space)
        fill = preds_t.set_index("location_id")["pred_natural"]
        m = work["year_id"] == t
        work.loc[m, "_yhat_nat"] = work.loc[m, "location_id"].map(fill).to_numpy()

        n_dropped[t] = int(len(frame_t) - len(preds_t))
        out.append(preds_t)

    preds = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    meta = {
        "forecast_years": forecast_years,
        "train_lo": train_lo,
        "lag_years": L,
        "n_dropped_per_year": n_dropped,
        "restrict_to_seen_countries": restrict_to_seen_countries,
        "exogenous_lags_observed_only": [l.column for l in lag_list if not l.roll],
        "note": ("exogenous (non-response) lags are observed-only by the PfPR-AR "
                 "architecture decision, not an unhandled case"),
    }
    return RollingForecastResult(preds, gam, train_end, lag_list, meta)


def compare_modes(
    data: pd.DataFrame,
    spec,
    lag: LagCovariate,
    *,
    train_lo: int,
    train_end: int,
    forecast_years: Iterable[int],
    outcome: fit.Outcome = fit.PFPR,
    factor_lam=None,
) -> tuple[fit.FitResult, RollingForecastResult]:
    """Run the observed-covariate OOS (``fit.fit_predict`` + ``fit.Temporal``) and
    the compounded roll on the same window + same training rows.

    Returns ``(observed, rolled)``. The correctness gate is that
    ``rolled.predictions[recursion_depth == 0]`` reproduces ``observed`` on shared
    ``(location_id, year_id)`` — so the roll is run with
    ``restrict_to_seen_countries=True`` to match ``fit_predict``'s test-row set
    (the flag's only real use). The actual forecast never sets it.
    """
    data = data.copy()
    fyears = sorted(int(y) for y in forecast_years)

    # ensure the observed lag column exists for the fit_predict path
    src = data[["A0_location_id", "year_id", "population", lag.source_var]]
    data = data.drop(columns=[lag.column], errors="ignore").merge(
        _a0_lag_lookup(src, lag), on=_KEYS, how="left")

    tw = fit.Temporal(train_lo, train_end, fyears[0], fyears[-1], name="observed")
    observed = fit.fit_predict(data, spec, outcome, tw, factor_lam=factor_lam)
    rolled = rolling_forecast(
        data, spec, lag, train_end=train_end, forecast_years=fyears, train_lo=train_lo,
        outcome=outcome, factor_lam=factor_lam, restrict_to_seen_countries=True)
    return observed, rolled
