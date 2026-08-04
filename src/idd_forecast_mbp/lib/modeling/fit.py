"""Fit + predict a pyGAM spec under IS / within-country / temporal evaluation.

Reproduces the fold logic of ``run_oos()`` in
``03_modeling/select_malaria_models_rocket.r``:

- ``within`` : n-fold CV with folds assigned *within* each country (FE-present,
  so every country is in every training fold).
- ``Temporal`` : one split — fit ``[train_lo, train_hi]``, predict
  ``[test_lo, test_hi]``; test rows whose country is absent from the training
  years are dropped (no factor level to predict). ``TEMP_A`` / ``TEMP_B``
  mirror ``OOS_VERSIONS`` in the orchestrator.
- ``IS`` : fit on all rows, predict on all rows.

The response is a parameter (:class:`Outcome`), so the *same* code fits PfPR and
the downstream incidence / mortality models. :func:`fit_predict` returns per-row
predictions (location_id, year_id, fold, obs, pred, obs_natural, pred_natural),
so by-fold and pooled metrics both fall out. ``fit_gam`` / ``predict_gam`` are
the low-level split-agnostic primitives the Phase-3 chain (and a future
forecaster) build on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pygam import LinearGAM

from idd_forecast_mbp.lib.modeling import specs as specs_mod
from idd_forecast_mbp.lib.utils.transforms import expit

A0_COL = "A0_af"  # integer country key (from data.apply_transforms)


@dataclass(frozen=True)
class Outcome:
    """A response to fit: the model-space column, the natural-space observed
    column used for scoring, and the inverse link to back-transform predictions."""
    response: str        # model-space column the GAM fits (logit/log)
    natural_col: str     # observed natural-space column for scoring
    link: str            # "logit" | "log"


PFPR = Outcome("logit_malaria_pfpr", "malaria_pfpr", "logit")
INCIDENCE = Outcome("log_malaria_inc_rate", "malaria_inc_rate", "log")
MORTALITY = Outcome("log_malaria_mort_rate", "malaria_mort_rate", "log")


@dataclass(frozen=True)
class Temporal:
    """A temporal OOS split: fit [train_lo, train_hi], predict [test_lo, test_hi]."""
    train_lo: int
    train_hi: int
    test_lo: int
    test_hi: int
    name: str = "temporal"


# Mirror OOS_VERSIONS in fit_malaria_models_orchestrator.py.
TEMP_A = Temporal(2000, 2015, 2020, 2023, name="tempA")
TEMP_B = Temporal(2010, 2019, 2020, 2023, name="tempB")
OOS_VERSIONS = {"within": "within", "tempA": TEMP_A, "tempB": TEMP_B}


@dataclass
class FitResult:
    predictions: pd.DataFrame   # location_id, year_id, fold, obs, pred, obs_natural, pred_natural
    models: list                # fitted LinearGAM, one per fold (1 for IS/temporal)
    evaluation: str
    n_dropped: int              # rows dropped by the na.omit on (features + response)
    meta: dict = field(default_factory=dict)


def _inverse(link: str, x: np.ndarray) -> np.ndarray:
    """Back-transform model-space predictions to natural space."""
    if link == "logit":
        return expit(x)
    if link == "log":
        return np.exp(x)
    raise ValueError(f"unknown link {link!r} (use 'logit' or 'log')")


def _finite_mask(frame: pd.DataFrame, cols: list[str], response: str | None) -> np.ndarray:
    """True where the response (if given) and every feature column are finite."""
    mask = np.ones(len(frame), dtype=bool)
    if response is not None:
        mask &= np.isfinite(pd.to_numeric(frame[response], errors="coerce").to_numpy(float))
    for c in cols:
        mask &= np.isfinite(pd.to_numeric(frame[c], errors="coerce").to_numpy(float))
    return mask


def _model_frame(data: pd.DataFrame, cols: list[str], outcome: Outcome) -> tuple[pd.DataFrame, int]:
    """Keep the columns fit/score need and na.omit on (features + response)."""
    need = list(dict.fromkeys(
        cols + [outcome.response, outcome.natural_col, "location_id", "year_id", A0_COL]
    ))
    frame = data[need].copy()
    mask = _finite_mask(frame, cols, outcome.response)
    dropped = int((~mask).sum())
    return frame.loc[mask].reset_index(drop=True), dropped


def _fit_terms(frame: pd.DataFrame, cols: list[str], spec, response: str,
               *, factor_lam=None, gridsearch=False, lam_grid=None) -> LinearGAM:
    """Fit a LinearGAM on an already-clean frame (no internal na.omit)."""
    X = frame[cols].to_numpy(dtype=float)
    y = frame[response].to_numpy(dtype=float)
    terms = specs_mod.spec_to_terms(spec, factor_lam=factor_lam)
    gam = LinearGAM(terms)
    if gridsearch:
        grid = lam_grid if lam_grid is not None else np.logspace(-3, 3, 11)
        gam.gridsearch(X, y, lam=grid, progress=False)
    else:
        gam.fit(X, y)
    return gam


def _predict_rows(gam: LinearGAM, frame: pd.DataFrame, cols: list[str],
                  outcome: Outcome, fold) -> pd.DataFrame:
    """Predict on ``frame`` (dropping any non-finite-feature rows) -> per-row frame."""
    if len(frame) == 0:
        return pd.DataFrame(columns=["location_id", "year_id", "fold", "obs", "pred",
                                     "obs_natural", "pred_natural"])
    mask = _finite_mask(frame, cols, response=None)
    sub = frame.loc[mask]
    pred = np.asarray(gam.predict(sub[cols].to_numpy(dtype=float)), dtype=float)
    return pd.DataFrame({
        "location_id": sub["location_id"].to_numpy(),
        "year_id": sub["year_id"].to_numpy(),
        "fold": fold,
        "obs": sub[outcome.response].to_numpy(dtype=float),
        "pred": pred,
        "obs_natural": sub[outcome.natural_col].to_numpy(dtype=float),
        "pred_natural": _inverse(outcome.link, pred),
    })


# --- public split-agnostic primitives (used by fit_predict + the Phase-3 chain) ---

def fit_gam(data: pd.DataFrame, spec, outcome: Outcome = PFPR,
            *, factor_lam=None, gridsearch=False, lam_grid=None) -> LinearGAM:
    """Fit one GAM on ``data`` for ``spec``/``outcome`` (na.omit'd internally)."""
    cols = specs_mod.spec_columns(spec)
    frame, _ = _model_frame(data, cols, outcome)
    return _fit_terms(frame, cols, spec, outcome.response,
                      factor_lam=factor_lam, gridsearch=gridsearch, lam_grid=lam_grid)


def predict_gam(gam: LinearGAM, data: pd.DataFrame, spec, outcome: Outcome = PFPR,
                *, fold="predict") -> pd.DataFrame:
    """Predict a fitted GAM on ``data`` -> per-row frame (non-finite features dropped)."""
    cols = specs_mod.spec_columns(spec)
    need = list(dict.fromkeys(
        cols + [outcome.response, outcome.natural_col, "location_id", "year_id", A0_COL]
    ))
    return _predict_rows(gam, data[need].copy(), cols, outcome, fold)


# --- fold assignment ----------------------------------------------------------

def within_country_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """Assign each row a fold in [0, n_folds), balanced *within* each country.

    Mirrors run_oos's within_country strategy (ave over A0 of a shuffled
    rep_len). Uses numpy's RNG, so the exact assignment differs from R's, but
    the structure (each country split across folds) is identical.
    """
    rng = np.random.default_rng(seed)
    folds = np.empty(len(frame), dtype=int)
    for _, pos in frame.groupby(A0_COL).indices.items():
        f = np.resize(np.arange(n_folds), len(pos))
        rng.shuffle(f)
        folds[pos] = f
    return folds


def _splits(frame: pd.DataFrame, evaluation, n_folds: int, cv_seed: int):
    """Yield (fold_label, train_frame, test_frame) for the requested evaluation."""
    if evaluation == "IS":
        yield "IS", frame, frame
    elif evaluation == "within":
        folds = within_country_folds(frame, n_folds, cv_seed)
        for k in range(n_folds):
            yield k, frame.loc[folds != k], frame.loc[folds == k]
    elif isinstance(evaluation, Temporal):
        yr = frame["year_id"].to_numpy()
        train_mask = (yr >= evaluation.train_lo) & (yr <= evaluation.train_hi)
        test_mask = (yr >= evaluation.test_lo) & (yr <= evaluation.test_hi)
        seen = set(frame.loc[train_mask, A0_COL].unique())
        unseen = test_mask & ~frame[A0_COL].isin(seen).to_numpy()
        test_mask = test_mask & ~unseen
        yield evaluation.name, frame.loc[train_mask], frame.loc[test_mask]
    else:
        raise ValueError(f"unknown evaluation {evaluation!r} "
                         "(use 'IS', 'within', or a Temporal)")


def fit_predict(data: pd.DataFrame, spec, outcome: Outcome = PFPR, evaluation="IS",
                *, n_folds: int = 5, cv_seed: int = 42,
                factor_lam=None, gridsearch=False, lam_grid=None) -> FitResult:
    """Fit ``spec`` for ``outcome`` under ``evaluation`` and return per-row predictions.

    ``evaluation`` is ``"IS"``, ``"within"``, or a :class:`Temporal` (e.g.
    :data:`TEMP_A`). ``factor_lam`` overrides the country-factor penalty (small
    -> approximate unpenalized FE). Set ``gridsearch=True`` for the shared-``lam``
    sweep.
    """
    cols = specs_mod.spec_columns(spec)
    frame, dropped = _model_frame(data, cols, outcome)
    parts, models, meta = [], [], {}
    for label, train, test in _splits(frame, evaluation, n_folds, cv_seed):
        gam = _fit_terms(train, cols, spec, outcome.response,
                         factor_lam=factor_lam, gridsearch=gridsearch, lam_grid=lam_grid)
        models.append(gam)
        parts.append(_predict_rows(gam, test, cols, outcome, label))
        meta.setdefault("fold_sizes", {})[str(label)] = int(len(test))
    preds = pd.concat(parts, ignore_index=True) if parts else parts
    ev_name = evaluation.name if isinstance(evaluation, Temporal) else str(evaluation)
    return FitResult(predictions=preds, models=models, evaluation=ev_name,
                     n_dropped=dropped, meta=meta)
