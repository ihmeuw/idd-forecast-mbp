"""Scoring metrics for pyGAM malaria model selection.

One canonical residual-metric set, computed identically for any outcome
(PfPR, incidence, mortality) and any evaluation (in-sample or out-of-sample),
so a spec's skill is comparable across the whole chain. The formulas mirror
``is_block_metrics`` / ``run_oos`` in
``03_modeling/select_malaria_models_rocket.r``:

- model space (the logit/log response the GAM actually fits): ssr, rmse, mae, r_sq, r
- natural space (back-transformed prevalence / rate): the same set, prefixed ``nat_``

The headline PfPR selection metric ``oos_pfpr_r`` is ``nat_r`` here (Pearson r
between observed prevalence and back-transformed predicted prevalence).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays and keep only rows finite in BOTH (na.omit analog)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def residual_metrics(observed, predicted, *, prefix: str = "") -> dict:
    """RMSE / MAE / SSR / R^2 + Pearson r between observed and predicted.

    Space-agnostic: pass model-space vectors (logit/log) or natural-space
    vectors (prevalence/rate). NA/inf pairs are dropped before computing.
    ``r_sq`` is 1 - SSR/SST (can be negative for poor OOS predictions, matching
    the scam path); ``r`` is Pearson correlation in whatever space was passed.
    """
    obs, pred = _finite_pair(observed, predicted)
    n = int(obs.size)
    if n == 0:
        keys = ("n_obs", "ssr", "rmse", "mae", "r_sq", "r")
        out = {k: (0 if k == "n_obs" else np.nan) for k in keys}
        return {f"{prefix}{k}": v for k, v in out.items()} if prefix else out

    resid = obs - pred
    ssr = float(np.sum(resid ** 2))
    sst = float(np.sum((obs - obs.mean()) ** 2))
    out = {
        "n_obs": n,
        "ssr": ssr,
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "r_sq": (1.0 - ssr / sst) if sst > 0 else np.nan,
        "r": float(np.corrcoef(obs, pred)[0, 1]) if n > 1 else np.nan,
    }
    return {f"{prefix}{k}": v for k, v in out.items()} if prefix else out


def score_predictions(
    pred_frame: pd.DataFrame,
    *,
    obs_col: str = "obs",
    pred_col: str = "pred",
    nat_obs_col: str = "obs_natural",
    nat_pred_col: str = "pred_natural",
) -> dict:
    """Full metric set for a ``fit.FitResult.predictions`` frame.

    Model-space metrics from (obs, pred); natural-space metrics (prefixed
    ``nat_``) from (obs_natural, pred_natural) when those columns are present.
    """
    out = residual_metrics(pred_frame[obs_col], pred_frame[pred_col])
    if {nat_obs_col, nat_pred_col}.issubset(pred_frame.columns):
        out.update(residual_metrics(pred_frame[nat_obs_col], pred_frame[nat_pred_col], prefix="nat_"))
    return out


def score_by_fold(pred_frame: pd.DataFrame, *, fold_col: str = "fold", **kw) -> pd.DataFrame:
    """Per-fold metrics (one row per fold) + a pooled 'ALL' row.

    Pooled metrics are computed over all rows at once (not an average of the
    per-fold numbers), matching the scam ``oos_*`` columns which pool the
    held-out predictions across folds.
    """
    rows = []
    for fold, sub in pred_frame.groupby(fold_col, sort=False):
        rows.append({"fold": fold, **score_predictions(sub, **kw)})
    rows.append({"fold": "ALL", **score_predictions(pred_frame, **kw)})
    return pd.DataFrame(rows)


def score_by_depth(pred_frame: pd.DataFrame, *, depth_col: str = "recursion_depth", **kw) -> pd.DataFrame:
    """Per-recursion-depth metrics (one row per depth) + a pooled 'ALL' row.

    The companion to :func:`score_by_fold` for the compounded rolling forecast
    (``forecast.rolling_forecast``): the same metric set computed grouped by
    ``recursion_depth`` and pooled overall. ``n_obs`` / ``nat_n_obs`` are reported
    per depth so the thinning, noisier high-depth buckets are visible. Read
    ``nat_rmse`` (not ``nat_r``) across depth — ``nat_r`` isn't comparable across
    the different, shrinking row sets each depth bucket spans.
    """
    rows = []
    for depth, sub in pred_frame.groupby(depth_col, sort=True):
        rows.append({"recursion_depth": depth, **score_predictions(sub, **kw)})
    rows.append({"recursion_depth": "ALL", **score_predictions(pred_frame, **kw)})
    return pd.DataFrame(rows)


def model_stats(gam) -> dict:
    """In-sample-only GAM statistics (no OOS analog): AIC, edf, deviance, R^2.

    Pulled from ``LinearGAM.statistics_``; keys absent in a given pyGAM version
    come back as None rather than raising.
    """
    s = getattr(gam, "statistics_", None) or {}
    pr2 = s.get("pseudo_r2")
    if isinstance(pr2, dict):
        pr2 = pr2.get("explained_deviance")
    return {
        "aic": s.get("AIC"),
        "aicc": s.get("AICc"),
        "edf": s.get("edof"),
        "loglik": s.get("loglikelihood"),
        "deviance": s.get("deviance"),
        "scale": s.get("scale"),
        "pseudo_r2": pr2,
    }
