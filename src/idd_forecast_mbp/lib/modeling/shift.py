"""2023-anchor shift + the downstream incidence/mortality chain (Phase 3).

:func:`apply_shift` reproduces ``apply_shift()`` in
``04_forecasting/forecast_malaria_admin_2s_rocket.r``: per location, add a
constant so the prediction at the anchor year equals the observed anchor (in
model space — logit for PfPR, log for rates). :func:`downstream_chain` wires the
whole thing the way the production forecaster does:

    predict PfPR -> shift to observed-2023 logit PfPR -> feed the shifted PfPR
    into the inc/mort models -> (optionally) shift inc/mort to their own
    observed-2023 anchors -> score every outcome with the same metrics.

The inc/mort models are trained on *observed* PfPR (production semantics) and
predicted with the *shifted predicted* PfPR, isolating the effect of the PfPR
formulation on the downstream outcomes.

Open decision (surface at check-in): ``shift_downstream`` defaults to True
(mirror production: anchor inc/mort to observed 2023 too). Set False to score
the raw inc/mort model output instead. Evaluation is ``"IS"`` or a
:class:`fit.Temporal`; within-country CV is intentionally unsupported here — it
scatters a location's years across folds, so a single per-location 2023 anchor
isn't well defined (and it isn't a forecasting split).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idd_forecast_mbp.lib.modeling import fit, metrics
from idd_forecast_mbp.lib.modeling import specs as specs_mod

# Default downstream RHS = the production inc/mort formula
# (02_fit_final_malaria_models.r): s(logit_pfpr, k=10, mpi) + log_gdppc_mean + A0.
INC_SPEC = (
    specs_mod.Term("logit_malaria_pfpr", "mpi", 10),
    specs_mod.Term("log_gdppc_mean", "linear"),
    specs_mod.Term("A0_af", "factor"),
)
MORT_SPEC = INC_SPEC  # same RHS; the response differs (fit.MORTALITY)


def anchor_from_observed(data: pd.DataFrame, response: str, anchor_year: int = 2023) -> dict:
    """One observed model-space value per location at ``anchor_year`` (finite only)."""
    a = data.loc[data["year_id"] == anchor_year, ["location_id", response]].copy()
    a[response] = pd.to_numeric(a[response], errors="coerce")
    a = a[np.isfinite(a[response].to_numpy(dtype=float))].drop_duplicates("location_id")
    return dict(zip(a["location_id"].to_numpy(), a[response].to_numpy(dtype=float)))


def apply_shift(pred_frame: pd.DataFrame, anchor: dict, *,
                pred_col: str = "pred", anchor_year: int = 2023) -> np.ndarray:
    """Per-location shift of ``pred_col`` to hit the observed anchor at ``anchor_year``.

    shift(loc) = anchor[loc] - pred_at_anchor(loc); applied to every row of that
    location. Locations with no anchor, or no (finite) prediction at the anchor
    year, are left unshifted (shift 0). Returns an array aligned to ``pred_frame``.
    """
    at = pred_frame[pred_frame["year_id"] == anchor_year][["location_id", pred_col]]
    at = at.drop_duplicates("location_id").set_index("location_id")[pred_col]
    shift = {
        loc: anchor[loc] - float(at[loc])
        for loc in at.index
        if loc in anchor and np.isfinite(at[loc])
    }
    per_row = pred_frame["location_id"].map(shift).fillna(0.0).to_numpy(dtype=float)
    return pred_frame[pred_col].to_numpy(dtype=float) + per_row


def _chain_split(data: pd.DataFrame, evaluation):
    """(train, test) frames for the chain. IS -> (all, all); Temporal -> year windows."""
    if evaluation == "IS":
        return data, data
    if isinstance(evaluation, fit.Temporal):
        yr = data["year_id"].to_numpy()
        train = data[(yr >= evaluation.train_lo) & (yr <= evaluation.train_hi)]
        test = data[(yr >= evaluation.test_lo) & (yr <= evaluation.test_hi)]
        return train, test
    raise NotImplementedError(
        "downstream_chain supports 'IS' or a fit.Temporal split; within-country "
        "CV is intentionally unsupported (not a forecasting split)."
    )


def _fit_then_predict(train: pd.DataFrame, test: pd.DataFrame, spec, outcome, factor_lam):
    """Fit on ``train`` (its own na.omit), predict on ``test`` rows whose country
    was seen in training (pyGAM's factor term can't score unseen levels)."""
    cols = specs_mod.spec_columns(spec)
    train_frame, _ = fit._model_frame(train, cols, outcome)          # package-internal helper
    gam = fit._fit_terms(train_frame, cols, spec, outcome.response, factor_lam=factor_lam)
    seen = set(train_frame[fit.A0_COL].unique())
    need = list(dict.fromkeys(
        cols + [outcome.response, outcome.natural_col, "location_id", "year_id", fit.A0_COL]
    ))
    test_seen = test[test[fit.A0_COL].isin(seen)][need].copy()
    pred = fit._predict_rows(gam, test_seen, cols, outcome, "chain")
    return gam, pred


def _shift_outcome(pred: pd.DataFrame, data: pd.DataFrame, outcome, anchor_year: int) -> pd.DataFrame:
    """Anchor an outcome's predictions to observed ``anchor_year`` (keeps the raw in pred_raw)."""
    anchor = anchor_from_observed(data, outcome.response, anchor_year)
    pred = pred.copy()
    pred["pred_raw"] = pred["pred"]
    pred["pred"] = apply_shift(pred, anchor, pred_col="pred_raw", anchor_year=anchor_year)
    pred["pred_natural"] = fit._inverse(outcome.link, pred["pred"].to_numpy(dtype=float))
    return pred


def downstream_chain(data: pd.DataFrame, pfpr_spec, evaluation, *,
                     inc_spec=INC_SPEC, mort_spec=MORT_SPEC,
                     shift_downstream: bool = True, anchor_year: int = 2023,
                     factor_lam=None) -> dict:
    """Run PfPR -> shift -> inc/mort and score every outcome (see module docstring).

    ``data`` should be loaded with ``row_filter='final'`` (carries the log-rate
    responses). Returns ``{'pfpr'|'inc'|'mort': {'gam','pred','score'}, ...}``
    plus ``'evaluation'`` and ``'shift_downstream'``. PfPR is always shifted (it
    is the input to inc/mort); inc/mort are shifted only if ``shift_downstream``.
    """
    train, test = _chain_split(data, evaluation)

    # PfPR: fit -> predict -> shift to observed-2023 logit PfPR (always shifted).
    pfpr_gam, pfpr_pred = _fit_then_predict(train, test, pfpr_spec, fit.PFPR, factor_lam)
    pfpr_pred = _shift_outcome(pfpr_pred, data, fit.PFPR, anchor_year)

    # Substitute the shifted predicted PfPR into the test frame for inc/mort.
    sub = dict(zip(zip(pfpr_pred["location_id"], pfpr_pred["year_id"]), pfpr_pred["pred"]))
    test_sub = test.copy()
    test_sub[fit.PFPR.response] = [
        sub.get((loc, yr), np.nan)
        for loc, yr in zip(test_sub["location_id"], test_sub["year_id"])
    ]

    results = {
        "pfpr": {"gam": pfpr_gam, "pred": pfpr_pred, "score": metrics.score_predictions(pfpr_pred)},
    }
    for name, spec, outcome in (("inc", inc_spec, fit.INCIDENCE),
                                ("mort", mort_spec, fit.MORTALITY)):
        # trained on OBSERVED PfPR (train), predicted with the shifted predicted PfPR (test_sub)
        gam, pred = _fit_then_predict(train, test_sub, spec, outcome, factor_lam)
        if shift_downstream:
            pred = _shift_outcome(pred, data, outcome, anchor_year)
        results[name] = {"gam": gam, "pred": pred, "score": metrics.score_predictions(pred)}

    results["evaluation"] = evaluation.name if isinstance(evaluation, fit.Temporal) else str(evaluation)
    results["shift_downstream"] = shift_downstream
    return results


def chain_score_table(result: dict) -> pd.DataFrame:
    """Tidy the per-outcome scores from :func:`downstream_chain` into one frame."""
    rows = []
    for name in ("pfpr", "inc", "mort"):
        rows.append({"outcome": name, "evaluation": result["evaluation"], **result[name]["score"]})
    return pd.DataFrame(rows)
