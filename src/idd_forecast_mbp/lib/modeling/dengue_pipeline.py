"""Run a dengue :class:`Formulation` from fit through to anchored predictions.

This is the piece that reads the formulation contract and behaves accordingly:
it fits whatever outcomes the structure declares, in the order it declares,
feeds one outcome forward when the structure says to, anchors each in its own
model space, and derives the remaining outcome.

The prediction frame
--------------------
Predictions are produced on the **base age/sex cell only** — a
``(location, year)`` frame — even when ``as_id`` was in the design matrix. Under
a per-``(location, age, sex)`` anchor every link-space-additive, time-constant
term cancels exactly, so evaluating the model across ~50 age/sex cells builds a
50x larger design matrix whose entire extra contribution the anchor then
subtracts away. Age/sex arrives once, at the end, from the observed anchor.

See ``.claude/DECISIONS.md`` 2026-08-03. Two important qualifications:

- The cancellation applies to the **raked** result only. Unraked, an ``as_id``
  term absolutely does change the answer — which is why both are reported.
- ``as_id`` therefore belongs in the regression (it de-biases the coefficients on
  the time-varying covariates) and a model carrying it predicts on the full
  age/sex grid. An earlier version of this module raised on any ``as_id`` term in
  a spec; that conflated "should not widen the frame" with "may not be a term"
  and was wrong.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from idd_forecast_mbp.lib.modeling import fit as fit_mod
from idd_forecast_mbp.lib.modeling import specs as specs_mod
from idd_forecast_mbp.lib.modeling.anchor import apply_shift, compute_shift
from idd_forecast_mbp.lib.modeling.dengue_formulations import (
    AGE_SEX_FROM_REDISTRIBUTION,
    CFR,
    INCIDENCE,
    LINK,
    MORTALITY,
    RESPONSE_COLUMN,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idd_forecast_mbp.lib.data.dengue_inputs import DengueInputs
    from idd_forecast_mbp.lib.modeling.dengue_formulations import Formulation

#: Natural-space column produced per outcome, once the link is inverted.
NATURAL_COLUMN = {
    INCIDENCE: "dengue_inc_rate",
    MORTALITY: "dengue_mort_rate",
    CFR: "dengue_cfr",
}

#: The fed-forward outcome reuses the OBSERVED column name, so the same term reads
#: observed values at fit time and predicted values at predict time.
FORWARD_COLUMN = RESPONSE_COLUMN[MORTALITY]

#: How many unscorable factor levels to name in a warning before truncating.
_MAX_LEVELS_SHOWN = 8


@dataclass
class OutcomeFit:
    """One fitted outcome and its anchored prediction."""

    outcome: str
    model: Any
    #: (location_id, year_id, model-space anchored value, natural-space value)
    predictions: pd.DataFrame
    n_fit_rows: int
    n_unanchored: int
    #: Per-group additive shift learned from the PAST. The forecast reuses this
    #: rather than recomputing: the anchor is a statement about observed history,
    #: so it does not change because we extended the covariates forward.
    shift: pd.Series | None = None


@dataclass
class FormulationRun:
    """Everything a formulation produced on the past."""

    formulation: Formulation
    fits: dict[str, OutcomeFit]
    #: (location_id, year_id) with a natural-space column per outcome.
    rates: pd.DataFrame


def model_frame(
    inputs: DengueInputs, formulation: Formulation, outcome: str,
) -> pd.DataFrame:
    """The rows an outcome is fitted and predicted on, at its own response grain.

    ``base_cell``
        The reference age/sex cell — one row per (location, year). Every model
        covariate is age/sex-invariant (verified: one distinct value per
        location-year across all 50 cells), so the cell selects rows only.
    ``all_age``
        Same rows, responses replaced by the all-age series. The malaria-style
        approach: regress the all-age quantity, redistribute afterwards.
    ``age_sex``
        The full age/sex grid. Needed by any model carrying ``as_id``, which is
        the point of fitting CFR across age/sex.
    """
    grain = formulation.grain_for(outcome)
    frame = inputs.fit_frame
    if grain == "age_sex":
        return frame.copy()

    base = frame[
        (frame["age_group_id"] == formulation.reference_age_group_id)
        & (frame["sex_id"] == formulation.reference_sex_id)
    ].copy()
    if grain != "all_age":
        return base
    return base.drop(columns=list(RESPONSE_COLUMN.values()), errors="ignore").merge(
        all_age_responses(inputs), on=["location_id", "year_id"], how="left",
    )


def all_age_responses(inputs: DengueInputs) -> pd.DataFrame:
    """All-age model-space responses per (location, year), from the age/sex grid.

    Rates are collapsed in COUNT space — ``sum(rate * population) / sum(population)``
    — never averaged. Verified against the observed all-age artifact to 1.8e-15.
    """
    f = inputs.fit_frame
    parts = f.assign(
        _inc=f["dengue_inc_rate"] * f["population"],
        _mort=f["dengue_mort_rate"] * f["population"],
        _pop=f["population"],
    ).groupby(["location_id", "year_id"], as_index=False)[["_inc", "_mort", "_pop"]].sum()

    inc = parts["_inc"] / parts["_pop"]
    mort = parts["_mort"] / parts["_pop"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cfr = np.where(inc > 0, mort / inc, np.nan)
        return pd.DataFrame({
            "location_id": parts["location_id"],
            "year_id": parts["year_id"],
            RESPONSE_COLUMN[INCIDENCE]: np.where(inc > 0, np.log(inc), np.nan),
            RESPONSE_COLUMN[MORTALITY]: np.where(mort > 0, np.log(mort), np.nan),
            RESPONSE_COLUMN[CFR]: np.where(
                (cfr > 0) & (cfr < 1), np.log(cfr / (1 - cfr)), np.nan),
        })


def broadcast_to_cells(
    predicted: pd.DataFrame,
    inputs: DengueInputs,
    formulation: Formulation,
    outcome: str,
    *,
    value_col: str,
) -> pd.DataFrame:
    """Spread a per-(location, year) natural-space prediction across the age/sex grid.

    Used when the anchor is not supplying the age/sex pattern — the no-rake
    variant, and any all-age response.

    Incidence and mortality spread by the observed relative risks. CFR is a ratio
    rather than a distributable quantity, so it applies unchanged to every cell.

    For an all-age response the spread counts are then rescaled so they sum back to
    the predicted all-age quantity. That rescaling is what makes redistribution and
    the plain rr broadcast agree on *shares* — both give
    ``pop * rr / sum(pop * rr)`` — while differing on the total.
    """
    cells = inputs.fit_frame[
        ["location_id", "year_id", "age_group_id", "sex_id", "population", "rr_inc_as"]
    ]
    keys = ["location_id", "year_id"]
    out = cells.merge(
        predicted[[*keys, value_col]].rename(columns={value_col: "_all_age"}),
        on=keys, how="inner",
    )

    if outcome == CFR:
        out[value_col] = out["_all_age"]
        return out.drop(columns=["population", "rr_inc_as", "_all_age"])

    out[value_col] = out["_all_age"] * out["rr_inc_as"].fillna(0.0)

    if formulation.age_sex_source == AGE_SEX_FROM_REDISTRIBUTION:
        out["_spread_count"] = out[value_col] * out["population"]
        totals = out.groupby(keys, as_index=False).agg(
            _spread=("_spread_count", "sum"),
            _pop=("population", "sum"),
            _target_rate=("_all_age", "first"),
        )
        out = out.merge(totals, on=keys, how="left")
        target_count = out["_target_rate"] * out["_pop"]
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(out["_spread"] > 0, target_count / out["_spread"], 0.0)
        out[value_col] = out[value_col] * factor
        out = out.drop(columns=["_spread_count", "_spread", "_pop", "_target_rate"])

    return out.drop(columns=["population", "rr_inc_as", "_all_age"])


def observed_anchor_frame(
    inputs: DengueInputs, formulation: Formulation, outcome: str,
) -> pd.DataFrame:
    """Observed model-space values the anchor pulls towards, at the anchor grain.

    When the anchor is per age/sex cell this keeps the full age/sex grid; when it
    is per location it collapses to the reference cell. An all-age response anchors
    per location against the all-age observed series instead.
    """
    response = RESPONSE_COLUMN[outcome]
    keys = list(formulation.anchor_group)
    if formulation.grain_for(outcome) == "all_age":
        frame = model_frame(inputs, formulation, outcome)
    elif formulation.anchors_by_age_sex:
        frame = inputs.fit_frame
    else:
        frame = inputs.fit_frame
        frame = frame[
            (frame["age_group_id"] == formulation.reference_age_group_id)
            & (frame["sex_id"] == formulation.reference_sex_id)
        ]
    out = frame[[*keys, "year_id", response]].rename(columns={response: "observed"})
    return out[np.isfinite(out["observed"].to_numpy(dtype=float))]


def _fit_one(
    fit_frame: pd.DataFrame,
    terms: tuple[Any, ...],
    outcome: str,
    predict_frame: pd.DataFrame | None = None,
    *,
    id_cols: tuple[str, ...] = ("location_id", "year_id"),
) -> tuple[Any, pd.DataFrame, int]:
    """Fit on ``fit_frame``, predict on ``predict_frame`` (defaults to the same rows).

    The two frames differ when a covariate is OBSERVED at fit time but PREDICTED at
    predict time — the malaria pattern, where inc/mort are trained on observed PfPR
    and predicted with the shifted predicted PfPR. Training on a model's own output
    would fit the model to its own error.
    """
    spec_outcome = fit_mod.Outcome(
        RESPONSE_COLUMN[outcome], NATURAL_COLUMN[outcome], LINK[outcome],
    )
    columns = specs_mod.spec_columns(terms)
    response = RESPONSE_COLUMN[outcome]

    ok = np.isfinite(fit_frame[columns].to_numpy(dtype=float)).all(axis=1)
    fit_rows = fit_frame[ok & np.isfinite(fit_frame[response].to_numpy(dtype=float))]
    model = fit_mod.fit_gam(fit_rows, terms, spec_outcome)

    target = fit_frame if predict_frame is None else predict_frame
    usable = target[np.isfinite(target[columns].to_numpy(dtype=float)).all(axis=1)]

    # A factor level the fit never saw cannot be scored: pyGAM builds the factor
    # basis from the training levels and refuses anything outside that domain. This
    # happens whenever a country's response is entirely missing -- it drops out of
    # the fit but is still in the prediction frame. Rare pooled, common once the
    # fit is restricted to one super-region.
    #
    # Dropped rather than mapped to some other level. Silently reassigning a
    # country to a neighbour's fixed effect is the kind of error that produces
    # plausible numbers, and the malaria rocket needs a deliberate orphan-A0
    # fallback for exactly this reason.
    for term in terms:
        if getattr(term, "form", None) != "factor":
            continue
        column = getattr(term, "col", None)
        if column is None or column not in usable.columns:
            continue
        seen = set(fit_rows[column].unique())
        unscorable = ~usable[column].isin(seen)
        if unscorable.any():
            missing = sorted(set(usable.loc[unscorable, column].unique()))
            warnings.warn(
                f"{outcome}: {int(unscorable.sum()):,} prediction rows dropped -- "
                f"{column} level(s) {missing[:_MAX_LEVELS_SHOWN]}"
                f"{' ...' if len(missing) > _MAX_LEVELS_SHOWN else ''} never appeared in the fit "
                f"(no finite response for them)",
                RuntimeWarning, stacklevel=2)
            usable = usable[~unscorable]

    keep = [c for c in id_cols if c in usable.columns]
    predicted = usable[keep].copy()
    predicted["predicted"] = model.predict(usable[columns].to_numpy(dtype=float))
    return model, predicted, len(fit_rows)


def _fit_by_super_region(  # noqa: PLR0913
    fit_frame: pd.DataFrame,
    terms: tuple[Any, ...],
    outcome: str,
    predict_frame: pd.DataFrame | None = None,
    *,
    id_cols: tuple[str, ...] = ("location_id", "year_id"),
    group_col: str = "super_region_location_id",
) -> tuple[dict[int, Any], pd.DataFrame, int]:
    """One model per super-region; returns ``{super_region: model}`` and the union.

    Each sub-model only ever predicts its own super-region, so a factor level
    absent from that group is never scored — which is why the global ``A0_af``
    coding can be used as-is despite having gaps within a group (pyGAM builds the
    factor basis from the levels it sees).

    A group whose fit raises is skipped and reported through the row count rather
    than aborting the formulation: with as few as 168 rows, a thin group failing
    should not lose the other five.
    """
    models: dict[int, Any] = {}
    pieces: list[pd.DataFrame] = []
    n_fit_total = 0
    target = fit_frame if predict_frame is None else predict_frame

    for group in sorted(fit_frame[group_col].dropna().unique()):
        rows = fit_frame[fit_frame[group_col] == group]
        target_rows = target[target[group_col] == group]
        if rows.empty or target_rows.empty:
            continue
        try:
            model, predicted, n_fit = _fit_one(
                rows, terms, outcome, target_rows, id_cols=id_cols)
        except Exception as exc:  # noqa: BLE001 - a thin group must not lose the rest
            # Never silent: a dropped super-region changes what the aggregate
            # covers, so it has to be visible in the run log.
            warnings.warn(
                f"{outcome}: super-region {int(group)} failed to fit on "
                f"{len(rows):,} rows ({type(exc).__name__}: {exc}); it is EXCLUDED "
                f"from this formulation's predictions",
                RuntimeWarning, stacklevel=2)
            continue
        models[int(group)] = model
        pieces.append(predicted)
        n_fit_total += n_fit

    if not pieces:
        msg = (f"no super-region produced a usable {outcome} fit; every group "
               f"failed or was empty")
        raise ValueError(msg)
    return models, pd.concat(pieces, ignore_index=True), n_fit_total


def _anchor_one(
    predicted: pd.DataFrame,
    inputs: DengueInputs,
    formulation: Formulation,
    outcome: str,
) -> tuple[pd.DataFrame, int, pd.Series]:
    """Shift predictions onto the observed anchor, in this outcome's model space.

    When the anchor is per age/sex cell the prediction — which exists only for
    the base cell — is broadcast across the cells before shifting, so each cell
    lands on its own observed anchor.
    """
    if formulation.anchor is None:  # pragma: no cover - guarded by the caller
        msg = "_anchor_one called on an unraked formulation"
        raise ValueError(msg)
    keys = list(formulation.anchor_group)
    observed = observed_anchor_frame(inputs, formulation, outcome)

    already_per_cell = {"age_group_id", "sex_id"} <= set(predicted.columns)
    if formulation.anchors_by_age_sex and not already_per_cell:
        # A base-cell or all-age prediction has one row per (location, year); the
        # per-cell anchor needs one row per cell. An age_sex-grain prediction is
        # already there — broadcasting again would duplicate the key columns.
        cells = observed[keys].drop_duplicates()
        predicted = predicted.merge(cells, on="location_id", how="inner")

    shift = compute_shift(
        observed, predicted, formulation.anchor, group_cols=keys,
        obs_col="observed", pred_col="predicted",
    )
    n_unanchored = int(predicted[keys].drop_duplicates().shape[0] - shift.shape[0])
    anchored = apply_shift(predicted, shift, group_cols=keys, out_col="anchored")
    return anchored, n_unanchored, shift


def run_formulation(inputs: DengueInputs, formulation: Formulation) -> FormulationRun:
    """Fit, predict and anchor every outcome the structure declares.

    Raises if the formulation's grain disagrees with the loaded inputs, or if it
    asks for an age/sex interaction, which would break the base-cell prediction
    frame this function relies on.
    """
    if formulation.grain != inputs.grain:
        msg = (f"formulation {formulation.id!r} is specified at grain "
               f"{formulation.grain!r} but inputs were loaded at {inputs.grain!r}")
        raise ValueError(msg)

    fits: dict[str, OutcomeFit] = {}
    predicted_forward: pd.DataFrame | None = None

    for outcome in formulation.fitted_outcomes:
        terms = formulation.terms[outcome]
        fit_frame = model_frame(inputs, formulation, outcome)
        predict_frame = fit_frame
        if predicted_forward is not None:
            # Train on OBSERVED, predict with PREDICTED: same column, different
            # values. Fitting on the model's own output would fit its error.
            predict_frame = fit_frame.drop(
                columns=[FORWARD_COLUMN], errors="ignore").merge(
                predicted_forward, on=["location_id", "year_id"], how="left")
        id_cols = (("location_id", "year_id", "age_group_id", "sex_id")
                   if formulation.grain_for(outcome) == "age_sex"
                   else ("location_id", "year_id"))
        if formulation.by_super_region:
            model, predicted, n_fit = _fit_by_super_region(
                fit_frame, terms, outcome, predict_frame, id_cols=id_cols)
        else:
            model, predicted, n_fit = _fit_one(
                fit_frame, terms, outcome, predict_frame, id_cols=id_cols)
        shift: pd.Series | None = None
        if formulation.is_raked:
            anchored, n_unanchored, shift = _anchor_one(
                predicted, inputs, formulation, outcome)
        else:
            anchored, n_unanchored = predicted.rename(
                columns={"predicted": "anchored"}), 0
        anchored[NATURAL_COLUMN[outcome]] = fit_mod._inverse(  # noqa: SLF001
            LINK[outcome], anchored["anchored"].to_numpy(dtype=float),
        )
        fits[outcome] = OutcomeFit(outcome, model, anchored, n_fit, n_unanchored, shift)

        forward = formulation.spec.feeds_forward
        if forward == outcome:
            predicted_forward = (
                anchored[["location_id", "year_id", "anchored"]]
                .drop_duplicates(["location_id", "year_id"])
                .rename(columns={"anchored": FORWARD_COLUMN}))

    return FormulationRun(formulation, fits, _combine_rates(fits, formulation))


def run_formulations(
    inputs: DengueInputs,
    formulations: Sequence[Formulation],
    *,
    n_jobs: int = 1,
    backend: str = "loky",
) -> dict[str, FormulationRun]:
    """Run several formulations, optionally in parallel.

    Formulations are independent — each fits its own models on the same inputs and
    shares no state — so this is embarrassingly parallel over the list.

    Use ``backend="loky"`` (processes). Measured on the real FHS inputs, 8
    formulations: serial ~78 s, loky n_jobs=8 **7.6 s (10x)**, threading n_jobs=8
    **110 s — slower than serial**. The pyGAM fit is GIL-bound rather than
    numpy-bound, so threads contend instead of scaling; the cost of pickling
    ``inputs`` to each process is repaid many times over. ``n_jobs=1`` runs
    serially and skips joblib entirely, which keeps tracebacks readable while a
    formulation is being debugged.

    Returns ``{formulation.id: FormulationRun}``, insertion-ordered to match the
    input list.
    """
    if n_jobs == 1:
        return {f.id: run_formulation(inputs, f) for f in formulations}

    from joblib import Parallel, delayed

    runs = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(run_formulation)(inputs, f) for f in formulations
    )
    return {f.id: run for f, run in zip(formulations, runs, strict=True)}


def _combine_rates(
    fits: dict[str, OutcomeFit], formulation: Formulation,
) -> pd.DataFrame:
    """Join the fitted outcomes and compute the derived one, in natural space."""
    keys = [*formulation.anchor_group, "year_id"]
    combined: pd.DataFrame | None = None
    for outcome, result in fits.items():
        columns = [c for c in keys if c in result.predictions.columns]
        piece = result.predictions[[*columns, NATURAL_COLUMN[outcome]]]
        if combined is None:
            combined = piece
            continue
        # Mixed response grains: join on the keys the two pieces share, so a
        # per-(location, year) piece broadcasts across the finer piece's cells.
        shared = [c for c in combined.columns if c in piece.columns]
        combined = combined.merge(piece, on=shared, how="outer")
    if combined is None:  # pragma: no cover — a structure always fits something
        return pd.DataFrame()

    derived = formulation.derived_outcome
    if derived == MORTALITY:
        combined[NATURAL_COLUMN[MORTALITY]] = (
            combined[NATURAL_COLUMN[INCIDENCE]] * combined[NATURAL_COLUMN[CFR]]
        )
    elif derived == INCIDENCE:
        cfr = combined[NATURAL_COLUMN[CFR]].to_numpy(dtype=float)
        combined[NATURAL_COLUMN[INCIDENCE]] = np.where(
            cfr > 0, combined[NATURAL_COLUMN[MORTALITY]].to_numpy(dtype=float) / cfr,
            np.nan,
        )
    return combined
