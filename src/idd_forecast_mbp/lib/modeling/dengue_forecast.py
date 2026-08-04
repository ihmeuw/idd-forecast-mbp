"""Carry a fitted dengue formulation forward to 2100.

The forecast reuses everything the past run produced: the fitted models, and the
per-group anchor shift. Nothing is refitted and nothing is re-anchored — the
anchor is a statement about observed history, so extending the covariates forward
does not change it.

    past = run_formulation(inputs, formulation)          # fit + anchor on history
    fc   = forecast_formulation(past, inputs, formulation, ssp_scenario="ssp245")

The only thing that varies between forecast scenarios of the *same* fit is the
covariate path — including the year covariate, which is why a decayed time effect
needs no refit (see :mod:`~idd_forecast_mbp.lib.modeling.year_path`).

Draws
-----
Draws enter only through the climate covariates, so predictions come out
``(location, year, draw)``. The anchor shift carries no draw dimension and is
broadcast across them. Aggregation happens within each draw and the collapse to
mean/lower/upper happens last, in
:mod:`~idd_forecast_mbp.lib.processing.dengue_products`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from idd_forecast_mbp.lib.data.dengue_forecast_covariates import (
    build_prediction_frame,
    load_forecast_covariates,
)
from idd_forecast_mbp.lib.modeling import fit as fit_mod
from idd_forecast_mbp.lib.modeling import specs as specs_mod
from idd_forecast_mbp.lib.modeling.anchor import apply_shift
from idd_forecast_mbp.lib.modeling.dengue_formulations import (
    CFR,
    INCIDENCE,
    LINK,
    MORTALITY,
)
from idd_forecast_mbp.lib.modeling.dengue_pipeline import (
    FORWARD_COLUMN,
    NATURAL_COLUMN,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from idd_forecast_mbp.lib.data.dengue_inputs import DengueInputs
    from idd_forecast_mbp.lib.modeling.dengue_formulations import Formulation
    from idd_forecast_mbp.lib.modeling.dengue_pipeline import FormulationRun

#: Covariates that carry a draw dimension. A model reading none of these produces
#: the same value in every draw, so it is predicted once and broadcast — otherwise
#: an age/sex CFR at 382 locations x 78 years x 100 draws x 50 cells is 149M rows
#: of duplicated arithmetic.
from idd_forecast_mbp.lib.data.dengue_forecast_covariates import (
    DRAW_COVARIATES,
)

#: Age/sex identity columns, when a model needs the cell grid.
_CELL_COLUMNS = ("age_group_id", "sex_id", "as_id")


@dataclass
class DengueForecast:
    """One formulation carried forward under one ssp and one year path."""

    formulation: Formulation
    ssp_scenario: str
    year_path: str
    #: (location_id, year_id, draw) with a natural-space column per outcome.
    rates: pd.DataFrame
    n_locations: int
    n_dropped_unseen_country: int
    n_unanchored: int


def _age_sex_cells(inputs: DengueInputs) -> pd.DataFrame:
    """The (age, sex, as_id) grid, with the coding the FIT assigned.

    Recomputing ``as_id`` here would renumber the cells and hand each one another
    cell's effect, exactly as recomputing ``A0_af`` would for countries.
    """
    return (inputs.fit_frame[list(_CELL_COLUMNS)]
            .drop_duplicates().reset_index(drop=True))


def _substitute_year_columns(
    frame: pd.DataFrame, terms: Sequence[Any],
) -> pd.DataFrame:
    """Point year terms at the effective-year columns the forecast frame carries.

    A model fitted on ``year_sr_4`` must be predicted on the forecast frame's
    ``year_sr_4``, which already holds the effective year. A model fitted on a
    plain ``year_id`` reads ``year_id_effective`` instead — otherwise a decay
    would be silently ignored for that formulation.
    """
    out = frame
    for term in terms:
        column = getattr(term, "col", None)
        if column == "year_id" and "year_id_effective" in out.columns:
            out = out.assign(year_id=out["year_id_effective"])
    return out


def forecast_formulation(  # noqa: PLR0913
    past: FormulationRun,
    inputs: DengueInputs,
    formulation: Formulation,
    *,
    ssp_scenario: str,
    year_path: str | Callable[..., np.ndarray[Any, Any]] = "identity",
    year_path_kwargs: dict[str, Any] | None = None,
    inputs_path: Path | None = None,
    years: Sequence[int] | None = None,
    draws: Sequence[int] | None = None,
    covariates: pd.DataFrame | None = None,
) -> DengueForecast:
    """Predict every fitted outcome forward, reusing the past anchor.

    ``covariates`` lets a caller pass an already-loaded frame, so several year
    paths can be compared without re-reading the netCDF each time — the read and
    the admin-2 roll-up are the expensive part, and neither depends on the path.
    """
    if covariates is None:
        covariates = load_forecast_covariates(
            ssp_scenario, grain=formulation.grain, inputs_path=inputs_path,
            years=years, draws=draws, hierarchy=inputs.hierarchy,
            population=inputs.population)

    a0_code_map = (
        inputs.fit_frame[["A0_location_id", "A0_af"]]
        .drop_duplicates().set_index("A0_location_id")["A0_af"])

    prepared = build_prediction_frame(
        covariates, inputs.hierarchy, a0_code_map=a0_code_map,
        anchor_year=inputs.anchor_year, year_center=inputs.year_center,
        year_path=year_path,
        year_path_kwargs=year_path_kwargs, ssp_scenario=ssp_scenario,
        grain=formulation.grain)
    frame = prepared.frame

    keys = ["location_id", "year_id", "draw"]
    combined: pd.DataFrame | None = None
    n_unanchored = 0

    for outcome in formulation.fitted_outcomes:
        of = past.fits[outcome]
        terms = formulation.terms[outcome]
        columns = specs_mod.spec_columns(terms)
        predict_on = _substitute_year_columns(frame, terms)

        # A model carrying as_id is fitted across age/sex, so it must be predicted
        # there too; the covariates themselves are age/sex-invariant, so this is a
        # cross join rather than new data.
        needs_cells = any(c in _CELL_COLUMNS for c in columns)
        if needs_cells:
            predict_on = predict_on.merge(_age_sex_cells(inputs), how="cross")

        # Draw-free covariates -> predict once, broadcast later.
        draw_free = not any(c in DRAW_COVARIATES for c in columns)
        row_keys = ["location_id", "year_id"]
        if needs_cells:
            row_keys += ["age_group_id", "sex_id"]
        if not draw_free:
            row_keys += ["draw"]
        if draw_free:
            predict_on = predict_on.drop_duplicates(row_keys)

        missing = [c for c in columns if c not in predict_on.columns]
        if missing:
            msg = (f"formulation {formulation.id!r} needs {missing} to forecast "
                   f"{outcome}, but the 08b inputs do not provide them. Rebuild 08b "
                   f"with those covariates, or drop the terms.")
            raise KeyError(msg)

        usable = predict_on[
            np.isfinite(predict_on[columns].to_numpy(dtype=float)).all(axis=1)]
        predicted = usable[row_keys].copy()
        predicted["predicted"] = of.model.predict(
            usable[columns].to_numpy(dtype=float))

        if of.shift is not None:
            group_cols = list(formulation.anchor_group)
            already_per_cell = {"age_group_id", "sex_id"} <= set(predicted.columns)
            if {"age_group_id", "sex_id"} <= set(group_cols) and not already_per_cell:
                # Broadcasting a per-(location, year) prediction onto the anchor's
                # cells. A prediction that is ALREADY per cell must be left alone --
                # merging cells onto cells duplicates the key columns into _x/_y and
                # the group columns silently disappear.
                cells = (of.shift.index.to_frame(index=False)
                         if isinstance(of.shift.index, pd.MultiIndex)
                         else pd.DataFrame({group_cols[0]: of.shift.index}))
                predicted = predicted.merge(
                    cells.drop_duplicates(), on="location_id", how="inner")
            before = predicted[group_cols].drop_duplicates().shape[0]
            predicted = apply_shift(predicted, of.shift, group_cols=group_cols,
                                    pred_col="predicted", out_col="anchored")
            n_unanchored = max(
                n_unanchored, before - predicted[group_cols].drop_duplicates().shape[0])
        else:
            predicted = predicted.rename(columns={"predicted": "anchored"})

        predicted[NATURAL_COLUMN[outcome]] = fit_mod._inverse(  # noqa: SLF001
            LINK[outcome], predicted["anchored"].to_numpy(dtype=float))

        if formulation.spec.feeds_forward == outcome:
            # Predicted mortality replaces the observed column the fit used.
            forward_keys = [c for c in keys if c in predicted.columns]
            forward = (predicted[[*forward_keys, "anchored"]]
                       .drop_duplicates(forward_keys)
                       .rename(columns={"anchored": FORWARD_COLUMN}))
            frame = frame.drop(columns=[FORWARD_COLUMN], errors="ignore").merge(
                forward, on=forward_keys, how="left")

        piece_keys = [c for c in (*keys, "age_group_id", "sex_id")
                      if c in predicted.columns]
        piece = predicted[[*piece_keys, NATURAL_COLUMN[outcome]]]
        if combined is None:
            combined = piece
        else:
            shared = [c for c in combined.columns if c in piece.columns]
            combined = combined.merge(piece, on=shared, how="outer")

    rates = _derive_missing_outcome(combined, formulation)
    return DengueForecast(
        formulation=formulation, ssp_scenario=ssp_scenario,
        year_path=year_path if isinstance(year_path, str) else "custom",
        rates=rates, n_locations=prepared.n_locations,
        n_dropped_unseen_country=prepared.n_dropped_unseen_country,
        n_unanchored=n_unanchored)


def _derive_missing_outcome(
    combined: pd.DataFrame | None, formulation: Formulation,
) -> pd.DataFrame:
    """Compute whichever outcome the structure derives rather than regresses."""
    if combined is None:  # pragma: no cover - a structure always fits something
        return pd.DataFrame()
    out = combined
    derived = formulation.derived_outcome
    if derived == MORTALITY:
        out[NATURAL_COLUMN[MORTALITY]] = (
            out[NATURAL_COLUMN[INCIDENCE]] * out[NATURAL_COLUMN[CFR]])
    elif derived == INCIDENCE:
        cfr = out[NATURAL_COLUMN[CFR]].to_numpy(dtype=float)
        out[NATURAL_COLUMN[INCIDENCE]] = np.where(
            cfr > 0, out[NATURAL_COLUMN[MORTALITY]].to_numpy(dtype=float) / cfr, np.nan)
    return out
