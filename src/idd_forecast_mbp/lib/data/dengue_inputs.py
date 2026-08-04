"""Assemble the dengue fit frame from the stage-02 artifacts.

Extracted from ``reports/03_modeling/pygam_dengue_models_explore.ipynb`` (cells
3, 5, 7), which is the only place this assembly existed.

The pure transforms are separated from the reads so they can be unit tested
without the cluster artifacts, which run to 728 MB (past inputs) and 1.6 GB
(observed age/sex).

Grain
-----
``grain="fhs"`` is the default. Dengue has no genuine admin-2 data — admin-2 was
distributed down from national data using suitability — so descending to admin-2
and re-aggregating recovers no information that was not already national. See
``.claude/DECISIONS.md`` 2026-08-03. Practically it also means the fit frame is
a few hundred thousand rows rather than tens of millions, because the grain
filter pushes down into the parquet read.

Reference group
---------------
The base age/sex group is ``cause_map['dengue']`` (age_group_id 3, sex_id 1) and
is a *parameter*, not a constant: formulations are free to anchor on a different
cell. ``03_modeling/fit_dengue_models_explore.r`` hardcodes 7 / 2 with a
``<<CONFIRM>>`` marker and disagrees with every other source; 3 / 1 is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data.hierarchy import load_hierarchy
from idd_forecast_mbp.lib.utils.transforms import logit

if TYPE_CHECKING:
    from collections.abc import Sequence

URBAN_COLUMN = "weighted_1km_urban_threshold_300.0_simple_mean"

#: ``grain`` -> the hierarchy flag marking that grain's locations.
GRAIN_FLAG = {"fhs": "most_detailed_fhs", "lsae": "most_detailed_lsae"}

#: Keep urban strictly interior so its logit stays finite.
_URBAN_EPS = 1e-3


@dataclass(frozen=True)
class DengueInputs:
    """Everything a formulation needs from the past, at one grain."""

    fit_frame: pd.DataFrame
    observed_all_age: pd.DataFrame
    population: pd.DataFrame
    hierarchy: pd.DataFrame
    grain: str
    anchor_year: int
    reference_age_group_id: int
    reference_sex_id: int
    base_location_ids: np.ndarray[Any, Any]
    #: Year the time covariate is centred on. Fit-derived state: the forecast MUST
    #: reuse it, because a year term fitted on (year - 2011.5) evaluated at raw
    #: year is off by 2011.5 -- which for a slope of 0.04 is a factor of e^80.
    year_center: float = 0.0


# ---------------------------------------------------------------------------
# Pure transforms
# ---------------------------------------------------------------------------

def derive_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the model-space responses and transformed covariates.

    ``inc_rate == 0`` gives ``log(0) = -inf`` and a CFR outside ``(0, 1)`` gives
    NaN. Both are expected — the fit filters to finite rows — so the warnings are
    silenced rather than raised on a multi-million-row column. They are *not*
    silently repaired: an infinite or missing response drops that row from the
    fit, which is the intended behaviour and is visible in the fit row count.
    """
    out = df.copy()
    out["urban_fraction"] = np.clip(
        out[URBAN_COLUMN].astype(float), _URBAN_EPS, 1 - _URBAN_EPS,
    )
    # The 2025 first-submission spec enters urban as a LOGIT
    # (`final_models_dengue.r::mod_inc_base` uses logit_urban_1km_threshold_300),
    # so both forms are provided and a formulation picks one. The clip above is
    # what keeps this finite.
    out["logit_urban_fraction"] = logit(out["urban_fraction"])
    out["log_gdppc_mean"] = np.log(out["gdppc_mean"].astype(float))

    inc = out["dengue_inc_rate"].astype(float)
    with np.errstate(divide="ignore"):
        out["log_dengue_inc_rate"] = np.log(inc)

    with np.errstate(divide="ignore", invalid="ignore"):
        mort = out["dengue_mort_rate"].astype(float)
        out["dengue_cfr"] = np.where(inc > 0, mort / inc, np.nan)
        interior = (out["dengue_cfr"] > 0) & (out["dengue_cfr"] < 1)
        out["logit_dengue_cfr"] = np.where(interior, logit(out["dengue_cfr"]), np.nan)
        out["log_dengue_mort_rate"] = np.where(mort > 0, np.log(mort), np.nan)

    return out


def build_age_sex_rr(
    observed_age_sex: pd.DataFrame,
    *,
    anchor_year: int,
    reference_age_group_id: int,
    reference_sex_id: int,
) -> tuple[pd.DataFrame, np.ndarray[Any, Any]]:
    """Observed age/sex incidence relative risks, against the reference cell.

    The relative risk is each cell's observed incidence rate divided by the
    reference cell's, at ``anchor_year``, per location. The reference cell has
    ``rr == 1`` by construction.

    ``observed_age_sex`` must be restricted to the **FHS-most-detailed** set —
    always, whatever grain the model is fit at, because the pattern is defined
    per FHS location and inherited downward. Keying it to ``most_detailed_gbd``
    instead silently drops any country with no GBD-most-detailed unit (India has
    none), which makes that country predict flat zero and leaves its whole
    super-region unanchored.

    Returns the relative-risk table and the locations that have a usable
    reference observation (positive incidence count at the anchor year).
    """
    at_anchor = observed_age_sex[observed_age_sex["year_id"] == anchor_year]
    base = at_anchor[
        (at_anchor["age_group_id"] == reference_age_group_id)
        & (at_anchor["sex_id"] == reference_sex_id)
        & (at_anchor["dengue_inc_count"] > 0)
    ]
    base_location_ids = base["location_id"].unique()

    rr = (
        at_anchor[at_anchor["location_id"].isin(base_location_ids)]
        [["location_id", "age_group_id", "sex_id", "dengue_inc_rate"]]
        .merge(
            base[["location_id", "dengue_inc_rate"]]
            .rename(columns={"dengue_inc_rate": "_base_rate"}),
            on="location_id", how="left",
        )
    )
    rr["rr_inc_as"] = rr["dengue_inc_rate"] / rr["_base_rate"]
    rr = rr.rename(columns={"location_id": "fhs_location_id"})
    return rr[["fhs_location_id", "age_group_id", "sex_id", "rr_inc_as"]], base_location_ids


def attach_age_sex_rr(
    past: pd.DataFrame,
    rr: pd.DataFrame,
    base_location_ids: Sequence[int] | np.ndarray[Any, Any],
) -> pd.DataFrame:
    """Restrict ``past`` to locations with a reference observation and attach the rr.

    Each row inherits the age/sex pattern of its FHS-most-detailed parent, so at
    the FHS grain this is a self-join and at the admin-2 grain each admin-2 unit
    takes its FHS parent's pattern.

    Also assigns the two factor codes models use, as contiguous integers so the
    fit and the prediction share one mapping:

    ``A0_af``
        Country.
    ``as_id``
        Age/sex cell. Derived here rather than in a notebook so every consumer
        gets the same coding — a factor level that shifts between fit and predict
        silently reassigns effects.
    """
    out = past[past["fhs_location_id"].isin(base_location_ids)].copy()
    out = out.merge(rr, on=["fhs_location_id", "age_group_id", "sex_id"], how="left")
    out["A0_af"] = out["A0_location_id"].astype("category").cat.codes.astype("int64")
    out["as_id"] = (
        out["age_group_id"].astype(str) + "_" + out["sex_id"].astype(str)
    ).astype("category").cat.codes.astype("int64")
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_dengue_inputs(  # noqa: PLR0913
    grain: str = "fhs",
    *,
    anchor_year: int = 2023,
    reference_age_group_id: int | None = None,
    reference_sex_id: int | None = None,
    past_inputs_path: Path | None = None,
    observed_age_sex_path: Path | None = None,
    observed_all_age_path: Path | None = None,
    population_path: Path | None = None,
) -> DengueInputs:
    """Read the stage-02 artifacts and assemble the fit frame at ``grain``.

    The grain filter is pushed into the parquet read, so ``grain="fhs"`` never
    materialises the admin-2 rows.
    """
    if grain not in GRAIN_FLAG:
        msg = f"grain must be one of {tuple(GRAIN_FLAG)}; got {grain!r}"
        raise ValueError(msg)

    cause: dict[str, Any] = mbpc.cause_map["dengue"]
    ref_age = int(reference_age_group_id if reference_age_group_id is not None
                  else cause["reference_age_group_id"])
    ref_sex = int(reference_sex_id if reference_sex_id is not None
                  else cause["reference_sex_id"])

    past_inputs_path = past_inputs_path or (
        Path(mbpc.DEN_PAST_INPUTS_READ_PATH) / "dengue_past_inputs.parquet")
    observed_age_sex_path = observed_age_sex_path or (
        Path(mbpc.DEN_RAKED_AS_READ_PATH) / "as_full_dengue_df.parquet")
    observed_all_age_path = observed_all_age_path or (
        Path(mbpc.DEN_RAKED_AA_READ_PATH) / "aa_full_dengue_df.parquet")
    population_path = population_path or (
        Path(mbpc.POPULATION_READ_PATH) / "aa_2023_full_population_df.parquet")

    hierarchy = load_hierarchy()
    by_location = hierarchy.set_index("location_id")
    flag = GRAIN_FLAG[grain]
    grain_location_ids = hierarchy.loc[hierarchy[flag] == 1, "location_id"].to_numpy()

    past = pd.read_parquet(
        past_inputs_path,
        filters=[("location_id", "in", grain_location_ids.tolist())],
    )
    past["fhs_location_id"] = past["location_id"].map(by_location["fhs_location_id"])
    past = derive_model_columns(past)

    # The relative-risk universe is ALWAYS the FHS-most-detailed set, whatever
    # grain we fit at: rr is defined per FHS location and inherited downward, so
    # an admin-2 fit still reads its age/sex pattern from the FHS parent. Keying
    # this to the fit grain instead leaves an admin-2 fit with no rr at all.
    fhs_location_ids = hierarchy.loc[
        hierarchy[GRAIN_FLAG["fhs"]] == 1, "location_id"
    ].to_numpy()
    observed_age_sex = pd.read_parquet(
        observed_age_sex_path,
        filters=[("year_id", "==", anchor_year),
                 ("location_id", "in", fhs_location_ids.tolist())],
    )
    rr, base_location_ids = build_age_sex_rr(
        observed_age_sex, anchor_year=anchor_year,
        reference_age_group_id=ref_age, reference_sex_id=ref_sex,
    )

    fit_frame = attach_age_sex_rr(past, rr, base_location_ids)
    year_center = float(fit_frame["year_id"].mean())
    fit_frame["year_centered"] = fit_frame["year_id"] - year_center

    return DengueInputs(
        fit_frame=fit_frame,
        observed_all_age=pd.read_parquet(observed_all_age_path),
        population=pd.read_parquet(population_path),
        hierarchy=hierarchy,
        grain=grain,
        anchor_year=anchor_year,
        reference_age_group_id=ref_age,
        reference_sex_id=ref_sex,
        base_location_ids=base_location_ids,
        year_center=year_center,
    )
