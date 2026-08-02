"""Registry of malaria forecast-input covariates.

One entry per covariate column name → CovariateSpec describing what it is,
where its source lives, and how the builder should read it. The builder in
02_data_prep/08a_build_malaria_forecast_inputs.py dispatches on `kind` to
pick the right reader; the rest of the fields are documentation that lets
a reader of this file see at a glance where each variable comes from.

Dims are deterministic from `kind` — see DIMS_BY_KIND.

Adding a new covariate (e.g. rainfall): add one entry here. Dropping one:
omit it from the builder's --covariates CLI list. No schema change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal[
    "climate_draw",     # draw-varying, ssp-dependent (loc, year, draw)
    "climate_mean",     # ssp-dependent climate, collapsed to the ensemble mean (loc, year)
    "shared_scalar",    # non-draw, scalar per (loc, year)
    "flooding",         # non-draw, ssp-dependent, scalar per (loc, year)
    "suitability",      # draw-varying, ssp-dependent, single variant
    "dah",              # by (loc, year, dah_scenario), broadcast from A0
    "static_lookup",    # constant per location (loc,)
]

SourceKind = Literal["external", "made_in_repo"]

DIMS_BY_KIND: dict[str, tuple[str, ...]] = {
    "climate_draw":  ("location_id", "year_id", "draw"),
    "climate_mean":  ("location_id", "year_id"),
    "shared_scalar": ("location_id", "year_id"),
    "flooding":      ("location_id", "year_id"),
    "suitability":   ("location_id", "year_id", "draw"),
    "dah":           ("location_id", "year_id", "dah_scenario"),
    "static_lookup": ("location_id",),
}


@dataclass(frozen=True)
class CovariateSpec:
    """One covariate's metadata for the forecast-input builder.

    description:
        Short human-readable line — what the variable is and where it
        comes from.
    kind:
        Dispatch key; determines dims and which reader the builder calls.
    source_kind:
        "external" if the file was produced outside this repo (e.g. by
        rapidresponse, FHS, GBD); "made_in_repo" if a script in this
        repo writes it.
    source_path_attr:
        Name of the path attribute in `idd_forecast_mbp.constants` that
        resolves to the directory holding the source file. None for
        kinds that don't read from a single fixed location (climate,
        suitability — those template paths by ssp_scenario / hierarchy).
    source_filename:
        Filename inside `source_path_attr`. May contain `{ssp_scenario}`
        or `{lsae_hierarchy}` placeholders.
    source_script:
        Name of the script inside this repo that produced the file
        (when source_kind == "made_in_repo").
    ssp_dependent:
        True if the source file differs per SSP scenario — meaning the
        per-SSP forecast-input netCDFs will hold different values for
        this variable.
    rcp_filter:
        True if the source file contains all RCP scenarios as rows and
        the reader filters via ('scenario', '==', rcp_scenario).
    """
    description: str
    kind: Kind
    source_kind: SourceKind
    source_path_attr: str | None = None
    source_filename: str | None = None
    source_script: str | None = None
    ssp_dependent: bool = False
    rcp_filter: bool = False


# ── Climate (draw-varying, ssp-dependent) ─────────────────────────────────────
# All from rapidresponse climate-aggregates, one file per (var, ssp) at
# CLIMATE_AGGREGATES_PATH / {lsae_hierarchy} / {var}_{ssp_scenario}.parquet.
_CLIMATE_PATH_ATTR = "CLIMATE_AGGREGATES_PATH"
_CLIMATE_FILENAME  = "{var}_{ssp_scenario}.parquet"

_CLIMATE_SPECS = {
    "total_precipitation":   "Annual total precipitation; rapidresponse climate-aggregates.",
    "precipitation_days":    "Days per year with non-zero precipitation; rapidresponse.",
    "relative_humidity":     "Annual mean relative humidity (%); rapidresponse.",
    "wind_speed":            "Annual mean surface wind speed; rapidresponse.",
    "mean_temperature":      "Annual mean temperature (°C); rapidresponse.",
    "mean_low_temperature":  "Annual mean daily-low temperature (°C); rapidresponse.",
    "mean_high_temperature": "Annual mean daily-high temperature (°C); rapidresponse.",
    "days_over_30C":         "Days per year with mean temperature > 30°C; rapidresponse.",
}

# ── Shared scalars and flooding (non-draw) ────────────────────────────────────
# gdppc/ldipc are SSP-dependent via the 'scenario' column (rcp_filter=True);
# urban is not SSP-dependent; flooding has one file per SSP.

# ── COVARIATE_REGISTRY ────────────────────────────────────────────────────────
COVARIATE_REGISTRY: dict[str, CovariateSpec] = {
    # Climate (8 variables; built in a loop below)
    **{
        var: CovariateSpec(
            description=desc,
            kind="climate_draw",
            source_kind="external",
            source_path_attr=_CLIMATE_PATH_ATTR,
            source_filename=_CLIMATE_FILENAME.format(var=var, ssp_scenario="{ssp_scenario}"),
            ssp_dependent=True,
        )
        for var, desc in _CLIMATE_SPECS.items()
    },

    # mean_low_temperature: SINGLE-REALIZATION override of the climate_draw entry
    # above (later key wins). Stored as the ensemble mean over climate draws, so
    # it is loc x year (not loc x year x draw) — no draw-varying array added to the
    # forecast nc / rocket. Delete this entry to revert to draw-varying.
    "mean_low_temperature": CovariateSpec(
        description="Annual mean daily-low temperature (°C); rapidresponse. "
                    "Stored as the ensemble MEAN over climate draws (loc x year).",
        kind="climate_mean",
        source_kind="external",
        source_path_attr=_CLIMATE_PATH_ATTR,
        source_filename=_CLIMATE_FILENAME.format(var="mean_low_temperature", ssp_scenario="{ssp_scenario}"),
        ssp_dependent=True,
    ),

    # Suitability — draw-varying, single variant chosen at builder runtime.
    "malaria_suitability": CovariateSpec(
        description="Malaria temperature suitability (raw days; R derives logit). "
                    "Rapidresponse climate-aggregates; variant chosen via "
                    "--suitability_variant. Path resolved by "
                    "mbpc.get_malaria_suitability_path(variant, ssp, hierarchy).",
        kind="suitability",
        source_kind="external",
        source_path_attr=None,
        source_filename=None,
        ssp_dependent=True,
    ),

    # Dengue temperature suitability — draw-varying, SAME climate-aggregates layout as
    # the climate vars above (plain per-draw wide parquet; no variant selector, unlike
    # malaria_suitability). Registered as climate_draw so 08b reads it via the climate path.
    "dengue_suitability": CovariateSpec(
        description="Dengue temperature suitability (draw-varying; raw index, "
                    "predict-time derives any transform). Rapidresponse "
                    "climate-aggregates: CLIMATE_AGGREGATES_PATH/{lsae}/"
                    "dengue_suitability_{ssp_scenario}.parquet.",
        kind="climate_draw",
        source_kind="external",
        source_path_attr=_CLIMATE_PATH_ATTR,
        source_filename=_CLIMATE_FILENAME.format(var="dengue_suitability", ssp_scenario="{ssp_scenario}"),
        ssp_dependent=True,
    ),

    # GDP per capita — SSP-dependent via rcp_scenario filter.
    "gdppc_mean": CovariateSpec(
        description="GDP per capita (2020 USD); FHS team projection.",
        kind="shared_scalar",
        source_kind="external",
        source_path_attr="GDPPC_READ_PATH",
        source_filename="gdppc_mean.parquet",
        ssp_dependent=True,
        rcp_filter=True,
    ),

    # LDIPC — not used in current malaria forecasts, kept for future use.
    "ldipc_mean": CovariateSpec(
        description="Lag-distributed income per capita; FHS team projection. "
                    "Not in current malaria forecast covariate set.",
        kind="shared_scalar",
        source_kind="external",
        source_path_attr="LDIPC_READ_PATH",
        source_filename="ldipc_mean.parquet",
        ssp_dependent=True,
        rcp_filter=True,
    ),

    # Urban — population-weighted 1km urban fraction at multiple thresholds.
    # Not SSP-dependent. Source columns are in urban_threshold_{T}.0_simple_mean
    # parquet files; reader returns the non-id/non-population columns.
    "weighted_1km_urban_threshold_300.0_simple_mean": CovariateSpec(
        description="Population-weighted 1km-pixel urban fraction at density "
                    "threshold 300 ppl/km²; produced by pixel_urban_hierarchy.py "
                    "(stage 01).",
        kind="shared_scalar",
        source_kind="made_in_repo",
        source_path_attr="URBAN_READ_PATH",
        source_filename="urban_threshold_300.0_simple_mean.parquet",
        source_script="01_map_to_admin_2/pixel_urban_hierarchy.py",
        ssp_dependent=False,
    ),
    "weighted_1km_urban_threshold_1500.0_simple_mean": CovariateSpec(
        description="Population-weighted 1km-pixel urban fraction at density "
                    "threshold 1500 ppl/km²; produced by pixel_urban_hierarchy.py.",
        kind="shared_scalar",
        source_kind="made_in_repo",
        source_path_attr="URBAN_READ_PATH",
        source_filename="urban_threshold_1500.0_simple_mean.parquet",
        source_script="01_map_to_admin_2/pixel_urban_hierarchy.py",
        ssp_dependent=False,
    ),

    # Flooding — SSP-dependent; single file per SSP holds people_flood_days_*.
    # Source path is templated via the rapidresponse flooding output dir.
    "people_flood_days_per_capita": CovariateSpec(
        description="People-days of flooding per capita per year; "
                    "ensemble-mean from rapidresponse flooding pipeline "
                    "(fldfrc_shifted0.1_sum_{ssp}_mean_r1i1p1f1.parquet).",
        kind="flooding",
        source_kind="external",
        source_path_attr=None,  # path is templated outside the standard tree
        source_filename="fldfrc_shifted0.1_sum_{ssp_scenario}_mean_r1i1p1f1.parquet",
        ssp_dependent=True,
    ),

    # DAH — A0-level source, broadcast to A2 + scenario dim by build_dah_array.
    "mal_DAH_total_per_capita": CovariateSpec(
        description="Malaria DAH per capita (A0-broadcast). Source: dah_df.parquet "
                    "produced by 02_data_prep/make_dah_df.py. Scenario dim "
                    "(Baseline, Constant) added by build_dah_array.",
        kind="dah",
        source_kind="made_in_repo",
        source_path_attr="DAH_READ_PATH",
        source_filename="dah_df.parquet",
        source_script="02_data_prep/make_dah_df.py",
        ssp_dependent=False,
    ),

    # Med-consumption per capita — not in current forecast set, registered for
    # future use (already produced by make_med_consumppc_df.py).
    "med_consumppc": CovariateSpec(
        description="Median consumption per capita; produced in repo. "
                    "Not in current malaria forecast covariate set.",
        kind="shared_scalar",
        source_kind="made_in_repo",
        source_path_attr="MED_CONSUMPPC_READ_PATH",
        source_filename="med_consumppc_mean.parquet",
        source_script="02_data_prep/make_med_consumppc_df.py",
        ssp_dependent=True,
        rcp_filter=True,
    ),

    # Static lookups — needed by R at predict time (A0 fixed effect).
    "A0_location_id": CovariateSpec(
        description="Admin-0 location_id for each prediction location; derived "
                    "from the hierarchy. R uses paste0('A0_', A0_location_id) "
                    "to construct the A0_af fixed-effect factor.",
        kind="static_lookup",
        source_kind="made_in_repo",
        source_path_attr="HIERARCHY_READ_PATH",
        source_filename="full_hierarchy_2023_{lsae_hierarchy}.parquet",
        source_script="02_data_prep/01_make_full_hierarchy.py",
        ssp_dependent=False,
    ),
}


# Default covariate set for the malaria forecast-input builder.
# Matches the current winning model's predictors plus the static A0 lookup.
DEFAULT_MALARIA_FORECAST_COVARIATES: tuple[str, ...] = (
    "mal_DAH_total_per_capita",
    "gdppc_mean",
    "weighted_1km_urban_threshold_300.0_simple_mean",
    "people_flood_days_per_capita",
    "malaria_suitability",
    "mean_low_temperature",   # single-realization (climate_mean); used by formulations f1-f4
    "A0_location_id",
)


# Default covariate set for the dengue forecast-input builder (08b). Matches the
# pyGAM dengue explorer's predictors: base-incidence climate/urban/flood + gdppc (CFR)
# + the static A0 lookup. dengue_suitability + relative_humidity are draw-varying.
DEFAULT_DENGUE_FORECAST_COVARIATES: tuple[str, ...] = (
    "dengue_suitability",
    "relative_humidity",
    "weighted_1km_urban_threshold_300.0_simple_mean",
    "people_flood_days_per_capita",
    "gdppc_mean",
    "A0_location_id",
)
