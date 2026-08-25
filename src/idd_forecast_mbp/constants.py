import json
import os
from pathlib import Path

MODEL_ROOT = Path("/mnt/team/idd/pub/forecast-mbp")

REPO_ROOT = Path("/mnt/share/homes/bcreiner/repos")

# Run date: set IDD_RUN_DATE env var to override (e.g. for re-running a prior date).
# Format: YYYYMMDD. Multiple runs same day: set to YYYYMMDD_v2, etc.
# RUN_DATE: "20260405" First run after refactor
# RUN_DATE: "20260527" Run with 'new' gridded population: '2026_05_15.001'
GRIDDED_RD_2026_05_15 = "20260527"
RUN_DATE: str = os.environ.get("IDD_RUN_DATE", GRIDDED_RD_2026_05_15)
CLIMATE_COVARIATE_RUN_DATE: str = "2026_05_28"
MALARIA_SUITABILITY_RUN_DATE: str = "2026_05_27"
# Rapidresponse-aggregated population.parquet lives under the climate-aggregates
# tree but is published by a different upstream pipeline than the climate
# covariates, on a different cadence. Pinned separately so the two can advance
# independently. Sole consumer: LSAE_POP_PATH (defined below).
LSAE_POP_RUN_DATE: str = "2026_05_27"
# Rapidresponse flooding aggregates. The output dir
# /mnt/team/rapidresponse/pub/flooding/results/output/<hierarchy>/ holds older flat
# files; newer reruns land in a dated subdir <hierarchy>/<FLOODING_RUN_DATE>/.
# Every flooding read in this repo routes through this constant.
FLOODING_RUN_DATE: str = "20260530"
GRIDDED_POPULATION_BY_BLOCK_RUNDATE = "2026_05_15.001"
GRIDDED_POPULATION_RUNDATE = "2026_05_16"

# Whether the upstream FHS future-population file still contains location_id 44858
# (Ethiopia super-national, present in older FHS releases; split into 60908/95069/94364
# at source in newer releases). When True, 02a_fhs_population.py applies the historical
# split-redistribution to map 44858's future pop onto the three sub-nationals. When False
# (current FHS vintage, verified 2026-05-30: `44858 in future_fhs_pop_ds.location_id.values`
# returns False), that block is skipped because the data already has the three sub-nationals.
# Downstream scripts in 05_aggregation and 06_upload also have 44858 special-casing that
# should eventually gate on this flag (separate work; not yet done).
FHS_FUTURE_POP_HAS_44858: bool = False


# ── Stage root directories ────────────────────────────────────────────────────
# These are the node directories that contain dated run subdirs + current/ symlink.
# Do not use these directly for file I/O — use the write/read paths below.
_RAW_STAGE       = MODEL_ROOT / "01-raw_data"
_PROCESSED_STAGE = MODEL_ROOT / "02-processed_data"
_MODELING_STAGE  = MODEL_ROOT / "03-modeling_data"
_FORECASTING_STAGE = MODEL_ROOT / "04-forecasting_data"
_UPLOAD_STAGE    = MODEL_ROOT / "05-upload_data"
_VIZ_STAGE       = MODEL_ROOT / "06-visualization"
_FIGURES_STAGE   = MODEL_ROOT / "07-figures"
_MANUSCRIPT_STAGE  = MODEL_ROOT / "08-manuscript_material"
_PRESENTATION_STAGE = MODEL_ROOT / "10-presentation_material"


def _artifact_write(artifact_root: Path, run_date: str = None) -> Path:
    """Return the write path for an artifact: artifact_root/RUN_DATE."""
    return artifact_root / (run_date or RUN_DATE)


def _artifact_read(artifact_root: Path) -> Path:
    """Return the read path for an artifact via its current/ symlink."""
    return artifact_root / "current"


# ── Malaria model registry ────────────────────────────────────────────────────
# JSON registry of fitted malaria-model runs, written by
# 03_modeling/02_fit_final_malaria_models.r (via lib/model_registry.R). One array
# of run records; each carries a `best` flag and exactly one is True. This is the
# single source of truth for "which run is best" — both R and Python read it
# rather than hardcoding a model_date. Lives next to the {run_date}_malaria_models.RData.
MALARIA_MODEL_REGISTRY: Path = _MODELING_STAGE / "malaria_model_registry.json"


def read_malaria_model_registry(path: Path = MALARIA_MODEL_REGISTRY) -> list[dict]:
    """Return the malaria model run registry (list of run records), best-first.

    Empty list if the registry does not exist yet (i.e. no run has been recorded)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        recs = json.load(f)
    return recs or []


def get_malaria_model_run_date(best: bool = True, run_date: str | None = None,
                               path: Path = MALARIA_MODEL_REGISTRY) -> str:
    """Resolve a malaria-model run_date from the registry.

    - run_date set: verify it exists and return it.
    - best=True (default): return the run_date of the single entry flagged best.

    Raises if the registry is empty/missing, the date is absent, or the best flag
    is ambiguous (0 or >1 entries flagged)."""
    recs = read_malaria_model_registry(path)
    if not recs:
        raise FileNotFoundError(f"Malaria model registry is empty or missing: {path}")
    if run_date is not None:
        hits = [r for r in recs if str(r.get("run_date")) == str(run_date)]
        if not hits:
            raise KeyError(f"No malaria model run with run_date={run_date!r} in {path}")
        return str(hits[0]["run_date"])
    if best:
        best_hits = [r for r in recs if r.get("best") is True]
        if len(best_hits) == 0:
            raise ValueError(f"No malaria model flagged best=True in {path}")
        if len(best_hits) > 1:
            raise ValueError(
                f"Multiple malaria models flagged best=True in {path}; exactly one expected."
            )
        return str(best_hits[0]["run_date"])
    raise ValueError("get_malaria_model_run_date(): specify run_date=, or leave best=True.")


# ── External / read-only data (not versioned by this pipeline) ────────────────
RAW_DATA_PATH  = _RAW_STAGE          # GBD pulls, raw inputs — never written by pipeline
GBD_DATA_PATH  = RAW_DATA_PATH / "gbd"
# Versioned GBD pull written by 00_pull_raw_data/get_gbd_data (dated dir + `current`
# symlink). All GBD readers resolve through this, NOT the loose top-level files.
GBD_DATA_READ_PATH = GBD_DATA_PATH / "current"

# Malaria vaccine (RTS,S / R21) dose-3 / dose-4 coverage, received from the coverage
# modelers — dated dir + `current` symlink, resolved like the GBD pull above. No
# hierarchy split: rows are admin1 (subnat_id) within 37 countries, already keyed to
# the LSAE hierarchy, so there is nothing to split by. Read-only here — this pipeline
# never writes it, so there is deliberately no write path.
VACCINE_COVERAGE_PATH      = RAW_DATA_PATH / "malaria_vaccine_coverage"
VACCINE_COVERAGE_READ_PATH = _artifact_read(VACCINE_COVERAGE_PATH)
# Delivered file inside that node. A new vintage arrives as a new dated dir, so the
# filename is pinned with the run date it came with.
VACCINE_COVERAGE_FILE      = VACCINE_COVERAGE_READ_PATH / "2026_08_05_final_handoff_v2_edu_dtp3_lme.csv"

# Malaria vaccine efficacy curves, received from the VE modelers. Monthly VE by
# vaccine product for clinical ("case") and severe ("death") outcomes, split into
# dose-3-only and boosted (dose 3 + 4) columns. Delivered as a 2x2 factorial of
# curve-shape assumptions (linear vs log-linear interpolation x whether unboosted
# severe VE drops to 0 at the booster), one CSV per cell -- see FACTORIAL_README.md
# in the node. The variant is a modelling choice, so it is selected per run rather
# than defaulted; no hierarchy split (VE is per product, not per location).
VACCINE_EFFICACY_PATH      = RAW_DATA_PATH / "malaria_vaccine_efficacy"
VACCINE_EFFICACY_READ_PATH = _artifact_read(VACCINE_EFFICACY_PATH)
# ^ the received vintage, kept as the immutable record of what was handed over.
# From 2026-08-24 the curves are BUILT here from anchors, so the pipeline reads
# its own processed output (verified byte-identical to the received files).
VE_ANCHORS_PATH = Path(__file__).parent / "VE_ANCHORS.yaml"
VE_VARIANTS = ("linear_severe0", "linear_severeSmooth",
               "loglinear_severe0", "loglinear_severeSmooth")


def vaccine_efficacy_file(variant: str) -> Path:
    """Path to one VE factorial cell's CSV within the current node."""
    if variant not in VE_VARIANTS:
        raise ValueError(f"unknown VE variant {variant!r}; expected one of {VE_VARIANTS}")
    return VE_CURVES_READ_PATH / f"ve_{variant}.csv"

# Covariate data produced by the RapidResponse lsae pipeline. Lives flat in
# 02-processed_data/lsae_XXXX/ and is updated externally, not by this pipeline.
LSAE_HIERARCHY = "lsae_1285"
LSAE_INPUT_PATH = _PROCESSED_STAGE / LSAE_HIERARCHY

# ── 02-processed_data stage-01 pixel artifacts (GBD-release-tagged) ──────────
# pixel_main (per-block scratch) and pixel_hierarchy (per-hierarchy aggregates)
# outputs are tagged with the GBD release the source TIFFs were calibrated
# against. Currently 'GBD2023'. A future GBD2024 release would write to a
# sibling 'GBD2024/' tree without disturbing this one.
#
# Per-hierarchy artifact root: 02-processed_data/GBD2023/<hierarchy>/,
# versioned by RUN_DATE via _artifact_write / _artifact_read. Both per-block
# scratch and per-hierarchy aggregates share the same root for a given
# (release, hierarchy) — pixel_main writes <root>/<RUN_DATE>/<cov>/<block>/000.parquet
# and pixel_hierarchy writes <root>/<RUN_DATE>/<cov>_<stat>_<scenario>.parquet.
#
# This convention does NOT apply to pixel_urban_main / pixel_urban_hierarchy —
# urban is derived from population density and is not GBD-release-dependent.
PIXEL_GBD_RELEASE: str = "GBD2023"


def pixel_artifact_root(hierarchy: str) -> Path:
    """Artifact root for stage-01 pixel outputs at `hierarchy` under the
    current GBD release. Used by both pixel_main (per-block scratch) and
    pixel_hierarchy (per-hierarchy aggregates). Pass `finalize_artifact(...)`
    after a successful launch to update the current/ symlink."""
    return _PROCESSED_STAGE / PIXEL_GBD_RELEASE / hierarchy


def pixel_write_path(hierarchy: str, run_date: str | None = None) -> Path:
    """Versioned write path for stage-01 pixel outputs at `hierarchy`.
    Resolves to: 02-processed_data/GBD2023/<hierarchy>/<RUN_DATE>/."""
    return _artifact_write(pixel_artifact_root(hierarchy), run_date)


def pixel_read_path(hierarchy: str) -> Path:
    """Read path (via current/ symlink) for stage-01 pixel outputs at `hierarchy`.
    Resolves to: 02-processed_data/GBD2023/<hierarchy>/current/."""
    return _artifact_read(pixel_artifact_root(hierarchy))


CLIMATE_AGGREGATES_PATH = Path("/mnt/team/rapidresponse/pub/climate-aggregates") / CLIMATE_COVARIATE_RUN_DATE / "results"
# Canonical gridded population (location_id × year_id, ~1950-2100), produced
# by the rapidresponse team. Read-only here. Lives under the climate-aggregates
# tree but pinned to its own LSAE_POP_RUN_DATE so it advances independently of
# the climate-covariate vintage (different upstream pipelines, different cadences).
LSAE_POP_PATH = (
    Path("/mnt/team/rapidresponse/pub/climate-aggregates")
    / LSAE_POP_RUN_DATE / "results" / LSAE_HIERARCHY / "population.parquet"
)


# Gridded population by block, for use in urban pixel generation. Also produced by the
# rapidresponse team but updated more frequently, so we route through the population model
# modeling frame to get the correct run date for each block.

MODELING_FRAME_PATH = Path("/mnt/team/rapidresponse/pub/population-model/modeling/100m/modeling_frame.parquet")
GRIDDED_POPULATION_ROOT = f"/mnt/team/rapidresponse/pub/population-model/results/{GRIDDED_POPULATION_RUNDATE}"
GRIDDED_POPULATION_BY_BLOCK_PATH = Path(f"/mnt/team/rapidresponse/pub/population-model/modeling/100m/models/{GRIDDED_POPULATION_BY_BLOCK_RUNDATE}")

# Malaria suitability variants (malaria-specific, not universal across causes).
# Provenance: the 14 variant files (2 methods × 7 shifts) are produced by the
# climate-data repo at /mnt/share/homes/bcreiner/repos/climate-data/. This repo
# only consumes them via get_malaria_suitability_path() below.
MALARIA_SUITABILITY_METHODS = ["mordecai", "villena"]
MALARIA_SUITABILITY_SHIFTS = ["0_0", "p0_25", "p0_5", "p1_0", "m0_25", "m0_5", "m1_0"]
MALARIA_SUITABILITY_VARIANTS = [
    f"{method}_{shift}"
    for method in MALARIA_SUITABILITY_METHODS
    for shift in MALARIA_SUITABILITY_SHIFTS
]
MALARIA_SUITABILITY_VARIANT: str = "mordecai_0_0"

def get_malaria_suitability_path(variant: str, ssp_scenario: str, lsae_hierarchy: str) -> str:
    base = Path("/mnt/team/rapidresponse/pub/climate-aggregates") / MALARIA_SUITABILITY_RUN_DATE / "results" / lsae_hierarchy
    return str(base / f"malaria_{variant}_suitability_{ssp_scenario}.parquet")

# Age-specific FHS metadata (written by get_past_as_aa_fhs_outcomes.r, read-only here).
AGE_SPECIFIC_FHS_PATH = _PROCESSED_STAGE / "age_specific_fhs"

# ── 02-processed_data artifact roots ─────────────────────────────────────────
# Each artifact dir contains: {hierarchy}/RUN_DATE/, {hierarchy}/current -> RUN_DATE
# DAH has no hierarchy split (national-level data).
_A02_HIERARCHY   = _PROCESSED_STAGE / "hierarchy"   / LSAE_HIERARCHY
_A02_POPULATION  = _PROCESSED_STAGE / "population"  / LSAE_HIERARCHY
# Malaria VE curves, built by 02_data_prep/09_build_vaccine_efficacy_curves.py
# from VE_ANCHORS.yaml. No hierarchy split: VE is per product, not per location.
_A02_VE_CURVES   = _PROCESSED_STAGE / "malaria_vaccine_efficacy"
_A02_DAH         = _PROCESSED_STAGE / "covariates"  / "dah"
_A02_GDPPC       = _PROCESSED_STAGE / "covariates"  / "gdppc"        / LSAE_HIERARCHY
_A02_LDIPC       = _PROCESSED_STAGE / "covariates"  / "ldipc"        / LSAE_HIERARCHY
_A02_MED_CONSUMPPC = _PROCESSED_STAGE / "covariates" / "med_consumppc" / LSAE_HIERARCHY
_A02_URBAN       = _PROCESSED_STAGE / "urban"        / LSAE_HIERARCHY
_A02_MAL_RAKED_AA = _PROCESSED_STAGE / "malaria"    / "raked_aa" / LSAE_HIERARCHY
_A02_MAL_RAKED_AS = _PROCESSED_STAGE / "malaria"    / "raked_as" / LSAE_HIERARCHY
_A02_DEN_RAKED_AA = _PROCESSED_STAGE / "dengue"     / "raked_aa" / LSAE_HIERARCHY
_A02_DEN_RAKED_AS = _PROCESSED_STAGE / "dengue"     / "raked_as" / LSAE_HIERARCHY

# Write paths (current run)
HIERARCHY_WRITE_PATH    = _artifact_write(_A02_HIERARCHY)
POPULATION_WRITE_PATH   = _artifact_write(_A02_POPULATION)
VE_CURVES_WRITE_PATH    = _artifact_write(_A02_VE_CURVES)
DAH_WRITE_PATH          = _artifact_write(_A02_DAH)
GDPPC_WRITE_PATH        = _artifact_write(_A02_GDPPC)
LDIPC_WRITE_PATH        = _artifact_write(_A02_LDIPC)
MED_CONSUMPPC_WRITE_PATH = _artifact_write(_A02_MED_CONSUMPPC)
URBAN_WRITE_PATH        = _artifact_write(_A02_URBAN)
MAL_RAKED_AA_WRITE_PATH = _artifact_write(_A02_MAL_RAKED_AA)
MAL_RAKED_AS_WRITE_PATH = _artifact_write(_A02_MAL_RAKED_AS)
DEN_RAKED_AA_WRITE_PATH = _artifact_write(_A02_DEN_RAKED_AA)
DEN_RAKED_AS_WRITE_PATH = _artifact_write(_A02_DEN_RAKED_AS)

# Read paths (via current/ symlink)
HIERARCHY_READ_PATH    = _artifact_read(_A02_HIERARCHY)
POPULATION_READ_PATH   = _artifact_read(_A02_POPULATION)
VE_CURVES_READ_PATH    = _artifact_read(_A02_VE_CURVES)
DAH_READ_PATH          = _artifact_read(_A02_DAH)
GDPPC_READ_PATH        = _artifact_read(_A02_GDPPC)
LDIPC_READ_PATH        = _artifact_read(_A02_LDIPC)
MED_CONSUMPPC_READ_PATH = _artifact_read(_A02_MED_CONSUMPPC)
URBAN_READ_PATH        = _artifact_read(_A02_URBAN)
MAL_RAKED_AA_READ_PATH = _artifact_read(_A02_MAL_RAKED_AA)
MAL_RAKED_AS_READ_PATH = _artifact_read(_A02_MAL_RAKED_AS)
DEN_RAKED_AA_READ_PATH = _artifact_read(_A02_DEN_RAKED_AA)
DEN_RAKED_AS_READ_PATH = _artifact_read(_A02_DEN_RAKED_AS)

# ── 03-modeling_data artifact roots ──────────────────────────────────────────
_A03_MAL_MODELING    = _MODELING_STAGE / "malaria" / "modeling_dfs"   / LSAE_HIERARCHY
_A03_DEN_MODELING    = _MODELING_STAGE / "dengue"  / "modeling_dfs"   / LSAE_HIERARCHY
_A03_MAL_PAST_INPUTS = _MODELING_STAGE / "malaria" / "past_inputs_nc" / LSAE_HIERARCHY
_A03_DEN_PAST_INPUTS = _MODELING_STAGE / "dengue"  / "past_inputs_nc" / LSAE_HIERARCHY

# Write paths
MAL_MODELING_WRITE_PATH    = _artifact_write(_A03_MAL_MODELING)
DEN_MODELING_WRITE_PATH    = _artifact_write(_A03_DEN_MODELING)
MAL_PAST_INPUTS_WRITE_PATH = _artifact_write(_A03_MAL_PAST_INPUTS)
DEN_PAST_INPUTS_WRITE_PATH = _artifact_write(_A03_DEN_PAST_INPUTS)

# Read paths
MAL_MODELING_READ_PATH     = _artifact_read(_A03_MAL_MODELING)
DEN_MODELING_READ_PATH     = _artifact_read(_A03_DEN_MODELING)
MAL_PAST_INPUTS_READ_PATH  = _artifact_read(_A03_MAL_PAST_INPUTS)
DEN_PAST_INPUTS_READ_PATH  = _artifact_read(_A03_DEN_PAST_INPUTS)

# Dengue fit-location set (written by 06a; a modeling concern → under 03-modeling).
_A03_DEN_FIT_LOCATIONS = _MODELING_STAGE / "dengue" / "fit_locations" / LSAE_HIERARCHY
DEN_FIT_LOCATIONS_WRITE_PATH = _artifact_write(_A03_DEN_FIT_LOCATIONS)
DEN_FIT_LOCATIONS_READ_PATH  = _artifact_read(_A03_DEN_FIT_LOCATIONS)

# Dengue location-selection thresholds (A0-level all-age counts; strictly > ).
# Exploratory defaults = 0.0 (any nonzero qualifies); tighten later.
dengue_fit_mort_threshold  = 0.0
dengue_fit_inc_threshold   = 0.0
dengue_pred_mort_threshold = 0.0
dengue_pred_inc_threshold  = 0.0

# ── 04-forecasting_data artifact roots ───────────────────────────────────────
_A04_MAL_FORECAST_LOCATIONS = _FORECASTING_STAGE / "malaria" / "prediction_locations" / LSAE_HIERARCHY
_A04_MAL_FORECAST_INPUTS    = _FORECASTING_STAGE / "malaria" / "forecast_inputs"      / LSAE_HIERARCHY
_A04_MAL_FORECAST_OUTPUTS   = _FORECASTING_STAGE / "malaria" / "forecast_outputs"     / LSAE_HIERARCHY

MAL_FORECAST_LOCATIONS_WRITE_PATH = _artifact_write(_A04_MAL_FORECAST_LOCATIONS)
MAL_FORECAST_LOCATIONS_READ_PATH  = _artifact_read(_A04_MAL_FORECAST_LOCATIONS)

_A04_DEN_FORECAST_LOCATIONS = _FORECASTING_STAGE / "dengue" / "prediction_locations" / LSAE_HIERARCHY
DEN_FORECAST_LOCATIONS_WRITE_PATH = _artifact_write(_A04_DEN_FORECAST_LOCATIONS)
DEN_FORECAST_LOCATIONS_READ_PATH  = _artifact_read(_A04_DEN_FORECAST_LOCATIONS)

_A04_DEN_FORECAST_INPUTS = _FORECASTING_STAGE / "dengue" / "forecast_inputs" / LSAE_HIERARCHY
DEN_FORECAST_INPUTS_WRITE_PATH = _artifact_write(_A04_DEN_FORECAST_INPUTS)
DEN_FORECAST_INPUTS_READ_PATH  = _artifact_read(_A04_DEN_FORECAST_INPUTS)

MAL_FORECAST_INPUTS_WRITE_PATH = _artifact_write(_A04_MAL_FORECAST_INPUTS)
MAL_FORECAST_INPUTS_READ_PATH  = _artifact_read(_A04_MAL_FORECAST_INPUTS)

# Raked draw-level forecast predictions (written by forecast_malaria_admin_2s_rocket.r).
MAL_FORECAST_OUTPUTS_WRITE_PATH = _artifact_write(_A04_MAL_FORECAST_OUTPUTS)

# Malaria vaccine cohort coverage applied to age-sex population: dose-3/dose-4-ever
# fractions, waning-adjusted protection, and headcounts by location x year x
# age_group_id x sex_id. Written by 04_forecasting/apply_vaccine_coverage_to_population.py
# from the raw coverage node (VACCINE_COVERAGE_FILE) + POPULATION_READ_PATH.
_A04_MAL_VACCINE_COHORTS = _FORECASTING_STAGE / "malaria" / "vaccine_cohorts" / LSAE_HIERARCHY
MAL_VACCINE_COHORTS_WRITE_PATH = _artifact_write(_A04_MAL_VACCINE_COHORTS)
MAL_VACCINE_COHORTS_READ_PATH  = _artifact_read(_A04_MAL_VACCINE_COHORTS)
# Variant is in the filename, not a column: all rows of one run share it.
MAL_VACCINE_COHORTS_FILENAME_TEMPLATE = (
    "vaccine_cohort_coverage_ve_{variant}_{products}.parquet"
)
# Product-rollout scenarios: as delivered, or counterfactual R21 everywhere.
# "projected" = the delivered coverage series, which is itself an LME projection
# to 2100 -- NOT observed data. "all_r21" is the counterfactual product swap.
PRODUCT_SCENARIOS = ("projected", "all_r21")

MAL_FORECAST_OUTPUTS_READ_PATH  = _artifact_read(_A04_MAL_FORECAST_OUTPUTS)

# ── 05-products artifact roots (finished forecast products) ───────────────────
# One artifact node per forecast RUN, so each run versions and finalizes
# independently: <cause>/<hierarchy>/<run_key>/<RUN_DATE>/ + current -> RUN_DATE.
_PRODUCTS_STAGE   = MODEL_ROOT / "05-products"
_A05_MAL_PRODUCTS = _PRODUCTS_STAGE / "malaria" / LSAE_HIERARCHY
_A05_DEN_PRODUCTS = _PRODUCTS_STAGE / "dengue"  / LSAE_HIERARCHY


def mal_products_root(run_key: str) -> Path:
    """Artifact root for one malaria forecast run's finished products."""
    return _A05_MAL_PRODUCTS / run_key


def mal_products_write_path(run_key: str, run_date: str = None) -> Path:
    """Dated write path for one malaria forecast run's finished products."""
    return _artifact_write(mal_products_root(run_key), run_date)


def mal_products_read_path(run_key: str) -> Path:
    """current/ read path for one malaria forecast run's finished products."""
    return _artifact_read(mal_products_root(run_key))


def den_products_root(run_key: str) -> Path:
    """Artifact root for one dengue forecast run's finished products."""
    return _A05_DEN_PRODUCTS / run_key


def den_products_write_path(run_key: str, run_date: str = None) -> Path:
    """Dated write path for one dengue forecast run's finished products."""
    return _artifact_write(den_products_root(run_key), run_date)


def den_products_read_path(run_key: str) -> Path:
    """current/ read path for one dengue forecast run's finished products."""
    return _artifact_read(den_products_root(run_key))

# ── Stage-level paths (stages 04–10, not yet artifact-structured) ─────────────
FORECASTING_DATA_PATH = _FORECASTING_STAGE / RUN_DATE
UPLOAD_DATA_PATH      = _UPLOAD_STAGE      / RUN_DATE

# ── Previous-submission comparators (read-only) ───────────────────────────────
# Runs that current work is compared against. Centralised here so stage scripts
# carry no absolute paths; bump the dates when a new comparator supersedes these.
_UPLOAD_FOLDERS = _UPLOAD_STAGE / "upload_folders"
FIRST_SUBMISSION_UPLOAD_DATE = "2025_08_28"   # first submission = burden comparator
PREVIOUS_COVARIATE_UPLOAD_DATE = "2025_08_11" # arm carrying cov_ds_*.nc


def previous_upload_path(upload_date: str) -> Path:
    """One previous-run arm directory under 05-upload_data/upload_folders/."""
    return _UPLOAD_FOLDERS / upload_date


#: Default burden comparator: the first-submission run directory.
FIRST_SUBMISSION_RUN_PATH = previous_upload_path(FIRST_SUBMISSION_UPLOAD_DATE)
#: Default covariate comparator: previous-run covariate netCDF (no draw dimension).
PREVIOUS_COVARIATE_NC = (
    previous_upload_path(PREVIOUS_COVARIATE_UPLOAD_DATE) / "cov_ds_Baseline.nc"
)

# Source for the scenario-varying GDP per capita forecasts (external, read-only).
GDPPC_SOURCE_PATH = (
    Path("/mnt/share/resource_tracking/forecasting/poverty")
    / "climate_2025_income_distribution_forecasts"
    / "V5_consumption_forecasting_admin2_scenarios"
    / "LSAEadmin2_gdppc_mean_forecasts_scenarios_2010PPP.csv"
)
VISUALIZATION_PATH    = _VIZ_STAGE         / RUN_DATE
FIGURES_PATH          = _FIGURES_STAGE     / RUN_DATE
MANUSCRIPT_PATH       = _MANUSCRIPT_STAGE  / RUN_DATE
PRESENTATION_PATH     = _PRESENTATION_STAGE / RUN_DATE

FORECASTING_DATA_READ_PATH = _FORECASTING_STAGE / "current"

FHS_RESULTS_PATH = Path('/mnt/share/forecasting/data/9/future')
RR_PATH = Path('/mnt/team/rapidresponse/pub/malaria-denv')

repo_name = "idd-forecast-mbp"
package_name = "idd_forecast_mbp"


# Constants
# hierarchies = ["lsae_1209", "gbd_2021", "lsae_1285", "gbd_2023"]
hierarchies = [LSAE_HIERARCHY, "gbd_2023"]
hierarchies = [LSAE_HIERARCHY]
# Year-range constants. MODELING and FORECAST overlap on the bridge year(s)
# intentionally: forecasts include the last observed year so observed values
# can be linked to predicted values (verification, scaling, model-vs-data
# plots). FUTURE is the strictly-future window with no observed-data overlap.
#
# Invariants (verified at module load below):
#   MODELING_YEARS ∪ FUTURE_YEARS  = ALL_YEARS   (disjoint partition)
#   MODELING_YEARS ∩ FUTURE_YEARS  = ∅
#   FORECAST_YEARS \ MODELING_YEARS = FUTURE_YEARS
MODELING_YEARS  = list(range(2000, 2024))  # 2000-2023 (24 yrs) — observed window; models are fit on this
FORECAST_YEARS  = list(range(2023, 2101))  # 2023-2100 (78 yrs) — forecast output window; includes bridge year(s)
FUTURE_YEARS    = list(range(2024, 2101))  # 2024-2100 (77 yrs) — strictly future; FORECAST_YEARS minus the bridge
ALL_YEARS       = list(range(2000, 2101))  # 2000-2100 (101 yrs) — union of MODELING_YEARS and FUTURE_YEARS
EXTENDED_YEARS  = list(range(1970, 2101))  # 1970-2100 (131 yrs) — wider window for rare pre-2000 GBD pulls

# GBD Constants
GBD_DATA_DATE = "20260713"
gbd_constants = {
        "gbd_location_set_id": 35,
        "fhs_location_set_id": 39,
        "release_2021_id": 9,
        "release_2023_id": 16,
        "como_2023_v": 1762,        # Checked 2026/7/13
        "codcorrect_2023_v": 528,   # Checked 2026/7/13
        "dalynator_2023_v": 102,    # Checked 2026/7/13
        "burdenator_2023_v": 395,   # Checked 2026/7/13
        "compare_2023_v": 8352      # Checked 2026/7/13
}
# como_2023_v = 1591
# codcorrect_2023_v = 461
# dalynator_2023_v = 96
# burdenator_2023_v = 360
# compare_2023_v = 8234


ages = 22
sexes = 3
dengue_id = 357
malaria_id = 345
malaria_pf_id = 856
malaria_pv_id = 857


aa_merge_variables = ["location_id", "year_id"]
as_merge_variables = ["location_id", "year_id", "age_group_id", "sex_id"]
#
draws = [f"{i:03d}" for i in range(100)]
fhs_draws = [f"draw_{i}" for i in range(100)]

# Maps for various constants
cause_map = {
    'malaria':{
        'cause_id': malaria_id,
        'reference_age_group_id': 3,
        'reference_sex_id': 1,
        'cause_name': 'Malaria',
        'fhs_cause_name': 'malaria'
    },
    'malaria_pf':{
        'cause_id': malaria_pf_id,
        'reference_age_group_id': 3,
        'reference_sex_id': 1,
        'cause_name': 'Malaria falciparum',
        'fhs_cause_name': 'malaria_falciparum'
    },
    'malaria_pv':{
        'cause_id': malaria_pv_id,
        'reference_age_group_id': 3,
        'reference_sex_id': 1,
        'cause_name': 'Malaria vivax',
        'fhs_cause_name': 'malaria_vivax'
    },
    'dengue': {
        'cause_id': dengue_id,
        'reference_age_group_id': 3,
        'reference_sex_id': 1,
        'cause_name': 'Dengue',
        'fhs_cause_name': 'ntd_dengue'
    }
}

# Malaria / dengue aggregate parquets are produced by pixel_hierarchy.py
# and live under the GBD-release-tagged versioned root.
# See pixel_read_path / pixel_write_path above.
_PIXEL_AGG_LSAE = pixel_read_path(LSAE_HIERARCHY)
malaria_variables = {
    "pfpr": f"{_PIXEL_AGG_LSAE}/malaria_pfpr_mean_cc_insensitive.parquet",
    "incidence": f"{_PIXEL_AGG_LSAE}/malaria_pf_inc_rate_mean_cc_insensitive.parquet",
    "mortality": f"{_PIXEL_AGG_LSAE}/malaria_pf_mort_rate_mean_cc_insensitive.parquet",
}
dengue_variables = {
    "dengue_suitability": f"{_PIXEL_AGG_LSAE}/dengue_suitability_mean_cc_insensitive.parquet"
}

modeling_measure_map = {
    "malaria": {
        "mortality": {
            "short": "malaria_mort_rate",
            "gbd_measure_id": 1,
            "gbd_metric_id": 3,
            "count_name": 'malaria_mort_count',
            "transformation": "log",
        },
        "incidence": {
            "short": "malaria_inc_rate",
            "gbd_measure_id": 6,
            "gbd_metric_id": 3,
            "count_name": 'malaria_inc_count',
            "transformation": "log"
        }
    },
    "dengue": {
        "incidence": {
            "short": "dengue_inc_rate",
            "gbd_measure_id": 6,
            "gbd_metric_id": 3,
            "count_name": 'dengue_inc_count',
            "transformation": "log"
        },
        "cfr": {
            "short": "dengue_cfr",
            "gbd_measure_id": [1,6],
            "gbd_metric_id": [3,3],
            "count_name": None,
            "transformation": "logit"
        }
    }
}

measure_map = {
    "mortality": {
        "measure_id": 1,
        "name": "mortality",
        "rate_name": "Mortality rate",
        "count_name": "Deaths",
        "short": "mort",
    },
    "incidence": {
        "measure_id": 6,
        "name": "incidence",
        "rate_name": "Incidence rate",
        "count_name": "Cases",
        "short": "inc",
    }
}


ds_coords = ['location_id', 'year_id', 'sex_id', 'age_group_id']

fhs_population_paths = {
    'ssp245': f'{FHS_RESULTS_PATH}/population/20250709_first_sub_rcp45_climate_ref_100d_hiv_shocks_covid_all/population_agg.nc',
    'ssp126': f'{FHS_RESULTS_PATH}/population/20250709_first_sub_rcp26_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all/population_agg.nc',
    'ssp585': f'{FHS_RESULTS_PATH}/population/20250709_first_sub_rcp85_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all/population_agg.nc',
}


full_measure_map = {
    "mortality": {
        "measure_id": 1,
        "name": "mortality",
        "rate_name": "Mortality rate",
        "count_name": "Deaths",
        "short": "mort",
        'fhs_name': 'death',
        'ssp126': {
            'rate': '20250709_first_sub_rcp26_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all_s8',
            'count': '20250709_first_sub_rcp26_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all_s8_num'
        },
        'ssp245': {
            'rate': '20250709_first_sub_rcp45_climate_ref_100d_hiv_shocks_covid_all_s8',
            'count': '20250709_first_sub_rcp45_climate_ref_100d_hiv_shocks_covid_all_s8_num'
        },
        'ssp585': {
            'rate': '20250709_first_sub_rcp85_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all_s8',
            'count': '20250709_first_sub_rcp85_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all_s8_num'
        }
    },
    "incidence": {
        "measure_id": 6,
        "name": "incidence",
        "rate_name": "Incidence rate",
        "count_name": "Cases",
        "short": "inc",
        'fhs_name': 'incidence',
        'ssp126': {
            'rate': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg',
            'count': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg_num'
        },
        'ssp245': {
            'rate': '20250719_rcp45_first_sub_climate_ref_scen0_agg',
            'count': '20250719_rcp45_first_sub_climate_ref_scen0_agg_num'
        },
        'ssp585': {
            'rate': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg',
            'count': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg_num'
        }
    },
    "daly": {
        "measure_id": 2,
        "name": "daly",
        "rate_name": "DALY rate",
        "count_name": "DALYs",
        "short": "daly",
        'fhs_name': 'daly',
        'ssp126': {
            'rate': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg',
            'count': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg_num'
        },
        'ssp245': {
            'rate': '20250719_rcp45_first_sub_climate_ref_agg',
            'count': '20250719_rcp45_first_sub_climate_ref_agg_num'
        },
        'ssp585': {
            'rate': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg',
            'count': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg_num'
        }
    },
    "yld": {
        "measure_id": 3,
        "name": "yld",
        "rate_name": "YLD rate",
        "count_name": "YLDs",
        "short": "yld",
        'fhs_name': 'yld',
        'ssp126': {
            'rate': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg',
            'count': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg_num'
        },
        'ssp245': {
            'rate': '20250719_rcp45_first_sub_climate_ref_scen0_agg',
            'count': '20250719_rcp45_first_sub_climate_ref_scen0_agg_num'
        },
        'ssp585': {
            'rate': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg',
            'count': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg_num'
        }
    },
    "yll": {
        "measure_id": 4,
        "name": "yll",
        "rate_name": "YLL rate",
        "count_name": "YLLs",
        "short": "yll",
        'fhs_name': 'yll',
        'ssp126': {
            'rate' : '20250709_rcp26_first_sub_climate_vector_borne_diseases_agg',
            'count': '20250709_rcp26_first_sub_climate_vector_borne_diseases_agg_num'
        },
        'ssp245': {
            'rate' : '20250709_rcp45_first_sub_climate_ref_agg',
            'count': '20250709_rcp45_first_sub_climate_ref_agg_num'
        },
        'ssp585': {
            'rate' : '20250709_rcp85_first_sub_climate_vector_borne_diseases_agg',
            'count': '20250709_rcp85_first_sub_climate_vector_borne_diseases_agg_num'
        }
    },
}

metric_map = {
    "rate": {
        "name": "rate",
        "metric_id": 3
    },
    "count": {
        "name": "count",
        "metric_id": 1
    },
}

age_type_map = {
    "all_age": {
        "name": "All Age",
        "age_type": "aa"
    },
    "age_specific": {
        "name": "Age-specific",
        "age_type": "as"
    }
}

ssp_scenario_map = {
    "ssp126": {
        "name": "RCP2.6",
        "rcp_scenario": "rcp26",
        "color": "#046C9A",
        "dhs_scenario": 66,
        "dhs_vbd_scenario": 75
    },
    "ssp245": {
        "name": "RCP4.5",
        "rcp_scenario": "rcp45",
        "color": "#E58601",
        "dhs_scenario": 0,
        "dhs_vbd_scenario": 0
    },
    "ssp585": {
        "name": "RCP8.5",
        "rcp_scenario": "rcp85",
        "color": "#A42820",
        "dhs_scenario": 54,
        "dhs_vbd_scenario": 76
    }
}

ssp_scenarios = {
    "ssp126": {
        "name": "RCP2.6",
        "rcp_scenario": "rcp26",
        "color": "#046C9A",
        "dhs_scenario": 66
    },
    "ssp245": {
        "name": "RCP4.5",
        "rcp_scenario": "rcp45",
        "color": "#E58601",
        "dhs_scenario": 0
    },
    "ssp585": {
        "name": "RCP8.5",
        "rcp_scenario": "rcp85",
        "color": "#A42820",
        "dhs_scenario": 54
    }
}

dah_scenarios = {
    "Baseline": {
        "name": "Baseline",
        "color": "#000000"
    },
    "Constant": {
        "name": "Constant",
        "color": "#5DADE2"
    },
    "Increasing": {
        "name": "Increasing",
        "color": "#27AE60"
    },
    "Decreasing": {
        "name": "Decreasing",
        "color": "#8E44AD"
    }
}


problematic_rule_map = {
    'malaria': {
        'incidence': {
            'count_raking_factor_max': 100,        	# Flag if raking factor
            'rate_max': {
                4: 1,
                5: 1
            },
            'count_raking_factor_conditional': 10, 	# Combined with rate condition below
            'rate_max_conditional': 0.2          	# Flag if raking factor > 10 AND rate > 0.2
        },
        'mortality': {
            'count_raking_factor_max': 10000000,        		# Flag if raking factor > 100
            'rate_max': {
                4: 1,
                5: 1
            },					    	# Flag if the rate > 1
            'count_raking_factor_conditional': 10000000,	# This combined with 1 means this is turned off
            'rate_max_conditional': 1         	    	# Flag if raking factor > 10 AND rate > 0.2
        }
    },
    'dengue': {
        'incidence': {
            'count_raking_factor_max': 100000,        	# Flag if raking factor > 100
            'rate_max': {
                4: 1/3,
                5: 1/3
            },
            'count_raking_factor_conditional': 100000, # This combined with 0 means this is turned off
            'rate_max_conditional': 1         	    # Flag if raking factor > 10 AND rate > 0.2
        },
        'mortality': {
            'count_raking_factor_max': 100000,        	# Flag if raking factor > 100
            'rate_max': {
                4: 0.0003,
                5: 0.0003
            },
            'count_raking_factor_conditional': 100000, # This combined with 0 means this is turned off
            'rate_max_conditional': .1        	        # Flag if raking factor > 10 AND rate > 0.2
        }
    }
}


covariate_map = {
    'suitability' : {
        'var': 'suitability',
        'ylabel': 'Suitability (days)',
        'title' : 'Temperature Suitability'
    },
    'gdppc_mean': {
        'var': 'gdppc_mean',
        'ylabel': 'GDP per Capita',
        'title' : 'GDP per Capita (in 2020 USD)'
    },
    'dah_pc': {
        'var': 'dah_pc',
        'ylabel': 'DAH per Capita',
        'title' : 'DAH per Capita (in 2020 USD)'
    },
    'flooding_pc': {
        'var': 'flooding_pc',
        'ylabel': 'Flood days per capita',
        'title' : 'flood days per capita'
    },
    'urbanization': {
        'var': 'urbanization',
        'ylabel': 'Urbanization (%)',
        'title' : 'Urbanization'
    }
}
