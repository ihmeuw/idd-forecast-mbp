"""
Readers for the malaria vaccine pipeline's inputs.

Kept separate from `lib/processing/vaccine_coverage.py` so the transforms stay
pure and unit-testable without touching shared storage, matching how this repo
splits `lib/data` (readers) from `lib/processing` (transforms).
"""
from pathlib import Path

import pandas as pd

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.processing.vaccine_cohort_fractions import VACCINE_RELEVANT_MAX_AGE
from idd_forecast_mbp.lib.processing.vaccine_coverage import validate_coverage_frame
from idd_forecast_mbp.lib.processing.vaccine_efficacy import VECurve


def load_age_groups(max_age: float = VACCINE_RELEVANT_MAX_AGE) -> pd.DataFrame:
    """Vaccine-relevant, most-detailed age groups with their real bounds.

    Bounds come from the pipeline's own age metadata rather than a hardcoded
    table, so they cannot drift from the population source.
    """
    path = rfc.AGE_SPECIFIC_FHS_PATH / "age_metadata.parquet"
    am = read_parquet_with_integer_ids(
        path,
        columns=["age_group_id", "age_group_name", "age_group_years_start",
                 "age_group_years_end", "most_detailed"],
    )
    am = am[(am["most_detailed"] == 1) & (am["age_group_years_start"] < max_age)]
    am = am.sort_values("age_group_years_start").reset_index(drop=True)
    if am.empty:
        raise ValueError(f"no most-detailed age groups below age {max_age} in {path}")
    return am


def load_coverage(coverage_csv: Path, ve: VECurve) -> pd.DataFrame:
    """Read the coverage table, validate it against the contract, coerce ID dtypes."""
    cov = pd.read_csv(coverage_csv)
    validate_coverage_frame(cov, ve, source=str(coverage_csv))
    cov["subnat_id"] = cov["subnat_id"].astype("int64")
    cov["year_id"] = cov["year_id"].astype("int64")
    return cov


def load_population(population_file: Path, location_ids: list[int],
                    age_group_ids: list[int], years: list[int]) -> pd.DataFrame:
    """Filtered read of the real age-sex population.

    Predicate pushdown keeps this to the requested slice -- the full file is
    ~2.9 GB / 258M rows and must never be loaded whole.
    """
    return read_parquet_with_integer_ids(
        population_file,
        columns=["location_id", "year_id", "age_group_id", "sex_id", "population"],
        filters=[
            ("location_id", "in", location_ids),
            ("age_group_id", "in", age_group_ids),
            ("year_id", "in", years),
        ],
    )
