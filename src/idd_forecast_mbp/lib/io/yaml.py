"""
YAML config utilities for the idd-forecast-mbp pipeline.

Extracted from: yaml_functions.py
"""

from __future__ import annotations

import yaml

from idd_forecast_mbp import constants as mbpc


def load_yaml_dictionary(yaml_path: str) -> dict:
    """Load the COVARIATE_DICT block from a YAML file.

    # Extracted from: yaml_functions.py:12
    """
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data['COVARIATE_DICT']


def parse_yaml_dictionary(covariate: str) -> dict:
    """Parse covariate configuration from the project COVARIATE_DICT.yaml.

    Looks up covariate by name and returns a standardized config dict with
    resolved years list and computed covariate_resolution.

    Parameters
    ----------
    covariate:
        Key in COVARIATE_DICT.yaml to look up.

    Returns
    -------
    dict with keys: covariate_name, covariate_resolution, years, synoptic,
    cc_sensitive, summary_statistic, path.

    # Extracted from: yaml_functions.py:18
    """
    YAML_PATH = mbpc.REPO_ROOT / mbpc.repo_name / 'src' / mbpc.package_name / 'COVARIATE_DICT.yaml'
    covariate_dict = load_yaml_dictionary(YAML_PATH)
    if covariate not in covariate_dict:
        raise ValueError(f"Covariate '{covariate}' not found in the dictionary.")
    covariate_entry = covariate_dict[covariate]

    covariate_resolution = (
        covariate_entry['covariate_resolution_numerator']
        / covariate_entry['covariate_resolution_denominator']
    )
    years = list(range(covariate_entry['year_start'], covariate_entry['year_end'] + 1))

    return {
        'covariate_name': covariate_entry['covariate_name'],
        'covariate_resolution': covariate_resolution,
        'years': years,
        'synoptic': covariate_entry['synoptic'],
        'cc_sensitive': covariate_entry['cc_sensitive'],
        'summary_statistic': covariate_entry['summary_statistic'],
        'path': covariate_entry['path'],
    }
