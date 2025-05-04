import yaml
from pathlib import Path
from idd_forecast_mbp import constants as rfc

# Is it better to nest these functions, have the material repeated or read both in every time. Most of the time we only need the second one
def load_yaml_dictionary(yaml_path: str) -> dict:
    # Read YAML
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return(yaml_data['COVARIATE_DICT'])

def parse_yaml_dictionary(covariate: str) -> dict:
    YAML_PATH = rfc.REPO_ROOT / rfc.repo_name / "src" / rfc.package_name /  "COVARIATE_DICT.yaml"
    # Extract covariate-specific config
    covariate_dict = load_yaml_dictionary(YAML_PATH)
    # Check if the covariate exists in the dictionary
    if covariate not in covariate_dict:
        raise ValueError(f"Covariate '{covariate}' not found in the dictionary.")
    # Extract the covariate entry
    covariate_entry = covariate_dict.get(covariate, [])

    covariate_resolution = covariate_entry['covariate_resolution_numerator'] / covariate_entry['covariate_resolution_denominator']

    years = list(range(covariate_entry['year_start'], covariate_entry['year_end'] + 1))

    # Build the return dict dynamically
    result = {
        "covariate": covariate_entry['covariate'],
        "covariate_resolution": covariate_resolution,
        "years": years,
        "synoptic": covariate_entry['synoptic'],
        "cc_sensitive": covariate_entry['cc_sensitive'],
        "summary_statistic": covariate_entry['summary_statistic']
    }

    return result