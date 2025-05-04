from pathlib import Path

MODEL_ROOT = Path("/mnt/team/idd/pub/forecast-mbp")

REPO_ROOT = Path("/mnt/share/homes/bcreiner/repos")

repo_name = "idd-forecast-mbp"
package_name = "idd_forecast_mbp"


# Constants
years = list(range(1970, 2101))
past_years = list(range(1970, 2024))
future_years = list(range(2024, 2101))

# GBD Constants
release_2021_id = 9
release_2023_id = 16
como_2023_v = 1591
codcorrec_2023t_v = 461
dalynator_2023_v = 96
burdenator_2023_v = 360
compare_2023_v = 8234
ages = 22
sexes = 3
dengue_id = 357
malaria_ids = 345



# LSAE Constanst