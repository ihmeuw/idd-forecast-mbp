import xarray as xr
from pathlib import Path
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.netcdf import write_netcdf

TEST_DIR = Path("/mnt/team/idd/pub/forecast-mbp/test_output/05-upload_data")

cause = "malaria"
ssp_scenario = "ssp126"
dah_scenario = "Baseline"
draw = "001"
run_date = "2025_08_28"

UPLOAD_DATA_PATH = mbpc.MODEL_ROOT / "05-upload_data"

yld_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yld_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
yll_path = f"{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_yll_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
output_path = TEST_DIR / f"full_as_{cause}_measure_daly_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"

yld_ds = xr.open_dataset(yld_path)
yll_ds = xr.open_dataset(yll_path)
yld_aligned, yll_aligned = xr.align(yld_ds['count_pred'], yll_ds['count_pred'], join='outer', fill_value=0)
daly_ds = (yld_aligned + yll_aligned).to_dataset(name='count_pred')
del yld_ds, yll_ds, yld_aligned, yll_aligned

write_netcdf(
    daly_ds,
    output_path,
    compression=True,
    compression_level=4,
    chunking=True,
    max_retries=3
)
print(f"Wrote {output_path}")
