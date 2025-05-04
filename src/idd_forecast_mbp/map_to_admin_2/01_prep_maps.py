



# Prep maps for pixel main
# Pixel main wants maps that look like this:
# years = list(range(1970, 2101))
# root = Path("/mnt/team/idd/pub/forecast-mbp/01-raw_data") / scenario
# ds_file = root / f"{covariate}.nc"
# ds = ds.rename({"lat": "latitude", "lon": "longitude", "time": "year", "value": "value"})

    # if cc_sensitive:
    #     scenarios = ["ssp126", "ssp245", "ssp585"]
    # else:
    #     scenarios = ["cc_insensitive"]
    
    # climate_slice, bounds_map = build_location_masks(block_key, hiearchy)

    # result_records = []
    # for scenario in scenarios:
    #     root = Path(DATA_PATH) / scenario 
    #     # check if model exists, if not, skip
    #     if not (root / f"{covariate}.nc").exists():
    #         continue

    #     ds_file = root / f"{covariate}.nc"
    #     ds = xr.open_dataset(ds_file)