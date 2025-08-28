
rm(list = ls())
#

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)
library(ncdf4)

"%ni%" <- Negate("%in%")
"%nlike%" <- Negate("%like%")

###########################################
args <- commandArgs(trailingOnly = TRUE)
param_map_filepath <- "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/dengue_param_map.csv"

## Retrieving array task_id
task_id <- as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
task_id <- ifelse(is.na(task_id), 1, task_id)  # Default to 1 if not set
message(glue("Task ID: {task_id}"))
param_map <- fread(param_map_filepath)

draw_num <- param_map[task_id, draw_num]
ssp_scenario <- param_map[task_id, ssp_scenario]
counterfactual <- FALSE#param_map[task_id, counterfactual]
draw <- sprintf("%03d", draw_num)

covariates_to_hold <- c('gdppc', 'urban','suitability')
# 
# ssp_scenario <- "ssp245"
# draw = "066"


###########################################
REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"


last_year <- 2022

MODELING_DATA_PATH <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")


load(glue("{MODELING_DATA_PATH}/2025_06_29_dengue_models.RData"))

message(glue("SSP scenario: {ssp_scenario}, Draw: {draw}"))

input_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}.parquet")
# output_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet")
output_ncdf_path <- glue("{FORECASTING_DATA_PATH}/dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc")

vars_to_write <- c(
  'base_log_dengue_inc_rate_pred_raw', 'base_log_dengue_inc_rate',
  'logit_dengue_cfr_pred_raw',
  'log_gdppc_mean', 'logit_urban_1km_threshold_300','logit_dengue_suitability'
)





# as_md_dengue_modeling_df_path = glue("{MODELING_DATA_PATH}/as_md_dengue_modeling_df.parquet")
# as_dengue_df = as.data.frame(arrow::read_parquet(as_md_dengue_modeling_df_path))
# as_dengue_df$A0_af <- as.factor(as_dengue_df$A0_af)
# as_dengue_df$as_id = as.factor(as_dengue_df$as_id)

# base_md_dengue_modeling_df_path = glue("{MODELING_DATA_PATH}/base_md_dengue_modeling_df.parquet")
# base_df <- as.data.frame(arrow::read_parquet(base_md_dengue_modeling_df_path))

forecast_df <- as.data.frame(arrow::read_parquet(input_forecast_df_path))

# forecast_df = forecast_df[which(!is.na(forecast_df$log_gdppc_mean)),]

# forecast_df = forecast_df[which(forecast_df$location_id %in% unique(base_df$location_id)),]

forecast_df$dengue_suit_fraction <- forecast_df$dengue_suitability / 365
forecast_df$dengue_suit_fraction <- pmin(pmax(forecast_df$dengue_suit_fraction, 0.001), 0.999)
forecast_df$logit_dengue_suitability <- log(forecast_df$dengue_suit_fraction / (1 - forecast_df$dengue_suit_fraction))

# forecast_df$A0_af <- as.factor(forecast_df$A0_af)
# 

factor_col <- "A0_af"  # Replace with actual column name

## Fix for mod_inc_base
original_levels <- levels(mod_inc_base$model[[factor_col]])
model_coefs <- coef(mod_inc_base)
factor_coef_pattern <- paste0("^", factor_col)
factor_coefs <- model_coefs[grepl(factor_coef_pattern, names(model_coefs))]
lowest_effect_name <- names(factor_coefs)[which.min(factor_coefs)] 
lowest_level <- gsub(paste0("^", factor_col), "", lowest_effect_name)

current_levels <- unique(forecast_df[[factor_col]])
missing_levels <- setdiff(current_levels, original_levels)
locs_to_replace <- which(forecast_df[[factor_col]] %in% missing_levels)
forecast_df$A0_af[locs_to_replace] <- lowest_level



forecast_df$base_log_dengue_inc_rate_pred_raw <- predict(mod_inc_base, forecast_df)
print("Predicted base incidence")

## Fix for mod_inc_base
original_levels <- levels(mod_cfr_all$model[[factor_col]])
model_coefs <- coef(mod_cfr_all)
factor_coef_pattern <- paste0("^", factor_col)
factor_coefs <- model_coefs[grepl(factor_coef_pattern, names(model_coefs))]
lowest_effect_name <- names(factor_coefs)[which.min(factor_coefs)] 
lowest_level <- gsub(paste0("^", factor_col), "", lowest_effect_name)

forecast_df$A0_af[locs_to_replace] <- lowest_level


forecast_df$logit_dengue_cfr_pred_raw <- predict(mod_cfr_all, forecast_df)
print("Finished with predictions")

# shift_df <- forecast_df[which(forecast_df$year_id == last_year), c("location_id", "age_group_id", "sex_id", "base_log_dengue_inc_rate", "base_log_dengue_inc_rate_pred_raw", "logit_dengue_cfr", "logit_dengue_cfr_pred_raw")]
# shift_df$shift_inc <- shift_df$base_log_dengue_inc_rate - shift_df$base_log_dengue_inc_rate_pred_raw
# shift_df$shift_cfr <- shift_df$logit_dengue_cfr - shift_df$logit_dengue_cfr_pred_raw
# forecast_df <- merge(forecast_df, shift_df[,c("location_id", "age_group_id", "sex_id", "shift_inc", "shift_cfr")], by = c("location_id", "age_group_id", "sex_id"))
# 
# forecast_df$base_log_dengue_inc_rate_pred <- forecast_df$base_log_dengue_inc_rate_pred_raw + forecast_df$shift_inc
# forecast_df$logit_dengue_cfr_pred <- forecast_df$logit_dengue_cfr_pred_raw + forecast_df$shift_cfr


# forecast_df <- subset(forecast_df, select = -c(shift_inc, shift_cfr))
# 

#### 
print(glue("Draw: {draw}, SSP scenario: {ssp_scenario}"))
# print(glue("    - {forecast_df[which(forecast_df$location_id == 25364 & forecast_df$year_id == 2091),c('log_malaria_pf_mort_rate_pred', 'log_malaria_pf_inc_rate_pred')]}"))

####
# write_parquet(forecast_df, output_forecast_df_path)
print("fin")



# Get unique dimensions (keep your existing code)
locations <- sort(unique(forecast_df$location_id))
years <- sort(unique(forecast_df$year_id))
dim_location <- ncdim_def("location_id", units="id", vals=locations)
dim_year <- ncdim_def("year_id", units="year", vals=years)

# Sort data once
sorted_df <- forecast_df[order(forecast_df$location_id, forecast_df$year_id), ]

# Create all matrices in one go
all_matrices <- lapply(vars_to_write, function(varname) {
  matrix(sorted_df[[varname]], nrow = length(locations), ncol = length(years), byrow = TRUE)
})
names(all_matrices) <- vars_to_write

# Create NetCDF (your existing code structure)
var_defs <- lapply(vars_to_write, function(v) {
  ncvar_def(v, units="unknown", dim=list(dim_location, dim_year), missval=NA, prec="double")
})

nc <- nc_create(output_ncdf_path, var_defs)

# Write all variables (much faster)
for (v in vars_to_write) {
  ncvar_put(nc, v, all_matrices[[v]])
}

nc_close(nc)
Sys.chmod(output_ncdf_path, mode = "0775")






if (counterfactual){
  for (c_num in seq_along(covariates_to_hold)){
    output_ncdf_path <- glue("{FORECASTING_DATA_PATH}/dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions_hold_{covariates_to_hold[c_num]}.nc")
    forecast_dt <- as.data.table(arrow::read_parquet(input_forecast_df_path))
    
    cols_to_hold <- grep(covariates_to_hold[c_num], names(forecast_dt), value = TRUE)
    for (col_name in cols_to_hold) {
      # Get 2022 values for each location_id
      vals_2022 <- forecast_dt[year_id == 2022, .(location_id, val_2022 = get(col_name))]
      # Join 2022 values to all years >= 2023
      forecast_dt[year_id >= 2023 & year_id <= 2100, (col_name) := vals_2022[.SD, on = "location_id", x.val_2022]]
    }
    
    forecast_df <- as.data.frame(forecast_dt)
    # forecast_df = forecast_df[which(!is.na(forecast_df$log_gdppc_mean)),]
    
    # forecast_df = forecast_df[which(forecast_df$location_id %in% unique(base_df$location_id)),]
    
    forecast_df$dengue_suit_fraction <- forecast_df$dengue_suitability / 365
    forecast_df$dengue_suit_fraction <- pmin(pmax(forecast_df$dengue_suit_fraction, 0.001), 0.999)
    forecast_df$logit_dengue_suitability <- log(forecast_df$dengue_suit_fraction / (1 - forecast_df$dengue_suit_fraction))
    
    # forecast_df$A0_af <- as.factor(forecast_df$A0_af)
    # 
    
    factor_col <- "A0_af"  # Replace with actual column name
    
    ## Fix for mod_inc_base
    original_levels <- levels(mod_inc_base$model[[factor_col]])
    model_coefs <- coef(mod_inc_base)
    factor_coef_pattern <- paste0("^", factor_col)
    factor_coefs <- model_coefs[grepl(factor_coef_pattern, names(model_coefs))]
    lowest_effect_name <- names(factor_coefs)[which.min(factor_coefs)] 
    lowest_level <- gsub(paste0("^", factor_col), "", lowest_effect_name)
    
    current_levels <- unique(forecast_df[[factor_col]])
    missing_levels <- setdiff(current_levels, original_levels)
    locs_to_replace <- which(forecast_df[[factor_col]] %in% missing_levels)
    forecast_df$A0_af[locs_to_replace] <- lowest_level
    
    
    
    forecast_df$base_log_dengue_inc_rate_pred_raw <- predict(mod_inc_base, forecast_df)
    print("Predicted base incidence")
    
    ## Fix for mod_inc_base
    original_levels <- levels(mod_cfr_all$model[[factor_col]])
    model_coefs <- coef(mod_cfr_all)
    factor_coef_pattern <- paste0("^", factor_col)
    factor_coefs <- model_coefs[grepl(factor_coef_pattern, names(model_coefs))]
    lowest_effect_name <- names(factor_coefs)[which.min(factor_coefs)] 
    lowest_level <- gsub(paste0("^", factor_col), "", lowest_effect_name)
    
    forecast_df$A0_af[locs_to_replace] <- lowest_level
    
    
    forecast_df$logit_dengue_cfr_pred_raw <- predict(mod_cfr_all, forecast_df)
    print("Finished with predictions")
    
    # shift_df <- forecast_df[which(forecast_df$year_id == last_year), c("location_id", "age_group_id", "sex_id", "base_log_dengue_inc_rate", "base_log_dengue_inc_rate_pred_raw", "logit_dengue_cfr", "logit_dengue_cfr_pred_raw")]
    # shift_df$shift_inc <- shift_df$base_log_dengue_inc_rate - shift_df$base_log_dengue_inc_rate_pred_raw
    # shift_df$shift_cfr <- shift_df$logit_dengue_cfr - shift_df$logit_dengue_cfr_pred_raw
    # forecast_df <- merge(forecast_df, shift_df[,c("location_id", "age_group_id", "sex_id", "shift_inc", "shift_cfr")], by = c("location_id", "age_group_id", "sex_id"))
    # 
    # forecast_df$base_log_dengue_inc_rate_pred <- forecast_df$base_log_dengue_inc_rate_pred_raw + forecast_df$shift_inc
    # forecast_df$logit_dengue_cfr_pred <- forecast_df$logit_dengue_cfr_pred_raw + forecast_df$shift_cfr
    
    
    # forecast_df <- subset(forecast_df, select = -c(shift_inc, shift_cfr))
    # 
    
    #### 
    print(glue("Draw: {draw}, SSP scenario: {ssp_scenario}"))
    # print(glue("    - {forecast_df[which(forecast_df$location_id == 25364 & forecast_df$year_id == 2091),c('log_malaria_pf_mort_rate_pred', 'log_malaria_pf_inc_rate_pred')]}"))
    
    ####
    # write_parquet(forecast_df, output_forecast_df_path)
    print("fin")
    
    
    
    # Get unique dimensions (keep your existing code)
    locations <- sort(unique(forecast_df$location_id))
    years <- sort(unique(forecast_df$year_id))
    dim_location <- ncdim_def("location_id", units="id", vals=locations)
    dim_year <- ncdim_def("year_id", units="year", vals=years)
    
    # Sort data once
    sorted_df <- forecast_df[order(forecast_df$location_id, forecast_df$year_id), ]
    
    # Create all matrices in one go
    all_matrices <- lapply(vars_to_write, function(varname) {
      matrix(sorted_df[[varname]], nrow = length(locations), ncol = length(years), byrow = TRUE)
    })
    names(all_matrices) <- vars_to_write
    
    # Create NetCDF (your existing code structure)
    var_defs <- lapply(vars_to_write, function(v) {
      ncvar_def(v, units="unknown", dim=list(dim_location, dim_year), missval=NA, prec="double")
    })
    
    nc <- nc_create(output_ncdf_path, var_defs)
    
    # Write all variables (much faster)
    for (v in vars_to_write) {
      ncvar_put(nc, v, all_matrices[[v]])
    }
    
    nc_close(nc)
    Sys.chmod(output_ncdf_path, mode = "0775")
    
  }
}









print("fin")