rm(list = ls())
#

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)

"%ni%" <- Negate("%in%")
"%nlike%" <- Negate("%like%")

###########################################
args <- commandArgs(trailingOnly = TRUE)
param_map_filepath <- "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/malaria_param_map.csv"

## Retrieving array task_id
task_id <- as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
message(glue("Task ID: {task_id}"))
param_map <- fread(param_map_filepath)

draw_num <- param_map[task_id, draw_num]
ssp_scenario <- param_map[task_id, ssp_scenario]
dah_scenario_name <- param_map[task_id, dah_scenario_name]
draw <- sprintf("%03d", draw_num)
# 
# ssp_scenario <- "ssp245"
# draw = "073"
# dah_scenario = "Increasing"


###########################################

last_year <- 2022

data_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"
FORECASTING_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data"

load(file = glue("{data_path}/necessary_malaria_regression_models.RData"))

message(glue("DAH sceanrio: {dah_scenario_name}, SSP scenario: {ssp_scenario}, Draw: {draw}"))

input_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
output_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}_with_predictions.parquet")

forecast_df <- as.data.frame(arrow::read_parquet(input_forecast_df_path))
forecast_df$A0_af <- as.factor(forecast_df$A0_af)
forecast_df <- forecast_df[-which(is.na(forecast_df$people_flood_days)),]

forecast_df$malaria_pfpr[is.na(forecast_df$malaria_pfpr)] <- 0
forecast_df$logit_malaria_pfpr[is.na(forecast_df$logit_malaria_pfpr)] <- log(1e-10 / (1 - 1e-10))
# min_gdppc <- min(forecast_df$gdppc_mean, na.rm = TRUE)
# forecast_df$gdppc_mean[is.na(forecast_df$gdppc_mean)] <- min_gdppc
# min_log_gdppc <- min(forecast_df$log_gdppc_mean, na.rm = TRUE)
# forecast_df$log_gdppc_mean[is.na(forecast_df$log_gdppc_mean)] <- min_log_gdppc
#### 

forecast_df$logit_malaria_pfpr_pred_raw <- predict(malaria_pfpr_mod_1, forecast_df)
shift_df <- forecast_df[which(forecast_df$year_id == last_year), c("location_id", "logit_malaria_pfpr", "logit_malaria_pfpr_pred_raw")]
shift_df$shift <- shift_df$logit_malaria_pfpr - shift_df$logit_malaria_pfpr_pred_raw
forecast_df <- merge(forecast_df, shift_df[,c("location_id", "shift")], on = "location_id")
forecast_df$logit_malaria_pfpr_pred <- forecast_df$logit_malaria_pfpr_pred_raw + forecast_df$shift


forecast_df <- subset(forecast_df, select = -shift)

####
# Model 1

forecast_df$logit_malaria_pfpr_obs <- forecast_df$logit_malaria_pfpr
forecast_df$logit_malaria_pfpr <- forecast_df$logit_malaria_pfpr_pred

#### 

forecast_df$log_malaria_mort_rate_pred_raw <- predict(mortality_scam_mod, forecast_df)
forecast_df$log_malaria_inc_rate_pred_raw <- predict(incidence_scam_mod, forecast_df)



forecast_df <- subset(forecast_df, select = -logit_malaria_pfpr)


#### 


print(glue("Draw: {draw}, SSP scenario: {ssp_scenario}, DAH scenario: {dah_scenario_name}"))
# print(glue("    - {forecast_df[which(forecast_df$location_id == 25364 & forecast_df$year_id == 2091),c('log_malaria_pf_mort_rate_pred', 'log_malaria_pf_inc_rate_pred')]}"))

####
write_parquet(forecast_df, output_forecast_df_path)
