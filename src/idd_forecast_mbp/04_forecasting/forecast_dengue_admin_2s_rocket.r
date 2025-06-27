
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
param_map_filepath <- "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/dengue_param_map.csv"

## Retrieving array task_id
task_id <- as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
task_id <- ifelse(is.na(task_id), 1, task_id)  # Default to 1 if not set
message(glue("Task ID: {task_id}"))
param_map <- fread(param_map_filepath)

draw_num <- param_map[task_id, draw_num]
ssp_scenario <- param_map[task_id, ssp_scenario]
draw <- sprintf("%03d", draw_num)
# 
# ssp_scenario <- "ssp245"
# draw = "066"


###########################################
REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"


last_year <- 2022

MODELING_DATA_PATH <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")


load(file = glue("{MODELING_DATA_PATH}/final_dengue_regression_models.RData"))

message(glue("SSP scenario: {ssp_scenario}, Draw: {draw}"))

input_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}.parquet")
output_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet")

forecast_df <- as.data.frame(arrow::read_parquet(input_forecast_df_path))
forecast_df$A0_af <- as.factor(forecast_df$A0_af)
forecast_df$as_id <- as.factor(forecast_df$as_id)

forecast_df <- forecast_df[-which(is.na(forecast_df$base_log_dengue_inc_rate)),]


forecast_df$base_log_dengue_inc_rate_pred_raw <- predict(mod_inc_base, forecast_df)

forecast_df$logit_dengue_cfr_pred_raw <- predict(mod_cfr_all, forecast_df)


shift_df <- forecast_df[which(forecast_df$year_id == last_year), c("location_id", "age_group_id", "sex_id", "base_log_dengue_inc_rate", "base_log_dengue_inc_rate_pred_raw", "logit_dengue_cfr", "logit_dengue_cfr_pred_raw")]
shift_df$shift_inc <- shift_df$base_log_dengue_inc_rate - shift_df$base_log_dengue_inc_rate_pred_raw
shift_df$shift_cfr <- shift_df$logit_dengue_cfr - shift_df$logit_dengue_cfr_pred_raw
forecast_df <- merge(forecast_df, shift_df[,c("location_id", "age_group_id", "sex_id", "shift_inc", "shift_cfr")], on = c("location_id", "age_group_id", "sex_id"))

forecast_df$base_log_dengue_inc_rate_pred <- forecast_df$base_log_dengue_inc_rate_pred_raw + forecast_df$shift_inc
forecast_df$logit_dengue_cfr_pred <- forecast_df$logit_dengue_cfr_pred_raw + forecast_df$shift_cfr


forecast_df <- subset(forecast_df, select = -c(shift_inc, shift_cfr))
# 
message(max(forecast_df$logit_dengue_cfr_pred))
#### 
print(glue("Draw: {draw}, SSP scenario: {ssp_scenario}"))
# print(glue("    - {forecast_df[which(forecast_df$location_id == 25364 & forecast_df$year_id == 2091),c('log_malaria_pf_mort_rate_pred', 'log_malaria_pf_inc_rate_pred')]}"))

####
write_parquet(forecast_df, output_forecast_df_path)