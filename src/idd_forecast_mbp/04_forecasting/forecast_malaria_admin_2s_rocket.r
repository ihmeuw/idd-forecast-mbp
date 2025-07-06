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
task_id <- ifelse(is.na(as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))), 1, 
                  as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID')))
message(glue("Task ID: {task_id}"))
param_map <- fread(param_map_filepath)

# task_id = 10
draw_num <- param_map[task_id, draw_num]
ssp_scenario <- param_map[task_id, ssp_scenario]
dah_scenario_name <- param_map[task_id, dah_scenario_name]
draw <- sprintf("%03d", draw_num)
# 
# ssp_scenario <- "ssp245"
# draw = "073"
# dah_scenario_name = "Increasing"

rake_by_location <- function(df, variable){
  raw_pred_var <- glue("{variable}_pred_raw")
  pred_var <- glue("{variable}_pred")
  #as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
  shift_df <- df[which(df$year_id == df$year_to_rake_to), c("location_id", variable, raw_pred_var)]
  shift_df$shift <- shift_df[[variable]] - shift_df[[raw_pred_var]]
  df <- merge(df, shift_df[,c("location_id", "shift")], by = "location_id")
  df[[pred_var]] <- df[[raw_pred_var]] + df$shift
  df <- subset(df, select = -shift)
  return(df)
}
###########################################
REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"

last_year <- 2022

data_path <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")

# load(glue("{data_path}/2025_06_29_malaria_models.RData"))
load(glue("{data_path}/2025_07_03_malaria_models.RData"))
# load(file = glue("{data_path}/final_malaria_regression_models.RData"))

message(glue("DAH sceanrio: {dah_scenario_name}, SSP scenario: {ssp_scenario}, Draw: {draw}"))
input_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
output_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}_with_predictions.parquet")


forecast_df <- as.data.frame(arrow::read_parquet(input_forecast_df_path))
message(exp(min(forecast_df$log_aa_malaria_mort_rate, na.rm = TRUE)))
forecast_df$A0_af <- as.factor(forecast_df$A0_af)

forecast_df$malaria_suit_fraction <- forecast_df$malaria_suitability / 365
forecast_df$malaria_suit_fraction <- pmin(pmax(forecast_df$malaria_suit_fraction, 0.001), 0.999)
forecast_df$logit_malaria_suitability <- log(forecast_df$malaria_suit_fraction / (1 - forecast_df$malaria_suit_fraction))

forecast_df$logit_malaria_pfpr_pred_raw <- predict(malaria_pfpr_mod, forecast_df)
forecast_df <- rake_by_location(forecast_df, "logit_malaria_pfpr")

forecast_df$logit_malaria_pfpr_obs <- forecast_df$logit_malaria_pfpr
forecast_df$logit_malaria_pfpr <- forecast_df$logit_malaria_pfpr_pred

forecast_df$log_aa_malaria_mort_rate_pred_raw <- predict(mortality_scam_mod, forecast_df)
forecast_df <- rake_by_location(forecast_df, "log_aa_malaria_mort_rate")

forecast_df$log_aa_malaria_inc_rate_pred_raw <- predict(incidence_scam_mod, forecast_df)
forecast_df <- rake_by_location(forecast_df, "log_aa_malaria_inc_rate")  


forecast_df$log_base_malaria_mort_rate_pred_raw <- predict(mortality_base_scam_mod, forecast_df)
forecast_df <- rake_by_location(forecast_df, "log_base_malaria_mort_rate")

forecast_df$log_base_malaria_inc_rate_pred_raw <- predict(incidence_base_scam_mod, forecast_df)
forecast_df <- rake_by_location(forecast_df, "log_base_malaria_inc_rate")  

#### 
columns_to_kepp <- names(forecast_df)[which(names(forecast_df) %like% "malaria" & names(forecast_df) %nlike% "suit" & names(forecast_df) %nlike% "pfpr")]
columns_to_keep <- c(columns_to_kepp, "location_id", "year_id", "population", "aa_population", "year_to_rake_to")

forecast_df <- forecast_df[, columns_to_keep]
print(glue("Saving draw: {draw}, SSP scenario: {ssp_scenario}, DAH scenario: {dah_scenario_name}"))
# print(glue("    - {forecast_df[which(forecast_df$location_id == 25364 & forecast_df$year_id == 2091),c('log_malaria_pf_mort_rate_pred', 'log_malaria_pf_inc_rate_pred')]}"))




####
write_parquet(forecast_df, output_forecast_df_path)
print("fin")