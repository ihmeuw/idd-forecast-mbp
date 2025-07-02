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

draw_num <- param_map[task_id, draw_num]
ssp_scenario <- param_map[task_id, ssp_scenario]
dah_scenario_name <- param_map[task_id, dah_scenario_name]
draw <- sprintf("%03d", draw_num)



###########################################
REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"

last_year <- 2022

data_path <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")

# load(file = glue("{data_path}/final_malaria_regression_models.RData"))

message(glue("DAH sceanrio: {dah_scenario_name}, SSP scenario: {ssp_scenario}, Draw: {draw}"))

input_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
output_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}_with_options_with_predictions.parquet")

forecast_df <- as.data.frame(arrow::read_parquet(input_forecast_df_path))
forecast_df$A0_af <- as.factor(forecast_df$A0_af)

my_models_env <- new.env()
load(glue("{data_path}/my_models_new_2.RData"))#, envir = my_models_env)

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


n_mod <- length(which(ls() %like% 'malaria_pfpr_mod_'))
for (mod_num in 1:n_mod){
  message(glue("Running model {mod_num} for draw {draw} with SSP scenario {ssp_scenario} and DAH scenario {dah_scenario_name}"))
  pfpr_mod <- get(glue("malaria_pfpr_mod_{mod_num}"))
  #
  tmp_df <- forecast_df
  
  tmp_df$malaria_suit_fraction <- tmp_df$malaria_suitability / 365
  tmp_df$malaria_suit_fraction <- pmin(pmax(tmp_df$malaria_suit_fraction, 0.001), 0.999)
  tmp_df$logit_malaria_suitability <- log(tmp_df$malaria_suit_fraction / (1 - tmp_df$malaria_suit_fraction))

  tmp_df$logit_malaria_pfpr_pred_raw <- predict(pfpr_mod, tmp_df)
  tmp_df <- rake_by_location(tmp_df, "logit_malaria_pfpr")
  
  tmp_df$logit_malaria_pfpr_obs <- tmp_df$logit_malaria_pfpr
  tmp_df$logit_malaria_pfpr <- tmp_df$logit_malaria_pfpr_pred
  
  tmp_df$log_aa_malaria_mort_rate_pred_raw <- predict(mortality_scam_mod, tmp_df)
  tmp_df <- rake_by_location(tmp_df, "log_aa_malaria_mort_rate")
  
  tmp_df$log_aa_malaria_inc_rate_pred_raw <- predict(incidence_scam_mod, tmp_df)
  tmp_df <- rake_by_location(tmp_df, "log_aa_malaria_inc_rate")  
  
  forecast_df[[glue("logit_malaria_pfpr_pred_{mod_num}")]] <- tmp_df$logit_malaria_pfpr_pred
  forecast_df[[glue("log_aa_malaria_mort_rate_pred_{mod_num}")]] <- tmp_df$log_aa_malaria_mort_rate_pred
  forecast_df[[glue("log_aa_malaria_inc_rate_pred_{mod_num}")]] <- tmp_df$log_aa_malaria_inc_rate_pred
}

write_parquet(forecast_df, output_forecast_df_path)
print("fin")

















