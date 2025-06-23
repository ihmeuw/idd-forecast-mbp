rm(list = ls())

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)


"%ni%" <- Negate("%in%")
"%nlike%" <- Negate("%like%")

last_year <- 2022

data_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"
FORECASTING_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data"

load(file = glue("{data_path}/malaria_regression_models.RData"))

# draw <- "000"
# ssp_scenario <- "ssp245"
# dah_scenario_name <- "Baseline"

draws <- sprintf("%03d", 0:99)
ssp_scenarios <- c("ssp126", "ssp245", "ssp585")
dah_scenario_names <- c("Baseline", "Constant", "Decreasing", "Increasing")

for (dah_scenario_name in dah_scenario_names){
  for (draw in draws){
    for(ssp_scenario in ssp_scenarios){
      message(glue("DAH sceanrio: {dah_scenario_name}, SSP scenario: {ssp_scenario}, Draw: {draw}"))
      
      input_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
      output_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}_with_predictions.parquet")
      
      forecast_df <- arrow::read_parquet(input_forecast_df_path)
      forecast_df$A0_af <- as.factor(forecast_df$A0_af)
      
      #### 
      
      forecast_df$logit_malaria_pfpr_pred_raw <- predict(malaria_pfpr_mod, forecast_df)
      shift_df <- forecast_df[which(forecast_df$year_id == last_year), c("location_id", "logit_malaria_pfpr", "logit_malaria_pfpr_pred_raw")]
      shift_df$shift <- shift_df$logit_malaria_pfpr - shift_df$logit_malaria_pfpr_pred_raw
      forecast_df <- merge(forecast_df, shift_df[,c("location_id", "shift")], on = "location_id")
      forecast_df$logit_malaria_pfpr_pred <- forecast_df$logit_malaria_pfpr_pred_raw + forecast_df$shift
      forecast_df <- subset(forecast_df, select = -shift)
      
      ####
      
      forecast_df$logit_malaria_pfpr_obs <- forecast_df$logit_malaria_pfpr
      forecast_df$logit_malaria_pfpr <- forecast_df$logit_malaria_pfpr_pred
      
      #### 
      
      forecast_df$log_malaria_pf_mort_rate_pred_raw <- predict(malaria_mort_mod, forecast_df)
      shift_df <- forecast_df[which(forecast_df$year_id == last_year), c("location_id", "log_malaria_pf_mort_rate", "log_malaria_pf_mort_rate_pred_raw")]
      shift_df$shift <- shift_df$log_malaria_pf_mort_rate - shift_df$log_malaria_pf_mort_rate_pred_raw
      forecast_df <- merge(forecast_df, shift_df[,c("location_id", "shift")], on = "location_id")
      forecast_df$log_malaria_pf_mort_rate_pred <- forecast_df$log_malaria_pf_mort_rate_pred_raw + forecast_df$shift
      forecast_df <- subset(forecast_df, select = -shift)
      
      #### 
      
      forecast_df$log_malaria_pf_inc_rate_pred_raw <- predict(malaria_inc_mod, forecast_df)
      shift_df <- forecast_df[which(forecast_df$year_id == last_year), c("location_id", "log_malaria_pf_inc_rate", "log_malaria_pf_inc_rate_pred_raw")]
      shift_df$shift <- shift_df$log_malaria_pf_inc_rate - shift_df$log_malaria_pf_inc_rate_pred_raw
      forecast_df <- merge(forecast_df, shift_df[,c("location_id", "shift")], on = "location_id")
      forecast_df$log_malaria_pf_inc_rate_pred <- forecast_df$log_malaria_pf_inc_rate_pred_raw + forecast_df$shift
      forecast_df <- subset(forecast_df, select = -shift)
      
      print(glue("Draw: {draw}, SSP scenario: {ssp_scenario}, DAH scenario: {dah_scenario_name}"))
      print(glue("    - {forecast_df[which(forecast_df$location_id == 25364 & forecast_df$year_id == 2091),c('log_malaria_pf_mort_rate_pred', 'log_malaria_pf_inc_rate_pred')]}"))
      
      ####
      write_parquet(forecast_df, output_forecast_df_path)
    }
  }
}


tmp_loc_id <- 25355
tmp_df <- forecast_df[which(forecast_df$location_id == tmp_loc_id),]
plot(tmp_df$year_id, tmp_df$log_malaria_pf_mort_rate_pred_raw)
lines(tmp_df$year_id, tmp_df$log_malaria_pf_mort_rate)
lines(tmp_df$year_id, tmp_df$log_malaria_pf_mort_rate_pred, col = 2)












