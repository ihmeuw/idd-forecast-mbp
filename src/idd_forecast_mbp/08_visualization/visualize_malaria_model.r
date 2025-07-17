rm(list = ls())
#

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)

"%ni%" <- Negate("%in%")
"%nlike%" <- Negate("%like%")


ssp_scenario <- 'ssp245'
dah_scenario_name <- 'Baseline'
draw <- '000'

###########################################
REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"

last_year <- 2022

data_path <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")

load(glue("{data_path}/2025_07_03_malaria_models.RData"))


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