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
dah_scenario_name = 'Baseline'
draw = '077'

REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"
last_year <- 2022
data_path <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")


ssp126_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp126_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
ssp126_df <-as.data.frame(arrow::read_parquet(ssp126_df_path))
ssp126_df$A0_af <- as.factor(ssp126_df$A0_af)
ssp245_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp245_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
ssp245_df <-as.data.frame(arrow::read_parquet(ssp245_df_path))
ssp245_df$A0_af <- as.factor(ssp245_df$A0_af)
ssp585_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp585_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
ssp585_df <-as.data.frame(arrow::read_parquet(ssp585_df_path))
ssp585_df$A0_af <- as.factor(ssp585_df$A0_af)

past_data <- ssp585_df[-which(is.na(ssp585_df$malaria_pfpr)),]
past_data <- past_data[-which(is.na(past_data$gdppc_mean)),]

past_data$malaria_suit_fraction <- past_data$malaria_suitability / 365
past_data$malaria_suit_fraction <- pmin(pmax(past_data$malaria_suit_fraction, 0.001), 0.999)
past_data$logit_malaria_suitability <- log(past_data$malaria_suit_fraction / (1 - past_data$malaria_suit_fraction))


load(glue("{data_path}/my_models.RData"))
rm(list = ls(pattern = "malaria_pfpr_mod_"))

malaria_pfpr_mod_1 <- lm(logit_malaria_pfpr ~ malaria_suitability + 
                           log_gdppc_mean + 
                           mal_DAH_total_per_capita + 
                           people_flood_days_per_capita + 
                           A0_af,
                         data = past_data)

malaria_pfpr_mod_2 <- scam(logit_malaria_pfpr ~ s(malaria_suitability, k = 6, bs = "mpi") + 
                             s(log_gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations

malaria_pfpr_mod_3 <- scam(logit_malaria_pfpr ~ malaria_suitability + 
                             s(gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations

malaria_pfpr_mod_4 <- scam(logit_malaria_pfpr ~ s(malaria_suitability, k = 6, bs = "mpi") + 
                             s(gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations

malaria_pfpr_mod_5 <- lm(logit_malaria_pfpr ~ logit_malaria_suitability + 
                           log_gdppc_mean + 
                           mal_DAH_total_per_capita + 
                           people_flood_days_per_capita + 
                           A0_af,
                         data = past_data)

malaria_pfpr_mod_6 <- scam(logit_malaria_pfpr ~ s(logit_malaria_suitability, k = 6, bs = "mpi") + 
                             s(log_gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations

malaria_pfpr_mod_7 <- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
                             s(gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations

malaria_pfpr_mod_8 <- scam(logit_malaria_pfpr ~ s(logit_malaria_suitability, k = 6, bs = "mpi") + 
                             s(gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations


model_names <- c("malaria_pfpr_mod_1", "malaria_pfpr_mod_2", "malaria_pfpr_mod_3", "malaria_pfpr_mod_4",
                 "malaria_pfpr_mod_5", "malaria_pfpr_mod_6", "malaria_pfpr_mod_7", "malaria_pfpr_mod_8",
                 "mortality_scam_mod", "incidence_scam_mod")

save(list = model_names, file = glue("{data_path}/my_models_new_3.RData"))