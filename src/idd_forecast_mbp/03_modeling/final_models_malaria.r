rm(list = ls())

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)


"%ni%" <- Negate("%in%")
"%nlike%" <- Negate("%like%")

last_year <- 2022

MODELING_DATA_PATH <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"

aa_md_malaria_pfpr_modeling_df_path = glue("{MODELING_DATA_PATH}/aa_md_malaria_pfpr_modeling_df.parquet")
base_md_malaria_modeling_df_path = glue("{MODELING_DATA_PATH}/base_md_malaria_modeling_df.parquet")

# Read in a parquet file
malaria_df <- arrow::read_parquet(aa_md_malaria_pfpr_modeling_df_path)
malaria_df$A0_af <- as.factor(malaria_df$A0_af)


malaria_pfpr_mod <- lm(logit_malaria_pfpr ~ malaria_suitability + log_gdppc_mean + 
                         mal_DAH_total_per_capita + people_flood_days_per_capita + A0_af,
                       data = malaria_df)
summary(malaria_pfpr_mod)

malaria_df$log_malaria_mort_rate = log(malaria_df$malaria_mort_rate + 1e-6)
malaria_df$log_malaria_inc_rate = log(malaria_df$malaria_inc_rate + 1e-6)

mortality_scam_mod <- scam(log_malaria_mort_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = malaria_df)
incidence_scam_mod <- scam(log_malaria_inc_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = malaria_df)





dah_scenario_name = "Baseline"
draw = "000"
REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"
data_path <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")
ssp126_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp126_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
ssp126_df <-as.data.frame(arrow::read_parquet(ssp126_df_path))
ssp126_df$A0_af <- factor(ssp126_df$A0_af, levels = levels(malaria_df$A0_af))
ssp245_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp245_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
ssp245_df <-as.data.frame(arrow::read_parquet(ssp245_df_path))
ssp245_df$A0_af <- factor(ssp245_df$A0_af, levels = levels(malaria_df$A0_af))
ssp585_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp585_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
ssp585_df <-as.data.frame(arrow::read_parquet(ssp585_df_path))
ssp585_df$A0_af <- factor(ssp585_df$A0_af, levels = levels(malaria_df$A0_af))


ssp126_df <- ssp126_df[, c("location_id", "year_id", "malaria_suitability",
                           "logit_malaria_pfpr", "log_gdppc_mean", 
                           "mal_DAH_total_per_capita", "people_flood_days_per_capita", 
                           "A0_af", "population")]
ssp126_df = ssp126_df[complete.cases(ssp126_df), ]
ssp245_df <- ssp245_df[, c("location_id", "year_id", "malaria_suitability",
                           "logit_malaria_pfpr", "log_gdppc_mean", 
                           "mal_DAH_total_per_capita", "people_flood_days_per_capita", 
                           "A0_af", "population")]
ssp245_df = ssp245_df[complete.cases(ssp245_df), ]
ssp585_df <- ssp585_df[, c("location_id", "year_id", "malaria_suitability",
                           "logit_malaria_pfpr", "log_gdppc_mean", 
                           "mal_DAH_total_per_capita", "people_flood_days_per_capita", 
                           "A0_af", "population")]
ssp585_df = ssp585_df[complete.cases(ssp585_df), ]

ssp126_df$logit_malaria_pfpr = predict(malaria_pfpr_mod, ssp126_df)
ssp245_df$logit_malaria_pfpr = predict(malaria_pfpr_mod, ssp245_df)
ssp585_df$logit_malaria_pfpr = predict(malaria_pfpr_mod, ssp585_df)


ssp126_df$log_malaria_mort_rate_pred = predict(mortality_scam_mod, newdata = ssp126_df)
ssp245_df$log_malaria_mort_rate_pred = predict(mortality_scam_mod, newdata = ssp245_df)
ssp585_df$log_malaria_mort_rate_pred = predict(mortality_scam_mod, newdata = ssp585_df)

ssp126_df$malaria_mort_count = exp(ssp126_df$log_malaria_mort_rate_pred) * ssp126_df$population
ssp245_df$malaria_mort_count = exp(ssp245_df$log_malaria_mort_rate_pred) * ssp245_df$population
ssp585_df$malaria_mort_count = exp(ssp585_df$log_malaria_mort_rate_pred) * ssp585_df$population
agg_126_df = aggregate(malaria_mort_count ~ year_id, data = ssp126_df, FUN = sum)
agg_245_df = aggregate(malaria_mort_count ~ year_id, data = ssp245_df, FUN = sum)
agg_585_df = aggregate(malaria_mort_count ~ year_id, data = ssp585_df, FUN = sum)
plot(agg_126_df$year_id, agg_126_df$malaria_mort_count,
     type = "l", col = "blue", xlab = "Year", ylab = "Malaria Mortality Count",
     main = "Malaria Mortality Count by SSP Scenario")
lines(agg_245_df$year_id, agg_245_df$malaria_mort_count, col = "red")
lines(agg_585_df$year_id, agg_585_df$malaria_mort_count, col = "green")

location_of_interest <- malaria_df$location_id[which.max(malaria_df$log_malaria_mort_rate)]

tmp_ssp126_df <- ssp126_df[which(ssp126_df$location_id == location_of_interest),]
tmp_ssp245_df <- ssp245_df[which(ssp245_df$location_id == location_of_interest),]
tmp_ssp585_df <- ssp585_df[which(ssp585_df$location_id == location_of_interest),]

plot(tmp_ssp126_df$year_id, tmp_ssp126_df$logit_malaria_pfpr,
     type = "l", col = "blue", xlab = "Year", ylab = "Log Mortality Rate",
     main = paste("Location ID:", location_of_interest))
lines(tmp_ssp245_df$year_id, tmp_ssp245_df$logit_malaria_pfpr, col = "red")
lines(tmp_ssp585_df$year_id, tmp_ssp585_df$logit_malaria_pfpr, col = "green")







########################################################
########################################################
####                                                ####
####    Age-sex-specific mortality and incidence    ####
####                                                ####
########################################################
########################################################
base_malaria_df <- as.data.frame(arrow::read_parquet(base_md_malaria_modeling_df_path))
base_malaria_df$A0_af = as.factor(base_malaria_df$A0_af)

mortality_scam_mod <- scam(base_log_malaria_mort_rate ~ s(base_logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = base_malaria_df)



dah_scenario_name = "Baseline"
draw = "000"
REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"
data_path <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")
ssp126_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp126_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
ssp126_df <-as.data.frame(arrow::read_parquet(ssp126_df_path))
ssp126_df$A0_af <- factor(ssp126_df$A0_af, levels = levels(base_malaria_df$A0_af))
ssp245_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp245_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
ssp245_df <-as.data.frame(arrow::read_parquet(ssp245_df_path))
ssp245_df$A0_af <- factor(ssp245_df$A0_af, levels = levels(base_malaria_df$A0_af))
ssp585_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp585_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
ssp585_df <-as.data.frame(arrow::read_parquet(ssp585_df_path))
ssp585_df$A0_af <- factor(ssp585_df$A0_af, levels = levels(base_malaria_df$A0_af))


ssp126_df$base_logit_malaria_pfpr_raw = predict(malaria_pfpr_mod, ssp126_df)
ssp245_df$base_logit_malaria_pfpr_raw = predict(malaria_pfpr_mod, ssp245_df)
ssp585_df$base_logit_malaria_pfpr_raw = predict(malaria_pfpr_mod, ssp585_df)
ssp126_df$base_log_malaria_mort_rate_pred_raw = predict(mortality_scam_mod, ssp126_df)
ssp245_df$base_log_malaria_mort_rate_pred_raw = predict(mortality_scam_mod, ssp245_df)
ssp585_df$base_log_malaria_mort_rate_pred_raw = predict(mortality_scam_mod, ssp585_df)


location_of_interest <- base_malaria_df$location_id[which.max(base_malaria_df$base_log_malaria_mort_rate)]

tmp_ssp126_df <- ssp126_df[which(ssp126_df$location_id == location_of_interest),]
tmp_ssp245_df <- ssp245_df[which(ssp245_df$location_id == location_of_interest),]
tmp_ssp585_df <- ssp585_df[which(ssp585_df$location_id == location_of_interest),]

plot(tmp_ssp126_df$year_id, tmp_ssp126_df$logit_malaria_pfpr,
     type = "l", col = "blue", xlab = "Year", ylab = "Log Mortality Rate",
     main = paste("Location ID:", location_of_interest))
lines(tmp_ssp245_df$year_id, tmp_ssp245_df$logit_malaria_pfpr, col = "red")
lines(tmp_ssp585_df$year_id, tmp_ssp585_df$logit_malaria_pfpr, col = "green")












incidence_scam_mod <- scam(base_log_malaria_inc_rate ~ s(base_logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = base_malaria_df)


mortality_scam_mod <- scam(base_log_malaria_mort_rate ~ s(base_logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = base_malaria_df,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 1000))  # Limit iterations

incidence_scam_mod <- scam(base_log_malaria_inc_rate ~ s(base_logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = base_malaria_df,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 200))  # Limit iterations


save.image(file = glue("{MODELING_DATA_PATH}/final_malaria_regression_models_v2.RData"))
