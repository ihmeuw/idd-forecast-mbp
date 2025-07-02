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
past_data$logit_malaria_suitability <- log(past_data$malaria_suit_fraction / (1 - past_data$malaria_suit_fraction))
past_data_nz <- past_data[which(past_data$malaria_suitability > 0),]

load(glue("{data_path}/my_models.RData"))#, envir = my_models_env)

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

malaria_pfpr_mod_5 <- lm(logit_malaria_pfpr ~ malaria_suitability + 
                           log_gdppc_mean + 
                           mal_DAH_total_per_capita + 
                           people_flood_days_per_capita + 
                           A0_af,
                         data = past_data)

malaria_pfpr_mod_6 <- scam(logit_malaria_pfpr ~ s(malaria_suitability, k = 6, bs = "mpi") + 
                             s(log_gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations

malaria_pfpr_mod_7 <- scam(logit_malaria_pfpr ~ malaria_suitability + 
                             s(gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations

malaria_pfpr_mod_8 <- scam(logit_malaria_pfpr ~ s(malaria_suitability, k = 6, bs = "mpi") + 
                             s(gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations




mortality_scam_mod <- scam(log_aa_malaria_mort_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") + 
                             log_gdppc_mean + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations
incidence_scam_mod <- scam(log_aa_malaria_inc_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") + 
                             log_gdppc_mean + A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations

model_names <- c("malaria_pfpr_mod_1", "malaria_pfpr_mod_2", "malaria_pfpr_mod_3", "malaria_pfpr_mod_4",
                 "malaria_pfpr_mod_5", "malaria_pfpr_mod_6", "malaria_pfpr_mod_7", "malaria_pfpr_mod_8",
                 "mortality_scam_mod", "incidence_scam_mod")

save(list = model_names, file = glue("{data_path}/my_models_new.RData"))


my_models_env <- new.env()
load(glue("{data_path}/my_models.RData"))#, envir = my_models_env)


rake_by_location <- function(df, variable){
  raw_pred_var <- glue("{variable}_pred_raw")
  pred_var <- glue("{variable}_pred")
  #
  shift_df <- df[which(df$year_id == df$year_to_rake_to), c("location_id", variable, raw_pred_var)]
  shift_df$shift <- shift_df[[variable]] - shift_df[[raw_pred_var]]
  df <- merge(df, shift_df[,c("location_id", "shift")], by = "location_id")
  df[[pred_var]] <- df[[raw_pred_var]] + df$shift
  df <- subset(df, select = -shift)
  return(df)
}

n_mod <- 6
layout(matrix(1:(4*n_mod),4,n_mod))

draws = seq(0, 99, by = 10)
draws = sprintf("%03d", draws)
draw_results <- array(NA, dim = c(n_mod, 3, length(draws), 101))


for (d_num in seq_along(draws)){
  message(glue("Starting draw: {draws[d_num]}"))
  draw = draws[d_num]
  ssp126_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp126_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
  ssp126_df <-as.data.frame(arrow::read_parquet(ssp126_df_path))
  ssp126_df$A0_af <- as.factor(ssp126_df$A0_af)
  ssp245_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp245_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
  ssp245_df <-as.data.frame(arrow::read_parquet(ssp245_df_path))
  ssp245_df$A0_af <- as.factor(ssp245_df$A0_af)
  ssp585_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp585_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
  ssp585_df <-as.data.frame(arrow::read_parquet(ssp585_df_path))
  ssp585_df$A0_af <- as.factor(ssp585_df$A0_af)
  
  for (mod_num in 1:n_mod){
    message(glue("Starting model: {mod_num}"))
    pfpr_mod <- get(glue("malaria_pfpr_mod_{mod_num}"))
    #
    ssp126_df$logit_malaria_pfpr_pred_raw <- predict(pfpr_mod, ssp126_df)
    ssp126_df <- rake_by_location(ssp126_df, "logit_malaria_pfpr")
    ssp245_df$logit_malaria_pfpr_pred_raw <- predict(pfpr_mod, ssp245_df)
    ssp245_df <- rake_by_location(ssp245_df, "logit_malaria_pfpr")
    ssp585_df$logit_malaria_pfpr_pred_raw <- predict(pfpr_mod, ssp585_df)
    ssp585_df <- rake_by_location(ssp585_df, "logit_malaria_pfpr")
    
    
    ssp126_df$logit_malaria_pfpr_obs <- ssp126_df$logit_malaria_pfpr
    ssp126_df$logit_malaria_pfpr <- ssp126_df$logit_malaria_pfpr_pred
    
    
    ssp245_df$logit_malaria_pfpr_obs <- ssp245_df$logit_malaria_pfpr
    ssp245_df$logit_malaria_pfpr <- ssp245_df$logit_malaria_pfpr_pred
    
    
    ssp585_df$logit_malaria_pfpr_obs <- ssp585_df$logit_malaria_pfpr
    ssp585_df$logit_malaria_pfpr <- ssp585_df$logit_malaria_pfpr_pred
    
    ssp126_df$log_aa_malaria_mort_rate_pred_raw <- predict(mortality_scam_mod, ssp126_df)
    ssp245_df$log_aa_malaria_mort_rate_pred_raw <- predict(mortality_scam_mod, ssp245_df)
    ssp585_df$log_aa_malaria_mort_rate_pred_raw <- predict(mortality_scam_mod, ssp585_df)
    
    ssp126_df <- rake_by_location(ssp126_df, "log_aa_malaria_mort_rate")
    ssp245_df <- rake_by_location(ssp245_df, "log_aa_malaria_mort_rate")
    ssp585_df <- rake_by_location(ssp585_df, "log_aa_malaria_mort_rate")
    
    ssp126_df$mortality = exp(ssp126_df$log_aa_malaria_mort_rate_pred) * ssp126_df$aa_population
    ssp245_df$mortality = exp(ssp245_df$log_aa_malaria_mort_rate_pred) * ssp245_df$aa_population
    ssp585_df$mortality = exp(ssp585_df$log_aa_malaria_mort_rate_pred) * ssp585_df$aa_population
    past_data$mortality = exp(past_data$log_aa_malaria_mort_rate) * past_data$aa_population
    
    # Aggregate both mortality and population by year_id
    agg_ssp126_df <- aggregate(cbind(mortality, aa_population) ~ year_id, data = ssp126_df, FUN = sum)
    agg_ssp245_df <- aggregate(cbind(mortality, aa_population) ~ year_id, data = ssp245_df, FUN = sum)
    agg_ssp585_df <- aggregate(cbind(mortality, aa_population) ~ year_id, data = ssp585_df, FUN = sum)
    agg_past_df <- aggregate(cbind(mortality, aa_population) ~ year_id, data = past_data, FUN = sum)
    
    agg_ssp126_df$mortality_rate = 100000 * agg_ssp126_df$mortality / agg_ssp126_df$aa_population
    agg_ssp245_df$mortality_rate = 100000 * agg_ssp245_df$mortality / agg_ssp245_df$aa_population
    agg_ssp585_df$mortality_rate = 100000 * agg_ssp585_df$mortality / agg_ssp585_df$aa_population
    agg_past_df$mortality_rate = 100000 * agg_past_df$mortality / agg_past_df$aa_population
    
    agg_ssp126_df$mortality_pm = agg_ssp126_df$mortality / 1000000
    agg_ssp245_df$mortality_pm = agg_ssp245_df$mortality / 1000000
    agg_ssp585_df$mortality_pm = agg_ssp585_df$mortality / 1000000
    agg_past_df$mortality_pm = agg_past_df$mortality / 1000000
    
    draw_results[mod_num, 1, d_num, ] <- agg_ssp126_df$mortality
    draw_results[mod_num, 2, d_num, ] <- agg_ssp245_df$mortality
    draw_results[mod_num, 3, d_num, ] <- agg_ssp585_df$mortality
  }
}
















for (mod_num in 1:n_mod){
  message(glue("Starting model: {mod_num}"))
  pfpr_mod <- get(glue("malaria_pfpr_mod_{mod_num}"))
  #
  message("Predicting pfpr")
  message("ssp126")
  ssp126_df$logit_malaria_pfpr_pred_raw <- predict(pfpr_mod, ssp126_df)
  ssp126_df <- rake_by_location(ssp126_df, "logit_malaria_pfpr")
  message("ssp245")
  ssp245_df$logit_malaria_pfpr_pred_raw <- predict(pfpr_mod, ssp245_df)
  ssp245_df <- rake_by_location(ssp245_df, "logit_malaria_pfpr")
  message("ssp585")
  ssp585_df$logit_malaria_pfpr_pred_raw <- predict(pfpr_mod, ssp585_df)
  ssp585_df <- rake_by_location(ssp585_df, "logit_malaria_pfpr")
  
  
  ssp126_df$logit_malaria_pfpr_obs <- ssp126_df$logit_malaria_pfpr
  ssp126_df$logit_malaria_pfpr <- ssp126_df$logit_malaria_pfpr_pred
  
  
  ssp245_df$logit_malaria_pfpr_obs <- ssp245_df$logit_malaria_pfpr
  ssp245_df$logit_malaria_pfpr <- ssp245_df$logit_malaria_pfpr_pred
  
  
  ssp585_df$logit_malaria_pfpr_obs <- ssp585_df$logit_malaria_pfpr
  ssp585_df$logit_malaria_pfpr <- ssp585_df$logit_malaria_pfpr_pred
  
  message("Predicting mortality rate")
  message("ssp126")
  ssp126_df$log_aa_malaria_mort_rate_pred_raw <- predict(mortality_scam_mod, ssp126_df)
  message("ssp245")
  ssp245_df$log_aa_malaria_mort_rate_pred_raw <- predict(mortality_scam_mod, ssp245_df)
  message("ssp585")
  ssp585_df$log_aa_malaria_mort_rate_pred_raw <- predict(mortality_scam_mod, ssp585_df)
  
  ssp126_df <- rake_by_location(ssp126_df, "log_aa_malaria_mort_rate")
  ssp245_df <- rake_by_location(ssp245_df, "log_aa_malaria_mort_rate")
  ssp585_df <- rake_by_location(ssp585_df, "log_aa_malaria_mort_rate")
  
  ssp126_df$mortality = exp(ssp126_df$log_aa_malaria_mort_rate_pred) * ssp126_df$aa_population
  ssp245_df$mortality = exp(ssp245_df$log_aa_malaria_mort_rate_pred) * ssp245_df$aa_population
  ssp585_df$mortality = exp(ssp585_df$log_aa_malaria_mort_rate_pred) * ssp585_df$aa_population
  past_data$mortality = exp(past_data$log_aa_malaria_mort_rate) * past_data$aa_population
  
  # Aggregate both mortality and population by year_id
  agg_ssp126_df <- aggregate(cbind(mortality, aa_population) ~ year_id, data = ssp126_df, FUN = sum)
  agg_ssp245_df <- aggregate(cbind(mortality, aa_population) ~ year_id, data = ssp245_df, FUN = sum)
  agg_ssp585_df <- aggregate(cbind(mortality, aa_population) ~ year_id, data = ssp585_df, FUN = sum)
  agg_past_df <- aggregate(cbind(mortality, aa_population) ~ year_id, data = past_data, FUN = sum)
  
  agg_ssp126_df$mortality_rate = 100000 * agg_ssp126_df$mortality / agg_ssp126_df$aa_population
  agg_ssp245_df$mortality_rate = 100000 * agg_ssp245_df$mortality / agg_ssp245_df$aa_population
  agg_ssp585_df$mortality_rate = 100000 * agg_ssp585_df$mortality / agg_ssp585_df$aa_population
  agg_past_df$mortality_rate = 100000 * agg_past_df$mortality / agg_past_df$aa_population
  
  agg_ssp126_df$mortality_pm = agg_ssp126_df$mortality / 1000000
  agg_ssp245_df$mortality_pm = agg_ssp245_df$mortality / 1000000
  agg_ssp585_df$mortality_pm = agg_ssp585_df$mortality / 1000000
  agg_past_df$mortality_pm = agg_past_df$mortality / 1000000
  
  tmp_df = ssp126_df[which(ssp126_df$year_id == 2022),]
  location_of_interest <- tmp_df$location_id[which.max(tmp_df$log_aa_malaria_mort_rate)]
  location_of_interest <- 75758
  
  tmp_ssp126_df <- ssp126_df[which(ssp126_df$location_id == location_of_interest),]
  tmp_ssp245_df <- ssp245_df[which(ssp245_df$location_id == location_of_interest),]
  tmp_ssp585_df <- ssp585_df[which(ssp585_df$location_id == location_of_interest),]
  tmp_past_df <- past_data[which(past_data$location_id == location_of_interest),]
  
  plot(tmp_ssp126_df$year_id, tmp_ssp126_df$logit_malaria_pfpr_pred, col = 2, type = 'l', main = "Logit Malaria PFPR", ylim = c(-3,1.5))
  lines(tmp_ssp245_df$year_id, tmp_ssp245_df$logit_malaria_pfpr_pred, col = 3)
  lines(tmp_ssp585_df$year_id, tmp_ssp585_df$logit_malaria_pfpr_pred, col = 4)
  lines(tmp_past_df$year_id, tmp_past_df$logit_malaria_pfpr, lwd = 2)
  
  plot(tmp_ssp126_df$year_id, tmp_ssp126_df$log_aa_malaria_mort_rate_pred, col = 2, type = 'l', main = "Log Malaria Mortality Rate", ylim = c(-10,-5))
  lines(tmp_ssp245_df$year_id, tmp_ssp245_df$log_aa_malaria_mort_rate_pred, col = 3)
  lines(tmp_ssp585_df$year_id, tmp_ssp585_df$log_aa_malaria_mort_rate_pred, col = 4)
  lines(tmp_past_df$year_id, tmp_past_df$log_aa_malaria_mort_rate, lwd = 2)
  
  plot(agg_ssp126_df$year_id, agg_ssp126_df$mortality_pm, col = 2, type = 'l', main = "Malaria Mortality (in millions)", 
       ylim = c(0, 1), ylab = "Mortality per 1,000,000", xlab="Year")
  lines(agg_ssp245_df$year_id, agg_ssp245_df$mortality_pm, col = 3)
  lines(agg_ssp585_df$year_id, agg_ssp585_df$mortality_pm, col = 4)
  lines(agg_past_df$year_id, agg_past_df$mortality_pm, lwd = 2)
  
  plot(agg_ssp126_df$year_id, agg_ssp126_df$mortality_pm / agg_ssp126_df$mortality_pm, col = 2, 
       type = 'l', main = "Relative Differnece versus RCP 2.6", 
       ylim = c(0, 1.2), ylab = "Relative Difference", xlab="Year")
  lines(agg_ssp245_df$year_id, agg_ssp245_df$mortality_pm / agg_ssp126_df$mortality_pm, col = 3)
  lines(agg_ssp585_df$year_id, agg_ssp585_df$mortality_pm / agg_ssp126_df$mortality_pm, col = 4)
  # lines(agg_past_df$year_id, agg_past_df$mortality_pm / agg_ssp126_df$mortality_pm, lwd = 2)
}



#



















shift_df <- tmp_ssp126_df[which(tmp_ssp126_df$year_id == tmp_ssp126_df$year_to_rake), c("location_id", "logit_malaria_pfpr", "logit_malaria_pfpr_pred_raw")]
shift_df$shift <- shift_df$logit_malaria_pfpr - shift_df$logit_malaria_pfpr_pred_raw
tmp_ssp126_df <- merge(tmp_ssp126_df, shift_df[,c("location_id", "shift")], on = "location_id")
tmp_ssp126_df$logit_malaria_pfpr_pred <- tmp_ssp126_df$logit_malaria_pfpr_pred_raw + tmp_ssp126_df$shift
tmp_ssp126_df <- subset(tmp_ssp126_df, select = -shift)




current_pfpr_mod <- malaria_pfpr_mod
rm(malaria_pfpr_mod)
current_mortality_scam_mod <- mortality_scam_mod
rm(mortality_scam_mod)
current_incidence_scam_mod <- incidence_scam_mod
rm(incidence_scam_mod)
load(file = glue("{data_path}/necessary_malaria_regression_models.RData"))

exp(max(malaria_df$log_malaria_mort_rate))
malaria_df[which.max(malaria_df$log_malaria_mort_rate),]

par(mfrow=c(2,1))
plot(incidence_scam_mod)
plot(current_incidence_scam_mod)

dim(ssp126_df)

tmp_df = ssp126_df[which(ssp126_df$year_id == 2022),]
location_of_interest <- tmp_df$location_id[which.max(tmp_df$base_log_malaria_mort_rate )]

tmp_ssp126_df <- ssp126_df[which(ssp126_df$location_id == location_of_interest),]
tmp_ssp245_df <- ssp245_df[which(ssp245_df$location_id == location_of_interest),]
tmp_ssp585_df <- ssp585_df[which(ssp585_df$location_id == location_of_interest),]

tmp_ssp126_df$base_logit_malaria_pfpr_pred_raw <- predict(current_pfpr_mod, tmp_ssp126_df)
tmp_ssp245_df$base_logit_malaria_pfpr_pred_raw <- predict(current_pfpr_mod, tmp_ssp245_df)
tmp_ssp585_df$base_logit_malaria_pfpr_pred_raw <- predict(current_pfpr_mod, tmp_ssp585_df)
pfpr_126_old <- predict(malaria_pfpr_mod_1, tmp_ssp126_df)
pfpr_245_old <- predict(malaria_pfpr_mod_1, tmp_ssp245_df)
pfpr_585_old <- predict(malaria_pfpr_mod_1, tmp_ssp585_df)

plot(tmp_ssp126_df$year_id, tmp_ssp126_df$base_logit_malaria_pfpr_pred_raw, type = 'l', ylim = c(-5, 5), main = "Logit Malaria PFPR")
lines(tmp_ssp245_df$year_id, tmp_ssp245_df$base_logit_malaria_pfpr_pred_raw, col = 'red')
lines(tmp_ssp585_df$year_id, tmp_ssp585_df$base_logit_malaria_pfpr_pred_raw, col = 'blue')

lines(tmp_ssp126_df$year_id, pfpr_126_old, lty = 2, lwd = 2)
lines(tmp_ssp245_df$year_id, pfpr_245_old, col = 'red', lty = 2, lwd = 2)
lines(tmp_ssp585_df$year_id, pfpr_585_old, col = 'blue', lty = 2, lwd = 2)




shift_df <- tmp_ssp126_df[which(tmp_ssp126_df$year_id == tmp_ssp126_df$year_to_rake), c("location_id", "base_logit_malaria_pfpr", "base_logit_malaria_pfpr_pred_raw")]
shift_df$shift <- shift_df$base_logit_malaria_pfpr - shift_df$base_logit_malaria_pfpr_pred_raw
tmp_ssp126_df <- merge(tmp_ssp126_df, shift_df[,c("location_id", "shift")], on = "location_id")
tmp_ssp126_df$base_logit_malaria_pfpr_pred <- tmp_ssp126_df$base_logit_malaria_pfpr_pred_raw + tmp_ssp126_df$shift
tmp_ssp126_df <- subset(tmp_ssp126_df, select = -shift)

shift_df <- tmp_ssp245_df[which(tmp_ssp245_df$year_id == tmp_ssp245_df$year_to_rake), c("location_id", "base_logit_malaria_pfpr", "base_logit_malaria_pfpr_pred_raw")]
shift_df$shift <- shift_df$base_logit_malaria_pfpr - shift_df$base_logit_malaria_pfpr_pred_raw
tmp_ssp245_df <- merge(tmp_ssp245_df, shift_df[,c("location_id", "shift")], on = "location_id")
tmp_ssp245_df$base_logit_malaria_pfpr_pred <- tmp_ssp245_df$base_logit_malaria_pfpr_pred_raw + tmp_ssp245_df$shift
tmp_ssp245_df <- subset(tmp_ssp245_df, select = -shift)

shift_df <- tmp_ssp585_df[which(tmp_ssp585_df$year_id == tmp_ssp585_df$year_to_rake), c("location_id", "base_logit_malaria_pfpr", "base_logit_malaria_pfpr_pred_raw")]
shift_df$shift <- shift_df$base_logit_malaria_pfpr - shift_df$base_logit_malaria_pfpr_pred_raw
tmp_ssp585_df <- merge(tmp_ssp585_df, shift_df[,c("location_id", "shift")], on = "location_id")
tmp_ssp585_df$base_logit_malaria_pfpr_pred <- tmp_ssp585_df$base_logit_malaria_pfpr_pred_raw + tmp_ssp585_df$shift
tmp_ssp585_df <- subset(tmp_ssp585_df, select = -shift)

plot(tmp_ssp126_df$year_id, tmp_ssp126_df$base_logit_malaria_pfpr_pred, type = 'l', ylim = c(-5, 5), main = "Logit Malaria PFPR")
lines(tmp_ssp245_df$year_id, tmp_ssp245_df$base_logit_malaria_pfpr_pred, col = 'red')
lines(tmp_ssp585_df$year_id, tmp_ssp585_df$base_logit_malaria_pfpr_pred, col = 'blue')

lines(tmp_ssp126_df$year_id, pfpr_126_old, lty = 2, lwd = 2)
lines(tmp_ssp245_df$year_id, pfpr_245_old, col = 'red', lty = 2, lwd = 2)
lines(tmp_ssp585_df$year_id, pfpr_585_old, col = 'blue', lty = 2, lwd = 2)


tmp_ssp126_df$base_logit_malaria_pfpr_obs <- tmp_ssp126_df$base_logit_malaria_pfpr
tmp_ssp126_df$base_logit_malaria_pfpr <- tmp_ssp126_df$base_logit_malaria_pfpr_pred


tmp_ssp245_df$base_logit_malaria_pfpr_obs <- tmp_ssp245_df$base_logit_malaria_pfpr
tmp_ssp245_df$base_logit_malaria_pfpr <- tmp_ssp245_df$base_logit_malaria_pfpr_pred


tmp_ssp585_df$base_logit_malaria_pfpr_obs <- tmp_ssp585_df$base_logit_malaria_pfpr
tmp_ssp585_df$base_logit_malaria_pfpr <- tmp_ssp585_df$base_logit_malaria_pfpr_pred


tmp_ssp126_df$base_log_malaria_mort_rate_pred_raw <- predict(current_mortality_scam_mod, tmp_ssp126_df)
tmp_ssp126_df$base_log_malaria_mort_rate_pred_raw_old <- predict(mortality_scam_mod, tmp_ssp126_df)
tmp_ssp245_df$base_log_malaria_mort_rate_pred_raw <- predict(current_mortality_scam_mod, tmp_ssp245_df)
tmp_ssp245_df$base_log_malaria_mort_rate_pred_raw_old <- predict(mortality_scam_mod, tmp_ssp245_df)
tmp_ssp585_df$base_log_malaria_mort_rate_pred_raw <- predict(current_mortality_scam_mod, tmp_ssp585_df)
tmp_ssp585_df$base_log_malaria_mort_rate_pred_raw_old <- predict(mortality_scam_mod, tmp_ssp585_df)

plot(tmp_ssp126_df$year_id, 100000 * exp(tmp_ssp126_df$base_log_malaria_mort_rate_pred_raw), type = 'l', main = "Logit Malaria PFPR", ylim = c(0,1200), ylab = "Mortality rate per 100,000", xlab="Year")
lines(tmp_ssp245_df$year_id, 100000 * exp(tmp_ssp245_df$base_log_malaria_mort_rate_pred_raw), col = 'red')
lines(tmp_ssp585_df$year_id, 100000 * exp(tmp_ssp585_df$base_log_malaria_mort_rate_pred_raw), col = 'blue')

lines(tmp_ssp126_df$year_id, 100000 * exp(tmp_ssp126_df$base_log_malaria_mort_rate_pred_raw_old), lty = 2, lwd = 2)
lines(tmp_ssp245_df$year_id, 100000 * exp(tmp_ssp245_df$base_log_malaria_mort_rate_pred_raw_old), col = 'red', lty = 2, lwd = 2)
lines(tmp_ssp585_df$year_id, 100000 * exp(tmp_ssp585_df$base_log_malaria_mort_rate_pred_raw_old), col = 'blue', lty = 2, lwd = 2)


tmp_old_df <- as.data.frame(malaria_df[which(malaria_df$location_id == location_of_interest),])


tmp_ssp126_df$base_logit_malaria_pfpr <- pfpr_126_new
tmp_ssp126_df$base_logit_malaria_pfpr <- pfpr_126_new
tmp_ssp126_df$base_logit_malaria_pfpr <- pfpr_126_new







plot(tmp_ssp126_df$year_id, tmp_ssp126_df$malaria_suitability, type = 'l', ylim = c(0, 365))
lines(tmp_ssp245_df$year_id, tmp_ssp245_df$malaria_suitability, col = 'red')
lines(tmp_ssp585_df$year_id, tmp_ssp585_df$malaria_suitability, col = 'blue')



tmp_df <- tmp_ssp126_df[,c("logit_malaria_pfpr", "malaria_suitability", "log_gdppc_mean", "mal_DAH_total_per_capita", "people_flood_days_per_capita", "A0_af")]

predict(current_pfpr_mod, tmp_df)
#









output_forecast_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario_name}_draw_{draw}_with_predictions.parquet")

forecast_df <- as.data.frame(arrow::read_parquet(input_forecast_df_path))
forecast_df$A0_af <- as.factor(forecast_df$A0_af)
forecast_df <- forecast_df[-which(is.na(forecast_df$base_log_malaria_mort_rate)),]


forecast_df$logit_malaria_pfpr_pred_raw <- predict(malaria_pfpr_mod, forecast_df)


shift_df <- forecast_df[which(forecast_df$year_id == forecast_df$year_to_rake), c("location_id", "logit_malaria_pfpr", "logit_malaria_pfpr_pred_raw")]
shift_df$shift <- shift_df$logit_malaria_pfpr - shift_df$logit_malaria_pfpr_pred_raw
forecast_df <- merge(forecast_df, shift_df[,c("location_id", "shift")], on = "location_id")
forecast_df$logit_malaria_pfpr_pred <- forecast_df$logit_malaria_pfpr_pred_raw + forecast_df$shift
forecast_df <- subset(forecast_df, select = -shift)

####
# Model 1

forecast_df$base_logit_malaria_pfpr_obs <- forecast_df$base_logit_malaria_pfpr
forecast_df$base_logit_malaria_pfpr <- forecast_df$logit_malaria_pfpr_pred

#### 

forecast_df$base_log_malaria_mort_rate_pred_raw <- predict(mortality_scam_mod, forecast_df)
# Intercept shift base estimate
shift_df <- forecast_df[which(forecast_df$year_id == forecast_df$year_to_rake), c("location_id", "base_log_malaria_mort_rate", "base_log_malaria_mort_rate_pred_raw")]
shift_df$shift <- shift_df$base_log_malaria_mort_rate - shift_df$base_log_malaria_mort_rate_pred_raw
forecast_df <- merge(forecast_df, shift_df[,c("location_id", "shift")], on = "location_id")
forecast_df$base_log_malaria_mort_rate_pred <- forecast_df$base_log_malaria_mort_rate_pred_raw + forecast_df$shift
forecast_df <- subset(forecast_df, select = -shift)



forecast_df$base_log_malaria_inc_rate_pred_raw <- predict(incidence_scam_mod, forecast_df)
# Intercept shift base estimate
shift_df <- forecast_df[which(forecast_df$year_id == forecast_df$year_to_rake), c("location_id", "base_log_malaria_inc_rate", "base_log_malaria_inc_rate_pred_raw")]
shift_df$shift <- shift_df$base_log_malaria_inc_rate - shift_df$base_log_malaria_inc_rate_pred_raw
forecast_df <- merge(forecast_df, shift_df[,c("location_id", "shift")], on = "location_id")
forecast_df$base_log_malaria_inc_rate_pred <- forecast_df$base_log_malaria_inc_rate_pred_raw + forecast_df$shift
forecast_df <- subset(forecast_df, select = -shift)


forecast_df$base_logit_malaria_pfpr <- forecast_df$base_logit_malaria_pfpr_obs
forecast_df <- subset(forecast_df, select = -base_logit_malaria_pfpr_obs)


#### 


print(glue("Saving draw: {draw}, SSP scenario: {ssp_scenario}, DAH scenario: {dah_scenario_name}"))
# print(glue("    - {forecast_df[which(forecast_df$location_id == 25364 & forecast_df$year_id == 2091),c('log_malaria_pf_mort_rate_pred', 'log_malaria_pf_inc_rate_pred')]}"))

####
write_parquet(forecast_df, output_forecast_df_path)
print("fin")