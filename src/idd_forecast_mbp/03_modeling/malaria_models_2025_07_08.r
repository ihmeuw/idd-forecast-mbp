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

REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"
last_year <- 2022
data_path <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")

df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp126_dah_scenario_Baseline_draw_000.parquet")
df <-as.data.frame(arrow::read_parquet(df_path))
df$A0_af <- as.factor(df$A0_af)

past_data <- df[-which(is.na(df$malaria_pfpr)),]
past_data <- past_data[-which(is.na(past_data$gdppc_mean)),]

past_data$malaria_suit_fraction <- past_data$malaria_suitability / 365
past_data$malaria_suit_fraction <- pmin(pmax(past_data$malaria_suit_fraction, 0.001), 0.999)
past_data$logit_malaria_suitability <- log(past_data$malaria_suit_fraction / (1 - past_data$malaria_suit_fraction))



malaria_pfpr_mod <- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
                             s(gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations




mod_df <- past_data[which(past_data$aa_malaria_mort_rate > 0),]
mortality_scam_mod <- scam(log_aa_malaria_mort_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") + 
                             log_gdppc_mean + 
                             A0_af,
                           data = mod_df,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations

mod_df <- past_data[which(past_data$aa_malaria_inc_rate > 0),]
incidence_scam_mod <- scam(log_aa_malaria_inc_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") + 
                             log_gdppc_mean + A0_af,
                           data = mod_df,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations

mod_df <- past_data[which(past_data$base_malaria_mort_rate > 0),]
mortality_base_scam_mod <- scam(log_base_malaria_mort_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") + 
                                  log_gdppc_mean + 
                                  A0_af,
                                data = mod_df,
                                optimizer = "efs",      # Faster optimizer
                                control = list(maxit = 300))  # Limit iterations

mod_df <- past_data[which(past_data$base_malaria_inc_rate  > 0),]
incidence_base_scam_mod <- scam(log_base_malaria_inc_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") + 
                                  log_gdppc_mean + A0_af,
                                data = mod_df,
                                optimizer = "efs",      # Faster optimizer
                                control = list(maxit = 300))  # Limit iterations


model_names <- c("malaria_pfpr_mod", "mortality_scam_mod", "incidence_scam_mod", "mortality_base_scam_mod",
                 "incidence_base_scam_mod")

save(list = model_names, file = glue("{data_path}/2025_07_08_malaria_models.RData"))





percentiles = seq(0, 1, by = 0.05)
mal_dah_perc = sapply(percentiles, function(p) {
  quantile(past_data$mal_DAH_total_per_capita, p, na.rm = TRUE)
})

mal_dah_perc = unique(mal_dah_perc)

bin_df <- data.frame(bin_start = head(mal_dah_perc, -1),
                     bin_end = tail(mal_dah_perc, -1),
                     mean_residual = NA,
                     Q1 = NA,
                     Q3 = NA)
for (i in bin_df$bin_start){
  tmp_locs <- which(past_data$mal_DAH_total_per_capita >= i & 
                        past_data$mal_DAH_total_per_capita < (i + 0.01))
  bin_df$mean_residual[which(bin_df$bin_start == i)] <- mean(malaria_pfpr_mod$residuals[tmp_locs])
  bin_df$Q1[which(bin_df$bin_start == i)] <- quantile(malaria_pfpr_mod$residuals[tmp_locs], 0.25)
  bin_df$Q3[which(bin_df$bin_start == i)] <- quantile(malaria_pfpr_mod$residuals[tmp_locs], 0.75)
}


par(mfrow=c(3,1))
plot(malaria_pfpr_mod, select = 2)
plot(bin_df$bin_start+bin_df$bin_end, bin_df$mean_residual, type = 'n',xlim = c(0, max(bin_df$bin_end)), ylim = c(min(bin_df$Q1), max(bin_df$Q3)))
abline(h = 0, lty = 2)
for (i in seq_along(bin_df$bin_start)){
  lines(c(bin_df$bin_start[i], bin_df$bin_end[i]), 
        c(bin_df$mean_residual[i], bin_df$mean_residual[i]),
        col = "blue", lwd = 2)
  lines(rep((bin_df$bin_start[i] + bin_df$bin_end[i]) / 2, 2), 
          c(bin_df$Q1[i], bin_df$Q3[i]),
          col = "red", lwd = 2)
}
plot(bin_df$bin_start+1e-6, bin_df$mean_residual, type = 'n', xlim = c(min(bin_df$bin_start) + 1e-6, max(bin_df$bin_end)), ylim = c(min(bin_df$Q1), max(bin_df$Q3)), log = 'x')
abline(h = 0, lty = 2)
for (i in seq_along(bin_df$bin_start)){
  lines(c(bin_df$bin_start[i]+1e-6, bin_df$bin_end[i]), 
        c(bin_df$mean_residual[i], bin_df$mean_residual[i]),
        col = "blue", lwd = 2)
  lines(rep((bin_df$bin_start[i]+1e-6 + bin_df$bin_end[i]) / 2, 2), 
        c(bin_df$Q1[i], bin_df$Q3[i]),
        col = "red", lwd = 2)
}


