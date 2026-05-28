rm(list = ls())
#
library(languageserver)
library(httpgd)
library(rlang)

library(tidync)
require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)

"%ni%" <- Negate("%in%")
"%nlike%" <- Negate("%like%")

###########################################

nc_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/past_inputs_nc/lsae_1285/current/malaria_past_inputs.nc"

suit_variant_pick <- "mordecai_0_0"

src <- tidync(nc_path)

# loc x year (no draw)
ly_src <- activate(src, "malaria_pfpr")
ly_dt  <- as.data.table(hyper_tibble(ly_src))

# loc x year x draw (climate vars) -- all draws
lyd_src <- activate(src, "mean_temperature")
lyd_dt  <- as.data.table(hyper_tibble(lyd_src))

# loc x year x draw x suit_variant -- pick variant, keep all draws
lyds_src <- activate(src, "malaria_suitability")
lyds_src <- hyper_filter(lyds_src, suit_variant = suit_variant == suit_variant_pick)
lyds_dt  <- as.data.table(hyper_tibble(lyds_src))
lyds_dt[, suit_variant := NULL]

# A0_location_id (coord on location_id)
A0_src <- activate(src, "A0_location_id")
A0_dt  <- as.data.table(hyper_tibble(A0_src))

# merges (all data.table, fast)
past_data <- merge(lyd_dt,    lyds_dt, by = c("location_id", "year_id", "draw_id"), all.x = TRUE)
past_data <- merge(past_data, ly_dt,   by = c("location_id", "year_id"),            all.x = TRUE)
past_data <- merge(past_data, A0_dt,   by = "location_id",                          all.x = TRUE)

setDF(past_data)   # only at the end, before scam()































nc_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/past_inputs_nc/lsae_1285/current/malaria_past_inputs.nc"
src <- tidync(nc_path)

suit_variant_pick <- "mordecai_0_0"

src <- tidync(nc_path)

# loc x year grid (no draw)
ly_src   <- activate(src, "malaria_pfpr")
ly_df    <- as.data.frame(hyper_tibble(ly_src))

# loc x year x draw grid (climate vars) -- keep all draws
lyd_src  <- activate(src, "mean_temperature")
lyd_df   <- as.data.frame(hyper_tibble(lyd_src))

# loc x year x draw x suit_variant (malaria_suitability) -- pick variant, keep draws
lyds_src <- activate(src, "malaria_suitability")
lyds_src <- hyper_filter(lyds_src, suit_variant = suit_variant == suit_variant_pick)
lyds_df  <- as.data.frame(hyper_tibble(lyds_src))
lyds_df$suit_variant <- NULL

# A0_location_id (coord on location_id)
A0_src   <- activate(src, "A0_location_id")
A0_df    <- as.data.frame(hyper_tibble(A0_src))

full_df <- merge(lyd_df, lyds_df, by = c("location_id", "year_id", "draw_id"), all.x = TRUE)
full_df <- merge(full_df, ly_df,  by = c("location_id", "year_id"),            all.x = TRUE)
full_df <- merge(full_df, A0_df,  by = "location_id",                          all.x = TRUE)










###########################################
dah_scenario_name = 'Baseline'
draw = '077'

REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"
last_year <- 2022
data_path <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")

ssp585_df_path <- glue("{FORECASTING_DATA_PATH}/malaria_forecast_ssp_scenario_ssp585_dah_scenario_{dah_scenario_name}_draw_{draw}.parquet")
ssp585_df <-as.data.frame(arrow::read_parquet(ssp585_df_path))
ssp585_df$A0_af <- as.factor(ssp585_df$A0_af)

past_data <- ssp585_df[-which(is.na(ssp585_df$malaria_pfpr)),]
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

save(list = model_names, file = glue("{data_path}/2025_07_03_malaria_models.RData"))
