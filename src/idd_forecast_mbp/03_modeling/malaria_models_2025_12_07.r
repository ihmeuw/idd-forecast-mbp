rm(list = ls())
#

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)
library(ncdf4)
library(dplyr)

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

malaria_itn_path = '/ihme/forecasting/data/35/past/prevalence/malaria_itn/malaria_itn.nc'
malaria_act_path = '/ihme/forecasting/data/35/past/prevalence/malaria_drug/malaria_drug.nc'

############ 
#### ITN
############
# Open the NetCDF file
itn_nc <- nc_open(malaria_itn_path)

# Extract the dimensions
sex_ids <- ncvar_get(itn_nc, "sex_id")
age_group_ids <- ncvar_get(itn_nc, "age_group_id")
location_ids <- ncvar_get(itn_nc, "location_id")
year_ids <- ncvar_get(itn_nc, "year_id")

# Extract the malaria_itn variable
itn_data <- ncvar_get(itn_nc, "value")
nc_close(itn_nc)

# If you want the 0 scenario (index 2), extract just that slice
itn_data_slice <- itn_data[1, 1, , ]  # Adjust index as needed

# Convert to dataframe (no scenario column needed)
itn_df <- expand.grid(
  year_id = year_ids,
  location_id = location_ids
) %>%
  mutate(malaria_itn = as.vector(itn_data_slice))

# Merge with past_data (only on location_id and year_id)
past_data <- past_data %>%
  left_join(itn_df, 
            by = c("A0_location_id" = "location_id", 
                   "year_id" = "year_id"))

############ 
#### ACT
############
# Open the NetCDF file
act_nc <- nc_open(malaria_act_path)

# Extract the dimensions
sex_ids <- ncvar_get(itn_nc, "sex_id")
age_group_ids <- ncvar_get(itn_nc, "age_group_id")
location_ids <- ncvar_get(itn_nc, "location_id")
year_ids <- ncvar_get(itn_nc, "year_id")

# Extract the malaria_itn variable
act_data <- ncvar_get(act_nc, "value")
nc_close(act_nc)

# If you want the 0 scenario (index 2), extract just that slice
act_data_slice <- act_data[1, 1, , ]  # Adjust index as needed

# Convert to dataframe (no scenario column needed)
act_df <- expand.grid(
  year_id = year_ids,
  location_id = location_ids
) %>%
  mutate(malaria_act = as.vector(act_data_slice))

# Merge with past_data (only on location_id and year_id)
past_data <- past_data %>%
  left_join(act_df, 
            by = c("A0_location_id" = "location_id", 
                   "year_id" = "year_id"))

past_data$has_itn = ifelse(past_data$malaria_itn==0, 0, 1)
past_data$has_act = ifelse(past_data$malaria_act==0, 0, 1)
past_data$logit_malaria_itn = log(past_data$malaria_itn / (1 - past_data$malaria_itn))
past_data$logit_malaria_act = log(past_data$malaria_act / (1 - past_data$malaria_act))

past_data_w_interventions = past_data[which(past_data$has_itn == 1 & past_data$has_act == 1),]

new_malaria_pfpr_mod <- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
                               s(gdppc_mean, k = 4, bs = 'mpd') + 
                               s(malaria_itn, k = 4, bs = 'mpd') + 
                               s(malaria_act, k = 4, bs = 'mpd') + 
                               people_flood_days_per_capita + 
                               A0_af,
                             data = past_data,
                             optimizer = "efs",      # Faster optimizer
                             control = list(maxit = 200))  # Limit iterations

new_malaria_pfpr_mod_1 <- scam(logit_malaria_pfpr ~ 
                                 logit_malaria_suitability + 
                                 s(gdppc_mean, k = 4, bs = 'mpd') + 
                                 malaria_itn + 
                                 malaria_act+ 
                                 A0_af,
                               data = past_data_w_interventions,
                               optimizer = "efs") 

new_malaria_pfpr_mod_2 <- scam(logit_malaria_pfpr ~ 
                                 logit_malaria_suitability + 
                                 s(gdppc_mean, k = 4, bs = 'mpd') + 
                                 malaria_itn + 
                                 malaria_act+ 
                                 A0_af,
                               data = past_data,
                               optimizer = "efs") 

new_malaria_pfpr_mod_3 <- scam(logit_malaria_pfpr ~ 
                                 logit_malaria_suitability + 
                                 s(gdppc_mean, k = 4, bs = 'mpd') + 
                                 malaria_itn + 
                                 malaria_act+ 
                                 people_flood_days_per_capita +  
                                 A0_af,
                               data = past_data_w_interventions,
                               optimizer = "efs") 

new_malaria_pfpr_mod_4 <- scam(logit_malaria_pfpr ~ 
                                 logit_malaria_suitability + 
                                 s(gdppc_mean, k = 4, bs = 'mpd') + 
                                 malaria_itn + 
                                 malaria_act + 
                                 people_flood_days_per_capita +  
                                 A0_af,
                               data = past_data,
                               optimizer = "efs") 

new_malaria_pfpr_mod_5 <- lm(logit_malaria_pfpr ~ 
                               logit_malaria_suitability + 
                               gdppc_mean + 
                               malaria_itn + 
                               malaria_act+ 
                               people_flood_days_per_capita +  
                               A0_af,
                             data = past_data_w_interventions,
                             optimizer = "efs") 

new_malaria_pfpr_mod_6 <- lm(logit_malaria_pfpr ~ 
                               logit_malaria_suitability + 
                               gdppc_mean + 
                               malaria_itn + 
                               malaria_act + 
                               people_flood_days_per_capita +  
                               A0_af,
                             data = past_data,
                             optimizer = "efs") 


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


model_names <- c("new_malaria_pfpr_mod", "new_malaria_pfpr_mod_2", "new_malaria_pfpr_mod_4", "new_malaria_pfpr_mod_6",
                  "mortality_scam_mod", "incidence_scam_mod")


save(list = model_names, file = glue("{data_path}/2025_12_07_malaria_models.RData"))