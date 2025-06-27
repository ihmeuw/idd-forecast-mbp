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
                           data = base_malaria_df,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 200))  # Limit iterations

incidence_scam_mod <- scam(base_log_malaria_inc_rate ~ s(base_logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = base_malaria_df,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 200))  # Limit iterations


save.image(file = glue("{MODELING_DATA_PATH}/final_malaria_regression_models.RData"))
