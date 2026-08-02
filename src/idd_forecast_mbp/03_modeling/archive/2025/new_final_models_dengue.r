
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

as_md_dengue_modeling_df_path = glue("{MODELING_DATA_PATH}/as_md_dengue_modeling_df.parquet")
base_md_dengue_modeling_df_path = glue("{MODELING_DATA_PATH}/base_md_dengue_modeling_df.parquet")
rest_md_dengue_modeling_df_path = glue("{MODELING_DATA_PATH}/rest_md_dengue_modeling_df.parquet")

# Read in a parquet file
#

as_dengue_df = as.data.frame(arrow::read_parquet(as_md_dengue_modeling_df_path))
as_dengue_df$A0_af <- as.factor(as_dengue_df$A0_af)
as_dengue_df$as_id = as.factor(as_dengue_df$as_id)

base_dengue_df = as.data.frame(arrow::read_parquet(base_md_dengue_modeling_df_path))
base_dengue_df$A0_af <- as.factor(base_dengue_df$A0_af)
base_dengue_df$as_id = as.factor(base_dengue_df$as_id)

rest_dengue_df = as.data.frame(arrow::read_parquet(rest_md_dengue_modeling_df_path))
rest_dengue_df$A0_af <- as.factor(rest_dengue_df$A0_af)
rest_dengue_df$as_id = as.factor(rest_dengue_df$as_id)

base_dengue_df$dengue_suit_fraction <- base_dengue_df$dengue_suitability / 365
base_dengue_df$dengue_suit_fraction <- pmin(pmax(base_dengue_df$dengue_suit_fraction, 0.001), 0.999)
base_dengue_df$logit_dengue_suitability <- log(base_dengue_df$dengue_suit_fraction / (1 - base_dengue_df$dengue_suit_fraction))

mod_inc_base <- scam(base_log_dengue_inc_rate ~  s(logit_dengue_suitability, k = 6, bs = 'mpi') + 
                       s(logit_urban_1km_threshold_300, k = 6, bs = 'mpi') + 
                       people_flood_days_per_capita +A0_af,
                     data = base_dengue_df,
                     optimizer = "efs",
                     control = list(maxit = 300))  # Limit iterations 



mod_cfr_all <- lm(logit_dengue_cfr ~ log_gdppc_mean + as_id + A0_af,
                  data = as_dengue_df)



model_names <- c("mod_inc_base", "mod_cfr_all")

save(list = model_names, file = glue("{MODELING_DATA_PATH}/2025_06_29_dengue_models.RData"))







base_dengue_df$urban_1km_threshold_300 = invlogit(base_dengue_df$logit_urban_1km_threshold_300)


mod_inc_base1 <- scam(base_log_dengue_inc_rate ~  s(logit_dengue_suitability, k = 6, bs = 'mpi') + 
                       s(logit_urban_1km_threshold_300, k = 6, bs = 'mpi') + A0_af,
                     data = base_dengue_df,
                     optimizer = "efs",
                     control = list(maxit = 300))  # Limit iterations 

mod_inc_base2 <- scam(base_log_dengue_inc_rate ~  s(dengue_suitability, k = 6, bs = 'mpi') + 
                        s(urban_1km_threshold_300, k = 6, bs = 'mpi') + A0_af,
                      data = base_dengue_df,
                      optimizer = "efs",
                      control = list(maxit = 300))  # Limit iterations 



