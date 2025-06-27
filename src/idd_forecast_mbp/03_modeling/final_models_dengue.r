
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


mod_inc_base <- lm(base_log_dengue_inc_rate ~  dengue_suitability + relative_humidity + logit_urban_1km_threshold_300 + people_flood_days_per_capita +A0_af,
                     data = base_dengue_df) 

summary(mod_inc_base)


# 'incidence_no_income_temp_1_300': "log_dengue_inc_rate ~ dengue_suitability + total_precipitation + logit_urban_1km_threshold_300 + people_flood_days_per_capita + C(c)",

# mod_inc_base <- scam(base_log_dengue_inc_rate ~  s(dengue_suitability, k = 6, bs = 'mpi') + s(relative_humidity, k = 6, bs = 'mpi')  + people_flood_days_per_capita,
#                         data = base_dengue_df)  


mod_inc_rest <- lm(log_dengue_inc_rate ~ base_log_dengue_inc_rate + log_gdppc_mean + as_id+A0_af,
                   data = rest_dengue_df)



mod_cfr_all <- lm(logit_dengue_cfr ~ log_gdppc_mean + as_id + A0_af,
                  data = as_dengue_df)

summary(mod_cfr_all)

mod_cfr_base <- lm(base_logit_dengue_cfr ~ log_gdppc_mean + A0_af,
                   data = base_dengue_df)

mod_cfr_rest <- lm(logit_dengue_cfr ~ base_logit_dengue_cfr + as_id + A0_af,
                   data = rest_dengue_df)

summary(mod_cfr_base)
summary(mod_cfr_rest)




save.image(file = glue("{MODELING_DATA_PATH}/final_dengue_regression_models.RData"))
