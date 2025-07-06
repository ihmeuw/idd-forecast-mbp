
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
as_dengue_df <- as_dengue_df[which(as_dengue_df$dengue_mort_rate > 0),]
range(as_dengue_df$logit_dengue_cfr)
as_dengue_df$A0_af <- as.factor(as_dengue_df$A0_af)
as_dengue_df$as_id = as.factor(as_dengue_df$as_id)

base_dengue_df = as.data.frame(arrow::read_parquet(base_md_dengue_modeling_df_path))
base_dengue_df$A0_af <- as.factor(base_dengue_df$A0_af)
base_dengue_df$as_id = as.factor(base_dengue_df$as_id)

# load(glue("{MODELING_DATA_PATH}/2025_06_29_dengue_models.RData"))


rest_dengue_df = as.data.frame(arrow::read_parquet(rest_md_dengue_modeling_df_path))
rest_dengue_df$A0_af <- as.factor(rest_dengue_df$A0_af)
rest_dengue_df$as_id = as.factor(rest_dengue_df$as_id)

base_dengue_df$dengue_suit_fraction <- base_dengue_df$dengue_suitability / 365
base_dengue_df$dengue_suit_fraction <- pmin(pmax(base_dengue_df$dengue_suit_fraction, 0.001), 0.999)
base_dengue_df$logit_dengue_suitability <- log(base_dengue_df$dengue_suit_fraction / (1 - base_dengue_df$dengue_suit_fraction))


mod_inc_base <- scam(base_log_dengue_inc_rate ~  s(logit_dengue_suitability, k = 6, bs = 'mpi') + 
                       s(logit_urban_1km_threshold_300, k = 6, bs = 'mpi') + A0_af,
                     data = base_dengue_df,
                     optimizer = "efs",
                     control = list(maxit = 300))  # Limit iterations 



mod_cfr_all <- lm(logit_dengue_cfr ~ log_gdppc_mean + as_id + A0_af,
                  data = as_dengue_df)


write.csv(levels(as_dengue_df$as_id), file = glue("{MODELING_DATA_PATH}/as_id_levels.csv"), row.names = FALSE)
write.csv(coef(mod_cfr_all), file = glue("{MODELING_DATA_PATH}/mod_cfr_all_coefficients.csv"), row.names = TRUE)

strip_model <- function(model) {
  model$model <- NULL
  model$fitted.values <- NULL
  model$residuals <- NULL
  model$effects <- NULL
  model$qr$qr <- NULL
  model$linear.predictors <- NULL
  model$weights <- NULL
  model$prior.weights <- NULL
  model$data <- NULL
  model$family <- NULL
  model$deviance <- NULL
  model$aic <- NULL
  model$null.deviance <- NULL
  model$iter <- NULL
  model$df.residual <- NULL
  model$df.null <- NULL
  model$y <- NULL
  model$converged <- NULL
  model$boundary <- NULL
  
  # Keep essential components for prediction
  attr(model$terms, ".Environment") <- NULL
  
  return(model)
}

# Usage
stripped_mod_cfr_all <- strip_model(mod_cfr_all)

model_names <- c("mod_inc_base", "mod_cfr_all")

save(list = model_names, file = glue("{MODELING_DATA_PATH}/2025_06_29_dengue_models.RData"))
stripped_model_names <- c("mod_inc_base", "stripped_mod_cfr_all")
save(list = model_names, file = glue("{MODELING_DATA_PATH}/2025_06_29_dengue_models_stripped.RData"))







