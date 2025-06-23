
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

# Read in a parquet file
dengue_df <- as.data.frame(arrow::read_parquet(as_md_dengue_modeling_df_path))
dengue_df <- dengue_df[-which(dengue_df$age_group_id == 2),]

# dengue_df$A0_af <- as.factor(dengue_df$A0_location_id)
dengue_df$A0_af <- as.factor(dengue_df$A0_af)

# Translate this to R
# dengue_df["as_id"] = "a" + dengue_df["age_group_id"].astype(str) + "_s" + dengue_df["sex_id"].astype(str)
dengue_df$as_id <- paste0("a", dengue_df$age_group_id, "_s", dengue_df$sex_id)
dengue_df$as_id = as.factor(dengue_df$as_id)

min(dengue_df$dengue_mort_rate)

min(dengue_df$dengue_inc_rate)

# 
base_as_id = "a3_s1"
base_dengue_df = dengue_df[dengue_df$as_id == base_as_id,]
# rest_dengue_df = dengue_df[dengue_df$as_id != base_as_id,]
# 
# tmp_df <- base_dengue_df
# tmp_df$base_log_dengue_inc_rate <- tmp_df$log_dengue_inc_rate
# 
# rest_dengue_df = merge(rest_dengue_df, tmp_df[,c("location_id", "year_id", "base_log_dengue_inc_rate")],
#                        by = c("location_id", "year_id"), all.x = TRUE)
# 
# # 'incidence_no_income_temp_1_300': "log_dengue_inc_rate ~ dengue_suitability + total_precipitation + logit_urban_1km_threshold_300 + people_flood_days_per_capita + C(A0_af)",
tmp_df = base_dengue_df[which(base_dengue_df$dengue_mort_rate > 1e-5),]
mod_inc_base <- lm(log_dengue_inc_rate ~ dengue_suitability + total_precipitation + logit_urban_1km_threshold_300 + people_flood_days_per_capita + A0_af,
                        data = tmp_df)
coef_table <- summary(mod_inc_base)$coefficients
coef_table[!grepl("as_", rownames(coef_table)) & !grepl("A0_", rownames(coef_table)), ]
summary(mod_inc_base)$r.squared

# 'simple_plus': "log_dengue_inc_rate ~ base_as_dengue_inc_rate + C(as_id) + C(A0_af)",
mod_inc_rest <- gam(log_dengue_inc_rate ~ s(base_log_dengue_inc_rate) + as_id + A0_af,
                        data = rest_dengue_df)
#  + total_precipitation + logit_urban_1km_threshold_300 + people_flood_days_per_capita + 
tmp_df = dengue_df[which(dengue_df$dengue_inc_rate > 1e-6),]
mod_inc_all <- scam(log_dengue_inc_rate ~ s(dengue_suitability) + total_precipitation + logit_urban_1km_threshold_300 + people_flood_days_per_capita + as_id,
                  data = tmp_df)

# Get summary and filter out A0_af terms
coef_table <- summary(mod_inc_all)$coefficients
coef_table[!grepl("as_", rownames(coef_table)) & !grepl("A0_", rownames(coef_table)), ]
summary(mod_inc_base)$r.squared
# # Option 1:
# tmp_base_df <- base_dengue_df
# tmp_base_df$pred_inc = predict(mod_inc_base, tmp_base_df)
# pred_inc_base = mod_inc_base$fitted
# tmp_rest_df <- rest_dengue_df
# tmp_rest_df$base_log_dengue_inc_rate <- predict(mod_inc_base, tmp_rest_df)
# tmp_rest_df$pred_inc = predict(mod_inc_rest, tmp_rest_df)
# 
# opt_1 <- rbind(tmp_base_df[,c("pred_inc", "log_dengue_inc_rate")],
#                tmp_rest_df[,c("pred_inc", "log_dengue_inc_rate")])
# 
# cor(opt_1$pred_inc, opt_1$log_dengue_inc_rate, use = "pairwise.complete.obs")
# # plot(opt_1$log_dengue_inc_rate, opt_1$pred_inc, pch = 19, cex = 0.25, col = rgb(0,0,0,.01))
# cor(exp(opt_1$pred_inc), exp(opt_1$log_dengue_inc_rate), use = "pairwise.complete.obs")
# # plot(exp(opt_1$log_dengue_inc_rate), exp(opt_1$pred_inc), pch = 19, cex = 0.25, col = rgb(0,0,0,.01))
# 
# # Option 2:
# opt_2 = dengue_df
# opt_2$pred_inc = predict(mod_inc_all, opt_2)
# 
# cor(opt_2$pred_inc, opt_2$log_dengue_inc_rate, use = "pairwise.complete.obs")
# # plot(opt_2$log_dengue_inc_rate, opt_2$pred_inc, pch = 19, cex = 0.25, col = rgb(0,0,0,.01))
# cor(exp(opt_2$pred_inc), exp(opt_2$log_dengue_inc_rate), use = "pairwise.complete.obs")
# # plot(exp(opt_2$log_dengue_inc_rate), exp(opt_2$pred_inc), pch = 19, cex = 0.25, col = rgb(0,0,0,.01))
# 
# 
# 
# 
# mod_2_inc_rest <- gam(log_dengue_inc_rate ~ s(base_log_dengue_inc_rate, k = 10) + as_id + A0_af,
#                    data = rest_dengue_df)
# 
# summary(mod_inc_base)
# summary(mod_inc_rest)
# 
# plot(mod_2_inc_rest)






# scale = 0.00001
# dengue_df$dengue_pfpr_trimmed <- (1-scale) * dengue_df$dengue_pfpr
# 
# dengue_df$logit_dengue_pfpr = log(dengue_df$dengue_pfpr_trimmed / (1 - dengue_df$dengue_pfpr_trimmed))
# pfpr

# 
# dengue_pfpr_mod <- lm(logit_dengue_pfpr ~ dengue_suitability + log_gdppc_mean + 
#                          mal_DAH_total_per_capita + total_precipitation + people_flood_days_per_capita + A0_af,
#                        data = dengue_df)
# 
# # Get summary and filter out A0_af terms
# coef_table <- summary(dengue_pfpr_mod)$coefficients
# coef_table[!grepl("A0_af", rownames(coef_table)), ]

length(dengue_df$location_id[which(dengue_df$year_id == 2022)])
length(unique(dengue_df$location_id))


dengue_pfpr_mod_1 <- lm(logit_dengue_pfpr ~ dengue_suitability + log_gdppc_mean + 
                           mal_DAH_total_per_capita + people_flood_days_per_capita + A0_af,
                         data = dengue_df)

dengue_pfpr_mod_1 <- lm(logit_dengue_pfpr ~ dengue_suitability + log_gdppc_mean + 
                           mal_DAH_total_per_capita + total_precipitation + A0_af,
                         data = dengue_df)

# Get summary and filter out A0_af terms
coef_table <- summary(dengue_pfpr_mod_1)$coefficients
coef_table[!grepl("A0_af", rownames(coef_table)), ]

dengue_pfpr_mod_2 <- lm(logit_dengue_pfpr ~ dengue_suitability + gdppc_mean + 
                           mal_DAH_total_per_capita + people_flood_days_per_capita + A0_af,
                         data = dengue_df)

# Get summary and filter out A0_af terms
coef_table <- summary(dengue_pfpr_mod_1)$coefficients
coef_table[!grepl("A0_af", rownames(coef_table)), ]









measures = c("mortality", "incidence")
metrics = c("count", "rate")
cause = "dengue"


measure = measures[1]
metric = metrics[1]

data_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"
mortality_file_path <- glue("{data_path}/fhs_dengue_mortality_count_modeling_df_0.parquet")

# Read in a parquet file
dengue_mortality_df <- as.data.frame(arrow::read_parquet(mortality_file_path))
dengue_mortality_df <- dengue_mortality_df[which(!is.na(dengue_mortality_df$dengue_pfpr)),]
dengue_mortality_df$aa_val / dengue_mortality_df$dengue_mort_count

data_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"
incidence_file_path <- glue("{data_path}/fhs_{cause}_incidence_{metric}_modeling_df_0.parquet")

dengue_incidence_df <- as.data.frame(arrow::read_parquet(incidence_file_path))
dengue_incidence_df <- dengue_incidence_df[which(!is.na(dengue_incidence_df$dengue_pfpr)),]

dengue_mortality_df$log_risk = log(dengue_mortality_df$risk_as + 1e-6)
dengue_mortality_df$A0_af <- as.factor(dengue_mortality_df$A0_af)
dengue_mortality_df <- dengue_mortality_df[,c("location_id", "log_risk", "logit_dengue_pfpr", "log_gdppc_mean", "A0_af", "year_id")]
dengue_mortality_df <- dengue_mortality_df[complete.cases(dengue_mortality_df),]

mortality_scam_mod <- scam(log_risk ~ s(logit_dengue_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = dengue_mortality_df,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 200))  # Limit iterations

# Read in a parquet file

dengue_incidence_df$log_risk = log(dengue_incidence_df$risk_as + 1e-6)
dengue_incidence_df$A0_af <- as.factor(dengue_incidence_df$A0_af)
dengue_incidence_df <- dengue_incidence_df[,c("location_id", "log_risk", "logit_dengue_pfpr", "log_gdppc_mean", "A0_af")]
dengue_incidence_df <- dengue_incidence_df[complete.cases(dengue_incidence_df),]

incidence_scam_mod <- scam(log_risk ~ s(logit_dengue_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = dengue_incidence_df,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 200))  # Limit iterations

require(RColorBrewer)
log_gdppc_mean_seq <- seq(6.5,10, by = 0.5)
BINS <- seq(6.25, 10.25, by = 0.5)
BINS <- log(c(650, 1000, 1800, 3000,5000,8000,13500,22000))
COLS <- brewer.pal(length(log_gdppc_mean_seq), "Spectral")


layout(matrix(c(1,3,2,3),2,2,byrow = TRUE), widths = c(4,2))
par(mar = c(4.1, 0.3, 2.1, .3), oma = c(0,4.1,0,0))
############
## Mortality
############
bin_locs <- findInterval(dengue_mortality_df$log_gdppc_mean, BINS, rightmost.closed = TRUE, all.inside = TRUE)
plot(dengue_mortality_df$log_risk, mortality_scam_mod$fitted,
     col = COLS[bin_locs], pch = 16, cex = 0.75,
     xlab = "",
     ylab = "",
     main = "log mortality rate")
mtext("Observed", 1, line = 2.6, cex = 1.1)
mtext("Predicted", 2, line = 2.6, cex = 1.1)
############
## Incidence
############
bin_locs <- findInterval(dengue_incidence_df$log_gdppc_mean, BINS, rightmost.closed = TRUE, all.inside = TRUE)
plot(dengue_incidence_df$log_risk, incidence_scam_mod$fitted,
     col = COLS[bin_locs], pch = 16, cex = 0.75,
     xlab = "",
     ylab = "",
     main = "log incidence rate")
mtext("Observed", 1, line = 2.6, cex = 1.1)
mtext("Predicted", 2, line = 2.6, cex = 1.1)


plot(1, type = "n", axes = FALSE, ann = FALSE, ylim = c(-.25,1.25), xlim = c(0,1))
y_locs <- seq(0,1,length = length(BINS))[-1]
y_locs <- y_locs - diff(y_locs)[1] / 2
x_start <- 0.1
x_end <- 0.25
y_height <- diff(y_locs)[1] / 4
for (y_num in seq_along(y_locs)){
  # rect(x_start, y_locs[y_num] - y_height / 2,
  #      x_end, y_locs[y_num] + y_height / 2,
  #      col = COLS[y_num])
  points(x_start, y_locs[y_num], pch = 19, col = COLS[y_num], cex = 2.5)
  b_start <- format(exp(BINS[y_num]), big.mark = ",", scientific = FALSE)
  b_end <- format(exp(BINS[y_num+1]), big.mark = ",", scientific = FALSE)
  text(x_start+.05, y_locs[y_num], glue("{b_start} - {b_end}"), pos = 4)
}
text(-0.05, 1, "GDP per capita", pos = 4, cex = 1.2)



save.image(file = glue("{data_path}/necessary_dengue_regression_models.RData"))