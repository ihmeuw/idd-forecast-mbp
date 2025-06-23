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
as_md_malaria_modeling_df_path = glue("{MODELING_DATA_PATH}/as_md_malaria_modeling_df.parquet")
# 
# 
# file_path <- glue("{data_path}/malaria_stage_2_modeling_df.parquet")

# Read in a parquet file
malaria_df <- arrow::read_parquet(aa_md_malaria_pfpr_modeling_df_path)

# malaria_df$A0_af <- as.factor(malaria_df$A0_location_id)
malaria_df$A0_af <- as.factor(malaria_df$A0_af)

# scale = 0.00001
# malaria_df$malaria_pfpr_trimmed <- (1-scale) * malaria_df$malaria_pfpr
# 
# malaria_df$logit_malaria_pfpr = log(malaria_df$malaria_pfpr_trimmed / (1 - malaria_df$malaria_pfpr_trimmed))
# pfpr

# 
# malaria_pfpr_mod <- lm(logit_malaria_pfpr ~ malaria_suitability + log_gdppc_mean + 
#                          mal_DAH_total_per_capita + total_precipitation + people_flood_days_per_capita + A0_af,
#                        data = malaria_df)
# 
# # Get summary and filter out A0_af terms
# coef_table <- summary(malaria_pfpr_mod)$coefficients
# coef_table[!grepl("A0_af", rownames(coef_table)), ]

length(malaria_df$location_id[which(malaria_df$year_id == 2022)])
length(unique(malaria_df$location_id))
malaria_df$suit_fraction = malaria_df$malaria_suitability / 365
malaria_df$suit_fraction = pmax(pmin(malaria_df$suit_fraction, 0.999), 0.001)
malaria_df$logit_malaria_suitability = log(malaria_df$suit_fraction / (1 - malaria_df$suit_fraction))

malaria_pfpr_mod <- lm(logit_malaria_pfpr ~ malaria_suitability + log_gdppc_mean + 
                           mal_DAH_total_per_capita + people_flood_days_per_capita + A0_af,
                         data = malaria_df)


# malaria_pfpr_mod <- lm(logit_malaria_pfpr ~ logit_malaria_suitability + log_gdppc_mean + 
#                          mal_DAH_total_per_capita + people_flood_days_per_capita + A0_af,
#                        data = malaria_df)

# Get summary and filter out A0_af terms
coef_table <- summary(malaria_pfpr_mod)$coefficients
coef_table[!grepl("A0_af", rownames(coef_table)), ]

malaria_pfpr_mod_2 <- lm(logit_malaria_pfpr ~ logit_malaria_suitability + gdppc_mean +
                           mal_DAH_total_per_capita + people_flood_days_per_capita + A0_af,
                         data = malaria_df)

# Get summary and filter out A0_af terms
coef_table <- summary(malaria_pfpr_mod_2)$coefficients
coef_table[!grepl("A0_af", rownames(coef_table)), ]


cor(exp(malaria_df$logit_malaria_pfpr) / (1 + exp(malaria_df$logit_malaria_pfpr)), exp(malaria_pfpr_mod$fitted.values) / (1 + exp(malaria_pfpr_mod$fitted.values)))
# cor(exp(malaria_df$logit_malaria_pfpr) / (1 + exp(malaria_df$logit_malaria_pfpr)), exp(malaria_pfpr_mod_2$fitted.values) / (1 + exp(malaria_pfpr_mod_2$fitted.values)))


# a = exp(malaria_df$logit_malaria_pfpr) / (1 + exp(malaria_df$logit_malaria_pfpr))
# b1 = exp(malaria_pfpr_mod_1$fitted.values) / (1 + exp(malaria_pfpr_mod_1$fitted.values))
# b2 = exp(malaria_pfpr_mod_2$fitted.values) / (1 + exp(malaria_pfpr_mod_2$fitted.values))
# 
# median_a = median(a)
# low_locs = which(a < median_a)
# high_locs = which(a >= median_a)
# cor(a[low_locs], b1[low_locs])
# cor(a[high_locs], b1[high_locs])
# cor(a[low_locs], b2[low_locs])
# cor(a[high_locs], b2[high_locs])




########################################################
########################################################
####                                                ####
####    Age-sex-specific mortality and incidence    ####
####                                                ####
########################################################
########################################################

as_malaria_df <- as.data.frame(arrow::read_parquet(as_md_malaria_modeling_df_path))

reference_age_group_id = 6
reference_sex_id = 1

as_malaria_df <- as_malaria_df[which(as_malaria_df$age_group_id == reference_age_group_id & as_malaria_df$sex_id == reference_sex_id),
                               c("log_malaria_mort_rate", "log_malaria_inc_rate", "logit_malaria_pfpr", "log_gdppc_mean", "A0_af")]
as_malaria_df <- as_malaria_df[complete.cases(as_malaria_df),]

as_malaria_df <- as_malaria_df[-which(as_malaria_df$age_group_id == 2),]

as_malaria_df$A0_af <- as.factor(as_malaria_df$A0_af)

mortality_scam_mod <- scam(log_malaria_mort_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = as_malaria_df,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 200))  # Limit iterations

incidence_scam_mod <- scam(log_malaria_inc_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") + log_gdppc_mean + A0_af,
                           data = as_malaria_df,
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
bin_locs <- findInterval(as_malaria_df$log_gdppc_mean, BINS, rightmost.closed = TRUE, all.inside = TRUE)
plot(as_malaria_df$log_malaria_mort_rate, mortality_scam_mod$fitted,
     col = COLS[bin_locs], pch = 16, cex = 0.75,
     xlab = "",
     ylab = "",
     main = "log mortality rate")
mtext("Observed", 1, line = 2.6, cex = 1.1)
mtext("Predicted", 2, line = 2.6, cex = 1.1)
############
## Incidence
############
bin_locs <- findInterval(as_malaria_df$log_gdppc_mean, BINS, rightmost.closed = TRUE, all.inside = TRUE)
plot(as_malaria_df$log_malaria_inc_rate, incidence_scam_mod$fitted,
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



save.image(file = glue("{MODELING_DATA_PATH}/final_necessary_malaria_regression_models.RData"))
