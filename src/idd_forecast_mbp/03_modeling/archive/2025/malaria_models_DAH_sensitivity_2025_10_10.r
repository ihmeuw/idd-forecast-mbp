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

alt_malaria_pfpr_mod <- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
                               s(gdppc_mean, k = 6, bs = 'mpd') + 
                               people_flood_days_per_capita + 
                               A0_af,
                             data = past_data,
                             optimizer = "efs",      # Faster optimizer
                             control = list(maxit = 300))  # Limit iterations

invlogit = function(x) {
  exp(x) / (1 + exp(x))
}

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


past_data$logit_malaria_pfpr_pred <- predict(malaria_pfpr_mod, newdata = past_data)
past_data$logit_malaria_pfpr_alt_pred <- predict(alt_malaria_pfpr_mod, newdata = past_data)

pred_data = past_data
pred_data$logit_malaria_pfpr = past_data$logit_malaria_pfpr_pred
pred_data$log_aa_malaria_mort_rate_pred <- predict(mortality_scam_mod, newdata = pred_data)
pred_data$aa_malaria_mort_rate_pred <- exp(pred_data$log_aa_malaria_mort_rate_pred)
pred_data$aa_malaria_mort_count_pred <- pred_data$aa_malaria_mort_rate_pred * pred_data$aa_population
pred_data$log_aa_malaria_inc_rate_pred <- predict(incidence_scam_mod, newdata = pred_data)
pred_data$aa_malaria_inc_rate_pred <- exp(pred_data$log_aa_malaria_inc_rate_pred)
pred_data$aa_malaria_inc_count_pred <- pred_data$aa_malaria_inc_rate_pred * pred_data$aa_population


alt_pred_data = past_data
alt_pred_data$logit_malaria_pfpr = past_data$logit_malaria_pfpr_alt_pred
alt_pred_data$log_aa_malaria_mort_rate_pred <- predict(mortality_scam_mod, newdata = alt_pred_data)
alt_pred_data$aa_malaria_mort_rate_pred <- exp(alt_pred_data$log_aa_malaria_mort_rate_pred)
alt_pred_data$aa_malaria_mort_count_pred <- alt_pred_data$aa_malaria_mort_rate_pred * alt_pred_data$aa_population
alt_pred_data$log_aa_malaria_inc_rate_pred <- predict(incidence_scam_mod, newdata = alt_pred_data)
alt_pred_data$aa_malaria_inc_rate_pred <- exp(alt_pred_data$log_aa_malaria_inc_rate_pred)
alt_pred_data$aa_malaria_inc_count_pred <- alt_pred_data$aa_malaria_inc_rate_pred * alt_pred_data$aa_population

pred_data$adjustment_mort = pred_data$adjustment_inc = 0
alt_pred_data$adjustment_mort = alt_pred_data$adjustment_inc = 0
ULocs <- unique(pred_data$location_id)
for (unum in seq_along(ULocs)){
  tmp_locs <- which(pred_data$location_id == ULocs[unum])
  last_year_loc <- which(pred_data$location_id == ULocs[unum] & pred_data$year_id == max(pred_data$year_id))
  if (length(last_year_loc) == 0){
    pred_data$adjustment_mort[tmp_locs] <- 0
    pred_data$adjustment_inc[tmp_locs] <- 0
    alt_pred_data$adjustment_mort[tmp_locs] <- 0
    alt_pred_data$adjustment_inc[tmp_locs] <- 0
  } else {
    adj_mort <- pred_data$aa_malaria_mort_count[last_year_loc] / pred_data$aa_malaria_mort_count_pred[last_year_loc]
    adj_inc <- pred_data$aa_malaria_inc_count[last_year_loc] / pred_data$aa_malaria_inc_count_pred[last_year_loc]
    pred_data$adjustment_mort[tmp_locs] <- adj_mort
    pred_data$adjustment_inc[tmp_locs] <- adj_inc
    
    alt_adj_mort <- alt_pred_data$aa_malaria_mort_count[last_year_loc] / alt_pred_data$aa_malaria_mort_count_pred[last_year_loc]
    alt_adj_inc <- alt_pred_data$aa_malaria_inc_count[last_year_loc] / alt_pred_data$aa_malaria_inc_count_pred[last_year_loc]
    alt_pred_data$adjustment_mort[tmp_locs] <- alt_adj_mort
    alt_pred_data$adjustment_inc[tmp_locs] <- alt_adj_inc
  }
}

pred_data$adjusted_aa_malaria_mort_count_pred <- pred_data$aa_malaria_mort_count_pred * pred_data$adjustment_mort
pred_data$adjusted_aa_malaria_inc_count_pred <- pred_data$aa_malaria_inc_count_pred * pred_data$adjustment_inc
alt_pred_data$adjusted_aa_malaria_mort_count_pred <- alt_pred_data$aa_malaria_mort_count_pred * alt_pred_data$adjustment_mort
alt_pred_data$adjusted_aa_malaria_inc_count_pred <- alt_pred_data$aa_malaria_inc_count_pred * alt_pred_data$adjustment_inc

global_pred_data <- aggregate(cbind(aa_malaria_mort_count, aa_malaria_mort_count_pred, aa_malaria_inc_count, adjusted_aa_malaria_inc_count_pred,
                                    aa_malaria_inc_count_pred, adjusted_aa_malaria_mort_count_pred) ~ year_id, data = pred_data, sum)
global_alt_pred_data <- aggregate(cbind(aa_malaria_mort_count, aa_malaria_mort_count_pred, aa_malaria_inc_count, adjusted_aa_malaria_inc_count_pred,
                                        aa_malaria_inc_count_pred, adjusted_aa_malaria_mort_count_pred) ~ year_id, data = alt_pred_data, sum)

dir = '/mnt/team/idd/pub/forecast-mbp/10-presentation_material/malaria_pst_2025'
pdf(file=glue("{dir}/fit.pdf"), height = 5, width = 6)
mult = 100000
pretty_mult = '(in 100,000s)'
par(mfrow=c(2,2),
    mar=c(5.1,5.1,1.1,0.6))
plot(global_alt_pred_data$year_id, global_alt_pred_data$aa_malaria_mort_count/mult, type = 'l', 
     col = 1, lwd = 2, ylab = "", xlab = "",
     ylim=c(0, 1.1*max(global_pred_data$aa_malaria_mort_count/mult)))
lines(global_alt_pred_data$year_id, global_alt_pred_data$adjusted_aa_malaria_mort_count_pred/mult, type = 'l', col = 'blue', lwd = 2)
mtext(glue("Global malaria deaths
{pretty_mult}"), 2, line = 2.1)
mtext(glue("Year"), 1, line = 2.6)

plot(global_pred_data$year_id, global_pred_data$aa_malaria_mort_count/mult, 
     type = 'l', col = 1, lwd = 2, ylab = "", xlab = "",
     ylim=c(0, 1.1*max(global_pred_data$aa_malaria_mort_count/mult)))
lines(global_pred_data$year_id, global_pred_data$adjusted_aa_malaria_mort_count_pred/mult, type = 'l', col = 'blue', lwd = 2)

mtext(glue("Global malaria deaths
{pretty_mult}"), 2, line = 2.1)
mtext(glue("Year"), 1, line = 2.6)

YLIM=XLIM=range(c(global_alt_pred_data$aa_malaria_mort_count, 
                  global_alt_pred_data$adjusted_aa_malaria_mort_count_pred,
                  global_pred_data$aa_malaria_mort_count, 
                  global_pred_data$adjusted_aa_malaria_mort_count_pred))
YLIM=YLIM/mult
XLIM=XLIM/mult

plot(global_alt_pred_data$aa_malaria_mort_count/mult, global_alt_pred_data$adjusted_aa_malaria_mort_count_pred/mult, 
     xlab = "",
     ylab = "", pch = 16, col = 1, xlim=XLIM, ylim=YLIM)
mtext(glue("Predicted malaria deaths
{pretty_mult}"), 2, line = 2.1)
mtext(glue("Observed malaria deaths
{pretty_mult}"), 1, line = 3.6)
plot(global_pred_data$aa_malaria_mort_count/mult, global_pred_data$adjusted_aa_malaria_mort_count_pred/mult, 
     xlab = "",
     ylab = "", pch = 16, col = 1, xlim=XLIM, ylim=YLIM)
mtext(glue("Predicted malaria deaths
{pretty_mult}"), 2, line = 2.1)
mtext(glue("Observed malaria deaths
{pretty_mult}"), 1, line = 3.6)
dev.off()


cor(global_alt_pred_data$aa_malaria_mort_count, global_alt_pred_data$adjusted_aa_malaria_mort_count_pred)
cor(global_pred_data$aa_malaria_mort_count, global_pred_data$adjusted_aa_malaria_mort_count_pred)

cor(global_alt_pred_data$aa_malaria_inc_count, global_alt_pred_data$adjusted_aa_malaria_inc_count_pred)
cor(global_pred_data$aa_malaria_inc_count, global_pred_data$adjusted_aa_malaria_inc_count_pred)


model_names <- c("malaria_pfpr_mod", "alt_malaria_pfpr_mod", "mortality_scam_mod", "incidence_scam_mod")


save(list = model_names, file = glue("{data_path}/2025_10_10_malaria_models.RData"))

# save alt_pred_data, pred_data, global_alt_pred_data, and global_pred_data as parquet to data_path
arrow::write_parquet(as.data.table(alt_pred_data), glue("{data_path}/2025_10_10_alt_pred_data.parquet"))
arror::write_parquet(as.data.table(pred_data), glue("{data_path}/2025_10_10_pred_data.parquet"))
arrow::write_parquet(as.data.table(global_alt_pred_data), glue("{data_path}/2025_10_10_global_alt_pred_data.parquet"))
arrow::write_parquet(as.data.table(global_pred_data), glue("{data_path}/2025_10_10_global_pred_data.parquet"))

########################
########################
########################

REPO_DIR = "/mnt/team/idd/pub/forecast-mbp"
last_year <- 2022
data_path <- glue("{REPO_DIR}/03-modeling_data")
FORECASTING_DATA_PATH = glue("{REPO_DIR}/04-forecasting_data")
PROCESSED_DATA_PATH = glue("{REPO_DIR}/02-processed_data")
hierarchy_df_path = glue('{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet')
hierarchy_df <- as.data.frame(arrow::read_parquet(hierarchy_df_path))

figure_dir = '/mnt/team/idd/pub/forecast-mbp/10-presentation_material/malaria_pst_2025'

large_row = which.max(past_data$logit_malaria_pfpr[past_data$year_id == last_year])
location_id = past_data$location_id[large_row]
parent_id = hierarchy_df$parent_id[which(hierarchy_df$location_id == location_id)[1]]
A0_location_id = past_data$A0_location_id[large_row]
location_name = hierarchy_df$location_name[which(hierarchy_df$location_id == location_id)[1]]
state_name = hierarchy_df$location_name[which(hierarchy_df$location_id == parent_id)[1]]
country_name = hierarchy_df$location_name[which(hierarchy_df$location_id == A0_location_id)[1]]

gdp_vec = seq(0, 20000, length = 1001)
dah_vec = seq(0, 10, length = 1001)

alt_gdp_data <- alt_pred_data[rep(large_row,length(gdp_vec)),c("logit_malaria_suitability", "gdppc_mean", "people_flood_days_per_capita",
                                                       "A0_af")]


alt_gdp_data$gdppc_mean <- gdp_vec


pred_mat <- predict(alt_malaria_pfpr_mod, newdata = alt_gdp_data, se = TRUE)

draw_mat <- ilogit_draw_mat <- rel_pred_mat <- matrix(NA, nrow = nrow(alt_gdp_data), ncol = 1000)
for (d_num in seq_len(1000)){
  tmp_z = rnorm(1, 0, 1)
  draw_mat[,d_num] = pred_mat$fit + tmp_z * pred_mat$se.fit
  ilogit_draw_mat[,d_num] = invlogit(draw_mat[,d_num])
  rel_pred_mat[,d_num] = 100 * (ilogit_draw_mat[,d_num] / min(ilogit_draw_mat[,d_num]) - 1)
  tmp_z <- rnorm(nrow(alt_gdp_data), 0, 1)
  draw_mat[,d_num] = pred_mat$fit + tmp_z * pred_mat$se.fit
  ilogit_draw_mat[,d_num] = invlogit(draw_mat[,d_num])
}

alt_gdp_data$rel_pred <- rowMeans(rel_pred_mat)
alt_gdp_data$rel_pred_lower <- apply(rel_pred_mat, 1, quantile, probs = 0.025)
alt_gdp_data$rel_pred_upper <- apply(rel_pred_mat, 1, quantile, probs = 0.975)

alt_gdp_data$pred <- pred_mat$fit
alt_gdp_data$pred_lower <- pred_mat$fit - 1.96 * pred_mat$se.fit
alt_gdp_data$pred_upper <- pred_mat$fit + 1.96 * pred_mat$se.fit
alt_gdp_data$ilogit_pred <- invlogit(alt_gdp_data$pred)
alt_gdp_data$ilogit_pred_lower <- invlogit(alt_gdp_data$pred_lower)
alt_gdp_data$ilogit_pred_upper <- invlogit(alt_gdp_data$pred_upper)
alt_gdp_data$rel_pred_2 <- 100 * (alt_gdp_data$ilogit_pred / min(alt_gdp_data$ilogit_pred) - 1)
alt_gdp_data$rel_pred_lower_2 <- 100 * (alt_gdp_data$ilogit_pred_lower / min(alt_gdp_data$ilogit_pred_lower) - 1)
alt_gdp_data$rel_pred_upper_2 <- 100 * (alt_gdp_data$ilogit_pred_upper / min(alt_gdp_data$ilogit_pred_upper) - 1)


TCL = -1.75
MGP = c(0,-TCL,0)
subtitle_cex = 1.1
title_cex = 1.5

pdf(file = glue("{figure_dir}/alt_model_malaria_regression_scam_gdp_and_dah_vs_pfpr.pdf"), width = 5, height = 5)
par(oma=c(0,0,2,0))
YLIM <- c(min(alt_gdp_data$ilogit_pred_lower * 100), max(alt_gdp_data$ilogit_pred_upper * 100))
plot(alt_gdp_data$gdppc_mean, alt_gdp_data$ilogit_pred, type = 'n',
     axes = FALSE,
     xlab = "GDP per capita (2020 USD)", 
     ylab = "Predicted PfPR",
     ylim = YLIM)
box()
axis(2, pretty(YLIM))
axis(1, pretty(gdp_vec), paste0("$", pretty(gdp_vec)))

polygon(c(alt_gdp_data$gdppc_mean, rev(alt_gdp_data$gdppc_mean)), 
        c(alt_gdp_data$ilogit_pred_lower * 100, rev(alt_gdp_data$ilogit_pred_upper * 100)),
        col = rgb(0.5, 0.5, 0.8, alpha = 0.4),  # Light blue-gray
        border = NA)

lines(alt_gdp_data$gdppc_mean, alt_gdp_data$ilogit_pred * 100, col = "#2166ac", lwd = 3)
obs_dah = past_data$gdppc_mean[large_row]
alt_gdp_data_loc = which.min((obs_dah - alt_gdp_data$gdppc_mean)^2)
tmp_locs = alt_gdp_data_loc + -2:2
polygon(c(alt_gdp_data$gdppc_mean[tmp_locs], rev(alt_gdp_data$gdppc_mean[tmp_locs])),
        c(alt_gdp_data$ilogit_pred_lower[tmp_locs] * 100, rev(alt_gdp_data$ilogit_pred_upper[tmp_locs] * 100)),
        col = rgb(0.8, 0.5, 0.5, alpha = 0.4),  # Light red
        border = NA)
lines(alt_gdp_data$gdppc_mean[tmp_locs], alt_gdp_data$ilogit_pred[tmp_locs] * 100, col = "#b2182b", lwd = 3)
lines(rep(obs_dah, 2), c(0, alt_gdp_data$ilogit_pred_lower[alt_gdp_data_loc] * 100), col = "#b2182b", lty = 2)
axis(1, obs_dah, glue("${round(obs_dah,2)}"), tcl = TCL, col = "#b2182b",
     mgp = MGP)

#### ADDING LEGEND

legend(0.2 * max(gdp_vec), 0.975 * YLIM[2], title = glue('Example Location:
{location_name}, 
        {state_name}, {country_name}'),
       legend = c("Mean prediction", "95% CI"), lwd = c(3, NA), 
       title.adj = 0,
       pch = c(NA, 15), pt.cex = 2, 
       col = c("#2166ac", rgb(0.5, 0.5, 0.8, alpha = 0.4)), bty = 'n')

mtext(glue("GDP per capita vs PfPR"), side = 3, cex = subtitle_cex)
dev.off()






# logit_malaria_pfpr

gdp_data <- past_data[rep(large_row,length(gdp_vec)),c("logit_malaria_suitability", "gdppc_mean", 
                                                       "mal_DAH_total_per_capita", "people_flood_days_per_capita",
                                                       "A0_af")]
gdp_data$gdppc_mean <- gdp_vec



pred_mat <- predict(malaria_pfpr_mod, newdata = gdp_data, se = TRUE)

draw_mat <- ilogit_draw_mat <- rel_pred_mat <- matrix(NA, nrow = nrow(gdp_data), ncol = 1000)
for (d_num in seq_len(1000)){
  tmp_z = rnorm(1, 0, 1)
  draw_mat[,d_num] = pred_mat$fit + tmp_z * pred_mat$se.fit
  ilogit_draw_mat[,d_num] = invlogit(draw_mat[,d_num])
  rel_pred_mat[,d_num] = 100 * (ilogit_draw_mat[,d_num] / min(ilogit_draw_mat[,d_num]) - 1)
  tmp_z <- rnorm(nrow(gdp_data), 0, 1)
  draw_mat[,d_num] = pred_mat$fit + tmp_z * pred_mat$se.fit
  ilogit_draw_mat[,d_num] = invlogit(draw_mat[,d_num])
}

gdp_data$rel_pred <- rowMeans(rel_pred_mat)
gdp_data$rel_pred_lower <- apply(rel_pred_mat, 1, quantile, probs = 0.025)
gdp_data$rel_pred_upper <- apply(rel_pred_mat, 1, quantile, probs = 0.975)

gdp_data$pred <- pred_mat$fit
gdp_data$pred_lower <- pred_mat$fit - 1.96 * pred_mat$se.fit
gdp_data$pred_upper <- pred_mat$fit + 1.96 * pred_mat$se.fit
gdp_data$ilogit_pred <- invlogit(gdp_data$pred)
gdp_data$ilogit_pred_lower <- invlogit(gdp_data$pred_lower)
gdp_data$ilogit_pred_upper <- invlogit(gdp_data$pred_upper)
gdp_data$rel_pred_2 <- 100 * (gdp_data$ilogit_pred / min(gdp_data$ilogit_pred) - 1)
gdp_data$rel_pred_lower_2 <- 100 * (gdp_data$ilogit_pred_lower / min(gdp_data$ilogit_pred_lower) - 1)
gdp_data$rel_pred_upper_2 <- 100 * (gdp_data$ilogit_pred_upper / min(gdp_data$ilogit_pred_upper) - 1)



dah_data <- past_data[rep(large_row,length(dah_vec)),c("logit_malaria_suitability", "gdppc_mean", 
                                                       "mal_DAH_total_per_capita", "people_flood_days_per_capita",
                                                       "A0_af")]

dah_data$mal_DAH_total_per_capita <- dah_vec



pred_mat <- predict(malaria_pfpr_mod, newdata = dah_data, se = TRUE)

draw_mat <- ilogit_draw_mat <- rel_pred_mat <- matrix(NA, nrow = nrow(dah_data), ncol = 1000)
for (d_num in seq_len(1000)){
  tmp_z = rnorm(1, 0, 1)
  draw_mat[,d_num] = pred_mat$fit + tmp_z * pred_mat$se.fit
  ilogit_draw_mat[,d_num] = invlogit(draw_mat[,d_num])
  rel_pred_mat[,d_num] = 100 * (ilogit_draw_mat[,d_num] / min(ilogit_draw_mat[,d_num]) - 1)
  tmp_z <- rnorm(nrow(dah_data), 0, 1)
  draw_mat[,d_num] = pred_mat$fit + tmp_z * pred_mat$se.fit
  ilogit_draw_mat[,d_num] = invlogit(draw_mat[,d_num])
}

dah_data$rel_pred <- rowMeans(rel_pred_mat)
dah_data$rel_pred_lower <- apply(rel_pred_mat, 1, quantile, probs = 0.025)
dah_data$rel_pred_upper <- apply(rel_pred_mat, 1, quantile, probs = 0.975)

dah_data$pred <- pred_mat$fit
dah_data$pred_lower <- pred_mat$fit - 1.96 * pred_mat$se.fit
dah_data$pred_upper <- pred_mat$fit + 1.96 * pred_mat$se.fit
dah_data$ilogit_pred <- invlogit(dah_data$pred)
dah_data$ilogit_pred_lower <- invlogit(dah_data$pred_lower)
dah_data$ilogit_pred_upper <- invlogit(dah_data$pred_upper)
dah_data$rel_pred_2 <- 100 * (dah_data$ilogit_pred / min(dah_data$ilogit_pred) - 1)
dah_data$rel_pred_lower_2 <- 100 * (dah_data$ilogit_pred_lower / min(dah_data$ilogit_pred_lower) - 1)
dah_data$rel_pred_upper_2 <- 100 * (dah_data$ilogit_pred_upper / min(dah_data$ilogit_pred_upper) - 1)




TCL = -1.75
MGP = c(0,-TCL,0)
subtitle_cex = 1.1
title_cex = 1.5

pdf(file = glue("{figure_dir}/SI_Figure_malaria_regression_scam_gdp_and_dah_vs_pfpr.pdf"), width = 10, height = 10)
layout(matrix(c(1,2,3,4), 2, 2, byrow = FALSE))
par(oma=c(0,0,2,0))
YLIM <- c(min(gdp_data$ilogit_pred_lower * 100), max(gdp_data$ilogit_pred_upper * 100))
plot(gdp_data$gdppc_mean, gdp_data$ilogit_pred, type = 'n',
     axes = FALSE,
     xlab = "GDP per capita (2020 USD)", 
     ylab = "Predicted PfPR",
     ylim = YLIM)
box()
axis(2, pretty(YLIM))
axis(1, pretty(gdp_vec), paste0("$", pretty(gdp_vec)))

polygon(c(gdp_data$gdppc_mean, rev(gdp_data$gdppc_mean)), 
        c(gdp_data$ilogit_pred_lower * 100, rev(gdp_data$ilogit_pred_upper * 100)),
        col = rgb(0.5, 0.5, 0.8, alpha = 0.4),  # Light blue-gray
        border = NA)

lines(gdp_data$gdppc_mean, gdp_data$ilogit_pred * 100, col = "#2166ac", lwd = 3)
obs_dah = past_data$gdppc_mean[large_row]
gdp_data_loc = which.min((obs_dah - gdp_data$gdppc_mean)^2)
tmp_locs = gdp_data_loc + -2:2
polygon(c(gdp_data$gdppc_mean[tmp_locs], rev(gdp_data$gdppc_mean[tmp_locs])),
        c(gdp_data$ilogit_pred_lower[tmp_locs] * 100, rev(gdp_data$ilogit_pred_upper[tmp_locs] * 100)),
        col = rgb(0.8, 0.5, 0.5, alpha = 0.4),  # Light red
        border = NA)
lines(gdp_data$gdppc_mean[tmp_locs], gdp_data$ilogit_pred[tmp_locs] * 100, col = "#b2182b", lwd = 3)
lines(rep(obs_dah, 2), c(0, gdp_data$ilogit_pred_lower[gdp_data_loc] * 100), col = "#b2182b", lty = 2)
axis(1, obs_dah, glue("${round(obs_dah,2)}"), tcl = TCL, col = "#b2182b",
     mgp = MGP)

#### ADDING LEGEND

legend(0.4 * max(gdp_vec), 0.975 * YLIM[2], title = glue('Example Location:
{location_name}, 
        {state_name}, {country_name}'),
       legend = c("Mean prediction", "95% CI"), lwd = c(3, NA), 
       title.adj = 0,
       pch = c(NA, 15), pt.cex = 2, 
       col = c("#2166ac", rgb(0.5, 0.5, 0.8, alpha = 0.4)), bty = 'n')

mtext(glue("GDP per capita vs PfPR"), side = 3, cex = subtitle_cex)


YLIM = c(0, max(gdp_data$rel_pred_upper))

plot(gdp_data$gdppc_mean, gdp_data$rel_pred_upper, type = 'n',
     axes = FALSE,
     xlab = "GDP per capita per capita (2020 USD)", 
     ylab = "Percent increase in PfPR",
     ylim = YLIM)

box()
axis(2, pretty(YLIM))
axis(1, pretty(gdp_vec), paste0("$", pretty(gdp_vec)))

polygon(c(gdp_data$gdppc_mean, rev(gdp_data$gdppc_mean)), 
        c(gdp_data$rel_pred_lower, rev(gdp_data$rel_pred_upper)),
        col = rgb(0.5, 0.5, 0.8, alpha = 0.4),  # Light blue-gray
        border = NA)

lines(gdp_data$gdppc_mean, gdp_data$rel_pred, col = "#2166ac", lwd = 3)
mtext(glue("Relative increase in PfPR
$20,000 reference category"), side = 3, cex = subtitle_cex)
legend('topright',
       legend = c("Mean prediction", "95% CI"), lwd = c(3, NA), 
       title.adj = 0,
       pch = c(NA, 15), pt.cex = 2, 
       col = c("#2166ac", rgb(0.5, 0.5, 0.8, alpha = 0.4)), bty = 'n')

##
## DAH
##
YLIM <- c(min(dah_data$ilogit_pred_lower * 100), max(dah_data$ilogit_pred_upper * 100))
plot(dah_data$mal_DAH_total_per_capita, dah_data$ilogit_pred, type = 'n',
     axes = FALSE,
     xlab = "Malaria DAH per capita (2020 USD)", 
     ylab = "Predicted PfPR",
     ylim = YLIM)
box()
axis(2, pretty(YLIM))
axis(1, pretty(dah_vec), paste0("$", pretty(dah_vec)))

polygon(c(dah_data$mal_DAH_total_per_capita, rev(dah_data$mal_DAH_total_per_capita)), 
        c(dah_data$ilogit_pred_lower * 100, rev(dah_data$ilogit_pred_upper * 100)),
        col = rgb(0.5, 0.5, 0.8, alpha = 0.4),  # Light blue-gray
        border = NA)

lines(dah_data$mal_DAH_total_per_capita, dah_data$ilogit_pred * 100, col = "#2166ac", lwd = 3)
obs_dah = past_data$mal_DAH_total_per_capita[large_row]
dah_data_loc = which.min((obs_dah - dah_data$mal_DAH_total_per_capita)^2)
tmp_locs = dah_data_loc + -2:2
polygon(c(dah_data$mal_DAH_total_per_capita[tmp_locs], rev(dah_data$mal_DAH_total_per_capita[tmp_locs])),
        c(dah_data$ilogit_pred_lower[tmp_locs] * 100, rev(dah_data$ilogit_pred_upper[tmp_locs] * 100)),
        col = rgb(0.8, 0.5, 0.5, alpha = 0.4),  # Light red
        border = NA)
lines(dah_data$mal_DAH_total_per_capita[tmp_locs], dah_data$ilogit_pred[tmp_locs] * 100, col = "#b2182b", lwd = 3)
lines(rep(obs_dah, 2), c(0, dah_data$ilogit_pred_lower[dah_data_loc] * 100), col = "#b2182b", lty = 2)
axis(1, obs_dah, glue("${round(obs_dah,2)}"), tcl = TCL, col = "#b2182b",
     mgp = MGP)

#### ADDING LEGEND

legend(0.4 * max(dah_vec), 0.975 * YLIM[2], title = glue('Example Location:
{location_name}, 
        {state_name}, {country_name}'),
       legend = c("Mean prediction", "95% CI"), lwd = c(3, NA), 
       title.adj = 0,
       pch = c(NA, 15), pt.cex = 2, 
       col = c("#2166ac", rgb(0.5, 0.5, 0.8, alpha = 0.4)), bty = 'n')

mtext(glue("DAH per capita vs PfPR"), side = 3, cex = subtitle_cex)


YLIM = c(0, max(dah_data$rel_pred_upper))

plot(dah_data$mal_DAH_total_per_capita, dah_data$rel_pred_upper, type = 'n',
     axes = FALSE,
     xlab = "Malaria DAH per capita (2020 USD)", 
     ylab = "Percent increase in PfPR",
     ylim = YLIM)

box()
axis(2, pretty(YLIM))
axis(1, pretty(dah_vec), paste0("$", pretty(dah_vec)))

polygon(c(dah_data$mal_DAH_total_per_capita, rev(dah_data$mal_DAH_total_per_capita)), 
        c(dah_data$rel_pred_lower, rev(dah_data$rel_pred_upper)),
        col = rgb(0.5, 0.5, 0.8, alpha = 0.4),  # Light blue-gray
        border = NA)

lines(dah_data$mal_DAH_total_per_capita, dah_data$rel_pred, col = "#2166ac", lwd = 3)
mtext(glue("Relative increase in PfPR
$10 reference category"), side = 3, cex = subtitle_cex)

legend('topright',
       legend = c("Mean prediction", "95% CI"), lwd = c(3, NA), 
       title.adj = 0,
       pch = c(NA, 15), pt.cex = 2, 
       col = c("#2166ac", rgb(0.5, 0.5, 0.8, alpha = 0.4)), bty = 'n')

mtext("Malaria Regression - PfPR splines", 3, outer = TRUE, line = -2, cex = title_cex)


dev.off()





