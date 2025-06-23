rm(list = ls())

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)


"%ni%" <- Negate("%in%")
"%nlike%" <- Negate("%like%")


data_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"
file_path <- glue("{data_path}/dengue_stage_2_modeling_df.parquet")
lsae_hierarchy_path <- "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_lsae_1209.parquet"
gbd_hierarchy_path <- "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_gbd_2023.parquet"

lsae_full_hierarchy_df_path = "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/hierarchy_lsae_1209_full.parquet"
# Read in lsae_full_hierarchy_df_path
lsae_full_hierarchy_df <- arrow::read_parquet(lsae_full_hierarchy_df_path)


source("/mnt/share/homes/bcreiner/repos/idd-forecast-mbp/src/idd_forecast_mbp/modeling/helper_functions.r")
# Read in a parquet file
# Read in a parquet file
dengue_df <- arrow::read_parquet(file_path)
dengue_df$A0_af <- as.factor(dengue_df$A0_af)
dengue_df <- merge(dengue_df, lsae_full_hierarchy_df[,c("location_id", "super_region_id")], by = "location_id", all.x = TRUE)


##

# dengue_mortality_theshold <- 1
# dengue_mortality_rate_theshold <- 1e-7
# 
# dengue_df$yn <- 0
# dengue_df$yn[which(dengue_df$dengue_mort_rate > dengue_mortality_rate_theshold & 
#                      dengue_df$dengue_mort_count > dengue_mortality_theshold &
#                      dengue_df$dengue_suitability > 0)] <- 1
# 
# dengue_df <- dengue_df[which(dengue_df$yn == 1),]

dengue_df$A0_af <- as.factor(as.character(paste("A0", dengue_df$A0_location_id, sep = "_")))

income_covs <- names(dengue_df)[which(names(dengue_df) %like% "gdp" & names(dengue_df) %nlike% "A0_mean")]
water_covs <- c("relative_humidity")
water_covs <- as.vector(as.matrix((sapply(water_covs, function(x) names(dengue_df)[which(names(dengue_df) %like% x & names(dengue_df) %nlike% "A0_mean")]))))
urban_covs <- names(dengue_df)[which(names(dengue_df) %like% "urban" & names(dengue_df) %like% "logit" & names(dengue_df) %nlike% "A0_mean")]
flood_covs <- names(dengue_df)[which(names(dengue_df) %like% "capita" & names(dengue_df) %nlike% "A0_mean")]

Expectation_table <- data.frame(variable = c("dengue_suitability", income_covs, water_covs, urban_covs, flood_covs),
                                expectation = c(1, rep(0, length(income_covs)),
                                               rep(1, length(water_covs)),
                                               rep(1, length(urban_covs)),
                                               rep(1, length(flood_covs))))

denv_formula <- function(i_num, w_num, u_num, f_num, mult = TRUE){
  tmp_covs <- c("dengue_suitability", income_covs[i_num], water_covs[w_num], urban_covs[u_num], flood_covs[f_num])
  if (mult){
    tmp_formula <- glue("log_dengue_mort_rate ~ {income_covs[i_num]} + dengue_suitability * {water_covs[w_num]} + {urban_covs[u_num]} + {flood_covs[f_num]} + 
                    A0_af")
  } else {
  tmp_formula <- glue("log_dengue_mort_rate ~ {income_covs[i_num]} + dengue_suitability + {water_covs[w_num]} + {urban_covs[u_num]} + {flood_covs[f_num]} + 
                    A0_af")
  }
  return(list(as.formula(tmp_formula), tmp_covs))
}



eval_model <- function(vec, mult){
  denv_out <- denv_formula(vec[1],vec[2],vec[3],vec[4], mult)
  tmp_mod <- lm(denv_out[[1]], data = dengue_df)
  tmp_out <- summary(tmp_mod)[[4]]
  tmp_out <- tmp_out[which(rownames(tmp_out) %nlike% "A0_af"), 1:4]
  summary(tmp_mod)$r.squared
  
  tmp_df <- data.frame(Covariate = denv_out[[2]],
                       Estimate = NA,
                       StdError = NA,
                       t_value = NA,
                       p_value = NA)
  for (c_num in seq_along(denv_out[[2]])){
    tmp_df$Estimate[c_num] <- tmp_out[which(rownames(tmp_out) == denv_out[[2]][c_num]), 1]
    tmp_df$StdError[c_num] <- tmp_out[which(rownames(tmp_out) == denv_out[[2]][c_num]), 2]
    tmp_df$t_value[c_num] <- tmp_out[which(rownames(tmp_out) == denv_out[[2]][c_num]), 3]
    tmp_df$p_value[c_num] <- tmp_out[which(rownames(tmp_out) == denv_out[[2]][c_num]), 4]
  }
  
  tmp_df$expected <- sapply(tmp_df$Covariate, function(x)Expectation_table$expectation[which(Expectation_table$variable == x)])
  tmp_df$observed <- tmp_df$Estimate / abs(tmp_df$Estimate)
  tmp_df$issue_flag <- ifelse(tmp_df$observed * tmp_df$expected < 0, 1,0)
  tmp_df$issue <- ifelse(tmp_df$observed * tmp_df$expected < 0, "ISSUE", "")
  #
  if (sum(tmp_df$issue_flag) == 0){
    return(list(tmp_df,summary(tmp_mod)$r.squared, tmp_mod, mult))
  }
}

mod_list <- vector("list", 0)
for (i_num in seq_along(income_covs)){
  message(glue("Running model {i_num} of {length(income_covs)}"))
  for (w_num in seq_along(water_covs)){
    message(glue("Running model {i_num}.{w_num} of {length(income_covs)}.{length(water_covs)}"))
    for (u_num in seq_along(urban_covs)){
      for (f_num in seq_along(flood_covs)){
        for (mult in c(TRUE, FALSE)){
          mod_list[[length(mod_list) + 1]] <- eval_model(c(i_num, w_num, u_num, f_num), mult)
        }
      }
    }
  }
}





r_vec <- unlist(lapply(mod_list, function(x)x[[2]]))
a_vec <- unlist(lapply(mod_list, function(x)AIC(x[[3]])))
m_vec <- unlist(lapply(mod_list, function(x)x[[4]]))
COLS <- rep(1, length(r_vec))
COLS[hm] <- 2
plot(r_vec, m_vec, col = COLS, cex = 2)

for (h in seq_along(hm)){
  print(mod_list[[hm[h]]][[1]][,1:4])
}


#

coef <- data.frame(matrix(unlist(lapply(mod_list, function(x)x[[1]][,2])), ncol = length(mod_list[[1]][[1]][,2]), byrow = TRUE))
names(coef) <- c("dengue_suitability",
                 "income",
                 "water",
                 "urban",
                 "flood")

tmp_coef <- coef[which(coef$dengue_suitability > 0.001 & coef$water > 0.004),]
table(round(tmp_coef$water, 5))

plot(tmp_coef)
plot(coef$dengue_suitability, coef$income)


water_levels <- names(table(round(tmp_coef$water, 5)))

for (w_num in seq_along(water_levels)){
  assign(glue("hm_{w_num}"), which(round(coef$water, 5) == water_levels[w_num] & coef$dengue_suitability > 0.001))
}

hm_1 <- which(coef$water > 0.004 & round(coef$dengue_suitability, 4) == 0.0011)
hm_2 <- which(coef$water > 0.004 & round(coef$dengue_suitability, 4) == 0.0012)
hm_3 <- which(coef$water > 0.004 & round(coef$dengue_suitability, 4) == 0.0013)


for (h in seq_along(hm_1)){
  print(mod_list[[hm_1[h]]][[1]][,1:3])
  print(mod_list[[hm_1[h]]][[4]])
}

w_num <-5
hm_i <- get(glue("hm_{w_num}"))
for (h in seq_along(hm_i)){
  print(mod_list[[hm_i[h]]][[1]][,1:3])
  print(mod_list[[hm_i[h]]][[4]])
}


for (h in seq_along(hm_3)){
  print(mod_list[[hm_3[h]]][[1]][,1:4])
}






# Model 1:
# dengue_suitability
# log_gdppc_mean
# relative humidity (with or without centered)
# logit 1km 1500 urban threshold (with or without centered)
# people_flood_days_per_capita (with or without centered)
# Interaction between suit & humidity

# Model 2:
# dengue_suitability
# log_gdppc_mean
# relative humidity (with or without centered)
# logit 100m 1500 urban threshold (with or without centered)
# people_flood_days_per_capita (with or without centered)

# Model 3:
# dengue_suitability
# log_gdppc_mean
# relative humidity (with or without centered)
# logit 100m 300 urban threshold (with or without centered)
# people_flood_days_per_capita (with or without centered)


tmp_coef <- coef[which(coef$dengue_suitability > 0.004 & coef$water > 0.003),]
COLS <- rep(1, dim(tmp_coef)[1])
COLS[which(round(tmp_coef[,1], 5) == 0.00692)] <- 2
COLS[which(round(tmp_coef[,1], 5) == 0.00705)] <- 3
plot(tmp_coef, pch = 19, col = COLS)
plot(jitter(tmp_coef[,1]), jitter(tmp_coef[,4]), col = COLS)
#

tmp_covs <- c("dengue_suitability", income_covs[i_num], water_covs[w_num], urban_covs[u_num], flood_covs[f_num])






























mod_list <- vector("list", 0)
counter <- 1
for (i_num in seq_along(income_covs)){
  for (w_num in seq_along(water_covs)){
    for (u_num in seq_along(urban_covs)){
      for (f_num in seq_along(flood_covs)){
        tmp_formula <- glue("log_dengue_mort_rate ~ {income_covs[i_num]} * dengue_suitability + year + 
                    ADM0_NAME_af")
        tmp_mod <- lm(as.formula(tmp_formula), data = dengue_df)
        tmp_out <- summary(tmp_mod)[[4]]
        mod_list[[counter]] <- list(tmp_mod, tmp_out[which(rownames(tmp_out) %nlike% "ADM0"), 1:3], summary(tmp_mod)$r.squared)
        counter  <- counter + 1
      }
    }
  }
}

lapply(mod_list, function(x) x[[3]])
lapply(mod_list, function(x) x[[2]])





mod_1 <- lm(log_dengue_mort_rate ~ log_gdppc_mean * dengue_suitability + weighted_100m_urban_threshold_1500.0_simple_mean + 
              total_precipitation + people_flood_days_per_capita + 
             A0_af,
            data = dengue_df)
tmp_out <- summary(mod_1)[[4]]
tmp_out[which(rownames(tmp_out) %nlike% "A0_af"), c(1,4)]
summary(mod_1)$r.squared
###
#
# Without A0_af, with A0_af, centered with A0_af
#
###
lsae_full_hierarchy_df[which(lsae_full_hierarchy_df$location_id == 103),]
lsae_full_hierarchy_df[which(lsae_full_hierarchy_df$location_id %in% c(15, 26, 28)),]
15 26 28

mod1 <- lm(logit_malaria_pfpr ~ malaria_suitability_A0_centered*log_ldipc_mean_A0_centered + log_mal_DAH_total_per_capita + total_precipitation_A0_centered + 
              people_flood_days_per_capita_A0_centered*SR_af + A0_af, data = malaria_df)

tmp_df <- as.data.frame(malaria_df[which(malaria_df$A0_location_id == 338),])



mod2 <- lm(logit_malaria_pfpr ~ malaria_suitability_A0_centered+log_ldipc_mean_A0_centered + log_mal_DAH_total_per_capita + total_precipitation_A0_centered + 
             people_flood_days_per_capita_A0_centered + A0_af, data = malaria_df)


mod2 <- lm(logit_malaria_pfpr ~ malaria_suitability+log_ldipc_mean + mal_DAH_total_per_capita + total_precipitation + 
             people_flood_days_per_capita, data = malaria_df)




out <- summary(mod1)[[4]]
cbind(out[which(rownames(out) %nlike% "A0_af"), 1:3], round(out[which(rownames(out) %nlike% "A0_af"), 4], 3))
summary(mod1)$r.squared

out <- summary(mod2)[[4]]
cbind(out[which(rownames(out) %nlike% "A0_af"), 1:3], round(out[which(rownames(out) %nlike% "A0_af"), 4], 3))
summary(mod2)$r.squared



Covariate_options <- list(
  c("malaria_suitability","log_gdppc_mean","log_mal_DAH_total_per_capita", "total_precipitation",  "people_flood_days_per_capita"),
  c("malaria_suitability","log_gdppc_mean",
    "log_mal_DAH_intervention_per_capita", "log_mal_DAH_health_systems_per_capita", "log_mal_DAH_other_per_capita",
    "total_precipitation",  "people_flood_days_per_capita"),
  c("malaria_suitability","log_gdppc_mean",
    "log_mal_DAH_intervention_per_capita",
    "total_precipitation",  "people_flood_days_per_capita"),
  c("malaria_suitability","log_gdppc_mean",
    "log_mal_DAH_health_systems_per_capita", 
    "total_precipitation",  "people_flood_days_per_capita"),
  c("malaria_suitability","log_gdppc_mean",
    "log_mal_DAH_other_per_capita",
    "total_precipitation",  "people_flood_days_per_capita")
)

All_Results <- vector("list", length = length(Covariate_options))
for (c_num in seq_along(Covariate_options)){
  message("Running model ", c_num, " of ", length(Covariate_options))
  results <- compare_three_models(
    df = malaria_df,
    y_var = "logit_malaria_pfpr",
    covariate_names = Covariate_options[[c_num]],
    interaction_pairs = list(Covariate_options[[c_num]][1:2])
  )
  
  All_Results[[c_num]] <- list(results$summary_table,
                               c(AIC(results$model1),
                                 AIC(results$model2),
                                 AIC(results$model3)),
                               c(summary(results$model1)$r.sq,
                                 summary(results$model2)$r.sq,
                                 summary(results$model3)$r.sq))
}



All_Results

Expectation_table <- data.frame(variable = c("malaria_suitability",
                                            "log_gdppc_mean",
                                            "log_ldipc_mean",
                                            "elevation",
                                            "log_mal_DAH_total_per_capita",
                                            "log_malaria_DAH_intervention_per_capita", 
                                            "log_malaria_DAH_health_systems_per_capita", 
                                            "log_malaria_DAH_other_per_capita",
                                            "total_precipitation",
                                            "relative_humidity",
                                            "precipitation_days",
                                            "people_flood_days_per_capita",
                                            "malaria_suitability:log_gdppc_mean",
                                            "malaria_suitability:log_ldipc_mean",
                                            "R-squared"),
                                expectation = c(1, -1, -1, 0, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 1))

form <- function(numbers){
  numbers <- as.numeric(numbers)
  out <- sapply(numbers, function(x){
    if (is.na(x)){
      x <- ""
    } else if (x < 0.001){
      x <- formatC(x, format = "e", digits = 2)
    } else {
      x <- round(x, 3)
    }
    return(x)
  })
  return(out)
}

parse_model <- function(results_table){
  tmp_cov_list <- results_table$Covariate
  summary_df <- data.frame(covariate = tmp_cov_list)
  summary_df$expectation <- sapply(tmp_cov_list, function(x)Expectation_table$expectation[which(Expectation_table$variable == x)])
  #
  summary_df$value_1 <- results_table$Model1_Coef
  summary_df$p_value_1 <- form(results_table$Model1_p_value)
  tmp_direction <- summary_df$value_1 / abs(summary_df$value_1)
  summary_df$issues_1 <- ifelse(tmp_direction* summary_df$expectation < 0, "ISSUE", "")
  summary_df$value_2 <- results_table$Model2_Coef
  summary_df$p_value_2 <- form(results_table$Model2_p_value)
  tmp_direction <- summary_df$value_2 / abs(summary_df$value_2)
  summary_df$issues_2 <- ifelse(tmp_direction* summary_df$expectation < 0, "ISSUE", "")
  summary_df$value_3 <- results_table$Model2_Centered_Coef
  summary_df$p_value_3 <- form(results_table$Model2_Centered_p_value)
  tmp_direction <- summary_df$value_3 / abs(summary_df$value_3)
  summary_df$issues_3 <- ifelse(tmp_direction* summary_df$expectation < 0, "ISSUE", "")
  # return(summary_df)
  # if (any(summary_df$issues_3 == "ISSUE")){
  #
  # } else {
    return(summary_df[,c("covariate", "value_3", "p_value_3", "issues_3")])
  # }
  

}

for (c_num in seq_along(All_Results)){
  print(parse_model(All_Results[[c_num]][[1]]))
}

get_pchs <- function(vec){
  out <- sapply(vec, function(x){
    ifelse(x < 0.001, 8, ifelse(x < 0.05, 4, 1))
  })
  return(out)
}


UCov <- unique(unlist(Covariate_options))
require(RColorBrewer)
require(beeswarm, lib = "~/packages")
COLS <- brewer.pal(length(All_Results), "Paired")
# COLS <- rep(1,12)
par(mfrow=c(2,5))
for (c_num in seq_along(UCov)){
  tmp_cov <- UCov[c_num]
  expectation_row <- Expectation_table[which(Expectation_table$variable == tmp_cov),]
  # Create an empty dataframe with the following columns
  # Covariate, Model_number, Model1_Coef, Model2_Coef, Model2_Centered_Coef, Model1_p_value, Model2_p_value, Model2_Centered_p_value
  # Loop through the All_Results list and find the rows that match the covariate
  
  tmp_df = data.frame(Covariate = character(),
              Model_number = integer(),
              Model1_Coef = numeric(),
              Model2_Coef = numeric(),
              Model2_Centered_Coef = numeric(),
              Model1_p_value = numeric(),
              Model2_p_value = numeric(),
              Model2_Centered_p_value = numeric())
  
  for (m_num in seq_along(All_Results)){
    if (tmp_cov %in% All_Results[[m_num]][[1]]$Covariate){
      tmp_row <- which(All_Results[[m_num]][[1]]$Covariate == tmp_cov)
      tmp_df_row <- data.frame(Covariate = tmp_cov,
                               Model_number = m_num,
                               Model1_Coef = as.numeric(All_Results[[m_num]][[1]]$Model1_Coef[tmp_row]),
                               Model2_Coef = as.numeric(All_Results[[m_num]][[1]]$Model2_Coef[tmp_row]),
                               Model2_Centered_Coef = as.numeric(All_Results[[m_num]][[1]]$Model2_Centered_Coef[tmp_row]),
                               Model1_p_value = as.numeric(All_Results[[m_num]][[1]]$Model1_p_value[tmp_row]),
                               Model2_p_value = as.numeric(All_Results[[m_num]][[1]]$Model2_p_value[tmp_row]),
                               Model2_Centered_p_value = as.numeric(All_Results[[m_num]][[1]]$Model2_Centered_p_value[tmp_row]))
      tmp_df <- rbind(tmp_df, tmp_df_row)
    }
  }
  YLIM <- c(min(tmp_df$Model1_Coef, tmp_df$Model2_Coef, tmp_df$Model2_Centered_Coef), max(tmp_df$Model1_Coef, tmp_df$Model2_Coef, tmp_df$Model2_Centered_Coef))
  
  
  if (YLIM[1] < 0){
    YLIM[1] <- YLIM[1] - 0.05 * diff(YLIM)
  } else {
    YLIM[1] <- YLIM[1] - 0.05 * diff(YLIM)
  }
  if (YLIM[2] > 0){
    YLIM[2] <- YLIM[2] + 0.05 * diff(YLIM)
  } else {
    YLIM[2] <- YLIM[2] + 0.05 * diff(YLIM)
  }
  
  if (diff(sign(YLIM)) == 0){
    if (sign(YLIM[1]) == -1){
      YLIM[2] <- abs(YLIM[1])*0.1
    } else {
      YLIM[1] <- -abs(YLIM[2])*0.1
    }
  }
  
  plot(1, type = "n", xlim = c(0.5, 3.5),xaxs = "i", yaxs = "i", ylim = YLIM, ann = FALSE, axes = FALSE)
  box()
  axis(2, pretty(YLIM))
  mtext(2, text = "Coefficient", line = 2.5)
  mtext(3, text = tmp_cov, line = 0.5)
  if (expectation_row$expectation < 0){
    polygon(c(0.5,3.5,3.5,0.5),c(0,0, YLIM[2], YLIM[2]), col = rgb(1,0,0,.05), border = NA)
  } else if (expectation_row$expectation > 0){
    polygon(c(0.5,3.5,3.5,0.5),c(YLIM[1], YLIM[1], 0, 0), col = rgb(1,0,0,.05), border = NA)
  }
  abline(h = 0, lty = 2, lwd = 2)
  tmp_pch <- get_pchs(tmp_df$Model1_p_value)
  beeswarm(tmp_df$Model1_Coef , pwpch = tmp_pch, pwcol  = COLS[tmp_df$Model_number], xlab = "Model", ylab = "Covariate", main = tmp_cov,
           at = 1, add = TRUE)

  tmp_pch <- get_pchs(tmp_df$Model2_p_value)
  beeswarm(tmp_df$Model2_Coef, pwpch = tmp_pch, pwcol  = COLS[tmp_df$Model_number], xlab = "", ylab = "",
           at = 2, add = TRUE)
  tmp_pch <- get_pchs(tmp_df$Model2_Centered_p_value)
  beeswarm(tmp_df$Model2_Centered_Coef , pwpch = tmp_pch, pwcol  = COLS[tmp_df$Model_number], xlab = "", ylab = "",
           at = 3, add = TRUE)
  axis(1, at = 1:3, labels = c("Model 1", "Model 2", ""))
  axis(1, 3, "Model 2
Centered", line = 1.1, tick = FALSE)
  if (c_num == length(UCov)) legend("topright", legend = c("p-value < 0.001", "p-value < 0.05", "p-value > 0.05"), pch = c(8,4,1), bty = "n")
}
#

R_squared_df <-data.frame(Covariate_number = integer(),
                          Model_number = integer(),
                          R2 = numeric(),
                          AIC = numeric())
for (c_num in seq_along(All_Results)){
  tmp_aic <- All_Results[[c_num]][[2]]
  tmp_r_squared <- All_Results[[c_num]][[3]]
  
  
  tmp_df = data.frame(Covariate_number = rep(c_num, 3),
                      Model_number = 1:3,
                      R2 = tmp_r_squared,
                      AIC = tmp_aic)
  
  R_squared_df <- rbind(R_squared_df, tmp_df)
}

beeswarm(R_squared_df$R2~R_squared_df$Model_number, pwcol  = COLS[R_squared_df$Covariate_number], 
         xlab = "", ylab = "R-squared", main = "R-squared", pch = 19, ylim = c(0,1), labels = NA)
axis(1, at = 1:3, labels = c("Model 1", "Model 2", ""))
axis(1, 3, "Model 2
Centered", line = 1.1, tick = FALSE)





######

names(malaria_df)


require(scam)
mod_mort_1 <- scam(log_malaria_pf_mort_rate ~ s(logit_malaria_pfpr, k = 4, bs = "mpi") + A0_af, data = malaria_df)

mod_mort_2 <- scam(log_malaria_pf_mort_rate ~ s(logit_malaria_pfpr, k = 4, bs = "mpi") + log_mal_DAH_total_per_capita + A0_af, data = malaria_df)
# R-squared = 97.9%

mod_inc <- scam(log_malaria_pf_inc_rate ~ s(malaria_pfpr, k = 4) + A0_af, data = malaria_df)

