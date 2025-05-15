rm(list = ls())

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)


"%ni%" <- Negate("%in%")
"%nlike%" <- Negate("%like%")


data_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"
file_path <- glue("{data_path}/malaria_stage_2_modeling_df.parquet")
lsae_hierarchy_path <- "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_lsae_1209.parquet"
gbd_hierarchy_path <- "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_gbd_2023.parquet"


source("/mnt/share/homes/bcreiner/repos/idd-forecast-mbp/src/idd_forecast_mbp/modeling/helper_functions.r")
# Read in a parquet file
# Read in a parquet file
malaria_df <- arrow::read_parquet(file_path)

malaria_df$A0_af <- as.factor(as.character(paste("A0", malaria_df$A0_location_id, sep = "_")))

###
#
# Without A0_af, with A0_af, centered with A0_af
#
###

mod1 <- lm(logit_malaria_pfpr ~ malaria_suitability_A0_centered*log_ldipc_mean_A0_centered + log_mal_DAH_total_per_capita + total_precipitation_A0_centered + 
              people_flood_days_per_capita_A0_centered + A0_af, data = malaria_df)





out <- summary(mod1)[[4]]
cbind(out[which(rownames(out) %nlike% "A0_af"), 1:3], round(out[which(rownames(out) %nlike% "A0_af"), 4], 3))
summary(mod1)$r.squared



Covariate_options <- list(
  c("malaria_suitability","log_gdppc_mean","elevation","log_mal_DAH_total_per_capita", "total_precipitation",  "people_flood_days_per_capita"),
  c("malaria_suitability","log_gdppc_mean","elevation","log_mal_DAH_total_per_capita", "relative_humidity",  "people_flood_days_per_capita"),
  c("malaria_suitability","log_gdppc_mean","elevation","log_mal_DAH_total_per_capita", "precipitation_days", "people_flood_days_per_capita"),
  c("malaria_suitability","log_ldipc_mean","elevation","log_mal_DAH_total_per_capita", "total_precipitation",  "people_flood_days_per_capita"),
  c("malaria_suitability","log_ldipc_mean","elevation","log_mal_DAH_total_per_capita", "relative_humidity", "people_flood_days_per_capita"),
  c("malaria_suitability","log_ldipc_mean","elevation","log_mal_DAH_total_per_capita", "precipitation_days", "people_flood_days_per_capita"),
  c("malaria_suitability","log_gdppc_mean","log_mal_DAH_total_per_capita", "total_precipitation",  "people_flood_days_per_capita"),
  c("malaria_suitability","log_gdppc_mean","log_mal_DAH_total_per_capita", "relative_humidity",  "people_flood_days_per_capita"),
  c("malaria_suitability","log_gdppc_mean","log_mal_DAH_total_per_capita", "precipitation_days", "people_flood_days_per_capita"),
  c("malaria_suitability","log_ldipc_mean","log_mal_DAH_total_per_capita", "total_precipitation",  "people_flood_days_per_capita"),
  c("malaria_suitability","log_ldipc_mean","log_mal_DAH_total_per_capita", "relative_humidity", "people_flood_days_per_capita"),
  c("malaria_suitability","log_ldipc_mean","log_mal_DAH_total_per_capita", "precipitation_days", "people_flood_days_per_capita")
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
                                            "total_precipitation",
                                            "relative_humidity",
                                            "precipitation_days",
                                            "people_flood_days_per_capita",
                                            "malaria_suitability:log_gdppc_mean",
                                            "malaria_suitability:log_ldipc_mean",
                                            "R-squared"),
                                expectation = c(1, -1, -1, 0, -1, 1, 1, 1, 1, 0, 0, 1))

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
  return(summary_df)
  # # if (any(summary_df$issues_3 == "ISSUE")){
  # # 
  # # } else {
  #   return(summary_df[,c("covariate", "value_3", "p_value_3", "issues_3")])
  # # }
  

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


UCov <- setdiff(Expectation_table$variable, "R-squared")
require(RColorBrewer)
require(beeswarm, lib = "~/packages")
COLS <- brewer.pal(length(All_Results), "Paired")
# COLS <- rep(1,12)
par(mfrow=c(3,4))
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













