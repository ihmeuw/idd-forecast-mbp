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
location_ids <- ncvar_get(itn_nc, "location_id")
year_ids <- ncvar_get(itn_nc, "year_id")
scenarios <- ncvar_get(itn_nc, "scenario")

# Check what the scenario values are
print(scenarios)  # See what scenarios are available

# Extract the malaria_itn variable
itn_data <- ncvar_get(itn_nc, "malaria_itn")
nc_close(itn_nc)

# If you want the 0 scenario (index 2), extract just that slice
itn_data_slice <- itn_data[, , 2]  # Adjust index as needed

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
location_ids <- ncvar_get(act_nc, "location_id")
year_ids <- ncvar_get(act_nc, "year_id")
scenarios <- ncvar_get(act_nc, "scenario")

# Extract the malaria_act variable
act_data <- ncvar_get(act_nc, "malaria_act")
nc_close(act_nc)

# If you want the 0 scenario (index 2), extract just that slice
act_data_slice <- act_data[, , 2]  # Adjust index as needed

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

range(past_data$malaria_act, na.rm = TRUE)
new_malaria_pfpr_mod <- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
                               s(gdppc_mean, k = 6, bs = 'mpd') + 
                               s(malaria_itn, k = 6, bs = 'mpd') + 
                               s(malaria_act, k = 6, bs = 'mpd') + 
                               people_flood_days_per_capita + 
                               A0_af,
                             data = past_data,
                             optimizer = "efs",      # Faster optimizer
                             control = list(maxit = 300))  # Limit iterations

new_malaria_pfpr_mod <- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
                               s(gdppc_mean, k = 6, bs = 'mpd') + 
                               malaria_itn + 
                               s(malaria_act, k = 6, bs = 'mpd') + 
                               people_flood_days_per_capita + 
                               A0_af,
                             data = past_data,
                             optimizer = "efs",      # Faster optimizer
                             control = list(maxit = 300))  # Limit iterations



new_malaria_pfpr_mod_2 <- scam(logit_malaria_pfpr ~ 
                                 logit_malaria_suitability + 
                                 s(gdppc_mean, k = 6, bs = 'mpd') + 
                                 malaria_itn + 
                                 malaria_act+ 
                                 A0_af,
                               data = past_data) 

new_malaria_pfpr_mod_4 <- scam(logit_malaria_pfpr ~ 
                                 logit_malaria_suitability + 
                                 s(gdppc_mean, k = 6, bs = 'mpd') + 
                                 malaria_itn + 
                                 malaria_act + 
                                 people_flood_days_per_capita +  
                                 A0_af,
                               data = past_data) 

new_malaria_pfpr_mod_6 <- lm(logit_malaria_pfpr ~ 
                                 logit_malaria_suitability + 
                                 gdppc_mean + 
                                 malaria_itn + 
                                 malaria_act + 
                                 people_flood_days_per_capita +  
                                 A0_af,
                               data = past_data) 


extract_itn_act_summary <- function(model) {
  # Get model summary
  model_summary <- summary(model)
  
  # Determine model type
  model_class <- class(model)[1]
  
  # Extract R-squared based on model type
  if (model_class == "lm") {
    r_sq <- model_summary$r.squared
    adj_r_sq <- model_summary$adj.r.squared
    coef_table <- model_summary$coefficients
    
    cat("=== Linear Model (lm) Summary ===\n\n")
    cat("R-squared:", round(r_sq, 4), "\n")
    cat("Adjusted R-squared:", round(adj_r_sq, 4), "\n\n")
    
  } else if (model_class %in% c("gam", "scam")) {
    adj_r_sq <- model_summary$r.sq
    dev_expl <- model_summary$dev.expl
    coef_table <- model_summary$p.table
    
    cat("=== GAM/SCAM Model Summary ===\n\n")
    cat("Adjusted R-squared:", round(adj_r_sq, 4), "\n")
    cat("Deviance explained:", round(dev_expl, 4), "\n\n")
    
  } else {
    stop("Model type not recognized. Expected 'lm', 'gam', or 'scam'.")
  }
  
  # Extract ITN and ACT coefficients
  cat("ITN and ACT Coefficients:\n")
  cat("-----------------------------------\n")
  
  # Check if both variables exist in the model
  vars_to_extract <- c("malaria_itn", "malaria_act")
  vars_present <- vars_to_extract[vars_to_extract %in% rownames(coef_table)]
  
  if (length(vars_present) == 0) {
    cat("Neither malaria_itn nor malaria_act found in model.\n")
    return(invisible(NULL))
  }
  
  itn_act_coefs <- coef_table[vars_present, , drop = FALSE]
  print(itn_act_coefs)
  
  # Return results invisibly for potential further use
  if (model_class == "lm") {
    results <- list(
      model_type = "lm",
      r_squared = r_sq,
      adj_r_squared = adj_r_sq,
      itn_act_coefficients = itn_act_coefs
    )
  } else {
    results <- list(
      model_type = model_class,
      adj_r_squared = adj_r_sq,
      deviance_explained = dev_expl,
      itn_act_coefficients = itn_act_coefs
    )
  }
  
  invisible(results)
}


plot_model_fit <- function(model, model_name = NULL, ALPHA = 0.1) {
  # Helper function to calculate R-squared for a subset
  calculate_r_squared <- function(observed_y, fitted_y) {
    ss_res <- sum((observed_y - fitted_y)^2)
    ss_tot <- sum((observed_y - mean(observed_y))^2)
    # Handle cases where ss_tot is zero
    if (ss_tot == 0) {
      return(1)
    } else {
      return(1 - (ss_res / ss_tot))
    }
  }
  
  # Define inverse logit function
  inv_logit <- function(x) {
    exp(x) / (1 + exp(x))
  }
  
  # --- Setup and Data Preparation (unchanged) ---
  
  # Get model summary
  model_summary <- summary(model)
  model_char <- deparse(substitute(model))
  last_char <- substr(model_char, nchar(model_char), nchar(model_char))
  is_odd <- suppressWarnings(!is.na(as.numeric(last_char)) && as.numeric(last_char) %% 2 == 1)
  
  if (is_odd) {
    data_used <- past_data_w_interventions
  } else {
    data_used <- past_data
  }
  data_used$col = ifelse(data_used$has_itn == 1 & data_used$has_act == 1, 4,
                         ifelse(data_used$has_itn == 1 & data_used$has_act == 0, 3,
                                ifelse(data_used$has_itn == 0 & data_used$has_act == 1, 2, 1)))
  
  if (is.null(model_name)) {
    model_name <- deparse(substitute(model))
  }
  
  # Define colors and labels
  require(RColorBrewer)
  COLS = c("black", brewer.pal(4, "Set1")[-1])
  RAW_COLS = COLS
  COLS = adjustcolor(COLS, alpha = ALPHA)
  LEGEND_LABELS = c("No ITN/No ACT", "No ITN/ACT", "ITN/No ACT", "ITN/ACT")
  
  # Determine model type and extract R-squared (Full Model)
  model_class <- class(model)[1]
  
  if (model_class == "lm") {
    r_sq <- model_summary$r.squared
    adj_r_sq <- model_summary$adj.r.squared
    coef_table <- model_summary$coefficients
    r_sq_label <- paste0("Full R² (Logit) = ", round(r_sq, 4))
  } else if (model_class %in% c("gam", "scam")) {
    adj_r_sq <- model_summary$r.sq
    coef_table <- model_summary$p.table
    r_sq_label <- paste0("Full Adj. R² (Logit): ", round(adj_r_sq, 4))
  } else {
    stop("Model type not recognized. Expected 'lm', 'gam', or 'scam'.")
  }
  
  # Extract Coefficients (unchanged)
  vars_to_extract <- c("malaria_itn", "malaria_act")
  vars_present <- vars_to_extract[vars_to_extract %in% rownames(coef_table)]
  coef_labels <- c()
  if (length(vars_present) > 0) {
    for (var in vars_present) {
      var_coef <- coef_table[var, "Estimate"]
      var_pval <- coef_table[var, "Pr(>|t|)"]
      sig <- ifelse(var_pval < 0.001, "***", 
                    ifelse(var_pval < 0.01, "**", 
                           ifelse(var_pval < 0.05, "*", "")))
      var_label <- ifelse(var == "malaria_itn", "ITN", "ACT")
      coef_labels <- c(coef_labels, 
                       paste0(var_label, ": ", round(var_coef, 4), sig))
    }
  }
  
  # --- R-squared for Subsets Calculation ---
  obs_logit <- model$y
  fitted_logit <- model$fitted.values
  obs_original <- inv_logit(obs_logit)
  fitted_original <- inv_logit(fitted_logit)
  
  subset_defs <- list(
    "No ITN/ACT" = data_used$col == 1,
    "ACT only"   = data_used$col == 2,
    "ITN only"   = data_used$col == 3,
    "ITN & ACT"  = data_used$col == 4,
    "Any Intervention" = data_used$col >= 2
  )
  
  # Calculate R-squared values and create legend labels
  logit_r_sq_labels <- c()
  natural_r_sq_labels <- c()
  
  for (name in names(subset_defs)) {
    indices <- subset_defs[[name]]
    
    # Check if subset has data points
    if (sum(indices) > 0) {
      # Logit scale R-squared
      r_sq_logit_sub <- calculate_r_squared(obs_logit[indices], fitted_logit[indices])
      logit_r_sq_labels <- c(logit_r_sq_labels, 
                             paste0("R² (Logit) for ", name, ": ", round(r_sq_logit_sub, 4)))
      
      # Original scale R-squared
      r_sq_original_sub <- calculate_r_squared(obs_original[indices], fitted_original[indices])
      natural_r_sq_labels <- c(natural_r_sq_labels, 
                               paste0("R² (Natural) for ", name, ": ", round(r_sq_original_sub, 4)))
    }
  }
  
  # --- Plotting ---
  
  # Logit Scale Limits
  min_logit <- min(obs_logit, fitted_logit, na.rm = TRUE)
  max_logit <- max(obs_logit, fitted_logit, na.rm = TRUE)
  # Add a small buffer for visual clarity
  range_logit <- c(min_logit, max_logit) + c(-0.1, 0.1) 
  
  # Original Scale Limits
  min_original <- min(obs_original, fitted_original, na.rm = TRUE)
  max_original <- max(obs_original, fitted_original, na.rm = TRUE)
  # Add a small buffer for visual clarity
  range_original <- c(min_original, max_original) + c(-0.01, 0.01)
  
  # Set up 1x2 panel layout
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
  
  ## Panel 1: Logit scale
  plot(obs_logit, fitted_logit, cex = 0.25,
       xlab = 'Observed logit malaria PFPR', 
       ylab = 'Fitted logit malaria PFPR',
       xlim = range_logit,
       ylim = range_logit,
       col = COLS[data_used$col],
       main = paste0(model_name, " - Logit scale"))
  abline(0, 1, col = 'red')
  
  # Combine full model R^2, coefficients, and subset R^2 for logit legend
  legend_text_logit <- c(r_sq_label, coef_labels, "", "--- Sub-group R² (Logit) ---", logit_r_sq_labels)
  
  # Add legend with statistics (top-left)
  legend("topleft", 
         legend = legend_text_logit,
         bty = "n",
         cex = 0.7)
  
  # Add legend with colors (bottom-right)
  legend("bottomright",
         legend = LEGEND_LABELS,
         col = RAW_COLS,
         bty = "n",
         cex = 0.8,
         pch = 16,
         pt.cex = 1.5)
  
  ## Panel 2: Original scale (inverse logit)
  # Calculate R-squared on original scale for the whole dataset
  r_sq_original <- calculate_r_squared(obs_original, fitted_original)
  
  plot(obs_original, fitted_original, cex = 0.25,
       xlab = 'Observed malaria PFPR', 
       ylab = 'Fitted malaria PFPR',
       xlim = range_original,
       ylim = range_original,
       col = COLS[data_used$col],
       main = paste0(model_name, " - Original scale"))
  abline(0, 1, col = 'red')
  
  # Combine full model R^2 (natural) and subset R^2 for natural scale legend
  legend_text_natural <- c(paste0("Full R² (Natural): ", round(r_sq_original, 4)), 
                           "", "--- Sub-group R² (Natural) ---", natural_r_sq_labels)
  
  # Add legend with statistics (top-left)
  legend("topleft", 
         legend = legend_text_natural,
         bty = "n",
         cex = 0.7)
  
  # Add legend with colors (bottom-right)
  legend("bottomright",
         legend = LEGEND_LABELS,
         col = RAW_COLS,
         bty = "n",
         cex = 0.8,
         pch = 16,
         pt.cex = 1.5)
  
  # Reset plotting parameters
  par(mfrow = c(1, 1))
  
  # Return statistics invisibly (added subset R^2 to the results list)
  results_list <- list()
  if (model_class == "lm") {
    results_list$r_squared <- r_sq
    results_list$adj_r_squared <- adj_r_sq
  } else {
    results_list$adj_r_squared <- adj_r_sq
  }
  if (length(vars_present) > 0) {
    results_list$coefficients <- coef_table[vars_present, , drop = FALSE]
  }
  
  # The subset R^2 values are currently in the natural_r_sq_labels and logit_r_sq_labels lists
  # I will structure them into a more accessible format for the return value
  r_sq_subsets <- list()
  for (i in 1:length(names(subset_defs))) {
    r_sq_subsets[[names(subset_defs)[i]]]$logit_r_sq <- sub(".*: ", "", logit_r_sq_labels[i])
    r_sq_subsets[[names(subset_defs)[i]]]$original_r_sq <- sub(".*: ", "", natural_r_sq_labels[i])
  }
  results_list$r_sq_subsets <- r_sq_subsets
  
  invisible(results_list)
}

# Usage:
# plot_model_fit(new_malaria_pfpr_mod_3, "Model 3")
# Usage examples:
# plot_model_fit(new_malaria_pfpr_mod_1)
# plot_model_fit(new_malaria_pfpr_mod_4, "Model 4")
# plot_model_fit(new_malaria_pfpr_mod_6, "Model 6")

# Usage examples:
# plot_model_fit(new_malaria_pfpr_mod_1)
# plot_model_fit(new_malaria_pfpr_mod_4, "Model 4")
# plot_model_fit(new_malaria_pfpr_mod_6, "Model 6")


plot_model_fit(new_malaria_pfpr_mod_1, "Model 1")
plot_model_fit(new_malaria_pfpr_mod_2, "Model 2")
plot_model_fit(new_malaria_pfpr_mod_3, "Model 3")
plot_model_fit(new_malaria_pfpr_mod_4, "Model 4")
plot_model_fit(new_malaria_pfpr_mod_5, "Model 5")
plot_model_fit(new_malaria_pfpr_mod_6, "Model 6")
plot_model_fit(malaria_pfpr_mod, "DAH Model")



extract_itn_act_summary(new_malaria_pfpr_mod_1)
plot(new_malaria_pfpr_mod_1$y, new_malaria_pfpr_mod_1$fitted.values, cex = 0.25,
     xlab = 'Observed logit malaria PFPR', ylab = 'Fitted logit malaria PFPR', main = 'New malaria PFPR model 1')
abline(0,1,col='red')
extract_itn_act_summary(new_malaria_pfpr_mod_2)
plot(new_malaria_pfpr_mod_2$y, new_malaria_pfpr_mod_2$fitted.values, cex = 0.25,
     xlab = 'Observed logit malaria PFPR', ylab = 'Fitted logit malaria PFPR', main = 'New malaria PFPR model 2')
abline(0,1,col='red')
extract_itn_act_summary(new_malaria_pfpr_mod_3)
extract_itn_act_summary(new_malaria_pfpr_mod_4)
extract_itn_act_summary(new_malaria_pfpr_mod_5)
extract_itn_act_summary(new_malaria_pfpr_mod_6)

# Usage examples:
# extract_itn_act_summary(new_malaria_pfpr_mod_4)  # For GAM/SCAM
# extract_itn_act_summary(new_malaria_pfpr_mod_6)  # For lm
# 
# Or save the results:
# results <- extract_itn_act_summary(new_malaria_pfpr_mod_4)



malaria_pfpr_mod <- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
                             s(gdppc_mean, k = 6, bs = 'mpd') + 
                             s(mal_DAH_total_per_capita, k = 6, bs = 'mpd') + 
                             people_flood_days_per_capita + 
                             A0_af,
                           data = past_data,
                           optimizer = "efs",      # Faster optimizer
                           control = list(maxit = 300))  # Limit iterations


tmp_data <- past_data[rep(1,12),c("logit_malaria_suitability", "gdppc_mean", 
                          "mal_DAH_total_per_capita", "people_flood_days_per_capita",
                          "A0_af")]

tmp_data$A0_af = levels(past_data$A0_af)[2]
tmp_data$mal_DAH_total_per_capita[2:12] <- 0:10
tmp_data$pred <- predict(malaria_pfpr_mod, newdata = tmp_data)
# plot(tmp_data$mal_DAH_total_per_capita[2:12], tmp_data$pred[2:12], type = 'b', 
#      xlab = "Malaria DAH per capita", ylab = "Predicted logit malaria PFPR",
#      main = "Malaria PFPR vs Malaria DAH per capita")

invlogit = function(x) {
  exp(x) / (1 + exp(x))
}


df = data.frame(from = 0:9,
                to = 1:10,
                start = invlogit(tmp_data$pred[2:11]),
                end = invlogit(tmp_data$pred[3:12]),
                relative_reduction = 100*(1 - invlogit(tmp_data$pred[3:12]) / invlogit(tmp_data$pred[2:11])))

plot(df$from, df$relative_reduction)



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



mod_df <- past_data[which(past_data$base_malaria_mort_rate > 0),]
mortality_base_scam_mod <- scam(log_base_malaria_mort_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") +
                                  log_gdppc_mean +
                                  A0_af,
                                data = mod_df,
                                optimizer = "efs",      # Faster optimizer
                                control = list(maxit = 300))  # Limit iterations

mod_df <- past_data[which(past_data$base_malaria_inc_rate  > 0),]
incidence_base_scam_mod <- scam(log_base_malaria_inc_rate ~ s(logit_malaria_pfpr, k = 10, bs = "mpi") +
                                  log_gdppc_mean + A0_af,
                                data = mod_df,
                                optimizer = "efs",      # Faster optimizer
                                control = list(maxit = 300))  # Limit iterations

model_names <- c("malaria_pfpr_mod", "mortality_scam_mod", "incidence_scam_mod", "mortality_base_scam_mod",
                 "incidence_base_scam_mod")


save(list = model_names, file = glue("{data_path}/2025_07_26_malaria_models.RData"))





percentiles = seq(0, 1, by = 0.05)
mal_dah_perc = sapply(percentiles, function(p) {
  quantile(past_data$mal_DAH_total_per_capita, p, na.rm = TRUE)
})

mal_dah_perc = unique(mal_dah_perc)

bin_df <- data.frame(bin_start = head(mal_dah_perc, -1),
                     bin_end = tail(mal_dah_perc, -1),
                     mean_residual = NA,
                     Q1 = NA,
                     Q3 = NA)
for (i in bin_df$bin_start){
  tmp_locs <- which(past_data$mal_DAH_total_per_capita >= i & 
                        past_data$mal_DAH_total_per_capita < (i + 0.01))
  bin_df$mean_residual[which(bin_df$bin_start == i)] <- mean(malaria_pfpr_mod$residuals[tmp_locs])
  bin_df$Q1[which(bin_df$bin_start == i)] <- quantile(malaria_pfpr_mod$residuals[tmp_locs], 0.25)
  bin_df$Q3[which(bin_df$bin_start == i)] <- quantile(malaria_pfpr_mod$residuals[tmp_locs], 0.75)
}


par(mfrow=c(3,1))
plot(malaria_pfpr_mod, select = 2)
plot(bin_df$bin_start+bin_df$bin_end, bin_df$mean_residual, type = 'n',xlim = c(0, max(bin_df$bin_end)), ylim = c(min(bin_df$Q1), max(bin_df$Q3)))
abline(h = 0, lty = 2)
for (i in seq_along(bin_df$bin_start)){
  lines(c(bin_df$bin_start[i], bin_df$bin_end[i]), 
        c(bin_df$mean_residual[i], bin_df$mean_residual[i]),
        col = "blue", lwd = 2)
  lines(rep((bin_df$bin_start[i] + bin_df$bin_end[i]) / 2, 2), 
          c(bin_df$Q1[i], bin_df$Q3[i]),
          col = "red", lwd = 2)
}
plot(bin_df$bin_start+1e-6, bin_df$mean_residual, type = 'n', xlim = c(min(bin_df$bin_start) + 1e-6, max(bin_df$bin_end)), ylim = c(min(bin_df$Q1), max(bin_df$Q3)), log = 'x')
abline(h = 0, lty = 2)
for (i in seq_along(bin_df$bin_start)){
  lines(c(bin_df$bin_start[i]+1e-6, bin_df$bin_end[i]), 
        c(bin_df$mean_residual[i], bin_df$mean_residual[i]),
        col = "blue", lwd = 2)
  lines(rep(exp((log(bin_df$bin_start[i]+1e-6) + log(bin_df$bin_end[i])) / 2), 2), 
        c(bin_df$Q1[i], bin_df$Q3[i]),
        col = "red", lwd = 2)
}

wtf <- past_data[which(past_data$A0_af %in% c('A0_28', 'A0_215')),]



past_data$year_af <- as.factor(past_data$year_id)
malaria_pfpr_mod <- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
                           s(gdppc_mean, k = 6, bs = 'mpd') + 
                           s(mal_DAH_total_per_capita, k = 6, bs = 'mpd', by = year_af) + 
                           people_flood_days_per_capita + 
                           A0_af,
                         data = past_data,
                         optimizer = "efs",      # Faster optimizer
                         control = list(maxit = 300))  # Limit iterations




