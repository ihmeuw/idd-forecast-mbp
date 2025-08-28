rm(list = ls())
#

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)
library(dplyr)
library(purrr)
library(tibble)

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

past_data$log_people_flood_days_per_capita = log(past_data$people_flood_days_per_capita + 1e-6)
past_data$relative_humidity_fraction = past_data$relative_humidity / 100
past_data$logit_relative_humidity = log(past_data$relative_humidity_fraction / (1 - past_data$relative_humidity_fraction))
past_data$log_total_precipitation = log(past_data$total_precipitation)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define covariates and their properties
covariates <- list(
  covar1 = list(natural = "malaria_suit_fraction", transformed = "logit_malaria_suitability", 
                monotonic = "increasing", justify_monotonic = TRUE),
  covar2 = list(natural = "gdppc_mean", transformed = "log_gdppc_mean",
                monotonic = "decreasing", justify_monotonic = TRUE),
  covar3 = list(natural = "mal_DAH_total_per_capita", transformed = "log_mal_DAH_total_per_capita",
                monotonic = "decreasing", justify_monotonic = TRUE),
  covar4 = list(natural = "people_flood_days_per_capita", transformed = "log_people_flood_days_per_capita", 
                monotonic = "increasing", justify_monotonic = TRUE),
  covar5 = list(natural = "relative_humidity", transformed = "logit_relative_humidity",
                monotonic = NA, justify_monotonic = FALSE),
  covar6 = list(natural = "total_precipitation", transformed = "log_total_precipitation",
                monotonic = NA, justify_monotonic = FALSE),
  covar7 = list(natural = "urban_1km_threshold_300", transformed = "logit_urban_1km_threshold_300",
                monotonic = NA, justify_monotonic = FALSE, group = "urban"),
  covar8 = list(natural = "urban_1km_threshold_1500", transformed = "logit_urban_1km_threshold_1500",
                monotonic = NA, justify_monotonic = FALSE, group = "urban"),
  covar9 = list(natural = "urban_100m_threshold_300", transformed = "logit_urban_100m_threshold_300",
                monotonic = NA, justify_monotonic = FALSE, group = "urban"),
  covar10 = list(natural = "urban_100m_threshold_1500", transformed = "logit_urban_100m_threshold_1500",
                 monotonic = NA, justify_monotonic = FALSE, group = "urban")
)

exclusive_groups <- list(urban = c("covar7", "covar8", "covar9", "covar10"))

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

get_covariate_forms <- function(covar_info) {
  forms <- c("natural_linear", "transformed_linear", "natural_spline", "transformed_spline")
  if (covar_info$justify_monotonic && !is.na(covar_info$monotonic)) {
    forms <- c(forms, "natural_monotonic", "transformed_monotonic")
  }
  return(forms)
}

build_formula_term <- function(covar_name, form, covariates) {
  covar_info <- covariates[[covar_name]]
  
  switch(form,
         "natural_linear" = covar_info$natural,
         "transformed_linear" = covar_info$transformed,
         "natural_spline" = paste0("s(", covar_info$natural, ", k=6)"),
         "transformed_spline" = paste0("s(", covar_info$transformed, ", k=6)"),
         "natural_monotonic" = {
           bs_type <- if (covar_info$monotonic == "increasing") "mpi" else "mpd"
           paste0("s(", covar_info$natural, ", bs='", bs_type, "', k=6)")
         },
         "transformed_monotonic" = {
           bs_type <- if (covar_info$monotonic == "increasing") "mpi" else "mpd"
           paste0("s(", covar_info$transformed, ", bs='", bs_type, "', k=6)")
         }
  )
}

generate_factorial_specs <- function(covariates, exclusive_groups, include_fixed_effect = TRUE) {
  # Get independent covariates
  independent_covars <- names(covariates)[sapply(covariates, function(x) is.null(x$group) || is.na(x$group))]
  
  # Build form options for independent covariates
  independent_options <- list()
  for (covar in independent_covars) {
    independent_options[[covar]] <- get_covariate_forms(covariates[[covar]])
  }
  
  # Build form options for exclusive groups
  group_options <- list()
  for (group_name in names(exclusive_groups)) {
    group_forms <- c()
    for (member in exclusive_groups[[group_name]]) {
      member_forms <- get_covariate_forms(covariates[[member]])
      group_forms <- c(group_forms, paste(member, member_forms, sep = ":"))
    }
    group_options[[group_name]] <- group_forms
  }
  
  # Combine all options
  all_options <- c(independent_options, group_options)
  form_combinations <- expand.grid(all_options, stringsAsFactors = FALSE)
  
  # Fixed effect options
  fixed_effect_options <- if (include_fixed_effect) c(FALSE, TRUE) else FALSE
  
  # Generate specs
  specs <- list()
  spec_id <- 1
  
  for (fe in fixed_effect_options) {
    for (i in 1:nrow(form_combinations)) {
      included_covars <- c()
      forms <- list()
      
      # Process independent covariates
      for (covar in independent_covars) {
        form <- form_combinations[i, covar]
        included_covars <- c(included_covars, covar)
        forms[[covar]] <- form
      }
      
      # Process exclusive groups
      for (group_name in names(exclusive_groups)) {
        selected <- form_combinations[i, group_name]
        parts <- strsplit(selected, ":")[[1]]
        member <- parts[1]
        form <- parts[2]
        included_covars <- c(included_covars, member)
        forms[[member]] <- form
      }
      
      specs[[spec_id]] <- list(
        model_id = sprintf("model_%05d", spec_id),
        included_covars = included_covars,
        forms = forms,
        fixed_effect = fe
      )
      spec_id <- spec_id + 1
    }
  }
  
  return(specs)
}

build_formula <- function(spec, covariates, response_var = "y") {
  terms <- sapply(spec$included_covars, function(covar) {
    build_formula_term(covar, spec$forms[[covar]], covariates)
  })
  
  formula_str <- paste(response_var, "~", paste(terms, collapse = " + "))
  
  if (spec$fixed_effect) {
    formula_str <- paste(formula_str, "+ A0_af")
  }
  
  return(as.formula(formula_str))
}

fit_model <- function(spec, data, covariates, response_var = "y", family = gaussian()) {
  formula <- build_formula(spec, covariates, response_var)
  uses_monotonic <- any(grepl("mp[id]", as.character(formula)))
  
  tryCatch({
    if (uses_monotonic) {
      model <- scam(formula, data = data, family = family, 
                    optimizer = "efs", control = list(maxit = 300))
    } else {
      model <- gam(formula, data = data, family = family)
    }
    
    list(
      model_id = spec$model_id,
      formula = as.character(formula)[3],
      fixed_effect = spec$fixed_effect,
      converged = model$converged,
      aic = AIC(model),
      bic = BIC(model),
      deviance_explained = (model$null.deviance - deviance(model)) / model$null.deviance,
      model_object = model,
      error = NA
    )
  }, error = function(e) {
    list(
      model_id = spec$model_id,
      formula = as.character(formula)[3],
      fixed_effect = spec$fixed_effect,
      converged = FALSE,
      aic = NA, bic = NA, deviance_explained = NA,
      model_object = NULL,
      error = as.character(e)
    )
  })
}

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================
all_specs <- generate_factorial_specs(covariates, exclusive_groups, TRUE)

test_models <- function(data, n_models = 50, method = "random", response_var = "logit_malaria_pfpr", 
                        family = gaussian(), seed = 123, keep_models = TRUE) {
  set.seed(seed)
  
  if (method == "first") {
    # You'll need to define generate_limited_specs or use a different approach
    specs <- all_specs[1:min(n_models, length(all_specs))]
    cat("Generated", length(specs), "models for testing\n")
  } else {
    if (method == "random") {
      indices <- sample(length(all_specs), min(n_models, length(all_specs)))
    } else {
      step <- max(1, floor(length(all_specs) / n_models))
      indices <- seq(1, length(all_specs), by = step)[1:n_models]
    }
    specs <- all_specs[indices]
    cat("Testing", length(specs), "of", length(all_specs), "total models\n")
  }
  print("Fitting models...")
  # Time the fitting
  start_time <- Sys.time()
  results <- vector("list", length(specs))
  for (i in seq_along(specs)) {
    print(glue("Fitting model {i}/{length(specs)}: {specs[[i]]}"))
    results[[i]] <- fit_model(specs[[i]], data, covariates, response_var, family)
  }
  end_time <- Sys.time()
  
  print("Finished the fitting")
  # Summary
  total_time <- end_time - start_time
  successful <- sum(sapply(results, function(x) x$converged), na.rm = TRUE)
  
  cat("\nResults:\n")
  cat("Successful fits:", successful, "/", length(specs), "\n")
  cat("Total time:", round(as.numeric(total_time, units = "mins"), 2), "minutes\n")
  cat("Time per model:", round(as.numeric(total_time, units = "secs") / length(specs), 2), "seconds\n")
  
  if (successful > 0) {
    est_full_time <- as.numeric(total_time, units = "hours") * length(generate_factorial_specs(covariates, exclusive_groups, TRUE)) / length(specs)
    cat("Estimated full time:", round(est_full_time, 1), "hours\n")
  }
  
  # Show errors for failed models
  failed_results <- results[!sapply(results, function(x) x$converged)]
  if (length(failed_results) > 0) {
    cat("\nErrors in failed models:\n")
    for (i in 1:min(3, length(failed_results))) {
      cat("Model", failed_results[[i]]$model_id, ":", failed_results[[i]]$error, "\n")
    }
  }
  
  if (keep_models) {
    return(results)  # Return full results with model objects
  } else {
    return(map_dfr(results, ~{.x$model_object <- NULL; as_tibble(.x)}))
  }
}

run_full_models <- function(data, response_var = "logit_malaria_pfpr", family = gaussian(), 
                            save_models = TRUE, output_dir = "results") {
  
  if (save_models && !dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  specs <- all_specs
  cat("Fitting", length(specs), "models\n")
  
  results <- list()
  for (i in seq_along(specs)) {
    if (i %% 100 == 0) cat("Progress:", i, "/", length(specs), "\n")
    
    result <- fit_model(specs[[i]], data, covariates, response_var, family)
    results[[i]] <- result
    
    if (save_models && !is.null(result$model_object)) {
      saveRDS(result$model_object, file.path(output_dir, paste0(result$model_id, ".rds")))
    }
  }
  
  results_df <- map_dfr(results, ~{.x$model_object <- NULL; as_tibble(.x)})
  
  if (save_models) {
    write.csv(results_df, file.path(output_dir, "results.csv"), row.names = FALSE)
  }
  
  cat("Complete! Successful fits:", sum(results_df$converged, na.rm = TRUE), "/", nrow(results_df), "\n")
  return(results_df)
}

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

# # Test first:
test_results <- test_models(past_data, n_models = 2, response_var = "logit_malaria_pfpr", family = gaussian())
# 
# # If satisfied, run full analysis:
# full_results <- run_full_models(my_data, response_var = "outcome", family = binomial())
# Your model failed, so let's see why:
print(test_results[[1]])

# Or more specifically:
cat("Model ID:", test_results[[1]]$model_id, "\\n")
cat("Formula:", test_results[[1]]$formula, "\\n") 
cat("Error:", test_results[[1]]$error, "\\n")

# Check if ANY models succeeded:
successful_indices <- which(sapply(test_results, function(x) x$converged))
cat("Successful models:", length(successful_indices), "\\n")

if (length(successful_indices) > 0) {
  # Access a successful model
  good_model <- test_results[[successful_indices[1]]]
  summary(good_model$model_object)
} else {
  cat("No models converged. Common issues:\\n")
  cat("1. Check variable names match your data\\n")
  cat("2. Check for missing values\\n") 
  cat("3. Try simpler models first\\n")
}

# Look at your actual data structure:
head(your_data)
names(your_data)