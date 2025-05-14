center_by_location_df <- function(df, covariate_name) {
  dt <- as.data.table(df)
  
  # Check if the covariate exists in the data.table
  if (!(covariate_name %in% names(dt))) {
    stop(paste("Column", covariate_name, "not found in data.table"))
  }
  
  # Create a new column name for the centered values
  centered_col_name <- paste0(covariate_name, "_A0_centered")
  
  # Center the covariate by A0_location_id
  dt[, (centered_col_name) := get(covariate_name) - mean(get(covariate_name), na.rm = TRUE), 
     by = A0_location_id]
  
  return(as.data.frame(dt))
}
library(ggplot2)
library(patchwork)
library(viridis)
library(data.table)
library(dplyr)
# Extended function to run three models with proper summary table
compare_three_models <- function(df, y_var, covariate_names, interaction_pairs = NULL, 
                                 reference_location_id = NULL) {
  # Check inputs
  if (!("A0_af" %in% names(df))) {
    stop("A0_af factor variable not found in dataframe")
  }
  
  # Set reference level for A0_af if provided
  if (!is.null(reference_location_id)) {
    if (reference_location_id %in% levels(df$A0_af)) {
      df$A0_af <- relevel(df$A0_af, ref = reference_location_id)
      message("Reference location set to: ", reference_location_id)
    } else {
      warning("Reference location ID not found in A0_af levels. Using default reference.")
    }
  }
  
  # Create centered versions of all covariates
  df_dt <- as.data.table(copy(df))
  for (cov in covariate_names) {
    df_dt <- center_by_location_df(df_dt, cov)
  }
  
  # Get centered covariate names
  centered_covs <- paste0(covariate_names, "_A0_centered")
  
  # Validate interaction pairs if provided
  if (!is.null(interaction_pairs) && is.list(interaction_pairs)) {
    for (pair in interaction_pairs) {
      if (length(pair) != 2 || !all(pair %in% covariate_names)) {
        stop("Each interaction pair must contain exactly 2 valid covariate names")
      }
    }
    # Create centered interaction pairs
    centered_interaction_pairs <- lapply(interaction_pairs, function(pair) {
      c(paste0(pair[1], "_A0_centered"), paste0(pair[2], "_A0_centered"))
    })
  } else {
    centered_interaction_pairs <- NULL
  }
  
  # Create formula strings for the main covariates
  covariates_formula <- paste(covariate_names, collapse = " + ")
  centered_covariates_formula <- paste(centered_covs, collapse = " + ")
  
  # Add interaction terms if provided
  interaction_terms <- ""
  if (!is.null(interaction_pairs) && length(interaction_pairs) > 0) {
    interactions <- sapply(interaction_pairs, function(pair) {
      paste(pair[1], "*", pair[2])
    })
    interaction_terms <- paste(" + ", paste(interactions, collapse = " + "))
  }
  
  centered_interaction_terms <- ""
  if (!is.null(centered_interaction_pairs) && length(centered_interaction_pairs) > 0) {
    centered_interactions <- sapply(centered_interaction_pairs, function(pair) {
      paste(pair[1], "*", pair[2])
    })
    centered_interaction_terms <- paste(" + ", paste(centered_interactions, collapse = " + "))
  }
  
  # Complete formula strings
  formula1_str <- paste(y_var, "~", covariates_formula, interaction_terms)
  formula2_str <- paste(y_var, "~", covariates_formula, interaction_terms, "+ A0_af")
  formula3_str <- paste(y_var, "~", centered_covariates_formula, centered_interaction_terms, "+ A0_af")
  
  # Convert to formula objects
  formula1 <- as.formula(formula1_str)
  formula2 <- as.formula(formula2_str)
  formula3 <- as.formula(formula3_str)
  
  # Run regressions
  model1 <- lm(formula1, data = df_dt)
  model2 <- lm(formula2, data = df_dt)
  model3 <- lm(formula3, data = df_dt)
  
  # Extract coefficients and standard errors from models
  coefs1 <- coef(summary(model1))
  coefs2 <- coef(summary(model2))
  coefs3 <- coef(summary(model3))
  
  # Get R-squared values
  r2_1 <- summary(model1)$r.squared
  r2_2 <- summary(model2)$r.squared
  r2_3 <- summary(model3)$r.squared
  
  # Find the A0_af level with coefficient closest to zero in model2
  A0_coefs <- coefs2[grep("^A0_af", rownames(coefs2)), "Estimate"]
  if (length(A0_coefs) > 0) {
    closest_to_zero_idx <- which.min(abs(A0_coefs))
    closest_to_zero_level <- sub("A0_af", "", names(A0_coefs)[closest_to_zero_idx])
  } else {
    closest_to_zero_level <- NULL
  }
  
  # Create a restructured summary table with both non-centered and centered values on the same row
  # First, get the basic covariates (excluding interactions and A0_af)
  base_covars <- unique(c(
    covariate_names
  ))
  
  # Extract interaction terms if any
  if (!is.null(interaction_pairs)) {
    interaction_terms_names <- sapply(interaction_pairs, function(pair) {
      paste(pair[1], ":", pair[2], sep = "")
    })
    base_covars <- c(base_covars, interaction_terms_names)
  }
  
  # Create the restructured table
  results <- data.frame(
    Covariate = base_covars,
    Model1_Coef = NA,
    Model1_SE = NA,
    Model1_t_stat = NA,
    Model1_p_value = NA,
    Model2_Coef = NA,
    Model2_SE = NA,
    Model2_t_stat = NA,
    Model2_p_value = NA,
    Model2_Centered_Coef = NA,
    Model2_Centered_SE = NA,
    Model2_Centered_t_stat = NA,
    Model2_Centered_p_value = NA,
    stringsAsFactors = FALSE
  )
  
  # Fill in values from each model
  for (i in 1:nrow(results)) {
    var_name <- results$Covariate[i]
    
    # For Model 1 (no fixed effects)
    if (var_name %in% rownames(coefs1)) {
      results$Model1_Coef[i] <- coefs1[var_name, "Estimate"]
      results$Model1_SE[i] <- coefs1[var_name, "Std. Error"]
      results$Model1_t_stat[i] <- coefs1[var_name, "t value"]
      results$Model1_p_value[i] <- coefs1[var_name, "Pr(>|t|)"]
    }
    
    # For Model 2 (with fixed effects, non-centered)
    if (var_name %in% rownames(coefs2)) {
      results$Model2_Coef[i] <- coefs2[var_name, "Estimate"]
      results$Model2_SE[i] <- coefs2[var_name, "Std. Error"]
      results$Model2_t_stat[i] <- coefs2[var_name, "t value"]
      results$Model2_p_value[i] <- coefs2[var_name, "Pr(>|t|)"]
    }
    
    # For Model 3 (with fixed effects, centered)
    # Map the centered variable name to the original name
    if (var_name == "(Intercept)") {
      # Intercept
      if ("(Intercept)" %in% rownames(coefs3)) {
        results$Model2_Centered_Coef[i] <- coefs3["(Intercept)", "Estimate"]
        results$Model2_Centered_SE[i] <- coefs3["(Intercept)", "Std. Error"]
        results$Model2_Centered_t_stat[i] <- coefs3["(Intercept)", "t value"]
        results$Model2_Centered_p_value[i] <- coefs3["(Intercept)", "Pr(>|t|)"]
      }
    } else if (var_name %in% covariate_names) {
      # Regular covariates
      centered_var <- paste0(var_name, "_A0_centered")
      if (centered_var %in% rownames(coefs3)) {
        results$Model2_Centered_Coef[i] <- coefs3[centered_var, "Estimate"]
        results$Model2_Centered_SE[i] <- coefs3[centered_var, "Std. Error"]
        results$Model2_Centered_t_stat[i] <- coefs3[centered_var, "t value"]
        results$Model2_Centered_p_value[i] <- coefs3[centered_var, "Pr(>|t|)"]
      }
    } else if (!is.null(interaction_pairs)) {
      # Interaction terms
      for (pair in interaction_pairs) {
        interaction_term <- paste(pair[1], ":", pair[2], sep = "")
        centered_interaction <- paste0(pair[1], "_A0_centered:", pair[2], "_A0_centered")
        
        if (var_name == interaction_term && centered_interaction %in% rownames(coefs3)) {
          results$Model2_Centered_Coef[i] <- coefs3[centered_interaction, "Estimate"]
          results$Model2_Centered_SE[i] <- coefs3[centered_interaction, "Std. Error"]
          results$Model2_Centered_t_stat[i] <- coefs3[centered_interaction, "t value"]
          results$Model2_Centered_p_value[i] <- coefs3[centered_interaction, "Pr(>|t|)"]
        }
      }
    }
  }
  
  # Add R-squared values
  r2_df <- data.frame(
    Covariate = "R-squared",
    Model1_Coef = r2_1,
    Model1_SE = "",
    Model1_t_stat = "",
    Model1_p_value = "",
    Model2_Coef = r2_2,
    Model2_SE = "",
    Model2_t_stat = "",
    Model2_p_value = "",
    Model2_Centered_Coef = r2_3,
    Model2_Centered_SE = "",
    Model2_Centered_t_stat = "",
    Model2_Centered_p_value = ""
  )
  
  # Combine and return results
  final_results <- rbind(results, r2_df)
  
  # Return a list with models and the summary table
  return(list(
    model1 = model1,
    model2 = model2, 
    model3 = model3,
    summary_table = final_results,
    closest_to_zero_level = closest_to_zero_level,
    centered_df = df_dt
  ))
}


plot_models_grid_base <- function(model_results, interaction_pair, df, y_var, 
                                  n_grid = 50, 
                                  transform_type = NULL, 
                                  pretty_labels = NULL,
                                  color_ranges = list(original = NULL, transformed = NULL),
                                  gbd_hierarchy_path = NULL,
                                  show_original = TRUE,
                                  show_transformed = TRUE) {
  
  # Load GBD hierarchy if path is provided
  gbd_hierarchy_df <- NULL
  if (!is.null(gbd_hierarchy_path)) {
    tryCatch({
      gbd_hierarchy_df <- arrow::read_parquet(gbd_hierarchy_path)
      message("Successfully loaded GBD hierarchy data")
    }, error = function(e) {
      warning("Could not load GBD hierarchy file: ", e$message)
      gbd_hierarchy_path <- NULL
    })
  }
  
  # Extract models
  model1 <- model_results$model1  # Without factor
  model2 <- model_results$model2  # With factor, original variables
  model3 <- model_results$model3  # With factor, centered variables
  
  # Get centered data
  centered_df <- as.data.frame(model_results$centered_df)
  
  # Get closest-to-zero A0_af level
  closest_to_zero_level <- model_results$closest_to_zero_level
  
  # Get variable names
  x_var <- interaction_pair[1]
  y_var_plot <- interaction_pair[2]
  centered_x <- paste0(x_var, "_A0_centered")
  centered_y <- paste0(y_var_plot, "_A0_centered")
  
  # Set up pretty labels if provided
  if (is.null(pretty_labels)) {
    original_label <- y_var
    if (!is.null(transform_type)) {
      transformed_label <- paste0(transform_type, "(", y_var, ")")
    } else {
      transformed_label <- y_var
    }
  } else {
    original_label <- pretty_labels$original
    transformed_label <- pretty_labels$transformed
  }
  
  # Create grid for prediction - non-centered version
  x_range <- range(df[[x_var]], na.rm = TRUE)
  y_range <- range(df[[y_var_plot]], na.rm = TRUE)
  
  x_grid <- seq(x_range[1], x_range[2], length.out = n_grid)
  y_grid <- seq(y_range[1], y_range[2], length.out = n_grid)
  
  # Create grid for prediction - centered version
  if (centered_x %in% names(centered_df) && centered_y %in% names(centered_df)) {
    x_centered_range <- range(centered_df[[centered_x]], na.rm = TRUE)
    y_centered_range <- range(centered_df[[centered_y]], na.rm = TRUE)
    
    x_centered_grid <- seq(x_centered_range[1], x_centered_range[2], length.out = n_grid)
    y_centered_grid <- seq(y_centered_range[1], y_centered_range[2], length.out = n_grid)
  } else {
    # Fall back to calculating centered ranges if columns don't exist
    ref_loc <- closest_to_zero_level
    if (is.null(ref_loc)) {
      ref_loc <- as.character(names(sort(table(df$A0_location_id), decreasing = TRUE)[1]))
    }
    
    ref_dat <- subset(centered_df, A0_location_id == ref_loc)
    x_mean <- mean(ref_dat[[x_var]], na.rm = TRUE)
    y_mean <- mean(ref_dat[[y_var_plot]], na.rm = TRUE)
    
    x_centered_grid <- x_grid - x_mean
    y_centered_grid <- y_grid - y_mean
  }
  
  # Function to create prediction data frame
  create_prediction_df <- function(model, include_fixed_effects = FALSE, use_centered = FALSE) {
    # Create new data frame for prediction
    new_data <- data.frame(matrix(NA, nrow = length(x_grid) * length(y_grid), 
                                  ncol = length(names(df))))
    colnames(new_data) <- names(df)
    
    # Create the grid
    if (use_centered) {
      grid_points <- expand.grid(x = x_centered_grid, y = y_centered_grid)
      names(grid_points) <- c(centered_x, centered_y)
      
      # For centered models, we still need original variables
      ref_loc <- closest_to_zero_level
      if (is.null(ref_loc)) {
        ref_loc <- as.character(names(sort(table(df$A0_location_id), decreasing = TRUE)[1]))
      }
      ref_dat <- subset(centered_df, A0_location_id == ref_loc)
      x_mean <- mean(ref_dat[[x_var]], na.rm = TRUE)
      y_mean <- mean(ref_dat[[y_var_plot]], na.rm = TRUE)
      
      # Calculate original variables from centered ones
      grid_points[[x_var]] <- grid_points[[centered_x]] + x_mean
      grid_points[[y_var_plot]] <- grid_points[[centered_y]] + y_mean
    } else {
      grid_points <- expand.grid(x = x_grid, y = y_grid)
      names(grid_points) <- c(x_var, y_var_plot)
      
      # For non-centered models using centered variables
      if (centered_x %in% names(centered_df) && centered_y %in% names(centered_df)) {
        ref_loc <- closest_to_zero_level
        if (is.null(ref_loc)) {
          ref_loc <- as.character(names(sort(table(df$A0_location_id), decreasing = TRUE)[1]))
        }
        ref_dat <- subset(centered_df, A0_location_id == ref_loc)
        x_mean <- mean(ref_dat[[x_var]], na.rm = TRUE)
        y_mean <- mean(ref_dat[[y_var_plot]], na.rm = TRUE)
        
        grid_points[[centered_x]] <- grid_points[[x_var]] - x_mean
        grid_points[[centered_y]] <- grid_points[[y_var_plot]] - y_mean
      }
    }
    
    # Add grid variables to new_data
    for (col in names(grid_points)) {
      new_data[[col]] <- grid_points[[col]]
    }
    
    # Determine reference location for fixed effects models
    ref_loc <- NULL
    if(include_fixed_effects) {
      # Use the level closest to zero if available
      if(!is.null(closest_to_zero_level)) {
        ref_loc <- closest_to_zero_level
      } else {
        # Get most common location
        ref_loc <- as.character(names(sort(table(df$A0_location_id), decreasing = TRUE)[1]))
      }
      # Create factor for prediction
      ref_A0_af <- factor(ref_loc, levels = levels(df$A0_af))
      new_data$A0_af <- ref_A0_af
    }
    
    # Fill in mean values for other variables in the model
    model_vars <- names(model$model)
    for(var in model_vars) {
      # Skip the variables we're already handling
      if(var == x_var || var == y_var_plot || var == "A0_af" || 
         var == centered_x || var == centered_y || var == y_var) {
        next
      }
      
      # For other variables, use the mean value
      if(var %in% names(df)) {
        if(is.numeric(df[[var]])) {
          new_data[[var]] <- mean(df[[var]], na.rm = TRUE)
        } else if(is.factor(df[[var]])) {
          # For factors, use the first level
          new_data[[var]] <- levels(df[[var]])[1]
        }
      }
    }
    
    # For centered models that aren't using centered_x/centered_y directly
    if(use_centered && !(centered_x %in% names(model$model) && centered_y %in% names(model$model))) {
      # Add all centered covariates needed for the model
      cent_vars <- grep("_A0_centered$", model_vars, value = TRUE)
      
      for(cvar in cent_vars) {
        if(cvar != centered_x && cvar != centered_y) {
          # Use 0 for centered variables (= mean)
          new_data[[cvar]] <- 0
        }
      }
    }
    
    # Ensure proper factor levels for any categorical variables
    for(var in names(new_data)) {
      if(var %in% names(df) && is.factor(df[[var]])) {
        new_data[[var]] <- factor(new_data[[var]], levels = levels(df[[var]]))
      }
    }
    
    # Make predictions
    preds <- tryCatch({
      predict(model, newdata = new_data, se.fit = TRUE)
    }, error = function(e) {
      cat("Error in prediction for model:", deparse(substitute(model)), "\n")
      cat("Error message:", e$message, "\n")
      cat("Model variables:", paste(model_vars, collapse=", "), "\n")
      cat("Missing variables:", paste(setdiff(model_vars, names(new_data)), collapse=", "), "\n")
      # Return a matrix of NA values instead of stopping
      return(list(
        fit = rep(NA, nrow(new_data)),
        se.fit = rep(NA, nrow(new_data))
      ))
    })
    
    # Reshape predictions into a matrix for contour plotting
    if (!all(is.na(preds$fit))) {
      z_matrix <- matrix(preds$fit, nrow = length(if(use_centered) x_centered_grid else x_grid), 
                         ncol = length(if(use_centered) y_centered_grid else y_grid))
    } else {
      # Create a matrix of NA values if prediction failed
      z_matrix <- matrix(NA, nrow = length(if(use_centered) x_centered_grid else x_grid), 
                         ncol = length(if(use_centered) y_centered_grid else y_grid))
      warning("Prediction failed, using NA matrix for visualization")
    }
    
    return(list(
      x = if(use_centered) x_centered_grid else x_grid, 
      y = if(use_centered) y_centered_grid else y_grid, 
      z = z_matrix
    ))
  }
  
  # Create prediction matrices for contour plots
  pred_data1 <- create_prediction_df(model1, include_fixed_effects = FALSE, use_centered = FALSE)
  pred_data2 <- create_prediction_df(model2, include_fixed_effects = TRUE, use_centered = FALSE)
  pred_data3 <- create_prediction_df(model3, include_fixed_effects = TRUE, use_centered = TRUE)
  
  # Apply transformation if requested
  if (!is.null(transform_type)) {
    pred_data1_transformed <- pred_data1
    pred_data2_transformed <- pred_data2
    pred_data3_transformed <- pred_data3
    
    # Apply transformation based on the type
    if (transform_type == "log") {
      transform_safely <- function(x) {
        result <- log(x)
        result[is.nan(result) | is.infinite(result)] <- NA
        return(result)
      }
      pred_data1_transformed$z <- transform_safely(pred_data1$z)
      pred_data2_transformed$z <- transform_safely(pred_data2$z)
      pred_data3_transformed$z <- transform_safely(pred_data3$z)
      
    } else if (transform_type == "logit") {
      # Logit transformation
      logit <- function(p) {
        result <- log(p/(1-p))
        result[is.nan(result) | is.infinite(result)] <- NA
        return(result)
      }
      pred_data1_transformed$z <- logit(pred_data1$z)
      pred_data2_transformed$z <- logit(pred_data2$z)
      pred_data3_transformed$z <- logit(pred_data3$z)
      
    } else if (transform_type == "invlogit" || transform_type == "expit") {
      # Inverse logit transformation (expit)
      invlogit <- function(x) {
        result <- 1/(1+exp(-x))
        result[is.nan(result) | is.infinite(result)] <- NA
        return(result)
      }
      pred_data1_transformed$z <- invlogit(pred_data1$z)
      pred_data2_transformed$z <- invlogit(pred_data2$z)
      pred_data3_transformed$z <- invlogit(pred_data3$z)
      
    } else if (transform_type == "exp") {
      transform_safely <- function(x) {
        result <- exp(x)
        result[is.nan(result) | is.infinite(result)] <- NA
        return(result)
      }
      pred_data1_transformed$z <- transform_safely(pred_data1$z)
      pred_data2_transformed$z <- transform_safely(pred_data2$z)
      pred_data3_transformed$z <- transform_safely(pred_data3$z)
      
    } else if (transform_type == "sqrt") {
      transform_safely <- function(x) {
        result <- sqrt(x)
        result[is.nan(result) | is.infinite(result)] <- NA
        return(result)
      }
      pred_data1_transformed$z <- transform_safely(pred_data1$z)
      pred_data2_transformed$z <- transform_safely(pred_data2$z)
      pred_data3_transformed$z <- transform_safely(pred_data3$z)
    }
  } else {
    # If no transformation, just use the original data
    pred_data1_transformed <- pred_data1
    pred_data2_transformed <- pred_data2
    pred_data3_transformed <- pred_data3
  }
  
  # Handle NA values in prediction data safely
  all_z_values <- c(
    pred_data1$z[!is.na(pred_data1$z) & !is.infinite(pred_data1$z)], 
    pred_data2$z[!is.na(pred_data2$z) & !is.infinite(pred_data2$z)], 
    pred_data3$z[!is.na(pred_data3$z) & !is.infinite(pred_data3$z)]
  )
  
  if (length(all_z_values) == 0) {
    warning("No valid prediction values found. Using default range.")
    z_range <- c(-1, 1)  # Default range if no valid values
  } else {
    z_range <- if (!is.null(color_ranges$original)) 
      color_ranges$original 
    else 
      range(all_z_values, na.rm = TRUE)
  }
  
  contour_levels <- seq(z_range[1], z_range[2], length.out = 10)
  
  all_z_transformed_values <- c(
    pred_data1_transformed$z[!is.na(pred_data1_transformed$z) & !is.infinite(pred_data1_transformed$z)], 
    pred_data2_transformed$z[!is.na(pred_data2_transformed$z) & !is.infinite(pred_data2_transformed$z)], 
    pred_data3_transformed$z[!is.na(pred_data3_transformed$z) & !is.infinite(pred_data3_transformed$z)]
  )
  
  if (length(all_z_transformed_values) == 0) {
    warning("No valid transformed prediction values found. Using default range.")
    z_transformed_range <- c(0, 1)  # Default range if no valid values
  } else {
    z_transformed_range <- if (!is.null(color_ranges$transformed)) 
      color_ranges$transformed 
    else 
      range(all_z_transformed_values, na.rm = TRUE)
  }
  
  contour_transformed_levels <- seq(z_transformed_range[1], z_transformed_range[2], length.out = 10)
  
  # Create a custom color palette (similar to viridis plasma)
  n_colors <- 100
  color_palette <- colorRampPalette(c("#0D0887", "#5D01A6", "#9C179E", 
                                      "#CC4678", "#ED7953", "#FDB32F", "#F0F921"))(n_colors)
  
  # Extract fixed effects for comparison
  A0_effects_m2 <- coef(model2)[grep("^A0_af", names(coef(model2)))]
  A0_effects_m3 <- coef(model3)[grep("^A0_af", names(coef(model3)))]
  
  use_gbd_hierarchy <- FALSE
  super_region_colors <- NULL
  loc_data <- NULL
  
  # Process location data with GBD hierarchy if available
  if (!is.null(gbd_hierarchy_df)) {
    tryCatch({
      # Extract location IDs from model coefficients - with fixed cleaning
      loc_names <- gsub("A0_", "", gsub("^A0_af", "", names(A0_effects_m2)))
      
      # Convert to numeric for matching with location_id in GBD hierarchy
      loc_ids <- suppressWarnings(as.numeric(loc_names))
      valid_loc_ids <- !is.na(loc_ids)
      
      if (sum(valid_loc_ids) > 0) {
        # Join with GBD hierarchy to get location names and super-regions
        loc_data <- data.frame(
          location_id = loc_ids[valid_loc_ids],
          effect = as.numeric(A0_effects_m2)[valid_loc_ids],
          effect_centered = as.numeric(A0_effects_m3)[valid_loc_ids]
        )
        
        # Merge with GBD hierarchy
        loc_data <- merge(
          loc_data,
          gbd_hierarchy_df[, c("location_id", "location_name", "super_region_name", "super_region_id")],
          by = "location_id",
          all.x = TRUE
        )
        
        # Check if we have any successful matches
        if (nrow(loc_data) > 0 && sum(!is.na(loc_data$super_region_name)) > 0) {
          # First, calculate the minimum effect by super_region (safely)
          loc_data_with_sr <- loc_data[!is.na(loc_data$super_region_name), ]
          
          if (nrow(loc_data_with_sr) > 0) {
            super_region_mins <- aggregate(effect ~ super_region_name + super_region_id, 
                                           data = loc_data_with_sr, 
                                           FUN = min)
            
            # Sort the super-regions by their minimum values
            super_region_order <- super_region_mins[order(super_region_mins$effect), "super_region_id"]
            
            # Sort the location data by super-region (in order of lowest value) and then by effect within super-region
            loc_data$super_region_id_factor <- factor(
              loc_data$super_region_id, 
              levels = super_region_order
            )
            
            # Sort by super-region and then by effect within super-region
            loc_data <- loc_data[order(loc_data$super_region_id_factor, loc_data$effect), ]
            
            # Assign colors to super-regions (safely)
            unique_super_regions <- unique(loc_data$super_region_name[!is.na(loc_data$super_region_name)])
            super_region_colors <- rainbow(length(unique_super_regions), alpha = 0.2)
            names(super_region_colors) <- unique_super_regions
            
            # Mark that we can use the GBD hierarchy
            use_gbd_hierarchy <- TRUE
          }
        } else {
          warning("No matches found between model location IDs and GBD hierarchy")
        }
      } else {
        warning("No valid location IDs could be extracted from model coefficients")
      }
    }, error = function(e) {
      warning("Error processing GBD hierarchy data: ", e$message)
    })
  }
  
  # Calculate the number of plot sections needed based on what's shown
  num_sections <- 0
  if (show_original && !is.null(transform_type)) num_sections <- num_sections + 2  # Original plots + legend
  else if (show_original) num_sections <- num_sections + 2  # Original plots + legend
  
  if (show_transformed && !is.null(transform_type)) num_sections <- num_sections + 2  # Transformed plots + legend
  
  num_sections <- num_sections + 3  # Fixed effects plot (includes extra height)
  
  # Set up plotting layout based on what's shown
  if (show_original && show_transformed && !is.null(transform_type)) {
    # Show both original and transformed
    layout_matrix <- matrix(c(
      1, 2, 3,     # Original contour plots
      4, 4, 4,     # Legend for original
      5, 6, 7,     # Transformed contour plots
      8, 8, 8,     # Legend for transformed
      9, 9, 9     # Fixed effects plot (row 1)
    ), nrow = 5, byrow = TRUE)
    layout(layout_matrix, heights = c(1, 0.25, 1, 0.25, 2.5))
  } else if (show_original && !show_transformed) {
    # Original only
    layout_matrix <- matrix(c(
      1, 2, 3,     # Contour plots
      4, 4, 4,     # Legend
      5, 5, 5,     # Fixed effects plot (row 1)
      5, 5, 5,     # Fixed effects plot (row 2)
      5, 5, 5      # Fixed effects plot (row 3)
    ), nrow = 5, byrow = TRUE)
    layout(layout_matrix, heights = c(1, 0.2, 1.5, 1.5, 1.5))
  } else if (show_transformed && !show_original && !is.null(transform_type)) {
    # Transformed only
    layout_matrix <- matrix(c(
      1, 2, 3,     # Transformed contour plots
      4, 4, 4,     # Legend for transformed
      5, 5, 5,     # Fixed effects plot (row 1)
      5, 5, 5,     # Fixed effects plot (row 2)
      5, 5, 5      # Fixed effects plot (row 3)
    ), nrow = 5, byrow = TRUE)
    layout(layout_matrix, heights = c(1, 0.2, 1.5, 1.5, 1.5))
  } else {
    # Fixed effects only
    layout_matrix <- matrix(c(
      1, 1, 1,     # Fixed effects plot (row 1)
      1, 1, 1,     # Fixed effects plot (row 2)
      1, 1, 1      # Fixed effects plot (row 3)
    ), nrow = 3, byrow = TRUE)
    layout(layout_matrix, heights = c(1.5, 1.5, 1.5))
  }
  
  # Set common plot margins to maximize plot area
  old_par <- par(no.readonly = TRUE)
  par(mar = c(3, 3, 2, 1), mgp = c(1.8, 0.5, 0), cex.main = 0.9, cex.axis = 0.8)
  
  # Function to plot a contour map
  plot_contour_map <- function(pred_data, title, x_lab, y_lab, z_values, z_levels) {
    # Check if we have valid data to plot
    if (all(is.na(pred_data$z))) {
      # Create an empty plot with a message
      plot(0, 0, type = "n", xlim = range(pred_data$x), ylim = range(pred_data$y),
           xlab = x_lab, ylab = y_lab, main = title)
      text(mean(range(pred_data$x)), mean(range(pred_data$y)), "No valid prediction data",
           cex = 1.2, col = "darkgray")
      return()
    }
    
    # Replace any remaining NA values with the mean for plotting
    z_mean <- mean(pred_data$z, na.rm = TRUE)
    if (is.nan(z_mean)) z_mean <- 0  # Handle case where all values are NA
    
    z_matrix <- pred_data$z
    z_matrix[is.na(z_matrix) | is.infinite(z_matrix)] <- z_mean
    
    # Create a color mapping for the z values
    z_range <- range(z_values, na.rm = TRUE)
    z_breaks <- seq(z_range[1], z_range[2], length.out = n_colors + 1)
    
    # Handle potential NAs in color mapping
    z_colors <- cut(z_matrix, breaks = z_breaks, include.lowest = TRUE)
    z_colors_matrix <- matrix(as.numeric(z_colors), nrow = nrow(z_matrix))
    
    # Plot the image with the color-mapped z values
    image(pred_data$x, pred_data$y, z_colors_matrix, 
          col = color_palette, 
          xlab = x_lab, ylab = y_lab,
          main = title,
          axes = FALSE)
    
    # Add contour lines (safely)
    tryCatch({
      contour(pred_data$x, pred_data$y, z_matrix, 
              levels = z_levels, 
              add = TRUE, col = "white", lwd = 0.5)
    }, error = function(e) {
      warning("Could not draw contour lines: ", e$message)
    })
    
    # Add axes
    axis(1)
    axis(2)
    box()
  }
  
  # Plot original data contours if requested
  if (show_original) {
    # Plot 1: No Location Fixed Effects
    plot_contour_map(
      pred_data1,
      "Model 1: No Location Fixed Effects",
      x_var,
      y_var_plot,
      all_z_values,
      contour_levels
    )
    
    # Plot 2: With Location Fixed Effects
    plot_contour_map(
      pred_data2,
      "Model 2: With Location Fixed Effects",
      x_var,
      y_var_plot,
      all_z_values,
      contour_levels
    )
    
    # Plot 3: With Location Fixed Effects + Centered
    plot_contour_map(
      pred_data3,
      "Model 3: With Location Fixed Effects + Centered",
      if(centered_x %in% names(centered_df)) centered_x else x_var,
      if(centered_y %in% names(centered_df)) centered_y else y_var_plot,
      all_z_values,
      contour_levels
    )
    
    # Add a color legend for original data
    par(mar = c(1, 0, 0, 0))
    plot(0, 0, type = "n", axes = FALSE, xlab = "", ylab = "", xlim = c(0, 1), ylim = c(-0.5, 0.5))
    
    # Create the color legend
    legend_x <- seq(0.2, 0.8, length.out = n_colors)
    legend_y <- rep(0.05, n_colors)
    
    # Draw color boxes
    rect_width <- legend_x[2] - legend_x[1]
    rect_height <- 0.4
    for(i in 1:n_colors) {
      rect(legend_x[i] - rect_width/2, legend_y[i] - rect_height/2,
           legend_x[i] + rect_width/2, legend_y[i] + rect_height/2,
           col = color_palette[i], border = NA)
    }
    
    # Add text labels
    text(0.2, 0.05 - rect_height, round(min(z_range), 2), adj = c(0.5, 1), cex = 1)
    text(0.8, 0.05 - rect_height, round(max(z_range), 2), adj = c(0.5, 1), cex = 1)
    text(0.5, 0.05 - 1.1*rect_height, original_label, adj = c(0.5, 0), cex = 1.5)
  }
  
  # Reset margins for contour plots if needed
  if (show_transformed) {
    par(mar = c(3, 3, 2, 1), mgp = c(1.8, 0.5, 0), cex.main = 0.9, cex.axis = 0.8)
  }
  
  # Plot transformed data if requested
  if (show_transformed && !is.null(transform_type)) {
    # Plot 1: No Location Fixed Effects (Transformed)
    plot_contour_map(
      pred_data1_transformed,
      paste0("Model 1: No Location Fixed Effects (", transform_type, ")"),
      x_var,
      y_var_plot,
      all_z_transformed_values,
      contour_transformed_levels
    )
    
    # Plot 2: With Location Fixed Effects (Transformed)
    plot_contour_map(
      pred_data2_transformed,
      paste0("Model 2: With Location Fixed Effects (", transform_type, ")"),
      x_var,
      y_var_plot,
      all_z_transformed_values,
      contour_transformed_levels
    )
    
    # Plot 3: With Location Fixed Effects + Centered (Transformed)
    plot_contour_map(
      pred_data3_transformed,
      paste0("Model 3: With Location Fixed Effects + Centered (", transform_type, ")"),
      if(centered_x %in% names(centered_df)) centered_x else x_var,
      if(centered_y %in% names(centered_df)) centered_y else y_var_plot,
      all_z_transformed_values,
      contour_transformed_levels
    )
    
    # Add a color legend for transformed data
    par(mar = c(1, 0, 0, 0))
    plot(0, 0, type = "n", axes = FALSE, xlab = "", ylab = "", xlim = c(0, 1), ylim = c(-0.5, 0.5))
    
    # Create the color legend
    legend_x <- seq(0.2, 0.8, length.out = n_colors)
    legend_y <- rep(0.05, n_colors)
    
    # Draw color boxes
    rect_width <- legend_x[2] - legend_x[1]
    rect_height <- 0.4
    for(i in 1:n_colors) {
      rect(legend_x[i] - rect_width/2, legend_y[i] - rect_height/2,
           legend_x[i] + rect_width/2, legend_y[i] + rect_height/2,
           col = color_palette[i], border = NA)
    }
    
    # Add text labels
    text(0.2, 0.05 - rect_height, round(min(z_transformed_range), 2), adj = c(0.5, 1), cex = 1)
    text(0.8, 0.05 - rect_height, round(max(z_transformed_range), 2), adj = c(0.5, 1), cex = 1)
    text(0.5, 0.05 - 1.1*rect_height, transformed_label, adj = c(0.5, 0), cex = 1.5)
  }
  
  # Plot fixed effects with super region colors and labels
  # Use larger margins for better display of country names
  par(mar = c(12, 4, 4, 1))
  
  if (use_gbd_hierarchy && !is.null(loc_data) && nrow(loc_data) > 0) {
    # Create a vector to store the super region boundaries
    super_region_boundaries <- c()
    current_super_region <- NULL
    
    # Find the boundaries between super regions
    for (i in 1:nrow(loc_data)) {
      if (is.na(loc_data$super_region_name[i])) next  # Skip rows with missing super region
      
      if (is.null(current_super_region) || current_super_region != loc_data$super_region_name[i]) {
        if (!is.null(current_super_region)) {
          super_region_boundaries <- c(super_region_boundaries, i - 0.5)
        }
        current_super_region <- loc_data$super_region_name[i]
      }
    }
    
    # Add the final boundary
    super_region_boundaries <- c(super_region_boundaries, nrow(loc_data) + 0.5)
    
    # Plot the fixed effects
    plot(1:nrow(loc_data), loc_data$effect, 
         type = "n",  # Start with an empty plot
         xlab = "", ylab = "Fixed Effect Estimate",
         main = "Comparison of Location Fixed Effects",
         xaxt = "n", # Turn off x-axis, will add custom labels
         ylim = range(c(loc_data$effect, loc_data$effect_centered), na.rm = TRUE))
    
    # Shade regions by super region color
    last_boundary <- 0.5  # Starting boundary
    for (i in 1:length(super_region_boundaries)) {
      if (i <= length(unique(loc_data$super_region_name[!is.na(loc_data$super_region_name)]))) {
        # Get super region name for this section
        sr_section <- unique(loc_data$super_region_name[!is.na(loc_data$super_region_name)])
        if (length(sr_section) >= i) {
          sr_name <- sr_section[i]
          start_x <- last_boundary
          end_x <- super_region_boundaries[i]
          
          # Draw the background rectangle
          if (!is.na(sr_name) && sr_name %in% names(super_region_colors)) {
            rect(start_x, par("usr")[3], end_x, par("usr")[4], 
                 col = super_region_colors[sr_name], border = NA)
          }
          
          # Update last boundary
          last_boundary <- end_x
        }
      }
    }
    
    # Add vertical lines at super region boundaries
    for (boundary in super_region_boundaries) {
      if (boundary > 0.5 && boundary < nrow(loc_data) + 0.5) {
        abline(v = boundary, col = "gray", lty = 2, lwd = 1)
      }
    }
    
    # Add points and lines for the effects
    points(1:nrow(loc_data), loc_data$effect, 
           col = "blue", pch = 16, cex = 0.8)
    lines(1:nrow(loc_data), loc_data$effect, 
          col = "blue", lwd = 1)
    
    # Add points and lines for centered effects
    points(1:nrow(loc_data), loc_data$effect_centered, 
           col = "red", pch = 17, cex = 0.8)
    lines(1:nrow(loc_data), loc_data$effect_centered, 
          col = "red", lwd = 1)
    
    # Add ALL location names at an angle - positioned so top right is under the point
    # Determine appropriate angle
    location_labels <- loc_data$location_name
    text_angle <- 45  # Default angle
    if (length(location_labels) > 40) {
      text_angle <- 60  # Vertical for many locations
    } else if (length(location_labels) > 20) {
      text_angle <- 60  # Steeper angle for more locations
    }
    
    #  Calculate x,y positioning for text
    text_x <- 1:length(location_labels)
    text_y <- par("usr")[3] - (par("usr")[4]-par("usr")[3])*0.04
    
    # Show ALL location names at the specified angle
    # Use adj=c(1,0) to position the END of text under each point
    text(text_x, text_y, 
         ifelse(!is.na(location_labels), as.character(location_labels), ""),
         srt = text_angle,         # Angled text
         adj = c(1, 0),           # Right-bottom alignment to position end of text under point
         cex = 0.8,               # Reasonable font size
         xpd = TRUE)              # Allow text outside plot area
    
  } else {
    # If no GBD hierarchy data available, use the original simple plot
    loc_names <- gsub("^A0_af", "", names(A0_effects_m2))
    loc_order <- order(A0_effects_m2)
    
    m2_df <- data.frame(
      location = loc_names[loc_order],
      effect = A0_effects_m2[loc_order]
    )
    
    m3_df <- data.frame(
      location = loc_names[loc_order],
      effect = A0_effects_m3[loc_order]
    )
    
    # Plot fixed effects
    plot(1:length(loc_order), m2_df$effect, 
         type = "b", col = "blue", pch = 16, 
         xlab = "", ylab = "Fixed Effect Estimate",
         main = "Comparison of Location Fixed Effects",
         xaxt = "n", # Turn off x-axis, will add custom labels
         ylim = range(c(m2_df$effect, m3_df$effect), na.rm = TRUE),
         cex.main = 1.2, cex = 1.0)
    
    # Add model 3 line
    lines(1:length(loc_order), m3_df$effect, 
          type = "b", col = "red", pch = 17, cex = 1.0)
    
    # Add ALL location labels at an angle - positioned so top right is under the point
    text_angle <- 45
    if (length(loc_order) > 40) {
      text_angle <- 90
    } else if (length(loc_order) > 20) {
      text_angle <- 60
    }
    
    text(1:length(loc_order), 
         par("usr")[3] - (par("usr")[4]-par("usr")[3])*0.04,
         m2_df$location,
         srt = text_angle,
         adj = c(0, 0),   # Left-bottom alignment to position top-right under point
         cex = 0.8,
         xpd = TRUE)
  }
  
  # Add legend
  legend("topleft", 
         legend = c("Model 2: Original Variables", "Model 3: Centered Variables"),
         col = c("blue", "red"), 
         pch = c(16, 17), 
         lty = 1, 
         cex = 1.0,  # Larger text in legend
         pt.cex = 1.2, # Larger symbols
         bty = "n")
  
  # Reset par settings
  par(old_par)
  
  # Return invisible results
  if (!is.null(transform_type)) {
    invisible(list(
      pred1 = pred_data1,
      pred2 = pred_data2,
      pred3 = pred_data3,
      pred1_transformed = pred_data1_transformed,
      pred2_transformed = pred_data2_transformed,
      pred3_transformed = pred_data3_transformed,
      fixed_effects = if (use_gbd_hierarchy) loc_data else list(m2_df = m2_df, m3_df = m3_df)
    ))
  } else {
    invisible(list(
      pred1 = pred_data1,
      pred2 = pred_data2,
      pred3 = pred_data3,
      fixed_effects = if (use_gbd_hierarchy) loc_data else list(m2_df = m2_df, m3_df = m3_df)
    ))
  }
}