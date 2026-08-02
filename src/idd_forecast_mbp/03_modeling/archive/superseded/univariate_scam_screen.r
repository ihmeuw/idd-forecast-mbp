### univariate_scam_screen.r
### Fits each covariate/smooth combination individually against logit_malaria_pfpr.
### Two models per combo: (1) covariate only, (2) covariate + A0_af.
### Outputs a CSV with deviance explained, AIC, R-squared, and timing.

require(glue)
require(scam)
require(arrow)
require(data.table)

# -------------------------- Data loading --------------------------

parquet_path      <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/past_inputs_nc/lsae_1285/current/malaria_past_inputs.parquet"
suit_variant_pick <- "mordecai_0_0"

past_data <- as.data.frame(arrow::read_parquet(parquet_path))
past_data$A0_af <- as.factor(past_data$A0_location_id)

# Drop NAs in key columns
for (v in c("malaria_pfpr", "gdppc_mean", "mal_DAH_total_per_capita")) {
  past_data <- past_data[!is.na(past_data[[v]]), ]
}

# Derived columns
suit_col <- paste0("malaria_suitability_", suit_variant_pick)
past_data$malaria_suit_fraction     <- pmin(pmax(past_data[[suit_col]] / 365, 0.001), 0.999)
past_data$logit_malaria_suitability <- log(past_data$malaria_suit_fraction / (1 - past_data$malaria_suit_fraction))
past_data$do30_fraction             <- pmin(pmax(past_data$days_over_30C / 365, 0.001), 0.999)
past_data$logit_do30                <- log(past_data$do30_fraction / (1 - past_data$do30_fraction))
past_data$rh_fraction               <- pmin(pmax(past_data$relative_humidity / 100, 0.001), 0.999)
past_data$logit_relative_humidity   <- log(past_data$rh_fraction / (1 - past_data$rh_fraction))

for (cov in c("mal_DAH_total_per_capita", "gdppc_mean", "ldipc_mean", "med_consumppc")) {
  past_data[[paste0("log_", cov)]] <- log(past_data[[cov]])
}

# Logit response
past_data$malaria_pfpr_clipped <- pmin(pmax(past_data$malaria_pfpr, 0.001), 0.999)
past_data$logit_malaria_pfpr   <- log(past_data$malaria_pfpr_clipped / (1 - past_data$malaria_pfpr_clipped))

message(glue("Data loaded: {nrow(past_data)} rows"))

# -------------------------- Define covariate/form combos --------------------------

K_DEFAULT <- 6

combos <- list(
  list(var = "mal_DAH_total_per_capita",     form = "linear"),
  list(var = "mal_DAH_total_per_capita",     form = "mpd"),
  list(var = "log_mal_DAH_total_per_capita", form = "linear"),
  list(var = "gdppc_mean",                   form = "linear"),
  list(var = "gdppc_mean",                   form = "mpd"),
  list(var = "log_gdppc_mean",               form = "linear"),
  list(var = "ldipc_mean",                   form = "linear"),
  list(var = "ldipc_mean",                   form = "mpd"),
  list(var = "log_ldipc_mean",               form = "linear"),
  list(var = "med_consumppc",                form = "linear"),
  list(var = "med_consumppc",                form = "mpd"),
  list(var = "log_med_consumppc",            form = "linear"),
  list(var = "weighted_1km_urban_threshold_300.0_simple_mean",   form = "linear"),
  list(var = "weighted_1km_urban_threshold_300.0_simple_mean",   form = "mpd"),
  list(var = "weighted_1km_urban_threshold_1500.0_simple_mean",  form = "linear"),
  list(var = "weighted_1km_urban_threshold_1500.0_simple_mean",  form = "mpd"),
  list(var = "weighted_100m_urban_threshold_1500.0_simple_mean", form = "linear"),
  list(var = "weighted_100m_urban_threshold_1500.0_simple_mean", form = "mpd"),
  list(var = "people_flood_days_per_capita", form = "linear"),
  list(var = "people_flood_days_per_capita", form = "mpi"),
  list(var = "people_flood_days_per_capita", form = "cv"),
  list(var = "total_precipitation",          form = "linear"),
  list(var = "total_precipitation",          form = "mpi"),
  list(var = "total_precipitation",          form = "cv"),
  list(var = "precipitation_days",           form = "linear"),
  list(var = "precipitation_days",           form = "mpi"),
  list(var = "precipitation_days",           form = "cv"),
  list(var = "relative_humidity",            form = "linear"),
  list(var = "relative_humidity",            form = "mpi"),
  list(var = "logit_relative_humidity",      form = "linear"),
  list(var = "logit_relative_humidity",      form = "mpi"),
  list(var = "mean_temperature",             form = "linear"),
  list(var = "mean_temperature",             form = "cv"),
  list(var = "mean_low_temperature",         form = "linear"),
  list(var = "mean_low_temperature",         form = "cv"),
  list(var = "mean_high_temperature",        form = "linear"),
  list(var = "mean_high_temperature",        form = "cv"),
  list(var = "days_over_30C",                form = "linear"),
  list(var = "days_over_30C",                form = "cv"),
  list(var = "logit_do30",                   form = "linear"),
  list(var = "malaria_suitability_mordecai_0_0", form = "linear"),
  list(var = "malaria_suitability_mordecai_0_0", form = "mpi"),
  list(var = "logit_malaria_suitability",    form = "linear")
)

# -------------------------- Fit loop --------------------------

build_term <- function(var, form) {
  if (form == "linear") return(var)
  sprintf("s(%s, k = %d, bs = '%s')", var, K_DEFAULT, form)
}

results <- vector("list", length(combos) * 2)
idx <- 0L

for (i in seq_along(combos)) {
  combo <- combos[[i]]
  var  <- combo$var
  form <- combo$form
  term <- build_term(var, form)

  message(glue("[{i}/{length(combos)}] {term}"))

  # Drop NAs for this covariate
  df_sub <- past_data[!is.na(past_data[[var]]), ]

  for (with_fe in c(FALSE, TRUE)) {
    rhs <- if (with_fe) paste(term, "+ A0_af") else term
    fml <- as.formula(paste("logit_malaria_pfpr ~", rhs))

    t0 <- Sys.time()
    fit <- tryCatch(
      scam(fml, data = df_sub, optimizer = "efs", control = list(maxit = 300)),
      error = function(e) e
    )
    elapsed <- as.numeric(Sys.time() - t0, units = "secs")

    idx <- idx + 1L

    if (inherits(fit, "error")) {
      results[[idx]] <- data.table(
        variable   = var,
        form       = form,
        term       = term,
        with_fe    = with_fe,
        dev_expl   = NA_real_,
        aic        = NA_real_,
        r_sq       = NA_real_,
        n_obs      = nrow(df_sub),
        elapsed_sec = elapsed,
        converged  = NA,
        error_msg  = conditionMessage(fit)
      )
    } else {
      sm <- summary(fit)
      results[[idx]] <- data.table(
        variable    = var,
        form        = form,
        term        = term,
        with_fe     = with_fe,
        dev_expl    = sm$dev.expl,
        aic         = AIC(fit),
        r_sq        = sm$r.sq,
        n_obs       = nrow(df_sub),
        elapsed_sec = elapsed,
        converged   = isTRUE(fit$conv),
        error_msg   = NA_character_
      )
    }
  }
}

# -------------------------- Save results --------------------------

out <- rbindlist(results)
out_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/scam_fits/lsae_1285/univariate_screen.csv"
fwrite(out, out_path)
message(glue("Done. Results written to {out_path}"))
message(glue("{sum(!is.na(out$dev_expl))} / {nrow(out)} fits succeeded."))



require(httpgd)
httpgd::hgd()
plot(1)
mod <- scam(logit_malaria_pfpr ~ s(malaria_suitability_mordecai_0_0, k = 6, bs = 'mpi') + A0_af, data = past_data, 
        optimizer = "bfgs", control = list(maxit = 300))

summary(mod)
plot(mod)
