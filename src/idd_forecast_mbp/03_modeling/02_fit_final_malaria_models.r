rm(list = ls())

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)

OPTIMIZER    <- "bfgs"
MAXIT        <- 300L

REPO_DIR <- "/mnt/team/idd/pub/forecast-mbp"

run_date    <- format(Sys.Date(), "%Y_%m_%d")
data_path   <- glue("{REPO_DIR}/03-modeling_data")          # define REPO_DIR earlier
FIT_OUTPUT_DIR = glue("{REPO_DIR}/03-modeling_data/malaria/scam_fits/lsae_1285/20260514_v2")
TASK_ID <- 23 # Winner had top borda, topsis, dominance using mae from no-fe IS and OOS

specs       <- readRDS(glue("{FIT_OUTPUT_DIR}/neighborhood_specs.rds"))
param_map   <- fread(glue("{FIT_OUTPUT_DIR}/param_map.csv"))
spec_index  <- param_map[TASK_ID, spec_index]
spec        <- specs[[spec_index]]



# -------------------------- Fit helper --------------------------
# Wraps scam() to consistently capture iter + convergence + timing + errors.

fit_scam_one <- function(fml, data, label = "fit") {
  t0 <- Sys.time()
  fit <- tryCatch(
    scam(fml, data = data, optimizer = OPTIMIZER, control = list(maxit = MAXIT)),
    error = function(e) e
  )
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (inherits(fit, "error")) {
    message(glue("  [{label}] ERROR after {sprintf('%.1f', elapsed)}s: {conditionMessage(fit)}"))
    return(list(fit = fit, iter = NA_integer_, converged = NA,
                elapsed = elapsed, error = TRUE,
                error_msg = conditionMessage(fit)))
  }
  iter <- tryCatch(as.integer(fit$iter), error = function(e) NA_integer_)
  conv <- isTRUE(fit$conv)
  message(glue("  [{label}] iter={iter} converged={conv} ({sprintf('%.1f', elapsed)}s)"))
  list(fit = fit, iter = iter, converged = conv,
       elapsed = elapsed, error = FALSE, error_msg = NA_character_)
}

# -------------------------- Formula --------------------------

K_DEFAULT <- 6
build_term <- function(var, form) {
  if (form == "linear") return(var)
  sprintf("s(%s, k = %d, bs = '%s')", var, K_DEFAULT, form)
}
build_formula <- function(spec, response) {
  rhs <- mapply(build_term, names(spec), unlist(spec), USE.NAMES = FALSE)
  reformulate(rhs, response = response)
}

pfpr_fml          <- build_formula(spec, "logit_malaria_pfpr")
formula_text <- deparse1(pfpr_fml)

inc_fml <-   reformulate('s(logit_malaria_pfpr, k = 10, bs = "mpi") + 
                             log_gdppc_mean + 
                             A0_af', response = "log_malaria_inc_rate")

mort_fml <-   reformulate('s(logit_malaria_pfpr, k = 10, bs = "mpi") + 
                             log_gdppc_mean + 
                             A0_af', response = "log_malaria_mort_rate")

# -------------------------- Load + clean past data --------------------------

parquet_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/past_inputs_nc/lsae_1285/current/malaria_past_inputs.parquet"
suit_variant_pick <- "mordecai_0_0"

past_data <- as.data.frame(arrow::read_parquet(parquet_path))
past_data$A0_af <- as.factor(past_data$A0_location_id)

nan_toss <- function(df, var) {
  to_toss <- which(is.na(df[var]))
  if (length(to_toss)) df[-to_toss, ] else df
}

past_data <- nan_toss(past_data, "malaria_pfpr")
past_data <- nan_toss(past_data, "gdppc_mean")
past_data <- nan_toss(past_data, "mal_DAH_total_per_capita")
past_data <- nan_toss(past_data, "malaria_inc_rate")
past_data <- nan_toss(past_data, "malaria_mort_rate")

suit_col <- paste0("malaria_suitability_", suit_variant_pick)
past_data$malaria_suit_fraction     <- past_data[[suit_col]] / 365
past_data$malaria_suit_fraction     <- pmin(pmax(past_data$malaria_suit_fraction, 0.001), 0.999)
past_data$logit_malaria_suitability <- log(past_data$malaria_suit_fraction / (1 - past_data$malaria_suit_fraction))
past_data$do30_fraction <- past_data$days_over_30C / 365
past_data$do30_fraction <- pmin(pmax(past_data$do30_fraction, 0.001), 0.999)
past_data$logit_do30    <- log(past_data$do30_fraction / (1 - past_data$do30_fraction))
past_data$rh_fraction <- past_data$relative_humidity / 100
past_data$rh_fraction <- pmin(pmax(past_data$rh_fraction, 0.001), 0.999)
past_data$logit_relative_humidity <- log(past_data$rh_fraction / (1 - past_data$rh_fraction))

log_covs <- c("mal_DAH_total_per_capita", "gdppc_mean", "ldipc_mean", "med_consumppc", "malaria_inc_rate", "malaria_mort_rate")
for (cov in log_covs) {
  past_data[[paste0("log_", cov)]] <- log(past_data[[cov]])
}

# ========================== pfpr fit ==========================
pfpr_mod <- fit_scam_one(pfpr_fml, past_data, label = "pfpr_fit")$fit


mort_mod <- fit_scam_one(mort_fml, past_data, label = "mort_fit")$fit
inc_mod <- fit_scam_one(inc_fml, past_data, label = "inc_fit")$fit



model_names <- c("pfpr_mod", "mort_mod", "inc_mod")


save(list = model_names, file = glue("{data_path}/{run_date}_malaria_models.RData"))