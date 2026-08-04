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
data_path   <- glue("{REPO_DIR}/03-modeling_data")   

pfpr_fml <- reformulate('logit_malaria_suitability + 
                             s(gdppc_mean, k = 6, bs = "mpd") + 
                             s(mal_DAH_total_per_capita, k = 6, bs = "mpd") + 
                             people_flood_days_per_capita + 
                             A0_af', response = "logit_malaria_pfpr")

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