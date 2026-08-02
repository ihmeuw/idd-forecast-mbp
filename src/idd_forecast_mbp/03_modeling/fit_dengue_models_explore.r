
rm(list = ls())

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)
require(ggplot2)
require(httpgd)

OPTIMIZER    <- "bfgs"
MAXIT        <- 300L
N_FOLDS      <- 5L
CV_SEED      <- 42L
TESTING      <- FALSE

# ============================================================================
# fit_dengue_models_explore.r — interactive dengue exploration: in-sample
# predicted vs observed all-age incidence/mortality, faceted by super-region.
#
# Flow:
#   incidence: fit (base + rest) -> predict age-sex rates -> aggregate to all-age
#              -> aggregate up the hierarchy -> plot AA inc vs observed
#   mortality: fit CFR -> predict -> mort = inc * CFR -> rake -> aggregate -> plot
#
# Model structure mirrors final_models_dengue.r. Two steps are stubbed with a
# sensible first pass and marked <<MIRROR>> — they should be made exact against
# the production workers as_dengue_shifts.py (AS extrapolation) and
# rake_dengue.py (rake). Other <<CONFIRM>>: the dengue reference age group and
# the CFR definition.
# ============================================================================

REPO_DIR <- "/mnt/team/idd/pub/forecast-mbp"
LSAE     <- "lsae_1285"

parquet_path     <- glue("{REPO_DIR}/03-modeling_data/dengue/past_inputs_nc/{LSAE}/current/dengue_past_inputs.parquet")
hierarchy_path   <- glue("{REPO_DIR}/02-processed_data/hierarchy/{LSAE}/current/full_hierarchy_2023_{LSAE}.parquet")
observed_aa_path <- glue("{REPO_DIR}/02-processed_data/dengue/raked_aa/{LSAE}/current/aa_full_dengue_df.parquet")
observed_as_path <- glue("{REPO_DIR}/02-processed_data/dengue/raked_as/{LSAE}/current/as_full_dengue_df.parquet")
age_metadata_path <- glue("{REPO_DIR}/01-raw_data/gbd/current/age_metadata.parquet")

hierarchy_df <- as.data.frame(arrow::read_parquet(hierarchy_path))

observed_aa_df = as.data.frame(arrow::read_parquet(observed_aa_path))

observed_as_df = as.data.frame(arrow::read_parquet(observed_as_path))
observed_as_df$most_detailed_gbd = hierarchy_df$most_detailed_gbd[match(observed_as_df$location_id, hierarchy_df$location_id)]
observed_as_df = observed_as_df[which(observed_as_df$year_id == 2023 & observed_as_df$most_detailed_gbd == 1),]


past_data <- as.data.frame(arrow::read_parquet(parquet_path))
past_data$gbd_location_id <- hierarchy_df$gbd_location_id[match(past_data$location_id, hierarchy_df$location_id)]
past_data$A0_af <- as.factor(past_data$A0_location_id)
past_data$r_af <- as.factor(past_data$region_location_id)
past_data$sr_af <- as.factor(past_data$super_region_location_id)

suit_col  <- "dengue_suitability"
urban_col <- "weighted_1km_urban_threshold_300.0_simple_mean"
past_data$dengue_suit_fraction     <- past_data[[suit_col]] / 365
past_data$dengue_suit_fraction     <- pmin(pmax(past_data$dengue_suit_fraction, 0.001), 0.999)
past_data$urban_fraction = pmin(pmax(past_data[[urban_col]], 0.001), 0.999)
past_data$logit_dengue_suitability <- log(past_data$dengue_suit_fraction / (1 - past_data$dengue_suit_fraction))
past_data$logit_urban_fraction <- log(past_data$urban_fraction / (1 - past_data$urban_fraction))
past_data$do30_fraction <- past_data$days_over_30C / 365
past_data$do30_fraction <- pmin(pmax(past_data$do30_fraction, 0.001), 0.999)
past_data$logit_do30    <- log(past_data$do30_fraction / (1 - past_data$do30_fraction))
past_data$rh_fraction <- past_data$relative_humidity / 100
past_data$rh_fraction <- pmin(pmax(past_data$rh_fraction, 0.001), 0.999)
past_data$logit_relative_humidity <- log(past_data$rh_fraction / (1 - past_data$rh_fraction))
past_data$cfr <- fifelse(past_data$dengue_inc_rate > 0, past_data$dengue_mort_rate / past_data$dengue_inc_rate, NA_real_)
past_data$logit_cfr <- log(past_data$cfr / (1 - past_data$cfr))

log_covs <- c("dengue_inc_rate", "gdppc_mean", "ldipc_mean")
for (cov in log_covs) {
  past_data[[paste0("log_", cov)]] <- log(past_data[[cov]])
}

apply_shift <- function(dt, raw_col, obs_dt, obs_col, rake_years_vec) {
  ry      <- unname(rake_years_vec[as.character(dt$location_id)])  # each row's loc rake year
  at_rake <- dt[dt$year_id == ry, .(location_id, raw_at_rake = get(raw_col))]  # one row/loc
  sh      <- merge(at_rake, obs_dt, by = "location_id", all.x = TRUE)
  sh[, shift := get(obs_col) - raw_at_rake]
  dt[[raw_col]] + sh$shift[match(dt$location_id, sh$location_id)]
}






# <<CONFIRM>> dengue reference (base) age group: the age group the base model is
# fit on; the rest model predicts the others relative to it. Pull from
# mbpc.cause_map['dengue'] / the dengue modeling-df builder.
library(data.table)
setDT(observed_as_df); setDT(past_data)   # observed_as_df already filtered to 2023 & most_detailed_gbd

reference_age_group_id <- 7
reference_sex_id        <- 2

# --- reference (base) group + age-sex relative risk ---
base_md_gbd_df <- observed_as_df[age_group_id == reference_age_group_id &
                                 sex_id == reference_sex_id & dengue_inc_count > 0]
base_md_gbd_location_ids <- unique(base_md_gbd_df$location_id)

rr_df <- observed_as_df[location_id %in% base_md_gbd_location_ids]
rr_df[base_md_gbd_df, base_inc_rate := i.dengue_inc_rate, on = "location_id"]
rr_df[, rr_inc_as := dengue_inc_rate / base_inc_rate]
setnames(rr_df, "location_id", "gbd_location_id")

# --- restrict past_data to base locations; attach rr_inc_as (fast update-join) ---
tmp_past_data <- past_data[gbd_location_id  %in% base_md_gbd_location_ids]
tmp_past_data[rr_df, rr_inc_as := i.rr_inc_as,
              on = c("gbd_location_id", "age_group_id", "sex_id")]

# --- base modelling frame + fit ---
base_inc_data <- tmp_past_data[age_group_id == reference_age_group_id &
                               sex_id == reference_sex_id & dengue_inc_rate > 0]
inc_mod <- scam(log_dengue_inc_rate ~ s(dengue_suitability, k = 6, bs = 'mpi', by  = sr_af) + urban_fraction +
                relative_humidity + A0_af, data = base_inc_data)


# inc_mod <- scam(log_dengue_inc_rate ~  s(dengue_suitability, k = 6, bs = 'mpi') + 
#         s(urban_fraction, k = 6, bs = 'mpi') + 
#         s(relative_humidity, k = 6, bs = 'mpi') +
#         people_flood_days_per_capita + A0_af,
#     data = base_inc_data, optimizer = OPTIMIZER, control = list(maxit = 5))
base_inc_data[, base_log_dengue_inc_rate_pred := fitted(inc_mod)]

# --- shift: anchor the predicted base log-rate to observed at the rake year ---
rake_year      <- 2023L                          # <<SET: year to anchor to observed>>
locs           <- unique(base_inc_data$location_id)
rake_years_vec <- setNames(rep(rake_year, length(locs)), as.character(locs))
obs_dt         <- base_inc_data[year_id == rake_year,
                                .(location_id, obs_base_log = log_dengue_inc_rate)]
base_inc_data[, base_log_dengue_inc_rate_pred :=
                apply_shift(base_inc_data, "base_log_dengue_inc_rate_pred",
                            obs_dt, "obs_base_log", rake_years_vec)]

# --- broadcast the (shifted) base prediction to all age/sex (fast update-join) ---
tmp_past_data[base_inc_data, base_log_dengue_inc_rate_pred := i.base_log_dengue_inc_rate_pred,
              on = c("location_id", "year_id")]

# --- age-sex incidence ---
tmp_past_data[, dengue_inc_rate_pred  := exp(base_log_dengue_inc_rate_pred) * rr_inc_as]
tmp_past_data[, dengue_inc_count_pred := population * dengue_inc_rate_pred]

tmp_past_data[, super_region_id := hierarchy_df$super_region_id[match(gbd_location_id, hierarchy_df$location_id)]]
tmp_sr_aa = tmp_past_data[, .(
  dengue_inc_count_pred = sum(dengue_inc_count_pred, na.rm = TRUE)
), by = .(super_region_id, year_id)]

tmp_sr_aa[obs_aa, `:=`(
  dengue_inc_count_obs = i.dengue_inc_count,
  dengue_inc_rate_obs = i.dengue_inc_rate,
  population_obs       = i.population
), on = c("super_region_id" = "location_id", "year_id")]

tmp_sr_aa[, dengue_inc_rate_pred := dengue_inc_count_pred / population_obs]
tmp_sr_aa[, super_region_name := hierarchy_df$super_region_name[match(super_region_id, hierarchy_df$location_id)]]


httpgd::hgd()


par(mfrow = c(3, 2), mar = c(4, 4, 2, 1))
for (sr in unique(tmp_sr_aa$super_region_id)) {
  d <- tmp_sr_aa[super_region_id == sr][order(year_id)]
  sr_name <- d$super_region_name[1]
  plot(d$year_id, d$dengue_inc_rate_obs, type = "l",
       ylim = range(c(d$dengue_inc_rate_obs, d$dengue_inc_rate_pred), na.rm = TRUE),
       xlab = "Year", ylab = "Inc rate", main = sr_name)
  lines(d$year_id, d$dengue_inc_rate_pred, col = "red")
  legend("topleft", c("observed", "predicted"), col = c("black", "red"), lty = 1, bty = "n")
}


# What is the correlation between predicted and observed all-age incidence rates across super-regions and years?
cor(tmp_sr_aa$dengue_inc_rate_pred, tmp_sr_aa$dengue_inc_rate_obs, use = "complete.obs")
lm_fit <- lm(dengue_inc_rate_obs ~ dengue_inc_rate_pred, data = tmp_sr_aa)
summary(lm_fit)$coef[2,1]


