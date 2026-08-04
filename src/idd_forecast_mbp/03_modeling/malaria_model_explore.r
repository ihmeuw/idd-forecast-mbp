rm(list = ls())

require(glue)
require(mgcv)
require(scam)
require(arrow)
require(data.table)
require(parallel)

OPTIMIZER    <- "efs"
MAXIT        <- 300L
N_FOLDS      <- 5L
CV_SEED      <- 42L
TESTING      <- FALSE


parquet_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/past_inputs_nc/lsae_1285/current/malaria_past_inputs.parquet"
suit_variant_pick <- "mordecai_0_0"


past_data <- as.data.frame(arrow::read_parquet(parquet_path))

past_data$malaria_inc_count <- past_data$malaria_inc_rate * past_data$population

nan_toss <- function(df, var) {
  to_toss <- which(is.na(df[var]))
  if (length(to_toss)) df[-to_toss, ] else df
}
past_data <- nan_toss(past_data, "malaria_pfpr")
past_data <- nan_toss(past_data, "gdppc_mean")
past_data <- nan_toss(past_data, "mal_DAH_total_per_capita")

suit_col <- paste0("malaria_suitability_", suit_variant_pick)
past_data$malaria_suit <- past_data[[suit_col]]
# Drop all columns with suit in them except the one we just made
past_data <- past_data[, !grepl("suit", names(past_data)) | names(past_data) == "malaria_suit"]

past_data$malaria_suit_fraction     <- past_data$malaria_suit / 365
past_data$malaria_suit_fraction     <- pmin(pmax(past_data$malaria_suit_fraction, 0.001), 0.999)
past_data$logit_malaria_suitability <- log(past_data$malaria_suit_fraction / (1 - past_data$malaria_suit_fraction))
past_data$do30_fraction <- past_data$days_over_30C / 365
past_data$do30_fraction <- pmin(pmax(past_data$do30_fraction, 0.001), 0.999)
past_data$logit_do30    <- log(past_data$do30_fraction / (1 - past_data$do30_fraction))
past_data$rh_fraction <- past_data$relative_humidity / 100
past_data$rh_fraction <- pmin(pmax(past_data$rh_fraction, 0.001), 0.999)
past_data$logit_relative_humidity <- log(past_data$rh_fraction / (1 - past_data$rh_fraction))




log_covs <- c("mal_DAH_total_per_capita", "gdppc_mean", "ldipc_mean", "med_consumppc")
for (cov in log_covs) {
  past_data[[paste0("log_", cov)]] <- log(past_data[[cov]])
}

sub_data <- past_data[which(past_data$malaria_inc_count >= 1 & past_data$malaria_pfpr >= 0.0001),]
past_data$A0_af <- as.factor(past_data$A0_location_id)
sub_data$A0_af <- as.factor(sub_data$A0_location_id)


ilogit <- function(x) 1 / (1 + exp(-x))



stripped_summary <- function(mod) {
  s <- summary(mod)
  # scam objects are class "scam"; mgcv gam is c("gam","glm","lm") — so this
  # MUST be checked before any inherits(mod, "lm"), or a gam falls into the lm path.
  is_gam_like <- inherits(mod, "scam") || inherits(mod, "gam")

  # Residuals. NOTE: mod$residuals are working residuals for gam/scam, but these
  # models are all gaussian + identity link on the logit-transformed response,
  # so working == response residuals here. (Would mislead under a non-identity link.)
  cat("\nResiduals:\n")
  print(summary(mod$residuals))

  if (is_gam_like) {
    # Parametric terms live in p.table (not $coefficients); drop the A0_ fixed effects
    cat("\nParametric coefficients:\n")
    p <- s$p.table[!grepl("A0_", rownames(s$p.table)), , drop = FALSE]
    printCoefmat(p, signif.stars = TRUE, signif.legend = FALSE)

    # Smooth terms have their own table (edf / Ref.df / F / p-value)
    cat("\nApproximate significance of smooth terms:\n")
    printCoefmat(s$s.table, has.Pvalue = TRUE, signif.stars = TRUE)

    # gam/scam report sqrt(scale) for residual SE, r.sq (adjusted), dev.expl; no F-stat
    df_resid <- if (!is.null(s$residual.df)) s$residual.df else mod$df.residual
    cat("\nResidual standard error:", round(sqrt(s$scale), 4),
        "on", round(df_resid, 1), "effective degrees of freedom\n")
    cat("Adjusted R-squared:", round(s$r.sq, 4),
        ",  Deviance explained:", paste0(round(100 * s$dev.expl, 2), "%"), "\n")

  } else {
    # original lm path
    cat("\nCoefficients:\n")
    cf <- s$coefficients[!grepl("A0_", rownames(s$coefficients)), , drop = FALSE]
    printCoefmat(cf, signif.stars = TRUE)
    cat("\nResidual standard error:", round(s$sigma, 4), "on", s$df[2], "degrees of freedom\n")
    cat("Multiple R-squared:", round(s$r.squared, 4),
        ",  Adjusted R-squared:", round(s$adj.r.squared, 4), "\n")
    cat("F-statistic:", sprintf("%.3e", s$fstatistic[1]), "on",
        s$fstatistic[2], "and", s$fstatistic[3], "DF,  p-value: < 2.2e-16\n")
  }

  # Natural-space R² — identical logic both ways; fitted + residuals reconstructs
  # the observed value on the logit scale, then ilogit() both back to prevalence
  fitted_natural   <- ilogit(mod$fitted.values)
  observed_natural <- ilogit(mod$fitted.values + mod$residuals)
  cor_natural      <- cor(fitted_natural, observed_natural)
  cat("\nR-squared in natural space:", round(cor_natural^2, 4), "\n")
}

mod_1 <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + 
              logit_malaria_suitability + mean_low_temperature + A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_2 <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + 
              s(malaria_suit, k = 6, bs = "mpi") + mean_low_temperature + A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_3 <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + 
            logit_malaria_suitability + s(mean_low_temperature, k = 4) + A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_4 <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + 
          s(malaria_suit, k = 6, bs = "mpi") + s(mean_low_temperature, k = 4) + A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_5 <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + 
            logit_malaria_suitability + A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_6 <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + 
          s(malaria_suit, k = 6, bs = "mpi") + A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))






stripped_summary(mod_1)
stripped_summary(mod_2)
stripped_summary(mod_3)
stripped_summary(mod_4)
stripped_summary(mod_5)
stripped_summary(mod_6)


require(httpgd)
httpgd::hgd()
plot(mod_1, pages = 1, scale = 0, rug = FALSE)
mtext("Model 1: logit_malaria_suitability + mean_low_temperature", side = 3, line = 2, cex = 0.9)
plot(mod_2, pages = 1, scale = 0, rug = FALSE)
mtext("Model 2: s(malaria_suit, k=6, bs='mpi') + mean_low_temperature", side = 3, line = 2, cex = 0.9)
plot(mod_3, pages = 1, scale = 0, rug = FALSE)
mtext("Model 3: logit_malaria_suitability + s(mean_low_temperature, k=4)", side = 3, line = 2, cex = 0.9)
plot(mod_4, pages = 1, scale = 0, rug = FALSE)
mtext("Model 4: s(malaria_suit, k=6, bs='mpi') + s(mean_low_temperature, k=4)", side = 3, line = 2, cex = 0.9)
plot(mod_5, pages = 1, scale = 0, rug = FALSE)
mtext("Model 5: logit_malaria_suitability + A0_af", side = 3, line = 2, cex = 0.9)
plot(mod_6, pages = 1, scale = 0, rug = FALSE)
mtext("Model 6: s(malaria_suit, k=6, bs='mpi') + A0_af", side = 3, line = 2, cex = 0.9)

cor(sub_data$logit_malaria_suitability, sub_data$mean_low_temperature)
cor(sub_data$malaria_suit, sub_data$mean_low_temperature)


# Model 1 is Model 5 + mean_low_temperature
# Model 2 is Model 6 + mean_low_temperature
# Model 3 is Model 5 + s(mean_low_temperature, k=4)
# Model 4 is Model 6 + s(mean_low_temperature, k=4)

sub_data$model_1_fit <- mod_1$fitted.values
sub_data$model_1_res <- mod_1$residuals
sub_data$model_2_fit <- mod_2$fitted.values
sub_data$model_2_res <- mod_2$residuals
sub_data$model_3_fit <- mod_3$fitted.values
sub_data$model_3_res <- mod_3$residuals
sub_data$model_4_fit <- mod_4$fitted.values
sub_data$model_4_res <- mod_4$residuals
sub_data$model_5_fit <- mod_5$fitted.values
sub_data$model_5_res <- mod_5$residuals
sub_data$model_6_fit <- mod_6$fitted.values
sub_data$model_6_res <- mod_6$residuals

low_1_5 <- which(abs(sub_data$model_1_res) < abs(sub_data$model_5_res))
low_5_1 <- which(abs(sub_data$model_1_res) > abs(sub_data$model_5_res))

par(mfrow = c(2, 3))
hist(sub_data$malaria_suit[low_1_5], 
breaks = 30, main = "", xlab = "malaria_suit")
hist(sub_data$mean_low_temperature[low_1_5], 
breaks = 30, main = "Model 1 abs(res) < Model 5 abs(res)", xlab = "mean_low_temperature")
hist(sub_data$logit_malaria_pfpr[low_1_5], 
breaks = 30, main = "", xlab = "logit_malaria_pfpr")

hist(sub_data$malaria_suit[low_5_1], 
breaks = 30, main = "", xlab = "malaria_suit")

hist(sub_data$mean_low_temperature[low_5_1], 
breaks = 30, main = "Model 1 abs(res) > Model 5 abs(res)", xlab = "mean_low_temperature")
hist(sub_data$logit_malaria_pfpr[low_5_1], 
breaks = 30, main = "", xlab = "logit_malaria_pfpr")







mod_test <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + 
          s(malaria_suit, k = 6, bs = "mpi") + A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))











plot(mod_4  , pages = 1, scale = 0, rug = FALSE)
mod_0b <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + logit_malaria_suitability + mean_low_temperature + A0_af,
            data=sub_data, method = "REML", select = TRUE)
mod_1c <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + s(malaria_suit, k = 6, bs = "mpi") + mean_low_temperature + A0_af,
            data=sub_data, method = "REML", select = TRUE)

mod_1d <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + logit_malaria_suitability + s(mean_low_temperature, k = 4) + A0_af,
            data=sub_data, method = "REML", select = TRUE)
mod_1e <- scam(logit_malaria_pfpr ~ s(mal_DAH_total_per_capita, k = 4, bs = "mpd") + s(gdppc_mean, k = 4, bs = "mpd") + s(malaria_suit, k = 6, bs = "mpi") + s(mean_low_temperature, k = 4) + A0_af,
            data=sub_data, method = "REML", select = TRUE)

, method = "REML", select = TRUE)









# base <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/scam_prelim/lsae_1285/20260701_efs_v2"

# spec <- as.data.table(read_parquet(file.path(base, "spec_table.parquet")))  # spec_index, n_smooths, formula_text

# # If you already know the dead task id from the GUI (e.g. s37): task_id s<N> == spec_index N
# formula_text = spec[spec_index == 11, formula_text]
# # Find malaria_suitability_mordecai_0_0 in formula_text and replace with malaria_suit
# fml = as.formula(formula_text)


# mod <- scam(fml, data=sub_data, optimizer="efs", control=list(maxit=50))
# require(httpgd)
# httpgd::hgd()
# plot(mod, pages = 1, scale = 0, rug = FALSE)



library(data.table); library(arrow)

dir <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/scam_prelim/lsae_1285/20260706_efs"

# --- load the run: glob bundled files directly (no stale probe files in this dir) ---
files <- list.files(dir, "^select_summary_(IS|within|tempA|tempB)_n\\d+_bin\\d+\\.parquet$",
                     full.names = TRUE)
res <- rbindlist(lapply(files, function(f) as.data.table(read_parquet(f))),
                 use.names = TRUE, fill = TRUE)
res[, cell := sub("_n[0-9]+_bin[0-9]+$", "", task_id)]   # IS / within / tempA / tempB

# --- authoritative spec metadata (all 459 specs) ---
spec_tab <- as.data.table(read_parquet(file.path(dir, "spec_table.parquet")))

# --- IS metrics: one row per spec ---
is_dt <- res[cell == "IS", .(spec_index, is_aic, is_bic, is_dev_expl, is_pfpr_r)]

# --- OOS pfpr correlation: one column per cell ---
oos_dt <- dcast(res[cell != "IS"], spec_index ~ cell, value.var = "oos_pfpr_r")

# --- master per-spec table ---
tab <- Reduce(function(a, b) merge(a, b, by = "spec_index", all.x = TRUE),
              list(spec_tab, is_dt, oos_dt))
tab[, temporal_oos := rowMeans(.SD, na.rm = TRUE), .SDcols = c("tempA", "tempB")]

# coverage check + rank only on specs with all 3 OOS cells (avoids NaN-at-top)
message(sprintf("specs missing >=1 OOS cell: %d of %d",
                tab[is.na(within) | is.na(tempA) | is.na(tempB), .N], nrow(tab)))
tab_ok <- tab[!is.na(within) & !is.na(tempA) & !is.na(tempB)]
setorder(tab_ok, -temporal_oos)

# 1) TOP 15 by temporal OOS (the selection metric) — is AIC agreeing?
tab_ok[, .(spec_index, n_smooths, is_aic, is_pfpr_r,
           within, tempA, tempB, temporal_oos)][1:15]

# 2) By complexity: does more n_smooths buy OOS skill? (the flatness question)
tab_ok[, .(n_specs           = .N,
           aic_min           = min(is_aic, na.rm = TRUE),
           temporal_oos_max  = max(temporal_oos, na.rm = TRUE),
           temporal_oos_mean = mean(temporal_oos, na.rm = TRUE),
           within_mean       = mean(within, na.rm = TRUE)),
       by = n_smooths][order(n_smooths)]

# 3) AIC winner vs OOS winner — the mismatch, in one place
rbind(
  cbind(pick = "AIC-best", tab_ok[which.min(is_aic),       .(spec_index, n_smooths, is_aic, temporal_oos)]),
  cbind(pick = "OOS-best", tab_ok[which.max(temporal_oos), .(spec_index, n_smooths, is_aic, temporal_oos)])
)

# 4) Flatness, quantified: spread across the top 20 temporal-OOS specs
sorted <- sort(tab_ok$temporal_oos, decreasing = TRUE)
data.table(best = sorted[1], rank20 = sorted[20],
           spread_top20 = sorted[1] - sorted[20],
           median = median(tab_ok$temporal_oos, na.rm = TRUE))





library(data.table)

## --- covariate-family presence, parsed from the formula string ---
## (DAH / gdppc / A0 are always-in, so they don't vary — omit them)
fam <- list(
  suitability = "malaria_suit|logit_malaria_suitability",
  temperature = "mean_(low_)?temperature",
  humidity    = "relative_humidity",
  urban       = "weighted_1km_urban",
  precip      = "total_precipitation"
)
for (k in names(fam)) tab[, (k) := grepl(fam[[k]], formula_text)]

## --- Q1a: MARGINAL — mean temporal_OOS with vs without each family ---
q1a <- rbindlist(lapply(names(fam), function(k) {
  p <- tab[[k]]
  data.table(family = k, n_with = sum(p),
             oos_with    = mean(tab$temporal_oos[p]),
             oos_without = mean(tab$temporal_oos[!p]),
             delta       = mean(tab$temporal_oos[p]) - mean(tab$temporal_oos[!p]))
}))[order(-delta)]
q1a

## --- Q1b: ADJUSTED — each family's effect holding the others constant ---
## suitability dominates; this isolates what each term adds *given* the rest.
fit <- lm(temporal_oos ~ suitability + temperature + humidity + urban + precip, data = tab)
round(summary(fit)$coefficients, 5)

## --- Q1c: STRATIFY each non-suit covariate by suitability presence ---
strat <- rbindlist(lapply(setdiff(names(fam), "suitability"), function(k) {
  tab[, .(family = k, .N,
          oos_with    = mean(temporal_oos[get(k)]),
          oos_without = mean(temporal_oos[!get(k)]),
          delta       = mean(temporal_oos[get(k)]) - mean(temporal_oos[!get(k)])),
      by = .(suit_present = suitability)]
}))
strat[order(family, -suit_present)]


## --- Q1 by WINDOW: split the with/without delta across within / tempA / tempB ---
## temporal_oos averaged tempA+tempB; this shows whether an effect is window-specific.

# long OOS by cell, joined to the family-presence flags already on `tab`
oos_long <- merge(
  res[cell %in% c("within", "tempA", "tempB"), .(spec_index, cell, oos_pfpr_r)],
  tab[, c("spec_index", names(fam)), with = FALSE],
  by = "spec_index"
)

# with/without delta, one column per window
by_window <- rbindlist(lapply(names(fam), function(k) {
  oos_long[, .(family = k,
               delta = mean(oos_pfpr_r[get(k)]) - mean(oos_pfpr_r[!get(k)])),
           by = cell]
}))
dcast(by_window, family ~ cell, value.var = "delta")[, .(family, within, tempA, tempB)]

## temperature delta by window AND suit presence (is the substitution in both windows?)
oos_long[, .(temp_delta = mean(oos_pfpr_r[temperature]) - mean(oos_pfpr_r[!temperature]),
             n_with     = sum(temperature)),
         by = .(cell, suit_present = suitability)][order(cell, -suit_present)]




## ============================================================
## DROP TEST — does the FRONTIER need each covariate? (max, not mean)
## Droppable if the best model WITHOUT it matches the best WITH it.
## ============================================================
frontier <- rbindlist(lapply(names(fam), function(k) {
  oos_long[, .(family = k,
               best_with    = max(oos_pfpr_r[get(k)]),
               best_without = max(oos_pfpr_r[!get(k)]),
               gain         = max(oos_pfpr_r[get(k)]) - max(oos_pfpr_r[!get(k)])),
           by = cell]
}))

# gain > 0  -> the best model in that cell USES the covariate  (keep)
# gain <= 0 -> best model doesn't need it                       (drop candidate)
dcast(frontier, family ~ cell, value.var = "gain")[, .(family, within, tempA, tempB)]

## definitive drop verdict on the temporal windows
NOISE <- 0.0005   # ~ within-fold pfpr_r sd floor; tune to taste
verdict <- dcast(frontier[cell %in% c("tempA", "tempB")], family ~ cell, value.var = "gain")
verdict[, drop := tempA <= NOISE & tempB <= NOISE]
verdict[]














base_is_ss_res = efs$is_deviance[which(efs$spec_index == 1)]

efs <- res[optimizer == "efs"]
efs[order(is_deviance),
    .(spec_index,
      n_smooths = is_n_smooths,
      conv      = is_converged,
      is_r_sq,        
      oos_r_sq,
      is_pfpr_r,      # PfPR-space correlation
      is_mae,         # logit-space MAE
      is_pfpr_mae,
      add_r_sq = 1 - is_deviance / base_is_ss_res)] 

efs[order(-oos_pfpr_r),
    .(spec_index,
      n_smooths = is_n_smooths,
      conv      = is_converged,
      oos_r_sq,        # logit-space R^2
      oos_pfpr_r,      # PfPR-space correlation
      oos_mae,         # logit-space MAE
      oos_pfpr_mae)]   # PfPR-space MAE



efs[, .(pfpr_r_spread  = max(oos_pfpr_r) - min(oos_pfpr_r),
        median_fold_sd = median(cv_fold_pfpr_r_sd, na.rm = TRUE))]


efs[order(-oos_pfpr_r),
    .(spec_index, n_smooths = is_n_smooths, oos_pfpr_r,
      fold_r_min = cv_fold_pfpr_r_min,   # worst fold — robustness
      fold_r_sd  = cv_fold_pfpr_r_sd,    # fold-to-fold volatility
      cv_fold_pfpr_r_per_fold)]          # the 5 raw values

efs[order(-cv_fold_pfpr_r_min),.(spec_index, oos_pfpr_r_sq = oos_pfpr_r^2,
      fold_r_min_sq = cv_fold_pfpr_r_min^2,
      fold_r_sd  = cv_fold_pfpr_r_sd,
      cv_fold_pfpr_r_per_fold)]



mod_0b <- scam(logit_malaria_pfpr ~ s(malaria_suit, k=6, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=4, bs='mpd') +
              s(gdppc_mean, k=4, bs='mpd') +
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))

mod_1a <- scam(logit_malaria_pfpr ~ s(malaria_suit, k=6, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=4, bs='mpd') +
              s(gdppc_mean, k=4, bs='mpd') +
              s(relative_humidity, k=6, bs='mpi') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))

mod_1b <- scam(logit_malaria_pfpr ~ s(malaria_suit, k=6) + 
              s(mal_DAH_total_per_capita, k=4, bs='mpd') +
              s(gdppc_mean, k=4, bs='mpd') +
              s(people_flood_days_per_capita, k = 4, bs = 'mpi') +
              s(weighted_1km_urban_threshold_300.0_simple_mean, k=4) + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))

            
mod_1c <- scam(logit_malaria_pfpr ~ s(malaria_suit, k=6, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=4, bs='mpd') +
              s(gdppc_mean, k=4, bs='mpd') +
              s(total_precipitation, k=6, bs='cv') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))

mod_2ab <- scam(logit_malaria_pfpr ~ s(malaria_suit, k=6, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=4, bs='mpd') +
              s(gdppc_mean, k=4, bs='mpd') +
              s(relative_humidity, k=6, bs='mpi') + 
              s(weighted_1km_urban_threshold_300.0_simple_mean, k=4, bs='cv') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))

mod_2ac <- scam(logit_malaria_pfpr ~ s(malaria_suit, k=6, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=4, bs='mpd') +
              s(gdppc_mean, k=4, bs='mpd') +
              s(relative_humidity, k=6, bs='mpi') + 
              s(total_precipitation, k=6, bs='cv') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))

            
mod_2bc <- scam(logit_malaria_pfpr ~ s(malaria_suit, k=6, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=4, bs='mpd') +
              s(gdppc_mean, k=4, bs='mpd') +
              s(weighted_1km_urban_threshold_300.0_simple_mean, k=4, bs='cv') + 
              s(total_precipitation, k=6, bs='cv') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))

mod_3abc <- scam(logit_malaria_pfpr ~ s(malaria_suit, k=6, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=4, bs='mpd') +
              s(gdppc_mean, k=4, bs='mpd') +
              s(relative_humidity, k=6, bs='mpi') + 
              s(weighted_1km_urban_threshold_300.0_simple_mean, k=4, bs='cv') + 
              s(total_precipitation, k=6, bs='cv') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))


require(httpgd)
httpgd::hgd()
plot(mod_0b, pages = 1, scale = 0, rug = FALSE)
plot(mod_1a, pages = 1, scale = 0, rug = FALSE)
plot(mod_1b, pages = 1, scale = 0, rug = FALSE)
plot(mod_1c, pages = 1, scale = 0, rug = FALSE)
plot(mod_2ab, pages = 1, scale = 0, rug = FALSE)
plot(mod_2ac, pages = 1, scale = 0, rug = FALSE)
plot(mod_2bc, pages = 1, scale = 0, rug = FALSE)
plot(mod_3abc, pages = 1, scale = 0, rug = FALSE)




stripped_summary(mod_1a)
stripped_summary(mod_1b)
stripped_summary(mod_1c)
stripped_summary(mod_2ab)
stripped_summary(mod_2ac)
stripped_summary(mod_2bc)
stripped_summary(mod_3abc)

seeds <- c(1, 2, 3)
for (s_num in seq_along(seeds)){
  seed <- seeds[s_num]
  set.seed(seed)
  sub_data[glue('fold_{seed}')] <- ave(
    seq_len(nrow(sub_data)),
    sub_data$A0_location_id,
    FUN = function(i) sample(rep_len(1:K, length(i)))
  )
}


K <- 5













table(sub_data$A0_location_id, sub_data$year_id)

sub_data$year_bin <- ifelse(sub_data$year_id <= 2014, "2000-2014",
                        ifelse(sub_data$year_id <= 2020, "2015-2020", "2021-2023"))

table(sub_data$A0_location_id, sub_data$year_bin)











cor(sub_data$total_precipitation, sub_data$precipitation_days)





















bad_dah <- which(sub_data$mal_DAH_total_per_capita == 0)
sub_data$log_mal_DAH_total_per_capita[bad_dah] <- 0.01 * min(sub_data$log_mal_DAH_total_per_capita[-bad_dah])
mod_0a <- lm(logit_malaria_pfpr ~ logit_malaria_suitability + log_mal_DAH_total_per_capita + log_gdppc_mean + logit_relative_humidity + A0_af,
            data=sub_data)

mod_0b <- lm(logit_malaria_pfpr ~ logit_malaria_suitability + log_mal_DAH_total_per_capita + log_gdppc_mean + logit_relative_humidity + weighted_1km_urban_threshold_300.0_simple_mean + total_precipitation + A0_af,
            data=sub_data)

mod_0c <- lm(logit_malaria_pfpr ~ logit_malaria_suitability + mal_DAH_total_per_capita + log_gdppc_mean + logit_relative_humidity + weighted_1km_urban_threshold_300.0_simple_mean + total_precipitation + A0_af,
            data=sub_data)
# print summary(mod_0) for all terms that do not contain 'A0_'
stripped_summary(mod_0a)
stripped_summary(mod_0b)
stripped_summary(mod_0c)

mod_1a <- scam(logit_malaria_pfpr ~ s(malaria_suit, k=6, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              s(relative_humidity, k=6, bs='mpi') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))

cor(ilogit(mod_1a$fitted.values), ilogit(mod_1a$y))^2
require(httpgd)
httpgd::hgd()
plot(mod_1a, pages = 1, scale = 0)

df <- sub_data[,c("logit_malaria_pfpr", "mal_DAH_total_per_capita", "gdppc_mean", "ldipc_mean",
  "weighted_1km_urban_threshold_300.0_simple_mean", "weighted_100m_urban_threshold_300.0_simple_mean", 
  "weighted_1km_urban_threshold_1500.0_simple_mean", "weighted_100m_urban_threshold_1500.0_simple_mean", 
  "people_flood_days_per_capita", "med_consumppc", "total_precipitation", "precipitation_days", "relative_humidity",
  "wind_speed", "mean_temperature", "mean_low_temperature", "mean_high_temperature", "days_over_30C", "malaria_suit",
  "A0_af")]

library(ranger)
rf <- ranger(
  logit_malaria_pfpr ~ .,
  data = df,
  num.trees = 500,
  mtry = floor((dim(df)[2]-1) / 3),      # default for regression is p/3, not sqrt(p)
  importance = "permutation",
  respect.unordered.factors = "order",
  num.threads = parallel::detectCores()
)

pred <- predict(rf, data = df)$predictions
rf$variable.importance
par(oma = c(0,10,0,0))
barplot(sort(rf$variable.importance), horiz = TRUE, las = 1,
        xlab = "Permutation importance")

# install.packages("pdp", lib = "~/packages")
library(pdp, lib.loc = "~/packages")

# one covariate
partial(rf, pred.var = "malaria_suit", train = df,
        grid.resolution = 20,          # <- 20 points is plenty for a shape
        plot = TRUE, plot.engine = "ggplot2")




# a factor — gives mean prediction per level
partial(rf, pred.var = "A0_af", train = df, plot = TRUE)




mod_1b<- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              s(relative_humidity, k=6, bs='mpi') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_2a <- scam(logit_malaria_pfpr ~ s(malaria_suit, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              logit_relative_humidity + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_2b<- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              logit_relative_humidity + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_3a <- scam(logit_malaria_pfpr ~ s(malaria_suit, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              s(people_flood_days_per_capita, k=6, bs='mpi') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_3b<- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              s(people_flood_days_per_capita, k=6, bs='mpi') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_3c <- scam(logit_malaria_pfpr ~ s(malaria_suit, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              s(people_flood_days_per_capita, k=6, bs='cv') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
mod_3d <- scam(logit_malaria_pfpr ~ logit_malaria_suitability + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              s(people_flood_days_per_capita, k=6, bs='cv') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))



base_formula_text <- "logit_malaria_pfpr ~ s(malaria_suit, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              A0_af"

base_formula <- as.formula(base_formula_text)

full_flooding_formula_text <- paste0(base_formula_text, " + s(people_flood_days_per_capita, k=6, bs='mpi')")
post_flooding_formula_text <- "residuals ~ s(people_flood_days_per_capita, k=6, bs='mpi')"

full_logit_humidity_formula_text <- paste0(base_formula_text, " + s(logit_relative_humidity, k=6, bs='mpi')")
post_logit_humidity_formula_text <- "residuals ~ s(logit_relative_humidity, k=6, bs='mpi')"

full_humidity_formula_text <- paste0(base_formula_text, " + s(relative_humidity, k=6, bs='mpi')")
post_humidity_formula_text <- "residuals ~ s(relative_humidity, k=6, bs='mpi')"

full_both_formula_text <- paste0(base_formula_text, " + s(people_flood_days_per_capita, k=6, bs='mpi') + s(relative_humidity, k=6, bs='mpi')")
post_both_formula_text <- "residuals ~ s(people_flood_days_per_capita, k=6, bs='mpi') + s(relative_humidity, k=6, bs='mpi')"



sub_full_flooding_fit <- scam(as.formula(full_flooding_formula_text),
            data=sub_data, optimizer="efs")
full_flooding_fit <- scam(as.formula(full_flooding_formula_text),
            data=past_data, optimizer="efs")












# safety: ensure ilogit is in scope inside forked workers


last_year_df = past_data[which(past_data$year_id == 2023),]
par(mfrow = c(1, 2))
hist(last_year_df$malaria_pfpr, breaks = 30, main = "2023 malaria_pfpr", xlab = "malaria_pfpr")
hist(last_year_df$logit_malaria_pfpr, breaks = 30, main = "2023 logit_malaria_pfpr", xlab = "logit_malaria_pfpr")

sub_df = past_data[which(past_data$malaria_inc_count >= .1),]
hist(sub_df$malaria_pfpr, breaks = 30, main = "malaria_pfpr with inc > 0.5", xlab = "malaria_pfpr")
hist(sub_df$logit_malaria_pfpr, breaks = 30, main = "logit_malaria_pfpr with inc > 0.5", xlab = "logit_malaria_pfpr")

for (i in 1:10){
  threshold = 1*10^-i
  print(glue("checking pfpr < 1e-{i}"))
  wtf <- sub_df[which(sub_df$malaria_pfpr < threshold),]
  print(dim(wtf))
  #print(table(wtf$year_id))
  print(sum(wtf$malaria_inc_count))
  print(sum(sub_df$malaria_inc_count))
}


windows <- list(
  list(train_max = 2008, test = 2009:2014),
  list(train_max = 2014, test = 2015:2020),
  list(train_max = 2020, test = 2021:2023)
)
models <- list(flooding = full_flooding_formula_text,
               humidity = full_humidity_formula_text)

jobs <- expand.grid(w = seq_along(windows), m = names(models), stringsAsFactors = FALSE)

run_one <- function(i) {
  w <- windows[[ jobs$w[i] ]]; ftext <- models[[ jobs$m[i] ]]
  tr <- past_data[past_data$year_id <= w$train_max, ]
  te <- past_data[past_data$year_id %in% w$test, ]

  # guard: only score test rows whose country was estimable in training
  te <- te[te$A0_location_id %in% unique(tr$A0_location_id), ]

  fit <- scam(as.formula(ftext), data = tr, optimizer = "efs")
  te$pred <- ilogit(predict(fit, newdata = te))
  hi <- te[te$malaria_pfpr >= quantile(te$malaria_pfpr, 0.9, na.rm = TRUE), ]

  list(window   = w$train_max,
       model    = jobs$m[i],
       tail_mae = mean(abs(hi$pred - hi$malaria_pfpr)),
       full_mae = mean(abs(te$pred - te$malaria_pfpr)),
       edf      = sum(summary(fit)$s.table[, "edf"]))
}

# --- run job 1 synchronously first to surface any real error ---
str(run_one(1))

# --- then run all jobs in parallel ---
res <- mclapply(seq_len(nrow(jobs)),
                function(i) tryCatch(run_one(i), error = function(e) conditionMessage(e)),
                mc.cores = nrow(jobs))

res


no_fe_mod <- scam(logit_malaria_pfpr ~ s(malaria_suit, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              s(relative_humidity, k=6, bs='mpi'),
            data=past_data, optimizer="efs")
summary(no_fe_mod)


base_fit <- scam(base_formula,
            data=past_data, optimizer="efs")
past_data$residuals <- base_fit$residuals
past_data$base_fitted <- base_fit$fitted.values



full_flooding_fit <- scam(as.formula(full_flooding_formula_text),
            data=past_data, optimizer="efs")
post_flooding_fit <- scam(as.formula(post_flooding_formula_text), data=past_data, optimizer="efs")
past_data$full_flooding_fitted <- full_flooding_fit$fitted.values
past_data$post_flooding_fitted <- post_flooding_fit$fitted.values + past_data$base_fitted

full_logit_humidity_fit <- scam(as.formula(full_logit_humidity_formula_text),
            data=past_data, optimizer="efs")
post_logit_humidity_fit <- scam(as.formula(post_logit_humidity_formula_text), data=past_data, optimizer="efs")
past_data$full_logit_humidity_fitted <- full_logit_humidity_fit$fitted.values
past_data$post_logit_humidity_fitted <- post_logit_humidity_fit$fitted.values + past_data$base_fitted



full_humidity_fit <- scam(as.formula(full_humidity_formula_text),
            data=past_data, optimizer="efs")
post_humidity_fit <- scam(as.formula(post_humidity_formula_text), data=past_data, optimizer="efs")

past_data$full_humidity_fitted <- full_humidity_fit$fitted.values
past_data$post_humidity_fitted <- post_humidity_fit$fitted.values + past_data$base_fitted


# Both


full_both_fit <- scam(as.formula(full_both_formula_text),
            data=past_data, optimizer="efs")
post_both_fit <- scam(as.formula(post_both_formula_text), data=past_data, optimizer="efs")

past_data$full_both_fitted <- full_both_fit$fitted.values
past_data$post_both_fitted <- post_both_fit$fitted.values + past_data$base_fitted



past_data$full_flooding_fitted_natural <- ilogit(past_data$full_flooding_fitted)
past_data$post_flooding_fitted_natural <- ilogit(past_data$post_flooding_fitted)

past_data$full_logit_humidity_fitted_natural <- ilogit(past_data$full_logit_humidity_fitted)
past_data$post_logit_humidity_fitted_natural <- ilogit(past_data$post_logit_humidity_fitted)

past_data$full_humidity_fitted_natural <- ilogit(past_data$full_humidity_fitted)
past_data$post_humidity_fitted_natural <- ilogit(past_data$post_humidity_fitted)

past_data$full_both_fitted_natural <- ilogit(past_data$full_both_fitted)
past_data$post_both_fitted_natural <- ilogit(past_data$post_both_fitted)


print('Flooding model')
# cor(past_data$full_flooding_fitted_natural, past_data$malaria_pfpr)
# cor(past_data$post_flooding_fitted_natural, past_data$malaria_pfpr)
# cor(past_data$full_flooding_fitted, past_data$logit_malaria_pfpr)
# cor(past_data$post_flooding_fitted, past_data$logit_malaria_pfpr)
# full_flooding_fit$aic
mean(abs(past_data$full_flooding_fitted_natural - past_data$malaria_pfpr))
mean(abs(past_data$post_flooding_fitted_natural - past_data$malaria_pfpr))
median(abs(past_data$full_flooding_fitted_natural - past_data$malaria_pfpr))
median(abs(past_data$post_flooding_fitted_natural - past_data$malaria_pfpr))

print('Logit humidity model')
# cor(past_data$full_logit_humidity_fitted_natural, past_data$malaria_pfpr)
# cor(past_data$post_logit_humidity_fitted_natural, past_data$malaria_pfpr)
# cor(past_data$full_logit_humidity_fitted, past_data$logit_malaria_pfpr)
# cor(past_data$post_logit_humidity_fitted, past_data$logit_malaria_pfpr)
# full_logit_humidity_fit$aic
mean(abs(past_data$full_logit_humidity_fitted_natural - past_data$malaria_pfpr))
mean(abs(past_data$post_logit_humidity_fitted_natural - past_data$malaria_pfpr))
median(abs(past_data$full_logit_humidity_fitted_natural - past_data$malaria_pfpr))
median(abs(past_data$post_logit_humidity_fitted_natural - past_data$malaria_pfpr))

print('Humidity model')
# cor(past_data$full_humidity_fitted_natural, past_data$malaria_pfpr)
# cor(past_data$post_humidity_fitted_natural, past_data$malaria_pfpr)
# cor(past_data$full_humidity_fitted, past_data$logit_malaria_pfpr)
# cor(past_data$post_humidity_fitted, past_data$logit_malaria_pfpr)
# full_humidity_fit$aic
mean(abs(past_data$full_humidity_fitted_natural - past_data$malaria_pfpr))
mean(abs(past_data$post_humidity_fitted_natural - past_data$malaria_pfpr))
median(abs(past_data$full_humidity_fitted_natural - past_data$malaria_pfpr))
median(abs(past_data$post_humidity_fitted_natural - past_data$malaria_pfpr))

print('Both model')
# cor(past_data$full_both_fitted_natural, past_data$malaria_pfpr)
# cor(past_data$post_both_fitted_natural, past_data$malaria_pfpr)
# cor(past_data$full_both_fitted, past_data$logit_malaria_pfpr)
# cor(past_data$post_both_fitted, past_data$logit_malaria_pfpr)
# full_both_fit$aic
mean(abs(past_data$full_both_fitted_natural - past_data$malaria_pfpr))
mean(abs(past_data$post_both_fitted_natural - past_data$malaria_pfpr))
median(abs(past_data$full_both_fitted_natural - past_data$malaria_pfpr))
median(abs(past_data$post_both_fitted_natural - past_data$malaria_pfpr))




covs <- c("malaria_suit", "mal_DAH_total_per_capita", "gdppc_mean",
          "people_flood_days_per_capita", "relative_humidity")

split_year <- 2015  # whatever your train/test boundary is
train <- past_data[past_data$year_id <= split_year, ]
test  <- past_data[past_data$year_id >  split_year, ]

support_check <- function(train, test, covs) {
  do.call(rbind, lapply(covs, function(v) {
    rng   <- range(train[[v]], na.rm = TRUE)
    tv    <- test[[v]]
    below <- mean(tv < rng[1], na.rm = TRUE)
    above <- mean(tv > rng[2], na.rm = TRUE)
    data.frame(cov = v,
               train_min = rng[1], train_max = rng[2],
               test_min  = min(tv, na.rm = TRUE),
               test_max  = max(tv, na.rm = TRUE),
               frac_below = below, frac_above = above,
               frac_oos   = below + above)
  }))
}

support_check(train, test, covs)
hi <- test[test$malaria_pfpr >= quantile(test$malaria_pfpr, 0.9, na.rm = TRUE), ]

support_check(train, hi, covs)



hi_train <- train[train$malaria_pfpr >= quantile(train$malaria_pfpr, 0.9, na.rm=TRUE), ]
hi_test  <- test [test$malaria_pfpr  >= quantile(test$malaria_pfpr,  0.9, na.rm=TRUE), ]

sort(table(hi_train$A0_location_id) / nrow(hi_train), decreasing = TRUE)[1:10]
sort(table(hi_test$A0_location_id)  / nrow(hi_test),  decreasing = TRUE)[1:10]




windows <- c(2008, 2014, 2020)
fit_one <- function(tmax, ftext) {
  scam(as.formula(ftext), data = past_data[past_data$year_id <= tmax, ], optimizer = "efs")
}

# grid to evaluate each smooth on a common covariate sequence
newgrid <- function(var, n = 200) {
  rng <- range(past_data[[var]], na.rm = TRUE)
  seq(rng[1], rng[2], length.out = n)
}

# example: humidity smooth stability across windows
require(ggplot2)
smooth_by_window <- function(ftext, term_var, term_label) {
  do.call(rbind, lapply(windows, function(tmax) {
    fit <- fit_one(tmax, ftext)
    # hold other covariates at median, vary term_var
    nd <- past_data[1, , drop = FALSE][rep(1, 200), ]
    for (v in c("malaria_suit","mal_DAH_total_per_capita","gdppc_mean",
                "people_flood_days_per_capita","relative_humidity")) {
      if (v %in% names(nd)) nd[[v]] <- median(past_data[[v]], na.rm = TRUE)
    }
    nd[[term_var]] <- newgrid(term_var)
    nd$A0_af <- factor(levels(past_data$A0_af)[1], levels = levels(past_data$A0_af))
    data.frame(x = nd[[term_var]],
               fit = predict(fit, newdata = nd),
               window = factor(tmax), term = term_label)
  }))
}

hum <- smooth_by_window(full_humidity_formula_text, "relative_humidity", "humidity")
fld <- smooth_by_window(full_flooding_formula_text, "people_flood_days_per_capita", "flooding")

ggplot(rbind(hum, fld), aes(x, fit, colour = window)) +
  geom_line(linewidth = 0.7) +
  facet_wrap(~term, scales = "free_x") +
  labs(y = "partial logit contribution") + theme_minimal()







require(httpgd)
httpgd::hgd()
plot(fit, pages = 1, scale = 0)

par(mfrow = c(3, 2))
term = "people_flood_days_per_capita"
full_mod = full_flooding_fit
post_mod = post_flooding_fit

label <- paste0("s(", term, ")")

# grid over flooding; other covariates are irrelevant to the flooding term
# under type="terms", so just repeat any row and overwrite flooding
fl <- seq(min(past_data[[term]]), max(past_data[[term]]), length = 200)
nd <- past_data[rep(1, length(fl)), , drop = FALSE]
nd[[term]] <- fl

full_t <- predict(full_mod, newdata = nd)
base_t <- predict(base_fit, newdata = nd)
post_t <- predict(post_mod, newdata = nd)

full_f1 <- full_t
post_f1 <- base_t + post_t

YLIM <- range(c(full_f1, post_f1))

plot(fl, full_f1, type = "l", col = "black", xlab = term, ylab = term, ylim = YLIM)
lines(fl, post_f1, col = "red")

term = "logit_relative_humidity"
full_mod = full_logit_humidity_fit
post_mod = post_logit_humidity_fit

label <- paste0("s(", term, ")")

fl <- seq(min(past_data[[term]]), max(past_data[[term]]), length = 200)
nd <- past_data[rep(1, length(fl)), , drop = FALSE]
nd[[term]] <- fl

full_t <- predict(full_mod, newdata = nd)
base_t <- predict(base_fit, newdata = nd)
post_t <- predict(post_mod, newdata = nd)

full_f1 <- full_t
post_f1 <- base_t + post_t

YLIM <- range(c(full_f1, post_f1))

plot(fl, full_f1, type = "l", col = "black", xlab = term, ylab = term, ylim = YLIM)
lines(fl, post_f1, col = "red")
#
#
term = "relative_humidity"
full_mod = full_humidity_fit
post_mod = post_humidity_fit

label <- paste0("s(", term, ")")

fl <- seq(min(past_data[[term]]), max(past_data[[term]]), length = 200)
nd <- past_data[rep(1, length(fl)), , drop = FALSE]
nd[[term]] <- fl

full_t <- predict(full_mod, newdata = nd)
base_t <- predict(base_fit, newdata = nd)
post_t <- predict(post_mod, newdata = nd)

full_f1 <- full_t
post_f1 <- base_t + post_t

YLIM <- range(c(full_f1, post_f1))

plot(fl, full_f1, type = "l", col = "black", xlab = term, ylab = term, ylim = YLIM)
lines(fl, post_f1, col = "red")
#
#

full_mod = full_both_fit
post_mod = post_both_fit
term = "people_flood_days_per_capita"

label <- paste0("s(", term, ")")

fl <- seq(min(past_data[[term]]), max(past_data[[term]]), length = 200)
nd <- past_data[rep(1, length(fl)), , drop = FALSE]
nd[[term]] <- fl

full_t <- predict(full_mod, newdata = nd)
base_t <- predict(base_fit, newdata = nd)
post_t <- predict(post_mod, newdata = nd)

full_f1 <- full_t
post_f1 <- base_t + post_t
YLIM <- range(c(full_f1, post_f1))

plot(fl, full_f1, type = "l", col = "black", xlab = term, ylab = term, ylim = YLIM)
lines(fl, post_f1, col = "red")

term = "relative_humidity"

label <- paste0("s(", term, ")")

fl <- seq(min(past_data[[term]]), max(past_data[[term]]), length = 200)
nd <- past_data[rep(1, length(fl)), , drop = FALSE]
nd[[term]] <- fl

full_t <- predict(full_mod, newdata = nd)
base_t <- predict(base_fit, newdata = nd)
post_t <- predict(post_mod, newdata = nd)

full_f1 <- full_t
post_f1 <- base_t + post_t
YLIM <- range(c(full_f1, post_f1))

plot(fl, full_f1, type = "l", col = "black", xlab = term, ylab = term, ylim = YLIM)
lines(fl, post_f1, col = "red")









full_fl <- full_t$fit[, label];  full_se <- full_t$se.fit[, label]
post_fl <- post_t$fit[, label];  post_se <- post_t$se.fit[, label]

matplot(fl, cbind(full_fl, post_fl), type = "l", lty = 1,
        col = c("black", "red"), xlab = term, ylab = "centered flooding effect")
legend("topleft", c("full", "post"), col = c("black", "red"), lty = 1)


par(mfrow = c(1, 1))
term <- "people_flood_days_per_capita"

fl  <- seq(min(past_data[[term]]), max(past_data[[term]]), length = 200)

# one reference row for the non-flooding covariates, held fixed across the grid
ref <- past_data[1, , drop = FALSE]          # or build from column medians/means
nd  <- ref[rep(1, length(fl)), , drop = FALSE]
nd[[term]] <- fl

# FULL model: flooding + base estimated jointly -> predict the whole LP directly
full_pred <- predict(full_mod, newdata = nd, type = "link", se.fit = TRUE)

ilogit <- function(x) exp(x) / (1 + exp(x))
# POST-HOC model: base prediction at ref (constant in flooding) + flooding-on-residuals
base_const <- as.numeric(predict(base_fit, newdata = nd))[1]

post_pred   <- predict(post_mod, newdata = nd)
posthoc_fit <- base_const + post_pred

matplot(fl, cbind(full_pred$fit, posthoc_fit), type = "l", lty = 1,
        col = c("black", "red"),
        xlab = term, ylab = "logit_malaria_pfpr (base held at reference)")
legend("topleft", c("full", "post-hoc"), col = c("black", "red"), lty = 1)


mod <- lm(full_flooding_fitted ~ post_flooding_fitted, data = past_data)
summary(mod)



plot(past_data$people_flood_days_per_capita, resid(mod),
     xlab = "flooding", ylab = "full − fitted-from-post")






# fit <- scam(logit_malaria_pfpr ~ s(malaria_suit, k=6, bs="mpi") + 
#               s(mal_DAH_total_per_capita, k=6, bs="mpd") +
#               s(gdppc_mean, k=6, bs="mpd") +
#               s(relative_humidity, k=6, bs="mpi") + 
#               A0_af,
#             data=past_data, optimizer="efs", control=list(maxit=50))
# summary(fit)

# require(httpgd)
# httpgd::hgd()
# plot(fit, pages = 1, scale = 0)


fit <- scam(logit_malaria_pfpr ~ s(malaria_suit, bs = 'mpi') + 
              mal_DAH_total_per_capita +
              gdppc_mean +
              relative_humidity + 
              A0_af,
            data=past_data)
summary(fit)$r.squared

library(data.table)
setDT(past_data)

covars <- c("malaria_suit", "mal_DAH_total_per_capita",
            "gdppc_mean", "relative_humidity")  # candidate set
yvar   <- "logit_malaria_pfpr"

# within-transform: subtract A0_af mean from y and every covariate
vars <- c(yvar, covars)
past_data[, paste0(vars, "_w") :=
            lapply(.SD, function(x) x - mean(x)), by = A0_af, .SDcols = vars]


fit_w <- scam(logit_malaria_pfpr_w ~ s(malaria_suit, bs = 'mpi') + mal_DAH_total_per_capita_w +
              gdppc_mean_w + relative_humidity_w - 1, data = past_data)
summary(fit_w)$r.squared


# require(httpgd)
httpgd::hgd()
par(mfrow = c(2, 1))
plot(fit)
plot(fit_w)
