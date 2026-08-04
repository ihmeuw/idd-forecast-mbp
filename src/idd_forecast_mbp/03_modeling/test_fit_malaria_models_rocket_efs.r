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
past_data$malaria_suit_fraction     <- past_data[[suit_col]] / 365
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














cnt_full <- table(past_data$A0_location_id)
cnt_sub  <- table(sub_data$A0_location_id)

# countries that survive into the intensity fit, and how thin they get
retained <- data.frame(
  loc      = names(cnt_sub),
  n_full   = as.integer(cnt_full[names(cnt_sub)]),
  n_sub    = as.integer(cnt_sub),
  frac     = round(as.integer(cnt_sub) / as.integer(cnt_full[names(cnt_sub)]), 3)
)
retained <- retained[order(retained$n_sub), ]
head(retained, 20)        # the thinnest survivors — these are the FE risk
forecast_countries <- unique(past_data$A0_location_id[past_data$year_id == 2023])
setdiff(forecast_countries, as.integer(names(cnt_sub)))

bad_dah <- which(sub_data$mal_DAH_total_per_capita == 0)
sub_data$log_mal_DAH_total_per_capita[bad_dah] <- 0.01 * min(sub_data$log_mal_DAH_total_per_capita[-bad_dah])
mod_0 <- lm(logit_malaria_pfpr ~ logit_malaria_suitability + log_mal_DAH_total_per_capita + log_gdppc_mean +
              logit_relative_humidity + A0_af,
            data=sub_data)

mod_0b <- lm(logit_malaria_pfpr ~ logit_malaria_suitability + log_mal_DAH_total_per_capita + log_gdppc_mean +
              logit_relative_humidity + weighted_1km_urban_threshold_300.0_simple_mean + total_precipitation + A0_af,
            data=sub_data)

mod_1a <- scam(logit_malaria_pfpr ~ s(malaria_suit, bs = 'mpi') + 
              s(mal_DAH_total_per_capita, k=6, bs='mpd') +
              s(gdppc_mean, k=6, bs='mpd') +
              s(relative_humidity, k=6, bs='mpi') + 
              A0_af,
            data=sub_data, optimizer="efs", control=list(maxit=50))
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
ilogit <- function(x) 1 / (1 + exp(-x))

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
