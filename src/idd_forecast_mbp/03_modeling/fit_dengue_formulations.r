library(arrow)
library(data.table)
library(mgcv)
library(scam)
library(httpgd)

CFG_FILE <- Sys.getenv("IDD_R_CONFIG", unset = path.expand("~/.idd_forecast_mbp.R"))
sys.source(CFG_FILE, envir = globalenv())
p   <- function(...) file.path(MODEL_ROOT, ...)
OUT <- p("07-figures", format(Sys.Date(), "%Y%m%d"), "dengue_formulation_comparison")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

LSAE        <- "lsae_1285"
ANCHOR_YEAR <- 2023
BASE_AG     <- 7
BASE_SG     <- 1
URB_EPS     <- 1e-3
URBAN_COL   <- "weighted_1km_urban_threshold_300.0_simple_mean"
K_SET       <- 3:5

COVS <- c("gdppc_mean", "ldipc_mean", "weighted_1km_urban_threshold_300.0_simple_mean",
          "weighted_100m_urban_threshold_300.0_simple_mean",
          "weighted_1km_urban_threshold_1500.0_simple_mean",
          "weighted_100m_urban_threshold_1500.0_simple_mean",
          "people_flood_days", "people_flood_days_per_capita", "total_precipitation",
          "precipitation_days", "relative_humidity", "wind_speed", "mean_temperature",
          "mean_low_temperature", "mean_high_temperature", "days_over_30C", "dengue_suitability")

logit <- function(x) log(x / (1 - x))

add_derived <- function(dt) {
  dt[, urban_fraction := pmin(pmax(get(URBAN_COL), URB_EPS), 1 - URB_EPS)]
  dt[, `:=`(logit_urban_fraction = logit(urban_fraction),
            log_gdppc_mean       = log(gdppc_mean),
            log_dengue_inc_rate  = ifelse(dengue_inc_rate  > 0, log(dengue_inc_rate),  NA_real_),
            log_dengue_mort_rate = ifelse(dengue_mort_rate > 0, log(dengue_mort_rate), NA_real_),
            dengue_cfr           = ifelse(dengue_inc_rate  > 0, dengue_mort_rate / dengue_inc_rate, NA_real_))]
  dt[, logit_dengue_cfr := ifelse(dengue_cfr > 0 & dengue_cfr < 1, logit(dengue_cfr), NA_real_)]
  dt[]
}

hierarchy_dt <- as.data.table(read_parquet(
  p("02-processed_data", "hierarchy", LSAE, "current", "full_hierarchy_2023_lsae_1285.parquet")))
past_dt <- as.data.table(read_parquet(
  p("03-modeling_data", "dengue", "past_inputs_nc", LSAE, "current", "dengue_past_inputs.parquet")))

past_dt <- past_dt[location_id %in% hierarchy_dt[most_detailed_fhs == 1, unique(location_id)]]
past_dt[, dengue_inc_count := dengue_inc_rate * population]
add_derived(past_dt)
past_dt <- merge(past_dt, hierarchy_dt[, .(location_id, super_region_id)],
                 by = "location_id", all.x = TRUE)

base_locs <- past_dt[year_id == ANCHOR_YEAR & age_group_id == BASE_AG & sex_id == BASE_SG &
                     dengue_inc_count > 0, unique(location_id)]
past_dt <- past_dt[location_id %in% base_locs]
past_dt[, `:=`(A0_af = factor(A0_location_id), sr_f = factor(super_region_id),
               year_centered = year_id - mean(year_id))]

keep_cols  <- c("location_id", "year_id", COVS, "year_centered", "A0_af", "sr_f")
past_aa_dt <- past_dt[, .(dengue_inc_rate  = sum(dengue_inc_rate  * population) / sum(population),
                          dengue_mort_rate = sum(dengue_mort_rate * population) / sum(population),
                          population       = sum(population)),
                      by = .(location_id, year_id)]
past_aa_dt <- merge(past_aa_dt,
                    unique(past_dt[age_group_id == BASE_AG & sex_id == BASE_SG, ..keep_cols]),
                    by = c("location_id", "year_id"))
add_derived(past_aa_dt)
past_aa_dt[, `:=`(dengue_inc_count = round(dengue_inc_rate * population),
                  log_population   = log(population))]

year_term <- function(k) {
  if (is.na(k)) "year_id*sr_f" else sprintf("s(year_id, k = %d, by = sr_f, bs = 'cr')", k)
}

fit_rate  <- function(k) gam(as.formula(paste("log_dengue_inc_rate ~ s(dengue_suitability, k = 4) +",
                             year_term(k), "+ relative_humidity + A0_af")), data = past_aa_dt)
fit_count <- function(k) gam(as.formula(paste("dengue_inc_count ~ s(dengue_suitability, k = 4) +",
                             year_term(k), "+ relative_humidity + A0_af + offset(log_population)")),
                             family = nb(), data = past_aa_dt)

spec       <- c(NA, K_SET)
mod_labels <- c("linear", paste0("k", K_SET))
rate_mods  <- setNames(lapply(spec, fit_rate),  mod_labels)
cnt_mods   <- setNames(lapply(spec, fit_count), mod_labels)

sr_lvls  <- levels(past_aa_dt$sr_f)
sr_names <- hierarchy_dt[match(as.integer(sr_lvls), location_id), location_name]

term_grid <- CJ(year_id = seq(min(past_aa_dt$year_id), max(past_aa_dt$year_id), length.out = 200),
                sr_f = sr_lvls)
term_grid[, `:=`(sr_f = factor(sr_f, levels = sr_lvls),
                 dengue_suitability = median(past_aa_dt$dengue_suitability, na.rm = TRUE),
                 relative_humidity  = median(past_aa_dt$relative_humidity,  na.rm = TRUE),
                 A0_af = past_aa_dt$A0_af[1], log_population = 0)]

linear_year_effect <- function(mod, g) {
  Xp <- predict(mod, g, type = "lpmatrix")
  Xp[, !grepl("^year_id", colnames(Xp))] <- 0
  for (lv in levels(g$sr_f)) {
    i <- g$sr_f == lv
    Xp[i, ] <- sweep(Xp[i, , drop = FALSE], 2, colMeans(Xp[i, , drop = FALSE]))
  }
  data.table(sr_f = g$sr_f, year_id = g$year_id,
             fit = as.vector(Xp %*% coef(mod)),
             se  = sqrt(rowSums((Xp %*% vcov(mod)) * Xp)))
}

plot_terms <- function(mods, title) {
  par(mfrow = c(length(mods), 7), oma = c(0, 0, 3, 0))
  for (m in mods) {
    plot(m, select = 1, scale = 0)
    if (length(m$smooth) == 1) {
      d <- linear_year_effect(m, term_grid)
      for (lv in sr_lvls) {
        dd <- d[sr_f == lv]
        matplot(dd$year_id, cbind(dd$fit, dd$fit + 2 * dd$se, dd$fit - 2 * dd$se), type = "l",
                lty = c(1, 2, 2), col = 1, xlab = "year_id", ylab = sprintf("year:sr_f%s", lv))
      }
    } else for (j in 2:7) plot(m, select = j, scale = 0)
  }
  mtext(c("dengue suitability", sr_names), side = 3, line = 0.5, outer = TRUE,
        at = (1:7 - 0.5) / 7, cex = 0.7, font = 2)
  mtext(title, side = 3, line = 1.8, outer = TRUE, font = 2)
}

plot_terms(rate_mods, "log-rate models")
plot_terms(cnt_mods,  "negative-binomial count models")

pred_dt <- copy(past_aa_dt)
pred_dt[, obs := dengue_inc_rate * population]
for (i in seq_along(mod_labels)) {
  pred_dt[, (paste0("rate_", i)) := exp(predict(rate_mods[[i]], newdata = pred_dt)) * population]
  pred_dt[, (paste0("cnt_",  i)) := predict(cnt_mods[[i]], newdata = pred_dt, type = "response")]
}

val_cols <- c("obs", paste0("rate_", seq_along(mod_labels)), paste0("cnt_", seq_along(mod_labels)))
by_sr    <- pred_dt[, lapply(.SD, sum), by = .(grp = as.character(sr_f), year_id), .SDcols = val_cols]
by_gl    <- pred_dt[, lapply(.SD, sum), by = .(year_id), .SDcols = val_cols][, grp := "GLOBAL"][]
agg_dt   <- rbind(by_gl, by_sr, use.names = TRUE)
setorder(agg_dt, grp, year_id)

grps  <- c("GLOBAL", sr_lvls)
glabs <- c("Global", sr_names)

par(mfrow = c(length(mod_labels), 7), mar = c(2.1, 2.1, 1.1, 0.1))
for (i in seq_along(mod_labels)){
  for (g in seq_along(grps)) {
    d <- agg_dt[grp == grps[g]]
    x = d$year_id
    y1 = d$obs
    y2 = d[[paste0("rate_", i)]]
    y3 = d[[paste0("cnt_", i)]]
    plot(x, y1, type = 'l', col = 1, 
      xlab = 'year_id', ylab = sprintf("%s count", mod_labels[i]),
      main = glabs[g], cex.main = 0.7)
    lines(x, y2, col = 'red')
    lines(x, y3, col = 'blue')
    if (i == 1 && g == 1)
      legend(2000,3e7, legend = c("obs", "rate", "count"),
       col = c("black", "red", "blue"), lwd = 1, bty = "n")

  }
}


#
g = 2

par(mfrow=c(2,4))
for (i in seq_along(mod_labels)){

  d <- agg_dt[grp == grps[g]]
  x = d$year_id
  y1 = d$obs
  y2 = d[[paste0("rate_", i)]]
  y3 = d[[paste0("cnt_", i)]]
  plot(x, y1, type = 'l', col = 1, 
    xlab = 'year_id', ylab = sprintf("%s count", mod_labels[i]),
    main = glabs[g], cex.main = 0.7)
  lines(x, y2, col = 'red')
  lines(x, y3, col = 'blue')
  if (i == 1 && g == 1)
    legend(2000,3e7, legend = c("obs", "rate", "count"),
      col = c("black", "red", "blue"), lwd = 1, bty = "n")


}


par(mfrow = c(length(mod_labels), 7), mar = c(2.1, 2.1, 1.1, 0.1))
for (i in seq_along(mod_labels)){
  for (g in seq_along(grps)) {
    d <- agg_dt[grp == grps[g]]
    x = d$year_id
    y1 = d$obs
    y2 = d[[paste0("rate_", i)]]
    y3 = d[[paste0("cnt_", i)]]
    plot(x, y1, type = 'l', col = 1, 
      xlab = 'year_id', ylab = sprintf("%s count", mod_labels[i]),
      main = glabs[g], cex.main = 0.7)
    lines(x, y2, col = 'red')
    lines(x, y3, col = 'blue')
    if (i == 1 && g == 1)
      legend(2000,3e7, legend = c("obs", "rate", "count"),
       col = c("black", "red", "blue"), lwd = 1, bty = "n")

  }
}




#
library(arrow)
library(data.table)
library(mgcv)
library(scam)
library(httpgd)

CFG_FILE <- Sys.getenv("IDD_R_CONFIG", unset = path.expand("~/.idd_forecast_mbp.R"))
sys.source(CFG_FILE, envir = globalenv())
p   <- function(...) file.path(MODEL_ROOT, ...)
OUT <- p("07-figures", format(Sys.Date(), "%Y%m%d"), "dengue_formulation_comparison")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

LSAE        <- "lsae_1285"
ANCHOR_YEAR <- 2023
BASE_AG     <- 7
BASE_SG     <- 1
URB_EPS     <- 1e-3
URBAN_COL   <- "weighted_1km_urban_threshold_300.0_simple_mean"
K_SET       <- 3:5
BY_F        <- "rg_f"
MEASURES    <- c("inc", "mort")

COVS <- c("gdppc_mean", "ldipc_mean", "weighted_1km_urban_threshold_300.0_simple_mean",
          "weighted_100m_urban_threshold_300.0_simple_mean",
          "weighted_1km_urban_threshold_1500.0_simple_mean",
          "weighted_100m_urban_threshold_1500.0_simple_mean",
          "people_flood_days", "people_flood_days_per_capita", "total_precipitation",
          "precipitation_days", "relative_humidity", "wind_speed", "mean_temperature",
          "mean_low_temperature", "mean_high_temperature", "days_over_30C", "dengue_suitability")

logit <- function(x) log(x / (1 - x))

add_derived <- function(dt) {
  dt[, urban_fraction := pmin(pmax(get(URBAN_COL), URB_EPS), 1 - URB_EPS)]
  dt[, `:=`(logit_urban_fraction = logit(urban_fraction),
            log_gdppc_mean       = log(gdppc_mean),
            log_dengue_inc_rate  = ifelse(dengue_inc_rate  > 0, log(dengue_inc_rate),  NA_real_),
            log_dengue_mort_rate = ifelse(dengue_mort_rate > 0, log(dengue_mort_rate), NA_real_),
            dengue_cfr           = ifelse(dengue_inc_rate  > 0, dengue_mort_rate / dengue_inc_rate, NA_real_))]
  dt[, logit_dengue_cfr := ifelse(dengue_cfr > 0 & dengue_cfr < 1, logit(dengue_cfr), NA_real_)]
  dt[]
}

hierarchy_dt <- as.data.table(read_parquet(
  p("02-processed_data", "hierarchy", LSAE, "current", "full_hierarchy_2023_lsae_1285.parquet")))
past_dt <- as.data.table(read_parquet(
  p("03-modeling_data", "dengue", "past_inputs_nc", LSAE, "current", "dengue_past_inputs.parquet")))

past_dt <- past_dt[location_id %in% hierarchy_dt[most_detailed_fhs == 1, unique(location_id)]]
past_dt[, dengue_inc_count := dengue_inc_rate * population]
add_derived(past_dt)
past_dt <- merge(past_dt, hierarchy_dt[, .(location_id, super_region_id, region_id)],
                 by = "location_id", all.x = TRUE)

base_locs <- past_dt[year_id == ANCHOR_YEAR & age_group_id == BASE_AG & sex_id == BASE_SG &
                     dengue_inc_count > 0, unique(location_id)]
past_dt <- past_dt[location_id %in% base_locs]
past_dt[, `:=`(A0_af = factor(A0_location_id), sr_f = factor(super_region_id),
               rg_f = factor(region_id), year_centered = year_id - mean(year_id))]

keep_cols  <- c("location_id", "year_id", COVS, "year_centered", "A0_af", "sr_f", "rg_f")
past_aa_dt <- past_dt[, .(dengue_inc_rate  = sum(dengue_inc_rate  * population) / sum(population),
                          dengue_mort_rate = sum(dengue_mort_rate * population) / sum(population),
                          population       = sum(population)),
                      by = .(location_id, year_id)]
past_aa_dt <- merge(past_aa_dt,
                    unique(past_dt[age_group_id == BASE_AG & sex_id == BASE_SG, ..keep_cols]),
                    by = c("location_id", "year_id"))
add_derived(past_aa_dt)
past_aa_dt[, `:=`(dengue_inc_count  = dengue_inc_rate  * population,
                  dengue_mort_count = dengue_mort_rate * population,
                  log_population    = log(population))]

year_term <- function(k) {
  if (is.na(k)) sprintf("year_id*%s", BY_F)
  else sprintf("s(year_id, k = %d, by = %s, bs = 'cr')", k, BY_F)
}

fit_rate <- function(k, measure) {
  f <- sprintf("log_dengue_%s_rate ~ s(dengue_suitability, k = 4) + %s + relative_humidity + A0_af",
               measure, year_term(k))
  gam(as.formula(f), data = past_aa_dt, weights = population)
}

fit_count <- function(k, measure) {
  f <- sprintf(paste("dengue_%s_count ~ s(dengue_suitability, k = 4) + %s +",
                     "relative_humidity + A0_af + offset(log_population)"),
               measure, year_term(k))
  gam(as.formula(f), family = nb(), data = past_aa_dt)
}

spec       <- c(NA, K_SET)
mod_labels <- c("linear", paste0("k", K_SET))
mods <- list()
for (ms in MEASURES) {
  mods[[ms]] <- list(rate  = setNames(lapply(spec, fit_rate,  measure = ms), mod_labels),
                     count = setNames(lapply(spec, fit_count, measure = ms), mod_labels))
}

by_lvls  <- levels(past_aa_dt[[BY_F]])
by_names <- hierarchy_dt[match(as.integer(by_lvls), location_id), location_name]
sr_lvls  <- levels(past_aa_dt$sr_f)
sr_names <- hierarchy_dt[match(as.integer(sr_lvls), location_id), location_name]

term_grid <- CJ(year_id = seq(min(past_aa_dt$year_id), max(past_aa_dt$year_id), length.out = 200),
                lvl = by_lvls)
setnames(term_grid, "lvl", BY_F)
term_grid[, (BY_F) := factor(get(BY_F), levels = by_lvls)]
term_grid[, `:=`(dengue_suitability = median(past_aa_dt$dengue_suitability, na.rm = TRUE),
                 relative_humidity  = median(past_aa_dt$relative_humidity,  na.rm = TRUE),
                 A0_af = past_aa_dt$A0_af[1], log_population = 0, population = 1)]

linear_year_effect <- function(mod, g) {
  Xp <- predict(mod, g, type = "lpmatrix")
  Xp[, !grepl("^year_id", colnames(Xp))] <- 0
  for (lv in by_lvls) {
    i <- g[[BY_F]] == lv
    Xp[i, ] <- sweep(Xp[i, , drop = FALSE], 2, colMeans(Xp[i, , drop = FALSE]))
  }
  data.table(lvl = g[[BY_F]], year_id = g$year_id,
             fit = as.vector(Xp %*% coef(mod)),
             se  = sqrt(rowSums((Xp %*% vcov(mod)) * Xp)))
}

grid_dim <- function(n) { nc <- ceiling(sqrt(n)); c(ceiling(n / nc), nc) }

plot_terms <- function(mod, title) {
  n <- 1 + length(by_lvls)
  par(mfrow = grid_dim(n), oma = c(0, 0, 2, 0), mar = c(2.1, 2.5, 1.6, 0.4), mgp = c(1.4, 0.4, 0))
  plot(mod, select = 1, scale = 0)
  if (length(mod$smooth) == 1) {
    d <- linear_year_effect(mod, term_grid)
    for (j in seq_along(by_lvls)) {
      dd <- d[lvl == by_lvls[j]]
      matplot(dd$year_id, cbind(dd$fit, dd$fit + 2 * dd$se, dd$fit - 2 * dd$se), type = "l",
              lty = c(1, 2, 2), col = 1, xlab = "year_id", ylab = "", main = by_names[j],
              cex.main = 0.8)
    }
  } else for (j in seq_along(by_lvls)) plot(mod, select = j + 1, scale = 0)
  mtext(title, side = 3, line = 0.5, outer = TRUE, font = 2)
}

for (ms in MEASURES) for (eng in c("rate", "count")) for (lb in mod_labels)
  plot_terms(mods[[ms]][[eng]][[lb]], sprintf("%s / %s / %s", ms, eng, lb))

pred_dt <- copy(past_aa_dt)
for (ms in MEASURES) {
  pred_dt[, (paste0("obs_", ms)) := get(paste0("dengue_", ms, "_rate")) * population]
  for (i in seq_along(mod_labels)) {
    pred_dt[, (sprintf("%s_rate_%d", ms, i)) :=
              exp(predict(mods[[ms]]$rate[[i]], newdata = pred_dt)) * population]
    pred_dt[, (sprintf("%s_cnt_%d", ms, i)) :=
              predict(mods[[ms]]$count[[i]], newdata = pred_dt, type = "response")]
  }
}

val_cols <- c(paste0("obs_", MEASURES),
              as.vector(outer(MEASURES, seq_along(mod_labels),
                              function(m, i) sprintf("%s_rate_%d", m, i))),
              as.vector(outer(MEASURES, seq_along(mod_labels),
                              function(m, i) sprintf("%s_cnt_%d", m, i))))

by_sr  <- pred_dt[, lapply(.SD, sum), by = .(grp = as.character(sr_f), year_id), .SDcols = val_cols]
by_gl  <- pred_dt[, lapply(.SD, sum), by = .(year_id), .SDcols = val_cols][, grp := "GLOBAL"][]
agg_dt <- rbind(by_gl, by_sr, use.names = TRUE)
setorder(agg_dt, grp, year_id)

grps  <- c("GLOBAL", sr_lvls)
glabs <- c("Global", sr_names)

for (ms in MEASURES) for (i in seq_along(mod_labels)) {
  par(mfrow = grid_dim(length(grps)), oma = c(0, 0, 2, 0),
      mar = c(2.1, 2.5, 1.6, 0.4), mgp = c(1.4, 0.4, 0))
  for (g in seq_along(grps)) {
    d  <- agg_dt[grp == grps[g]]
    y1 <- d[[paste0("obs_", ms)]]
    y2 <- d[[sprintf("%s_rate_%d", ms, i)]]
    y3 <- d[[sprintf("%s_cnt_%d",  ms, i)]]
    matplot(d$year_id, cbind(y1, y2, y3), type = "l", lty = 1,
            col = c("black", "red", "blue"), xlab = "year_id", ylab = "count",
            main = glabs[g], cex.main = 0.8)
  }
  mtext(sprintf("%s  |  %s  |  black obs, red rate, blue count", ms, mod_labels[i]),
        side = 3, line = 0.5, outer = TRUE, font = 2)
}






MS <- "inc"
LB <- "k4"
m  <- mods[[MS]]$rate[[LB]]
i  <- match(LB, mod_labels)

par(mfrow = grid_dim(1 + length(by_lvls)), oma = c(0, 0, 2, 0),
    mar = c(2.1, 2.5, 1.6, 0.4), mgp = c(1.4, 0.4, 0))
plot(m, select = 1, scale = 0)
for (j in seq_along(by_lvls)) plot(m, select = j + 1, scale = 0)
mtext(sprintf("%s / rate / %s  —  smooths", MS, LB), side = 3, line = 0.5, outer = TRUE, font = 2)

par(mfrow = grid_dim(length(grps)), oma = c(0, 0, 2, 0),
    mar = c(2.1, 2.5, 1.6, 0.4), mgp = c(1.4, 0.4, 0))
for (g in seq_along(grps)) {
  d <- agg_dt[grp == grps[g]]
  matplot(d$year_id, cbind(d[[paste0("obs_", MS)]], d[[sprintf("%s_rate_%d", MS, i)]]),
          type = "l", lty = 1, col = c("black", "red"),
          xlab = "year_id", ylab = "count", main = glabs[g], cex.main = 0.8)
}
mtext(sprintf("%s / rate / %s  —  black obs, red pred", MS, LB), side = 3, line = 0.5,
      outer = TRUE, font = 2)




MS <- "inc"
LB <- "k4"
m  <- mods[[MS]]$rate[[LB]]
i  <- match(LB, mod_labels)

png(file.path(OUT, sprintf("%s_rate_%s_smooths.png", MS, LB)),
    width = 24, height = 10, units = "in", res = 150, pointsize = 26)
par(mfrow = c(3,6), oma = c(0, 0, 3, 0),
    mar = c(3.2, 3.6, 2.4, 0.6), mgp = c(2, 0.7, 0), cex.main = 0.9, cex.lab = 0.9)
plot(m, select = 1, scale = 0, main = "dengue suitability", ylab = "")
for (j in seq_along(by_lvls))
  plot(m, select = j + 1, scale = 0, main = by_names[j], ylab = "", xlab = "year_id")
mtext(sprintf("%s / rate / %s  —  smooths", MS, LB), side = 3, line = 1, outer = TRUE, font = 2)
dev.off()

png(file.path(OUT, sprintf("%s_rate_%s_obs_vs_pred.png", MS, LB)),
    width = 24, height = 10, units = "in", res = 150, pointsize = 26)
par(mfrow = c(2,4), oma = c(0, 0, 3, 0),
    mar = c(3.2, 4.2, 2.4, 0.6), mgp = c(2.4, 0.7, 0), cex.main = 0.9, cex.lab = 0.9)
for (g in seq_along(grps)) {
  d <- agg_dt[grp == grps[g]]
  matplot(d$year_id, cbind(d[[paste0("obs_", MS)]], d[[sprintf("%s_rate_%d", MS, i)]]),
          type = "l", lty = 1, lwd = 2, col = c("black", "red"),
          xlab = "year_id", ylab = "count", main = glabs[g])
}
mtext(sprintf("%s / rate / %s  —  black obs, red pred", MS, LB), side = 3, line = 1,
      outer = TRUE, font = 2)
dev.off()


agg_rg <- pred_dt[, lapply(.SD, sum), by = .(grp = as.character(rg_f), year_id), .SDcols = val_cols]
setorder(agg_rg, grp, year_id)

png(file.path(OUT, sprintf("%s_rate_%s_obs_vs_pred_region.png", MS, LB)),
    width = 24, height = 18, units = "in", res = 150, pointsize = 26)
par(mfrow = grid_dim(length(by_lvls)), oma = c(0, 0, 3, 0),
    mar = c(3.2, 4.2, 2.4, 0.6), mgp = c(2.4, 0.7, 0), cex.main = 0.9, cex.lab = 0.9)
for (g in seq_along(by_lvls)) {
  d <- agg_rg[grp == by_lvls[g]]
  y <- cbind(d[[paste0("obs_", MS)]], d[[sprintf("%s_rate_%d", MS, i)]])
  matplot(d$year_id, y, type = "l", lty = 1, lwd = 2, col = c("black", "red"),
          ylim = c(0, max(y, na.rm = TRUE)),
          xlab = "year_id", ylab = "count", main = by_names[g])
}
mtext(sprintf("%s / rate / %s by region  —  black obs, red pred", MS, LB), side = 3, line = 1,
      outer = TRUE, font = 2)
dev.off()

agg_rg <- pred_dt[, c(.(population = sum(population)), lapply(.SD, sum)),
                  by = .(grp = as.character(rg_f), year_id), .SDcols = val_cols]
setorder(agg_rg, grp, year_id)

png(file.path(OUT, sprintf("%s_rate_%s_obs_vs_pred_region_rate.png", MS, LB)),
    width = 24, height = 18, units = "in", res = 150, pointsize = 26)
par(mfrow = grid_dim(length(by_lvls)), oma = c(0, 0, 3, 0),
    mar = c(3.2, 4.6, 2.4, 0.6), mgp = c(2.6, 0.7, 0), cex.main = 0.9, cex.lab = 0.9)
for (g in seq_along(by_lvls)) {
  d <- agg_rg[grp == by_lvls[g]]
  y <- cbind(d[[paste0("obs_", MS)]], d[[sprintf("%s_rate_%d", MS, i)]]) / d$population
  matplot(d$year_id, y, type = "l", lty = 1, lwd = 2, col = c("black", "red"),
          ylim = c(0, max(y, na.rm = TRUE)),
          xlab = "year_id", ylab = "rate", main = by_names[g])
}
mtext(sprintf("%s / rate / %s by region  —  black obs rate, red pred rate", MS, LB),
      side = 3, line = 1, outer = TRUE, font = 2)
dev.off()
















IND <- 163

past_dt <- as.data.table(read_parquet(
  p("03-modeling_data", "dengue", "past_inputs_nc", LSAE, "current", "dengue_past_inputs.parquet")))

past_dt <- past_dt[location_id %in% hierarchy_dt[most_detailed_fhs == 1, unique(location_id)]]
past_dt[, dengue_inc_count := round(dengue_inc_rate * population)]
add_derived(past_dt)
past_dt <- merge(past_dt, hierarchy_dt[, .(location_id, super_region_id, region_id)],
                 by = "location_id", all.x = TRUE)

base_locs <- past_dt[year_id == ANCHOR_YEAR & age_group_id == BASE_AG & sex_id == BASE_SG , unique(location_id)]
past_dt <- past_dt[location_id %in% base_locs]
past_dt[, `:=`(A0_af = factor(A0_location_id), sr_f = factor(super_region_id),
               rg_f = factor(region_id), year_centered = year_id - mean(year_id))]

keep_cols  <- c("location_id", "year_id", COVS, "year_centered", "A0_af", "sr_f", "rg_f")
past_aa_dt <- past_dt[, .(dengue_inc_rate  = sum(dengue_inc_rate  * population) / sum(population),
                          dengue_mort_rate = sum(dengue_mort_rate * population) / sum(population),
                          population       = sum(population)),
                      by = .(location_id, year_id)]
past_aa_dt <- merge(past_aa_dt,
                    unique(past_dt[age_group_id == BASE_AG & sex_id == BASE_SG, ..keep_cols]),
                    by = c("location_id", "year_id"))
add_derived(past_aa_dt)
past_aa_dt[, `:=`(dengue_inc_count  = round(dengue_inc_rate  * population),
                  dengue_mort_count = round(dengue_mort_rate * population),
                  log_population    = log(population))]

ind_locs <- past_aa_dt[A0_af == as.character(IND), unique(location_id)]
a1_map   <- hierarchy_dt[location_id %in% ind_locs, .(location_id, path_to_top_parent)]
a1_map[, a1_id := {
  v <- as.integer(strsplit(path_to_top_parent, ",")[[1]])
  k <- match(IND, v)
  if (is.na(k) || k == length(v)) location_id else v[k + 1L]
}, by = location_id]

ind_dt <- droplevels(past_aa_dt[A0_af == as.character(IND)])
ind_dt <- merge(ind_dt, a1_map[, .(location_id, a1_id)], by = "location_id")
ind_dt[, a1_f := factor(a1_id)]
a1_lvls  <- levels(ind_dt$a1_f)
a1_names <- hierarchy_dt[match(as.integer(a1_lvls), location_id), location_name]

m_ind <- gam(dengue_inc_count ~ s(dengue_suitability, k = 4) +
             s(year_id, a1_f, bs = "fs", k = 4) + relative_humidity +
             offset(log_population),
             family = nb(), data = ind_dt)

g_grid <- CJ(year_id = seq(min(ind_dt$year_id), max(ind_dt$year_id), length.out = 200),
             a1_f = a1_lvls)
g_grid[, a1_f := factor(a1_f, levels = a1_lvls)]
g_grid[, `:=`(dengue_suitability = median(ind_dt$dengue_suitability, na.rm = TRUE),
              relative_humidity  = median(ind_dt$relative_humidity,  na.rm = TRUE),
              log_population = 0)]
tm   <- predict(m_ind, g_grid, type = "terms")
ycol <- grep("year_id", colnames(tm), value = TRUE)[1]
g_grid[, fit := tm[, ycol]]

ind_dt[, pred_count := predict(m_ind, newdata = ind_dt, type = "response")]
agg_a1  <- ind_dt[, .(obs = sum(dengue_inc_count), pred = sum(pred_count)),
                  by = .(a1_f, year_id)][order(a1_f, year_id)]
agg_ind <- ind_dt[, .(obs = sum(dengue_inc_count), pred = sum(pred_count)),
                  by = year_id][order(year_id)]

Y <- dcast(g_grid, year_id ~ a1_f, value.var = "fit")

png(file.path(OUT, "india_a1_count_k4_smooths.png"),
    width = 18, height = 9, units = "in", res = 150, pointsize = 24)
par(mfrow = c(1, 2), oma = c(0, 0, 3, 0), mar = c(3.4, 4.2, 2.4, 0.6), mgp = c(2.4, 0.7, 0))
plot(m_ind, select = 1, scale = 0, main = "dengue suitability", ylab = "")
matplot(Y$year_id, as.matrix(Y[, -1, with = FALSE]),
        type = "l", lty = 1, lwd = 2, col = rainbow(length(a1_lvls)),
        xlab = "year_id", ylab = "year term (log scale)", main = "year curve by admin 1")
mtext("India / admin 1 / nb count / fs k=4", side = 3, line = 1, outer = TRUE, font = 2)
dev.off()

png(file.path(OUT, "india_a1_count_k4_obs_vs_pred.png"),
    width = 24, height = 18, units = "in", res = 150, pointsize = 26)
par(mfrow = grid_dim(1 + length(a1_lvls)), oma = c(0, 0, 3, 0),
    mar = c(3.2, 4.4, 2.4, 0.6), mgp = c(2.6, 0.7, 0), cex.main = 0.9, cex.lab = 0.9)
y <- cbind(agg_ind$obs, agg_ind$pred)
matplot(agg_ind$year_id, y, type = "l", lty = 1, lwd = 2, col = c("black", "red"),
        ylim = c(0, max(y, na.rm = TRUE)), xlab = "year_id", ylab = "count", main = "INDIA")
for (j in seq_along(a1_lvls)) {
  d <- agg_a1[a1_f == a1_lvls[j]]
  y <- cbind(d$obs, d$pred)
  matplot(d$year_id, y, type = "l", lty = 1, lwd = 2, col = c("black", "red"),
          ylim = c(0, max(y, na.rm = TRUE)), xlab = "year_id", ylab = "count",
          main = a1_names[j])
}
mtext("India admin 1  —  black obs, red pred", side = 3, line = 1, outer = TRUE, font = 2)
dev.off()



label_outbreaks <- function(dt, k_year = 4, k_suit = 4,
                            thresh = 3, max_iter = 8) {
  dt <- copy(dt)
  dt[, is_endemic := TRUE]           # start assuming all endemic
  prev <- NULL
  for (it in seq_len(max_iter)) {
    print(it)
    fit <- gam(dengue_inc_count ~ s(dengue_suitability, k = k_suit) +
                 s(year_id, a1_f, bs = "fs", k = k_year) +
                 relative_humidity + offset(log_population),
               family = nb(), data = dt[is_endemic == TRUE])
    # predict for ALL rows on the endemic fit
    mu  <- predict(fit, newdata = dt, type = "response")
    th  <- fit$family$getTheta(TRUE)          # NB dispersion
    var <- mu + mu^2 / th
    dt[, pearson := (dengue_inc_count - mu) / sqrt(var)]
    # outbreak = large POSITIVE excess only (one-sided)
    dt[, new_endemic := pearson < thresh]
    if (!is.null(prev) && all(dt$new_endemic == prev)) break
    prev <- dt$new_endemic
    dt[, is_endemic := new_endemic]
  }
  dt[, `:=`(is_outbreak = !is_endemic, pearson = NULL, new_endemic = NULL)]
  dt[]
}

ind_dt <- label_outbreaks(ind_dt)

m_endemic <- gam(dengue_inc_count ~ s(dengue_suitability, k = 4) +
                   s(year_id, a1_f, bs = "fs", k = 4) +
                   relative_humidity + offset(log_population),
                 family = nb(), data = ind_dt[is_endemic == TRUE])



# ziplss: first formula = count (Poisson) mean, second = P(presence) linear predictor
m_zi <- gam(list(
  dengue_inc_count ~ s(year_id, a1_f, bs = "fs", k = 4) +
                     s(dengue_suitability, k = 4) + offset(log_population),
  ~ s(year_id, a1_f, bs = "fs", k = 3)          # presence model
), family = ziplss(), data = ind_dt[is_endemic == TRUE])

# frequency: P(outbreak) per admin-1 per year
m_freq <- gam(is_outbreak ~ s(year_id, a1_f, bs = "fs", k = 3),
              family = binomial(), data = ind_dt)

# magnitude: outbreak excess over endemic baseline
ind_dt[, endemic_mu := predict(m_endemic, newdata = ind_dt, type = "response")]
ind_dt[is_outbreak == TRUE,
       excess := dengue_inc_count - endemic_mu]
# model excess (or ratio) with your heavy-tail machinery

d  <- ind_dt[is_endemic == TRUE]
mu <- predict(m_endemic, type = "response")         # trend + suitability + offset, in counts
th <- m_endemic$family$getTheta(TRUE)               # NB dispersion from the fit
d[, r_std := (dengue_inc_count - mu) / sqrt(mu + mu^2/th)]


var(d$r_std)                                          # ≈1 if NB variance is right
plot(mu, d$r_std, log = "x"); abline(h = 0)          # funnel = wrong variance scaling
plot(d$year_id, d$r_std); abline(h = 0)              # trend in resid = mean-trend misspecified
boxplot(r_std ~ a1_f, data = d)                       # differing spread = θ should vary by unit

# ---------------------------------------------------------------------------
# FORMULATIONS -- this is the edit surface.
#
# `engine` picks the fitter: "scam" whenever a monotone bs= is used, else "gam".
# `by = sr_f` is the whole reason this script exists.
# ---------------------------------------------------------------------------
FORMULATIONS <- list(

  `2025_run` = list(
    structure = "inc_cfr", engine = "scam", rake = "point",
    inc = "log_dengue_inc_rate ~ s(dengue_suitability, k = 6, bs = 'mpi') +
             logit_urban_fraction + A0_af",
    cfr = "logit_dengue_cfr ~ log_gdppc_mean + as_f + A0_af"
  ),

  `2025_run_w_time` = list(
    structure = "inc_cfr", engine = "scam", rake = "median",
    inc = "log_dengue_inc_rate ~ s(dengue_suitability, k = 6, bs = 'mpi') +
             logit_urban_fraction + year_centered:sr_f + A0_af",
    cfr = "logit_dengue_cfr ~ log_gdppc_mean + as_f + A0_af"
  ),

  # THE POINT OF THIS SCRIPT: per-super-region time SHAPE, everything else pooled.
  `2025_run_w_time_by_sr` = list(
    structure = "inc_cfr", engine = "gam", rake = "median",
    inc = "log_dengue_inc_rate ~ s(year_centered, by = sr_f, k = 5) + sr_f +
             s(dengue_suitability, k = 6) + logit_urban_fraction + A0_af",
    cfr = "logit_dengue_cfr ~ log_gdppc_mean + as_f + A0_af"
  ),

  `GBD-esque_w_time` = list(
    structure = "mort_then_inc", engine = "scam", rake = "median",
    mort = "log_dengue_mort_rate ~ s(dengue_suitability, k = 6, bs = 'mpi') +
              log_gdppc_mean + urban_fraction + year_centered:sr_f + A0_af",
    inc  = "log_dengue_inc_rate ~ log_dengue_mort_rate +
              s(dengue_suitability, k = 6, bs = 'mpi') + log_gdppc_mean +
              urban_fraction + year_centered:sr_f + A0_af"
  ),

  `GBD-esque_w_time_by_sr` = list(
    structure = "mort_then_inc", engine = "gam", rake = "median",
    mort = "log_dengue_mort_rate ~ s(year_centered, by = sr_f, k = 5) + sr_f +
              s(dengue_suitability, k = 6) + log_gdppc_mean + urban_fraction + A0_af",
    inc  = "log_dengue_inc_rate ~ log_dengue_mort_rate +
              s(year_centered, by = sr_f, k = 5) + sr_f +
              s(dengue_suitability, k = 6) + log_gdppc_mean + urban_fraction + A0_af"
  )
)

wanted <- if (nzchar(FORMULATION_SUBSET)) {
  trimws(strsplit(FORMULATION_SUBSET, ",")[[1]])
} else names(FORMULATIONS)

# ---------------------------------------------------------------------------
# Fitting + anchoring
# ---------------------------------------------------------------------------
fit_one <- function(formula_text, dat, engine) {
  f <- as.formula(gsub("\\s+", " ", formula_text))
  rows <- dat[complete.cases(dat[, all.vars(f), with = FALSE])]
  fitter <- if (engine == "scam") scam::scam else mgcv::gam
  list(model = fitter(f, data = rows), n_fit = nrow(rows), formula = f)
}

# Per-group additive shift in MODEL space, from observed over the rake window.
# `point` reproduces observed exactly at ANCHOR_YEAR; `median` uses the median of
# observed minus the median of predicted over RAKE_WINDOW.
anchor_shift <- function(dat, response, predicted, keys, mode) {
  d <- copy(dat)
  d[, .pred := predicted]
  win <- if (mode == "point") ANCHOR_YEAR else RAKE_WINDOW
  w <- d[year_id %in% win & is.finite(get(response)) & is.finite(.pred)]
  if (mode == "point") {
    w[, .(shift = get(response)[1] - .pred[1]), by = keys]
  } else {
    w[, .(shift = median(get(response)) - median(.pred)), by = keys]
  }
}

# Age/sex counts -> all-age counts -> every ancestor (count space) -> rates by
# that level's OWN population. Never sum a population to make a denominator.


run_formulation <- function(name, spec) {
  message(glue("\n=== {name} ({spec$structure}, {spec$engine}, rake={spec$rake}) ==="))
  keys_cell <- c("location_id", "age_group_id", "sex_id")

  if (spec$structure == "inc_cfr") {
    inc <- fit_one(spec$inc, BASE, spec$engine)
    cfr <- fit_one(spec$cfr, past, spec$engine)      # CFR across all age/sex
    message(glue("  inc n={inc$n_fit}  cfr n={cfr$n_fit}"))

    b <- copy(BASE); b[, .pred := as.numeric(predict(inc$model, newdata = b))]
    # broadcast the base-cell prediction onto cells, then anchor each cell
    cells <- past[, ..keys_cell] |> unique()
    bb <- merge(cells, b[, .(location_id, year_id, .pred)], by = "location_id",
                allow.cartesian = TRUE)
    bb <- merge(bb, past[, .(location_id, year_id, age_group_id, sex_id,
                             log_dengue_inc_rate, population, rr_inc_as)],
                by = c("location_id", "year_id", "age_group_id", "sex_id"))
    sh <- anchor_shift(bb, "log_dengue_inc_rate", bb$.pred, keys_cell, spec$rake)
    bb <- merge(bb, sh, by = keys_cell, all.x = TRUE)
    bb <- bb[is.finite(shift)]
    bb[, inc_rate := exp(.pred + shift)]

    c2 <- copy(past); c2 <- c2[complete.cases(c2[, all.vars(cfr$formula), with = FALSE])]
    c2[, .pred := as.numeric(predict(cfr$model, newdata = c2))]
    shc <- anchor_shift(c2, "logit_dengue_cfr", c2$.pred, keys_cell, spec$rake)
    c2 <- merge(c2, shc, by = keys_cell, all.x = TRUE)
    c2[, cfr := 1 / (1 + exp(-(.pred + shift)))]

    d <- merge(bb[, .(location_id, year_id, age_group_id, sex_id, inc_rate, population)],
               c2[, .(location_id, year_id, age_group_id, sex_id, cfr)],
               by = c("location_id", "year_id", "age_group_id", "sex_id"), all.x = TRUE)
    d[is.na(cfr), cfr := 0]                       # never invent deaths
    d[, inc_count  := inc_rate * population]
    d[, mort_count := inc_count * cfr]
    return(list(products = aggregate_products(d), models = list(inc = inc, cfr = cfr)))
  }

  # mort_then_inc: both all-age. Mortality is OBSERVED at fit and PREDICTED at
  # predict -- training on the model's own output would fit its error.
  aa <- past[, .(inc_rate = sum(dengue_inc_rate * population) / sum(population),
                 mort_rate = sum(dengue_mort_rate * population) / sum(population),
                 population = sum(population)),
             by = .(location_id, year_id)]
  aa <- merge(aa, unique(BASE[, .(location_id, year_id, dengue_suitability,
                                  urban_fraction, log_gdppc_mean, relative_humidity,
                                  year_centered, A0_af, sr_f)]),
              by = c("location_id", "year_id"))
  aa[, log_dengue_inc_rate  := ifelse(inc_rate  > 0, log(inc_rate),  NA_real_)]
  aa[, log_dengue_mort_rate := ifelse(mort_rate > 0, log(mort_rate), NA_real_)]

  mort <- fit_one(spec$mort, aa, spec$engine)
  aa[, .pred_mort := as.numeric(predict(mort$model, newdata = aa))]
  shm <- anchor_shift(aa, "log_dengue_mort_rate", aa$.pred_mort, "location_id", spec$rake)
  aa <- merge(aa, shm, by = "location_id", all.x = TRUE)
  aa[, mort_hat := .pred_mort + shift]

  inc <- fit_one(spec$inc, aa, spec$engine)        # fitted on OBSERVED mortality
  aa2 <- copy(aa)[, log_dengue_mort_rate := mort_hat]   # predicted at predict
  aa2[, .pred_inc := as.numeric(predict(inc$model, newdata = aa2))]
  shi <- anchor_shift(aa, "log_dengue_inc_rate",
                      as.numeric(predict(inc$model, newdata = aa)),
                      "location_id", spec$rake)
  aa2 <- merge(aa2[, !"shift"], shi, by = "location_id", all.x = TRUE)
  aa2[, inc_count  := exp(.pred_inc + shift) * population]
  aa2[, mort_count := exp(mort_hat) * population]
  message(glue("  mort n={mort$n_fit}  inc n={inc$n_fit}"))
  list(products = aggregate_products(
         aa2[, .(location_id, year_id, age_group_id = 22L, sex_id = 3L,
                 inc_count, mort_count)]),
       models = list(mort = mort, inc = inc))
}

# ---------------------------------------------------------------------------
# Run, score, plot
# ---------------------------------------------------------------------------
obs_levels <- obs_aa[, .(location_id, year_id,
                         obs_inc = dengue_inc_rate, obs_mort = dengue_mort_rate)]
results <- list(); metrics <- list()

for (nm in wanted) {
  res <- run_formulation(nm, FORMULATIONS[[nm]])
  results[[nm]] <- res
  m <- merge(res$products, obs_levels, by = c("location_id", "year_id"))
  m <- merge(m, hier[, .(location_id, level)], by = "location_id")
  for (lv in c(0, 1)) for (meas in c("inc", "mort")) {
    d <- m[level == lv & is.finite(get(paste0("obs_", meas))) &
             is.finite(get(paste0(meas, "_rate")))]
    if (nrow(d) < 3) next
    fitlm <- lm(get(paste0("obs_", meas)) ~ get(paste0(meas, "_rate")), data = d)
    metrics[[length(metrics) + 1]] <- data.table(
      formulation = nm, level = lv, measure = meas,
      r = cor(d[[paste0("obs_", meas)]], d[[paste0(meas, "_rate")]]),
      slope = coef(fitlm)[2], n = nrow(d))
  }
  # per-super-region year smooths, the reason for the by= term
  for (which_mod in names(res$models)) {
    mo <- res$models[[which_mod]]$model
    if (!any(grepl("year_centered", names(coef(mo))))) next
    png(file.path(OUT, glue("{nm}_{which_mod}_smooths.png")), 1400, 900, res = 110)
    tryCatch(plot(mo, pages = 1, scale = 0, se = TRUE, shade = TRUE,
                  main = glue("{nm} - {which_mod}")),
             error = function(e) plot.new())
    dev.off()
  }
}

met <- rbindlist(metrics)
fwrite(met, file.path(OUT, "formulation_metrics.csv"))
print(met)

# global + super-region timeseries, all formulations overlaid
sr_ids <- sort(unique(as.integer(as.character(past$sr_f))))
for (loc in c(1L, sr_ids)) {
  png(file.path(OUT, glue("timeseries_loc{loc}.png")), 1400, 800, res = 110)
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  for (meas in c("inc", "mort")) for (kind in c("count", "rate")) {
    col_nm <- paste0(meas, "_", kind)
    o <- obs_levels[location_id == loc][order(year_id)]
    ov <- if (kind == "rate") o[[paste0("obs_", meas)]] else
      o[[paste0("obs_", meas)]] * pop_aa[location_id == loc][order(year_id)]$population[seq_len(nrow(o))]
    ys <- lapply(wanted, function(nm) results[[nm]]$products[location_id == loc][order(year_id)][[col_nm]])
    plot(o$year_id, ov, type = "l", lwd = 3, col = "black",
         xlab = "year", ylab = col_nm, main = glue("loc {loc} - {col_nm}"),
         ylim = range(c(ov, unlist(ys)), na.rm = TRUE))
    for (k in seq_along(wanted)) {
      pr <- results[[wanted[k]]]$products[location_id == loc][order(year_id)]
      lines(pr$year_id, pr[[col_nm]], col = k + 1, lwd = 1.6)
    }
  }
  dev.off()
}
message(glue("\nwrote plots + formulation_metrics.csv to {OUT}"))
