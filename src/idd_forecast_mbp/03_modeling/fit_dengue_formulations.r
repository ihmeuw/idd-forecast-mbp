#!/usr/bin/env Rscript
# Dengue formulation comparison in mgcv / scam.
#
# WHY THIS EXISTS: pyGAM has no factor-`by` smooth. The Python notebook can give
# each super-region its own year SLOPE (a masked linear column works, because an
# out-of-group 0 contributes beta*0 regardless of beta), but it cannot give each
# super-region its own year SHAPE. Masking fails for a spline: s(0) is not zero,
# and 71-98% of rows sit at exactly 0, so the curve is dominated by other groups'
# responses. The only pyGAM workaround -- fitting separately per super-region --
# makes EVERY term per-super-region, when the point was to vary time alone.
#
# `s(year_centered, by = sr_f)` is the thing we actually want: per-group time
# smooths with suitability / urban / humidity / country still pooled.
#
# Run, batch:
#   export IDD_MODEL_ROOT=<the forecast-mbp data root>
#   Rscript 03_modeling/fit_dengue_formulations.r --out-dir <dir>
#
# Run, interactively (source it, or step through it line by line):
#   put MODEL_ROOT and OUT in ~/.idd_forecast_mbp.R -- see the config block below.
#   Nothing here needs a command line.

suppressPackageStartupMessages({
  library(optparse); library(arrow); library(data.table)
  library(mgcv); library(scam); library(glue)
})

# ---------------------------------------------------------------------------
# Configuration, resolved so the script runs identically batch or interactively.
#
#   Batch (non-interactive):  IDD_MODEL_ROOT in the environment, --out-dir on the
#                             command line. Unchanged from before.
#   Interactive:              ~/.idd_forecast_mbp.R supplies MODEL_ROOT and OUT.
#                             That file lives OUTSIDE the repo, which is how the
#                             absolute paths stay uncommitted. Create it once:
#                                 MODEL_ROOT <- "<the forecast-mbp data root>"
#                                 OUT        <- "<where plots and metrics go>"
#
# Anything already assigned in your session wins over the config file, so you can
# override MODEL_ROOT, OUT or FORMULATION_SUBSET at the console and re-source.
# The config file is read ONLY when interactive, so it can never shadow an
# explicit --out-dir in a batch run.
#
# optparse is likewise consulted only when non-interactive: parse_args() under
# RStudio picks up the front-end's own argv and then dies on a missing --out-dir,
# which is what made this file impossible to step through.
# ---------------------------------------------------------------------------
CFG_FILE <- path.expand("~/.idd_forecast_mbp.R")
if (interactive() && file.exists(CFG_FILE)) {
  .cfg <- new.env(); sys.source(CFG_FILE, envir = .cfg)
  for (.v in c("MODEL_ROOT", "OUT", "FORMULATION_SUBSET")) {
    if (!exists(.v, inherits = FALSE) && exists(.v, envir = .cfg)) {
      assign(.v, get(.v, envir = .cfg))
    }
  }
  rm(.cfg, .v)
}

if (!exists("MODEL_ROOT")) {
  MODEL_ROOT <- Sys.getenv("IDD_MODEL_ROOT", unset = NA_character_)
}
if (!exists("OUT") || !exists("FORMULATION_SUBSET")) {
  opt <- if (interactive()) {
    list(`out-dir` = NULL, formulations = "")
  } else {
    parse_args(OptionParser(option_list = list(
      make_option("--out-dir", type = "character",
                  help = "where to write plots + metrics"),
      make_option("--formulations", type = "character", default = "",
                  help = "comma-separated subset of formulation names; default all")
    )))
  }
  if (!exists("OUT")) OUT <- opt$`out-dir`
  if (!exists("FORMULATION_SUBSET")) FORMULATION_SUBSET <- opt$formulations
}

if (is.na(MODEL_ROOT) || !nzchar(MODEL_ROOT)) {
  stop("MODEL_ROOT is unset. Batch: export IDD_MODEL_ROOT. Interactive: set it in ",
       CFG_FILE, ", or assign MODEL_ROOT at the console.", call. = FALSE)
}
if (is.null(OUT) || !nzchar(OUT)) {
  stop("OUT is unset. Batch: pass --out-dir. Interactive: set it in ",
       CFG_FILE, ", or assign OUT at the console.", call. = FALSE)
}
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

LSAE <- "lsae_1285"
ANCHOR_YEAR  <- 2023
RAKE_WINDOW  <- 2014:2023
URBAN_COL    <- "weighted_1km_urban_threshold_300.0_simple_mean"

p <- function(...) file.path(MODEL_ROOT, ...)

# ---------------------------------------------------------------------------
# Load + assemble the fit frame. Mirrors lib/data/dengue_inputs.py so the two
# engines are comparable; any divergence here makes the comparison meaningless.
# ---------------------------------------------------------------------------
message("reading inputs ...")
hier <- as.data.table(read_parquet(
  p("02-processed_data", "hierarchy", LSAE, "current",
    "full_hierarchy_2023_lsae_1285.parquet")))
fhs_tab <- as.data.table(read_parquet(
  p("02-processed_data", "hierarchy", LSAE, "current",
    "lsae_1285_to_fhs_table.parquet")))

# FHS-most-detailed is the modelling grain: dengue admin-2 was distributed DOWN
# from national data via suitability, so descending buys no information.
fhs_locs <- fhs_tab[most_detailed_fhs == 1, unique(location_id)]

past <- as.data.table(read_parquet(
  p("03-modeling_data", "dengue", "past_inputs_nc", LSAE, "current",
    "dengue_past_inputs.parquet")))
past <- past[location_id %in% fhs_locs]

obs_as <- as.data.table(read_parquet(
  p("02-processed_data", "dengue", "raked_as", LSAE, "current",
    "as_full_dengue_df.parquet")))
obs_aa <- as.data.table(read_parquet(
  p("02-processed_data", "dengue", "raked_aa", LSAE, "current",
    "aa_full_dengue_df.parquet")))
pop_aa <- as.data.table(read_parquet(
  p("02-processed_data", "population", LSAE, "current",
    "aa_2023_full_population_df.parquet")))

# --- derived model columns (same transforms, same clip bounds) --------------
URB_EPS <- 1e-3
logit <- function(x) log(x / (1 - x))
past[, urban_fraction := pmin(pmax(get(URBAN_COL), URB_EPS), 1 - URB_EPS)]
past[, logit_urban_fraction := logit(urban_fraction)]
past[, log_gdppc_mean := log(gdppc_mean)]
past[, log_dengue_inc_rate  := ifelse(dengue_inc_rate  > 0, log(dengue_inc_rate),  NA_real_)]
past[, log_dengue_mort_rate := ifelse(dengue_mort_rate > 0, log(dengue_mort_rate), NA_real_)]
past[, dengue_cfr := ifelse(dengue_inc_rate > 0, dengue_mort_rate / dengue_inc_rate, NA_real_)]
past[, logit_dengue_cfr := ifelse(dengue_cfr > 0 & dengue_cfr < 1, logit(dengue_cfr), NA_real_)]

# --- observed age/sex relative risks at the anchor year ---------------------
# rr universe is ALWAYS FHS-most-detailed, whatever grain we fit at: the pattern
# is defined per FHS location and inherited downward.
obs_as <- obs_as[location_id %in% fhs_locs & year_id == ANCHOR_YEAR]
base_cell <- obs_as[age_group_id == 3 & sex_id == 1 & dengue_inc_count > 0]
base_locs <- base_cell[, unique(location_id)]
rr <- merge(obs_as[location_id %in% base_locs,
                   .(location_id, age_group_id, sex_id, dengue_inc_rate)],
            base_cell[, .(location_id, base_rate = dengue_inc_rate)],
            by = "location_id", all.x = TRUE)
rr[, rr_inc_as := dengue_inc_rate / base_rate]
rr <- rr[, .(fhs_location_id = location_id, age_group_id, sex_id, rr_inc_as)]

past[, fhs_location_id := location_id]
past <- past[fhs_location_id %in% base_locs]
past <- merge(past, rr, by = c("fhs_location_id", "age_group_id", "sex_id"), all.x = TRUE)

# --- factors and the centred year ------------------------------------------
# Centred, not raw: see the notebook. A by-factor smooth does not need masking,
# but centring keeps the two engines on the same covariate.
past <- merge(past, hier[, .(location_id, super_region_id)], by = "location_id", all.x = TRUE)
YEAR_CENTER <- mean(past$year_id)
past[, year_centered := year_id - YEAR_CENTER]
past[, A0_af := factor(A0_location_id)]
past[, as_f  := factor(paste0(age_group_id, "_", sex_id))]
past[, sr_f  := factor(super_region_id)]

BASE <- past[age_group_id == 3 & sex_id == 1]            # base-cell grain
message(glue("fit frame: {nrow(BASE)} base-cell rows, ",
             "{uniqueN(BASE$location_id)} locations, ",
             "{uniqueN(BASE$sr_f)} super-regions, centre {round(YEAR_CENTER,1)}"))

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
aggregate_products <- function(as_dt) {
  aa <- as_dt[, .(inc_count = sum(inc_count, na.rm = TRUE),
                  mort_count = sum(mort_count, na.rm = TRUE)),
              by = .(location_id, year_id)]
  paths <- hier[, .(location_id, path_to_top_parent)]
  aa <- merge(aa, paths, by = "location_id")
  long <- aa[, .(ancestor = as.integer(strsplit(path_to_top_parent, ",")[[1]])),
             by = .(location_id, year_id, inc_count, mort_count)]
  rolled <- long[, .(inc_count = sum(inc_count), mort_count = sum(mort_count)),
                 by = .(location_id = ancestor, year_id)]
  out <- merge(rolled, pop_aa[, .(location_id, year_id, population)],
               by = c("location_id", "year_id"), all.x = TRUE)
  out[, inc_rate  := inc_count  / population]
  out[, mort_rate := mort_count / population]
  out[]
}

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
