#!/usr/bin/env Rscript
# Evaluate each term of a fitted malaria model set (pfpr_mod, inc_mod, mort_mod) on a fine grid, for the
# scenario-gap decomposition (decompose_scenario_gap.py).
#
# The three models are additive on the link scale, so a term's shape is the change in the link prediction as
# that one covariate moves with every other covariate fixed. Constants (intercept, A0_af, the fixed values of
# the other covariates) drop out because the forecast shifts every admin-2 to its 2023 observation, so only
# differences of a term matter. Linear terms are written as the link change per unit.
#
# Grid sizes are multiples of 1000: scam 1.2-21 (the singularity image's version) mispredicts univariate
# shape-constrained smooths when the frame has n = 1 or n = 1 (mod 1000) rows.

suppressPackageStartupMessages({
  library(data.table); library(mgcv); library(scam); library(arrow); library(optparse)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--model-rdata", type = "character", help = "malaria_models.RData from the models node"),
  make_option("--out-dir", type = "character", help = "directory for term_grids.parquet + linear_terms.parquet"),
  make_option("--dah-max", type = "double", help = "upper end of the DAH-per-capita grid"),
  make_option("--gdp-min", type = "double", help = "lower end of the GDP-per-capita grid (log-spaced)"),
  make_option("--gdp-max", type = "double", help = "upper end of the GDP-per-capita grid (log-spaced)"),
  make_option("--logit-pfpr-min", type = "double", default = -40),
  make_option("--logit-pfpr-max", type = "double", default = 12),
  make_option("--n", type = "integer", default = 200000L, help = "grid points per term (multiple of 1000)"),
  # object names in the RData: the current fits use pfpr_mod / inc_mod / mort_mod; the 1st submission's (2025_07_08)
  # use malaria_pfpr_mod / incidence_scam_mod / mortality_scam_mod
  make_option("--pfpr-object", type = "character", default = "pfpr_mod"),
  make_option("--inc-object", type = "character", default = "inc_mod"),
  make_option("--mort-object", type = "character", default = "mort_mod")
)))
if (opt$n %% 1000L != 0L) stop("--n must be a multiple of 1000 (scam 1.2-21 prediction bug)")

e <- new.env()
load(opt$`model-rdata`, envir = e)
models <- list(pfpr = get(opt$`pfpr-object`, envir = e), inc = get(opt$`inc-object`, envir = e),
               mort = get(opt$`mort-object`, envir = e))

# every covariate any of the three models reads, at a fixed value; A0_af at the model's first level
fixed_frame <- function(mod, n) {
  data.table(
    mal_DAH_total_per_capita  = rep(0, n),
    gdppc_mean                = rep(1000, n),
    log_gdppc_mean            = rep(log(1000), n),
    logit_malaria_suitability = rep(0, n),
    wind_speed                = rep(0, n),
    people_flood_days_per_capita = rep(0, n),
    logit_malaria_pfpr        = rep(0, n),
    A0_af                     = factor(rep(mod$xlevels$A0_af[1], n), levels = mod$xlevels$A0_af)
  )
}

link_curve <- function(mod, var, grid) {
  nd <- fixed_frame(mod, length(grid))
  nd[[var]] <- grid
  as.numeric(predict(mod, newdata = nd))
}

# link change per unit of a linear term; zero when the model does not use the covariate
linear_slope <- function(mod, var) {
  nd <- fixed_frame(mod, 1000L)
  nd[[var]] <- rep(c(0, 1), 500L)
  y <- as.numeric(predict(mod, newdata = nd))
  y[2] - y[1]
}

grids <- list(
  pfpr = list(
    mal_DAH_total_per_capita = seq(0, opt$`dah-max`, length.out = opt$n),
    gdppc_mean = exp(seq(log(opt$`gdp-min`), log(opt$`gdp-max`), length.out = opt$n))
  ),
  inc  = list(logit_malaria_pfpr = seq(opt$`logit-pfpr-min`, opt$`logit-pfpr-max`, length.out = opt$n)),
  mort = list(logit_malaria_pfpr = seq(opt$`logit-pfpr-min`, opt$`logit-pfpr-max`, length.out = opt$n))
)
linear <- list(
  pfpr = c("logit_malaria_suitability", "wind_speed", "people_flood_days_per_capita"),
  inc  = "log_gdppc_mean",
  mort = "log_gdppc_mean"
)

term_rows <- list()
for (m in names(grids)) {
  for (v in names(grids[[m]])) {
    x <- grids[[m]][[v]]
    term_rows[[length(term_rows) + 1L]] <- data.table(model = m, covariate = v, x = x,
                                                      y = link_curve(models[[m]], v, x))
  }
}
lin_rows <- rbindlist(lapply(names(linear), function(m) {
  data.table(model = m, covariate = linear[[m]],
             slope = vapply(linear[[m]], function(v) linear_slope(models[[m]], v), numeric(1)))
}))

dir.create(opt$`out-dir`, recursive = TRUE, showWarnings = FALSE)
write_parquet(rbindlist(term_rows), file.path(opt$`out-dir`, "term_grids.parquet"))
write_parquet(lin_rows, file.path(opt$`out-dir`, "linear_terms.parquet"))
print(lin_rows)
message("wrote ", file.path(opt$`out-dir`, "term_grids.parquet"))
