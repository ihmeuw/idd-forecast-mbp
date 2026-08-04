#!/usr/bin/env Rscript
### plot_finalist_smooths.r
### Fitted-smooth / partial-dependence tooling for a hand-picked set of finalist specs.
### Refits each finalist on the FULL data (via the worker's load_past_data, so lags are
### empty -> unrestricted rows) and emits two things:
###
###   (A) SINGLE-WINNER SHAPE: one PDF per spec, plot.scam(pages=1) — the fitted smooths
###       for eyeballing whether the response shapes are scientifically sensible.
###
###   (B) ENSEMBLE-MEMBER DIVERSITY: for each covariate smoothed by >=2 finalists, overlay
###       the fitted (centered) term curves on a common grid, colored by spec, with the
###       outer deciles shaded as the extrapolation "edge" — plus an edge_divergence table
###       scoring how far the specs' smooths spread apart at the grid extremes. Members that
###       diverge at the edges add real forecast uncertainty (that's the 2100-relevant
###       region); members that overlap everywhere are near-duplicates that don't.
###
### Selection is NOT run here — you pass the finalist spec_indices in explicitly.
###
### Usage:
###   Rscript plot_finalist_smooths.r --run-dir <dir> --specs 46,12,... [--out-dir <dir>]
suppressPackageStartupMessages({
  library(optparse); library(arrow); library(data.table); library(glue); library(scam)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--run-dir", type = "character", help = "run dir (spec_table.parquet lives here)"),
  make_option("--specs",   type = "character", help = "comma-separated finalist spec_index list"),
  make_option("--worker",  type = "character",
              default = "src/idd_forecast_mbp/03_modeling/select_malaria_models_rocket.r",
              help = "path to select_malaria_models_rocket.r (sourced for load_past_data/fit_scam_one)"),
  make_option("--out-dir", type = "character", default = NA, help = "default <run-dir>/finalist_plots"),
  make_option("--maxit",   type = "integer",   default = 30L),
  make_option("--n-grid",  type = "integer",   default = 200L))))

stopifnot("need --run-dir" = !is.null(opt$`run-dir`),
          "need --specs"   = !is.null(opt$specs))
run_dir <- opt$`run-dir`
out_dir <- if (is.na(opt$`out-dir`)) file.path(run_dir, "finalist_plots") else opt$`out-dir`
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
sids <- as.integer(strsplit(opt$specs, ",")[[1]])

# Worker gives us the *identical* data pipeline (load_past_data) + fit_scam_one. The
# entrypoint guard means sourcing it does NOT run main(). lags is empty -> full data.
source(opt$worker)

message(glue("Loading full data via worker load_past_data() ..."))
dat <- load_past_data(past_inputs_parquet())
message(glue("  {nrow(dat)} rows"))

spec_tab <- as.data.table(read_parquet(file.path(run_dir, "spec_table.parquet")))

# --- refit each finalist on full data -------------------------------------------------
fits <- list()
for (sid in sids) {
  fml <- as.formula(spec_tab[spec_index == sid, formula_text])
  message(glue("Refitting spec {sid} ..."))
  res <- fit_scam_one(fml, dat, optimizer = "efs", maxit = opt$maxit, label = glue("spec{sid}"))
  fits[[as.character(sid)]] <- res$fit
}
fits <- Filter(Negate(is.null), fits)

# --- (A) per-spec fitted-smooth pages -------------------------------------------------
for (sid in names(fits)) {
  f <- fits[[sid]]
  n_sm <- length(f$smooth)
  if (!n_sm) next
  ncol <- min(n_sm, 3L); nrow <- ceiling(n_sm / ncol)
  pdf(file.path(out_dir, glue("spec_{sid}_smooths.pdf")), width = 4 * ncol, height = 4 * nrow + 0.5)
  plot(f, pages = 1, scale = 0, rug = FALSE, shade = TRUE)
  dev.off()
}
message(glue("(A) wrote per-spec smooth PDFs to {out_dir}"))

# --- (B) cross-spec overlay + edge-divergence -----------------------------------------
# Build a reference row (medians / first A0 level) and vary one covariate over a grid.
ref_row <- dat[1, , drop = FALSE]
num_cols <- names(dat)[vapply(dat, is.numeric, logical(1))]
for (nm in num_cols) ref_row[[nm]] <- median(dat[[nm]], na.rm = TRUE)
if ("A0_af" %in% names(dat)) ref_row$A0_af <- factor(levels(dat$A0_af)[1], levels = levels(dat$A0_af))

# map covariate -> the fits that smooth it (via each fit's smooth$term)
smoothed <- list()
for (sid in names(fits)) for (sm in fits[[sid]]$smooth)
  smoothed[[sm$term]] <- c(smoothed[[sm$term]], sid)

edge_rows <- list()
for (var in names(smoothed)) {
  specs_v <- unique(smoothed[[var]])
  if (length(specs_v) < 2) next            # overlay only makes sense for >=2 specs
  rng  <- range(dat[[var]], na.rm = TRUE)
  grid <- seq(rng[1], rng[2], length.out = opt$`n-grid`)
  nd   <- ref_row[rep(1, length(grid)), , drop = FALSE]; nd[[var]] <- grid
  curves <- sapply(specs_v, function(sid) {
    lab <- fits[[sid]]$smooth[[which(vapply(fits[[sid]]$smooth, function(s) s$term, "") == var)[1]]]$label
    predict(fits[[sid]], newdata = nd, type = "terms")[, lab]
  })
  # edge = outer 10% of the covariate range (where 2100 extrapolation lives)
  edge <- grid <= quantile(grid, 0.1) | grid >= quantile(grid, 0.9)
  spread_all  <- max(apply(curves, 1, function(z) diff(range(z))))
  spread_edge <- max(apply(curves[edge, , drop = FALSE], 1, function(z) diff(range(z))))
  edge_rows[[var]] <- data.table(covariate = var, n_specs = length(specs_v),
                                 spread_overall = spread_all, spread_edge = spread_edge)

  pdf(file.path(out_dir, glue("overlay_{var}.pdf")), width = 7, height = 5)
  matplot(grid, curves, type = "l", lty = 1, lwd = 2, col = seq_along(specs_v),
          xlab = var, ylab = "centered smooth contribution",
          main = glue("{var}: fitted smooth by spec (edge shaded)"))
  q <- quantile(grid, c(0.1, 0.9))
  abline(v = q, col = "grey70", lty = 3)
  rect(rng[1], par("usr")[3], q[1], par("usr")[4], col = rgb(0,0,0,0.05), border = NA)
  rect(q[2], par("usr")[3], rng[2], par("usr")[4], col = rgb(0,0,0,0.05), border = NA)
  legend("topleft", legend = glue("spec {specs_v}"), col = seq_along(specs_v), lty = 1, lwd = 2, bty = "n")
  dev.off()
}

if (length(edge_rows)) {
  edge_div <- rbindlist(edge_rows)[order(-spread_edge)]
  fwrite(edge_div, file.path(out_dir, "edge_divergence.csv"))
  message("(B) edge-divergence (higher spread_edge = more distinct extrapolation behavior):")
  print(edge_div)
} else {
  message("(B) no covariate is smoothed by >=2 finalists -> no overlay produced.")
}
message(glue("Done. Plots + edge_divergence.csv in {out_dir}"))
