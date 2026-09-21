# Build the frame the malaria scam/gam/lm fits are estimated on.
#
# One function, sourced by the selection worker and the final fitter (and by any caller
# that passes it as `--prep-script`). The contract is the name and signature below, plus
# the columns the returned frame carries. Every derived column is computed
# unconditionally; the formula decides which rows and columns matter (R's na.omit drops
# a row only when a column the formula names is NA). Non-finite results of a transform
# (log(0), log of a negative) are set to NA so that rule holds for them too.
#
# The response `logit_malaria_pfpr` is taken from the parquet as stored, never recomputed:
# one row has malaria_pfpr >= 1, where recomputation would give Inf.

MALARIA_LOG_COVARIATES <- c(
  "mal_DAH_total_per_capita", "gdppc_mean", "ldipc_mean", "med_consumppc",
  "malaria_inc_rate", "malaria_mort_rate"
)

clipped_logit <- function(fraction, floor = 0.001, ceiling = 0.999) {
  f <- pmin(pmax(fraction, floor), ceiling)
  log(f / (1 - f))
}

# The two suitability columns every formula names, from one variant. Called by the
# preparation for the run default and again per spec when a spec names another variant.
add_suit_terms <- function(frame, suit_variant) {
  suit_col <- paste0("malaria_suitability_", suit_variant)
  if (!suit_col %in% names(frame)) {
    stop("suit_variant '", suit_variant, "' -> column '", suit_col, "' not in past inputs")
  }
  frame$malaria_suit <- frame[[suit_col]]
  frame$logit_malaria_suitability <- clipped_logit(frame[[suit_col]] / 365)
  attr(frame, "suit_variant") <- suit_variant
  frame
}

prepare_malaria_fit_frame <- function(parquet_path, inc_count_min, pfpr_min,
                                      suit_variant = "mordecai_0_0") {
  stopifnot(is.numeric(inc_count_min), length(inc_count_min) == 1L,
            is.numeric(pfpr_min), length(pfpr_min) == 1L)
  if (!file.exists(parquet_path)) stop("past inputs not found: ", parquet_path)
  frame <- as.data.frame(arrow::read_parquet(parquet_path))
  n_read <- nrow(frame)

  frame$malaria_inc_count <- frame$malaria_inc_rate * frame$population
  for (cov in MALARIA_LOG_COVARIATES) {
    # log of a negative is NaN with a warning; the non-finite count below reports it instead
    frame[[paste0("log_", cov)]] <- suppressWarnings(log(frame[[cov]]))
  }
  frame$do30_fraction <- pmin(pmax(frame$days_over_30C / 365, 0.001), 0.999)
  frame$logit_do30    <- log(frame$do30_fraction / (1 - frame$do30_fraction))
  frame$rh_fraction   <- pmin(pmax(frame$relative_humidity / 100, 0.001), 0.999)
  frame$logit_relative_humidity <- log(frame$rh_fraction / (1 - frame$rh_fraction))

  frame <- add_suit_terms(frame, suit_variant)

  derived <- c(paste0("log_", MALARIA_LOG_COVARIATES), "logit_do30",
               "logit_relative_humidity", "logit_malaria_suitability")
  non_finite <- Reduce(`|`, lapply(derived, function(col) {
    x <- frame[[col]]
    is.infinite(x) | is.nan(x)
  }))
  if (any(non_finite)) {
    message(sprintf("prepare_malaria_fit_frame: %d rows carry a non-finite transform (set to NA)",
                    sum(non_finite)))
    for (col in derived) frame[[col]][!is.finite(frame[[col]])] <- NA
  }

  keep <- which(frame$malaria_inc_count >= inc_count_min & frame$malaria_pfpr >= pfpr_min)
  frame <- frame[keep, , drop = FALSE]
  frame$A0_af <- as.factor(frame$A0_location_id)

  message(sprintf("prepare_malaria_fit_frame: %d rows read -> %d after inc_count >= %s & pfpr >= %s; %d A0 levels",
                  n_read, nrow(frame), format(inc_count_min), format(pfpr_min), nlevels(frame$A0_af)))
  attr(frame, "n_read") <- n_read
  attr(frame, "n_non_finite_rows") <- sum(non_finite)
  attr(frame, "thresholds") <- c(inc_count_min = inc_count_min, pfpr_min = pfpr_min)
  attr(frame, "suit_variant") <- suit_variant
  frame
}
