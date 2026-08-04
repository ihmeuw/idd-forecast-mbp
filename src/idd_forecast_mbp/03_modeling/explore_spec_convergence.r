# explore_spec_convergence.r  -- run line by line (interactive)
# Reproduces a single no-FE screen fit from select_malaria_models_rocket.r
# (same data + transforms). Investigating why spec 1441 hits maxit=50 / conv=FALSE
# while spec 2315 (more smooths) converges in 1 iter.
#
# The cluster run set OPENBLAS_NUM_THREADS=8; set it before R starts if you want
# to match timing. The screen used K_DEFAULT=4, MAXIT=50, optimizer="bfgs".

library(scam)   # also attaches mgcv
library(arrow)

MAXIT <- 50L    # bump to 300 to see whether 1441 *eventually* converges

parquet_path <- "/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/past_inputs_nc/lsae_1285/current/malaria_past_inputs.parquet"

# ---- load + transforms (verbatim from load_past_data) ----
past_data <- as.data.frame(read_parquet(parquet_path))

nan_toss <- function(df, var) { i <- which(is.na(df[var])); if (length(i)) df[-i, ] else df }
past_data <- nan_toss(past_data, "malaria_pfpr")
past_data <- nan_toss(past_data, "gdppc_mean")
past_data <- nan_toss(past_data, "mal_DAH_total_per_capita")

past_data$malaria_suit_fraction     <- pmin(pmax(past_data$malaria_suitability_mordecai_0_0 / 365, 0.001), 0.999)
past_data$logit_malaria_suitability <- log(past_data$malaria_suit_fraction / (1 - past_data$malaria_suit_fraction))
past_data$do30_fraction <- pmin(pmax(past_data$days_over_30C / 365, 0.001), 0.999)
past_data$logit_do30    <- log(past_data$do30_fraction / (1 - past_data$do30_fraction))
past_data$rh_fraction   <- pmin(pmax(past_data$relative_humidity / 100, 0.001), 0.999)
past_data$logit_relative_humidity <- log(past_data$rh_fraction / (1 - past_data$rh_fraction))
for (cov in c("mal_DAH_total_per_capita", "gdppc_mean", "ldipc_mean", "med_consumppc"))
  past_data[[paste0("log_", cov)]] <- log(past_data[[cov]])

nrow(past_data)   # expect 313122
K_DEFAULT <- 6














RhpcBLASctl::blas_set_num_threads(8)


# ---- spec 1441: hit maxit=50, conv=FALSE, ~2.8 min ----
# FULL formula from param_map (4 smooths + 3 LINEAR terms). The linear terms are
# what differ from a smooths-only fit -- include them or you fit a different model.
fml_1441 <- logit_malaria_pfpr ~
  s(mal_DAH_total_per_capita, k = K_DEFAULT, bs = "mpd") +
  s(gdppc_mean,               k = K_DEFAULT, bs = "mpd") +
  weighted_100m_urban_threshold_1500.0_simple_mean +   # linear
  people_flood_days_per_capita +                       # linear
  s(total_precipitation,      k = K_DEFAULT, bs = "mpi") +
  s(relative_humidity,        k = K_DEFAULT, bs = "mpi") +
  mean_temperature                                     # linear

t0  <- Sys.time()
fit <- scam(fml_1441, data = past_data, optimizer = "bfgs", control = list(maxit = MAXIT))
difftime(Sys.time(), t0, units = "mins")

fit$iter        # iterations used (50 == hit the cap)
fit$conv        # scam convergence info
summary(fit)    # works on a live object (unlike a reloaded .rds)
plot(fit, pages = 1, scale = 0, shade = TRUE)

# ---- spec 2315: 6 smooths, converged in 1 iter, ~4.2 min -- swap in to compare ----
# fml_2315 <- logit_malaria_pfpr ~
#   s(mal_DAH_total_per_capita,                         k = 4, bs = "mpd") +
#   s(gdppc_mean,                                       k = 4, bs = "mpd") +
#   s(weighted_1km_urban_threshold_1500.0_simple_mean, k = 4, bs = "mpd") +
#   s(people_flood_days_per_capita,                     k = 4, bs = "mpi") +
#   s(precipitation_days,                               k = 4, bs = "mpi") +
#   relative_humidity +                                  # linear
#   s(mean_temperature,                                 k = 4, bs = "cv")
# t0  <- Sys.time()
# fit2 <- scam(fml_2315, data = past_data, optimizer = "bfgs", control = list(maxit = MAXIT))
# difftime(Sys.time(), t0, units = "mins"); fit2$iter; fit2$conv

# ---- spec 5318: did NOT converge on 4524 (iter=50, conv=FALSE, 538s) ----
# FULL formula from the 20260609 param_map: 4 smooths + 3 linear terms.
# NOTE: the cluster ran k=4 -- set K_DEFAULT <- 4 above to reproduce the
# non-convergence (at k=6 it may behave differently). Other 4524 non-convergers
# you could swap in: 2261, 9058, 11508 (the 900s one).

scale_vars <- c("mal_DAH_total_per_capita", "gdppc_mean",
                "weighted_1km_urban_threshold_300.0_simple_mean",
                "relative_humidity")

for (v in scale_vars) {
  past_data[[paste0(v, "_z")]] <- as.numeric(scale(past_data[[v]]))
}
fml_5318_gam_z <- logit_malaria_pfpr ~
  s(mal_DAH_total_per_capita_z,                        k = K_DEFAULT) +
  s(gdppc_mean_z,                                      k = K_DEFAULT) +
  s(weighted_1km_urban_threshold_300.0_simple_mean_z, k = K_DEFAULT) +
  people_flood_days_per_capita +
  precipitation_days +
  s(relative_humidity_z,                               k = K_DEFAULT) +
  mean_high_temperature

fit_5318_gam_z <- mgcv::gam(fml_5318_gam_z, data = past_data)

fml_5318_z <- logit_malaria_pfpr ~
  s(mal_DAH_total_per_capita_z,                        k = K_DEFAULT, bs = "mpd") +
  s(gdppc_mean_z,                                      k = K_DEFAULT, bs = "mpd") +
  s(weighted_1km_urban_threshold_300.0_simple_mean_z, k = K_DEFAULT, bs = "mpd") +
  people_flood_days_per_capita +
  precipitation_days +
  s(relative_humidity_z,                               k = K_DEFAULT, bs = "mpi") +
  mean_high_temperature

K_DEFAULT <- 4

fml_5318 <- logit_malaria_pfpr ~
  s(mal_DAH_total_per_capita,                        k = K_DEFAULT, bs = "mpd") +
  s(gdppc_mean,                                      k = K_DEFAULT, bs = "mpd") +
  s(weighted_1km_urban_threshold_300.0_simple_mean, k = K_DEFAULT, bs = "mpd") +
  people_flood_days_per_capita +
  precipitation_days +
  s(relative_humidity,                               k = K_DEFAULT, bs = "mpi") +
  mean_high_temperature

t0  <- Sys.time()
fit <- scam(fml_5318, data = past_data, optimizer = "efs")
print(difftime(Sys.time(), t0, units = "mins"))
print(fit$iter)        # iterations used (50 == hit the cap
print(fit$conv)   




t0  <- Sys.time()
fit <- scam(fml_5318_z, data = past_data, optimizer = "efs")
print(difftime(Sys.time(), t0, units = "mins"))
print(fit$iter)        # iterations used (50 == hit the cap
print(fit$conv)        # scam convergence info)



t0  <- Sys.time()
fit <- scam(fml_5318, data = past_data, optimizer = "efs")
print(difftime(Sys.time(), t0, units = "mins"))
print(fit$iter)        # iterations used (50 == hit the cap
print(fit$conv)        # scam convergence info)

t0  <- Sys.time()
fit <- scam(fml_5318, data = past_data, optimizer = "bfgs", control = list(maxit = MAXIT))
print(difftime(Sys.time(), t0, units = "mins"))
print(fit$iter)        # iterations used (50 == hit the cap
print(fit$conv)        # scam convergence info)





# fml_5318 <- logit_malaria_pfpr ~
#   s(mal_DAH_total_per_capita,                        k = K_DEFAULT, bs = "mpd") +
#   s(gdppc_mean,                                      k = K_DEFAULT, bs = "mpd") +
#   s(weighted_1km_urban_threshold_300.0_simple_mean, k = K_DEFAULT, bs = "mpd") +
#   people_flood_days_per_capita +     # linear
#   precipitation_days +               # linear
#   s(relative_humidity,                               k = K_DEFAULT, bs = "mpi") +
#   mean_high_temperature              # linear

t0  <- Sys.time()
# fit <- scam(fml_5318, data = past_data, optimizer = "bfgs", control = list(maxit = MAXIT))
fit <- scam(fml_5318_z, data = past_data)
difftime(Sys.time(), t0, units = "mins")
fit$iter; fit$conv
summary(fit)












# --- versions (cluster: scam 1.2.13, R 4.2.2) ---
R.version.string
packageVersion("scam"); packageVersion("mgcv")

# --- which BLAS/LAPACK is actually linked (the lib scam's linear algebra uses) ---
extSoftVersion()[["BLAS"]]
La_library()
sessionInfo()                       # BLAS/LAPACK lines at the very bottom

# --- cores R sees + BLAS thread env (cluster forced OPENBLAS_NUM_THREADS=8) ---
parallel::detectCores()
Sys.getenv(c("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"))

# --- threads BLAS will actually use (if RhpcBLASctl is installed) ---
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  cat("BLAS procs:", RhpcBLASctl::blas_get_num_procs(),
      " cores:", RhpcBLASctl::get_num_cores(), "\n")
}

# --- host + is this a Slurm allocation? + machine memory/cores ---
Sys.info()[c("nodename","sysname")]
Sys.getenv(c("SLURM_JOB_ID","SLURM_CPUS_PER_TASK","SLURM_MEM_PER_NODE"))
system("nproc"); system("free -h")





library(scam); library(arrow)
for (nt in c(1, 8, 64)) {
  RhpcBLASctl::blas_set_num_threads(nt)   # plus set the env var before R for a real test
  t0 <- Sys.time()
  fit <- scam(fml_1441, data = past_data, optimizer = "bfgs",
              control = list(maxit = 50))
  cat(sprintf("nt=%d iter=%d conv=%s %.1fs\n",
              nt, fit$iter, fit$conv, as.numeric(Sys.time()-t0)))
}
