# Dead ends

<!-- Append-only. Never delete or overwrite entries. -->

## YYYY-MM-DD: [approach name]
**What I tried:** One or two sentences.
**Why I stopped:** What failed or made it not worth continuing.
**Refs:** File paths or commit hashes if useful.

## 2026-03-30: combine_as_draws.py — parallel draw combining
**What I tried:** `06_upload/combine_as_draws.py` was an attempt to combine all draws into a single upload file more efficiently. `01_combine_as_draws_parallel.py` submits it as a cluster job.
**Why I stopped:** Script is broken mid-refactor — `list_of_dataarrays` and `draw_var_name` are used but never defined. Never got it working and moved on.
**Revisit when:** Working on "faster/more efficient" phase. The goal was presumably to replace or complement `create_and_combine_as_and_aa_draws.py` with a faster approach. Worth revisiting if draw combining becomes a bottleneck.
**Refs:** `src/idd_forecast_mbp/06_upload/combine_as_draws.py`, `src/idd_forecast_mbp/06_upload/01_combine_as_draws_parallel.py`

## 2026-05-29: lsae_1209 archaeology for 08's regression test
**What I tried:** Started designing a strict regression test for 08
that would compare its netCDF outputs against the 2025_08_11 per-draw
parquet goldens at `04-forecasting_data/malaria/lsae_1209/20250811/`,
pinning each upstream (DAH, gdppc, urban, flooding, climate, hierarchy)
to a 2025_08_11-era dated version.
**Why I stopped:** Bobby pulled me back ("Wait, what is the goal here?
To create the input for forecasting?"). 08 is a brand-new script — there
is no prior version of it to "regress against." The goal is to produce
forecast inputs for the new lsae_1285 + new-model run, not to reproduce
a year-old per-draw output. Pivoted to a smoke + merge-correctness test
against current lsae_1285 upstreams instead. The pinning archaeology
was discarded.
**Refs:** Session 2026-05-29.

## 2026-05-07: refactor bug — row filter dropped when extracting _endemic_location_ids
**What I tried:** Extracted endemic location selection into `_endemic_location_ids()`. The function applies the row-level filter (pfpr > 0, mort_count > 0, inc_count >= 0) internally to identify valid locations, but the caller then rebuilt `aa_sub` from the unfiltered `aa_df`, silently restoring 90K zero-pfpr rows (~23% of total). The logit on those rows produces -inf.
**Why I stopped:** Bug caught when the divide-by-zero warning appeared on re-run. Fixed by applying the same row filter to `aa_sub` in the caller.
**Watch for:** The same bug will exist in `06_build_dengue_past_inputs.py` when we build it — apply the row filter to the dengue equivalent of `aa_sub`, not just to the location ID selection step.
**Refs:** `src/idd_forecast_mbp/02_data_prep/05_build_malaria_past_inputs.py`

## 2026-05-12: scam summary() on deserialized objects
**What I tried:** Used `saveRDS()`/`readRDS()` and `save()`/`load()` to serialize scam model objects, then called `summary()` on the restored object.
**Why I stopped:** Fails with "data must be of a vector type, was NULL" — the model frame environment doesn't survive serialization. This is a fundamental limitation of scam/mgcv, not a bug in our code. Direct field access (`fit$deviance`, `coef(fit)`, `predict()`) still works fine. The rocket computes all needed metrics before saving.
**Refs:** Tested in interactive R session with fit from `20260512_v2` output.

## 2026-05-14: "Drop the 12 failed specs" recommendation
**What I tried:** When 12 specs timed out even after the threading fix, I
recommended dropping them as <1% of the grid that wouldn't change the
top-of-rankings anyway.
**Why I stopped:** Bobby correctly pushed back: this is a single-winner
model-selection exercise, so any spec without an OOS metric is a spec
that can't compete — "the failed ones might be the best." The fix is to
evaluate them, not exclude them. Pivoted to diagnosing the optimizer and
ultimately switching to BFGS (DECISIONS 2026-05-14).
**Refs:** Session 2026-05-14, `delete_once_model_selected.rmd`.

## 2026-05-14: "4+ shape-constrained smooths" as the failure fingerprint
**What I tried:** Hypothesized that specs with 4+ shape-constrained
smooths (`mpd`/`mpi`/`cv`) were dying because the EFS optimizer was
fighting the constraints. The 12 visibly-lost specs all had 4–5 such
smooths.
**Why I stopped:** Cross-tabulation of `neighborhood_specs` showed all 48
specs in the *6*-smooth bucket finished cleanly with EFS. If
constraint-count drove failure, the 6-bucket should fail most, not 0%.
Disproven by data. Real cause was EFS non-convergence on specific
covariate × constraint × data interactions, not constraint count.
**Refs:** Cross-tab in `delete_once_model_selected.rmd`, session 2026-05-14.

## 2026-05-27: Stage-01 owns its own pop (multiple iterations)
**What I tried:** Designed several approaches to fix the stale-pop bug in `pixel_hierarchy.py` / `pixel_urban_hierarchy.py`: (a) `_A02_PIXEL_POP` artifact + atomic write + drop the `if not exists` guard; (b) launcher-flagged single-writer with `--write-population` only on the first covariate task; (c) separate `LSAE_POP_RUN_DATE` constant for independent versioning of pop vs the broader climate-aggregates run date.
**Why I stopped:** Surfaced during conversation that the rapidresponse team already publishes the canonical aggregated `population.parquet` alongside their climate-aggregate outputs at the same path family we already read from. Owning a local copy was duplicate work and a staleness vector. Pivoted to "stage 01 doesn't write pop; 02b reads upstream via `LSAE_POP_PATH`" (DECISIONS 2026-05-27).
**Refs:** Earlier iterations preserved in deleted `.claude/DECISIONS_draft.md` (session-close cleanup).

## 2026-05-28: filter-repo --force on a dirty working tree
**What I tried:** Ran `git filter-repo --force --message-callback '<strip Claude trailers>'` to remove `Co-Authored-By: Claude` from 22 prior commits. The branch was local-only so force-push concerns didn't apply.
**Why I stopped (and won't again):** `--force` does NOT only bypass filter-repo's clean-tree check. It also performs an effective `git reset --hard` of the working tree AND expires the reflog. 67 modified tracked files / 4,745 line-changes wiped from disk with no git-side recovery path (no reflog, no stash). NFS hourly snapshot from 13:00 saved the work; rsync recovery → commit `da683cc`. New global rule landed in `~/.claude/CLAUDE.md § ⛔ ABSOLUTE BLOCK: destructive git ops require a clean working tree` and a callout in `~/.claude/STANDARDS.md § Git conventions`. Full post-mortem at `/mnt/share/homes/bcreiner/2026-05-28-claude-filter-repo-incident.md` for sharing with Anthropic.
**Refs:** Post-mortem at path above; corrections log entry at `~/.claude/projects/-mnt-share-homes-bcreiner-repos-idd-forecast-mbp/corrections/2026-05-28.md`; commit `f855cf1` (pre-incident, chore commit) → `da683cc` (post-recovery).

## 2026-06-01: Bumping CLIMATE_COVARIATE_RUN_DATE to "2026_05_27" to surface the new pop
**What I tried:** Set `CLIMATE_COVARIATE_RUN_DATE = "2026_05_27"` so
`LSAE_POP_PATH` would resolve to the new vintage's
`population.parquet` (the file lived under the same
`climate-aggregates/<DATE>/results/lsae_1285/` family). Verified the
target file existed before bumping.
**Why I stopped:** The 2026_05_27 climate-aggregates vintage uses a
different filename pattern for malaria suitability (14 variant-
prefixed files like `malaria_mordecai_0_0_suitability_ssp245.parquet`
instead of the un-prefixed `malaria_suitability_ssp245.parquet` that
the 2026_01_12 vintage used). Bumping `CLIMATE_COVARIATE_RUN_DATE`
broke `get_malaria_suitability_path` resolution as a side effect.
Reverted immediately. Real fix was to introduce `LSAE_POP_RUN_DATE`
as its own constant so pop and climate-covariates can advance
independently (DECISIONS 2026-06-01).
**Refs:** Reverted edit; DECISIONS 2026-06-01 entry on vintage
decoupling.

## 2026-06-01: float32-cast band-aid in read_shared_covariates
**What I tried:** `rcp_scenario = float(np.float32(rcp_scenario))` at
the top of `read_shared_covariates` to align the float64 comparand
with the float32 column representation in gdppc/ldipc/med_consumppc.
Technically worked — pyarrow's `==` filter matched the float32 rows
once both sides agreed on the same binary representation.
**Why I stopped:** Treats the symptom (precision mismatch) instead of
the underlying design mistake (storing categorical labels as floats).
Pivoted to writing `scenario` as string ("rcp26"/"rcp45"/"rcp85") in
the three economic parquets and updating `mbpc.ssp_scenarios` to
match (DECISIONS 2026-06-01). The cast band-aid was removed.
**Refs:** Two-step git history — band-aid lines first, then removed
once the string-RCP commits landed.

## 2026-06-03: setkey(suit_dt, draw) to cut stage-04 predict time
**What I tried:** Suspected the per-draw `suit_dt[draw == d]` (full scan of the
209.85M-row long suitability table, once per fork under mclapply) was the
predict-time bottleneck; keyed suit_dt on draw once in the parent (`setkey`,
then `suit_dt[.(d)]`) so each draw is a binary-search subset.
**Why I stopped:** Time-neutral — probe predict times unchanged (n=10 ~135s/wave
before and after). The real cost is the `scam predict()` itself (3 models ×
~774k rows/draw) plus memory-bandwidth saturation when 10 forks run at once
(5→10 forks: 50→135s), not the scan. KEPT the setkey anyway because it dropped
peak RSS ~8–10 GB (stopped each fork building its own secondary index on the
210M-row table). Parked real predict-speed levers: fewer cores, or chunked /
per-draw lazy reads instead of one giant long table.
**Refs:** `forecast_malaria_admin_2s_rocket.r` (make_predict_one_draw); probe
jobs 47817* (pre) vs 47830* (post); `04_forecasting/forecast_resource_log.csv`.

## 2026-06-09: base + "rest" double-regression for dengue age-sex incidence
**What I tried:** Scaffolded `fit_dengue_models_explore.r`'s `predict_incidence_as`
as a base model (reference age group) plus a "rest" regression predicting the other
age groups relative to the base (mirroring final_models_dengue.r's
mod_inc_base/mod_inc_rest), to produce age-sex rates.
**Why I stopped:** That's not how dengue is disaggregated. Production
(`disaggregate_age_sex_dengue`) spreads a single base rate with a fixed GBD age-sex
relative risk (`rr_inc_as`) — no rest regression. Rewrote the AS step to
`exp(base_log_rate) * rr_inc_as`. (Bobby: "I don't use the 'rest' model at all".)
**Refs:** `lib/processing/disaggregation.py`, `04_forecasting/as_dengue_shifts.py`,
`make_as_md_gbd_dengue_df.ipynb`.

## 2026-07-01: 2-stage IS-no-FE screen → cull → OOS
**What I tried:** Screen the wide spec grid on the cheap in-sample no-fixed-effects fit
(`is_no_fe_pfpr_r`), cull to survivors, then run the expensive OOS only on those (the
`01a` prelim launcher + `FIT_IS_NOFE` path).
**Why I stopped:** The no-FE fit selects covariates on the variance the country fixed
effects reclaim — the wrong objective. Deleted all no-FE machinery; went FE-present
throughout, with a two-cell design (IS once per spec; within/tempA/tempB OOS as separate
cells). Selection is on temporal OOS.
**Refs:** `select_malaria_models_rocket.r` (no-FE path removed); DECISIONS 2026-07-01.

## 2026-07-01: Fixed bundle size per cell type
**What I tried:** Bundle the fast selection cells with one fixed size per cell type
(e.g. is_cell ~4), sized off the low-n_smooths cells.
**Why I stopped:** Per-cell cost swings ~4× across n_smooths, so a fixed size of 4 at
n_smooths=6 (344 s/cell) = ~23-min mega-tasks — the opposite of the goal. Bundling must be
n_smooths-aware (`CALIB[(template, n_smooths)]`). Bobby caught it via a `time_per_task` table.
**Refs:** `fit_malaria_models_orchestrator.py` CALIB/`bundle_cells`; DECISIONS 2026-07-01.

## 2026-07-01: Ranking the malaria specs by in-sample AIC
**What I tried:** Rank specs by `is_aic` to pick the model.
**Why I stopped:** In-sample AIC favors the complex specs (n_smooths 5–7) by ΔAIC ~7000,
but those forecast WORSE on all three OOS experiments — it rewards the overfit. Selection
moved to temporal OOS + parsimony. AIC is fine as "best in-sample description," not "best
forecaster."
**Refs:** wf 595991 vet; DECISIONS 2026-07-01 (select on temporal OOS).

## 2026-07-01: CALIB ceilings from a 1-spec-per-level probe
**What I tried:** Size the per-(cell, n_smooths) runtime ceilings from a `--n-per-level 1`
probe (one arbitrary spec per n_smooths level).
**Why I stopped:** The single spec under-measured the slowest spec at each level, so several
full-run tasks exceeded the ceiling and TIMED OUT (jobmon auto-retried at ~1.5×). Fix: size
CALIB off the full run's per-level MAX, and probe with ≥2 specs per level going forward.
**Refs:** wf 595991 vet (`rt_frac ≥ 1.0` tasks); memory Malaria.

## 2026-07-07: Lagged admin-0 PfPR covariate as a country-FE *replacement* for malaria forecasting
**What I tried:** Replace the country FE with a smooth of the lagged admin-0 pop-weighted PfPR
(`data.add_a0_lag`, logit), motivated by "an FE can't be carried into the forecast." Built the
compounded rolling-forecast harness (`forecast.py`), swept lag×window×{mpi,smooth} vs an FE
reference, scored by recursion depth.
**Why I stopped:** The premise was false — under the 2023-anchor shift the FE cancels from the
forecast (DECISIONS 2026-07-07), so base+FE shift-anchored is already a fine forecaster. Two
further issues with the run as done: it scored RAW (pre-shift) predictions, not the shifted
trajectory; and it framed the lag as a *replacement* (no FE) rather than an *addition*.
**What survives:** the harness (`forecast.py`, `score_by_depth`, grid driver) is sound and
reusable — only the specs (keep FE, add lag) and scoring (shift first) change. Reusable for dengue.
**Refs:** `lib/modeling/forecast.py`; `tests/lib/modeling/test_forecast.py`; `~/roll_grid_run.py`;
`~/roll_forecast_grid/20260707/`; DECISIONS 2026-07-07.

## 2026-07-08: `conda run -n <env> python - <<HEREDOC` swallows stdout
**What I tried:** Running throwaway Python (aggregation / plotting) via
`conda run -n idd-forecast-mbp python - <<'PY' ... PY` piped to the shell.
**Why I stopped:** `conda run` buffered/swallowed the subprocess stdout+stderr — the command
returned exit 1 (or empty) with NO traceback visible, making errors undebuggable (burned two
iterations before noticing).
**What works instead:** call the env's python binary directly —
`/ihme/homes/bcreiner/miniconda/envs/idd-forecast-mbp/bin/python - <<'PY'` — it streams normally.
Use that for inline Python in this repo.

## 2026-07-08: `glue()` on a data.table column inside `dt[, `:=`(...)]`
**What I tried:** In the 04 launcher, `param_map[, model_run_date := glue("{MODEL_FIT_DATE}_{model_id}")]`.
**Why I stopped:** `glue()` evaluates its template in the calling frame, where `model_id` is a
data.table COLUMN, not a variable → runtime error `object 'model_id' not found` (only surfaces at
`source()`, not lint).
**What works instead:** `paste0(MODEL_FIT_DATE, "_", model_id)` — base string ops evaluate in the
data.table column scope.
**Refs:** `04_forecasting/01_forecast_malaria_admin_2s_launcher.r`.

## 2026-07-10: Flat base-group CFR broadcast across age/sex (dengue mortality)
**What I tried:** In the pygam dengue notebook, predict a single base-group CFR (`mod_cfr_base`
analog) and broadcast it unchanged across all age/sex for mortality (mort = inc × cfr_base).
**Why I stopped:** CFR varies enormously by age, so the base group (age 3/sex 1) is a terrible proxy
for all-age CFR — aggregate super-region CFR came out ~30× high (SE Asia), ~200× low (Sub-Saharan
Africa). The OLD pipeline never did this; it fit `as_id` (`mod_cfr`) and raked to observed age/sex.
Replaced by the age/sex `as_id` CFR (DECISIONS 2026-07-10).
**Also a dead end:** predicting/fitting the age/sex CFR at admin-2 (lsae) — ~29M-row frames /
design matrix OOM the kernel (a likely cause of the session-end crash). Fit at fhs + additive
offset instead.
**Refs:** `reports/03_modeling/pygam_dengue_models_explore.ipynb`; `04_forecasting/OLD_rake_dengue.py`.

## 2026-07-10: `arrow::set_io_thread_count(2L)` as the teardown-segfault fix
**What I tried:** Set arrow io threads 1->2 on arrow's warning that io<2 "may hang or crash", as the fix for the post-write `memory not mapped` segfault.
**Why I stopped:** VERIFIED it does not fix it — the calibration probe (io=2 worker, line 57) still segfaulted at `fin`. The crash is inherent arrow/jsonlite DLL teardown, independent of io thread count. io=2 kept as arrow's floor (harmless); real fix is output-gating (task success = output exists), not thread tuning.
**Refs:** `select_malaria_models_rocket.r:56-57`; `final_run_setup/idd_tools_findings.md`.

## 2026-07-10: Calibration probe as the sizing input for the final malaria run
**What I tried:** An 85-task idd-tools calibration probe (per-tier bundle sizes) to measure per-tier (a,b) and size the full run.
**Why I stopped:** It ran at ~85-way concurrency vs the real run's 5000 -> NFS load-contention unrepresentative (isolated load 2.7s vs 42.6s under the probe's contention). The census (599278) already has real-scale contended cost, strictly better for sizing. The probe's real value was the idd-tools machinery TEST (10 gaps -> inbox). Feeding probe+census to fit_from_probe_history mixes two contention regimes (noisy `a`).
**Refs:** `final_run_setup/probe_calibration.py`, `fit_from_probe.py`, `idd_tools_findings.md`.
