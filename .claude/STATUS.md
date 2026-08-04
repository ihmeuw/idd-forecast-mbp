# Project status
Updated: 2026-07-20

## Goals
Infectious disease forecasting pipeline for malaria and dengue, projecting
outcomes to 2100 under SSP climate scenarios + DAH funding scenarios.
Current phase: model-fitting on lsae_1285 inputs, then forecast-output
infrastructure design.

Long-term axes:
1. Migrate all pipeline scripts to `lib/` imports and `mbpc` alias (refactor —
   stages 01, 02, 04, 05, 06 done; legacy modules still in tree).
2. Node-level output versioning per STANDARDS.md (done; gaps fixed 2026-05-06).
3. Audit and update hardcoded input paths to latest data versions
   (PATH_CHECKLIST.md drives this).
4. Run reference scenario end-to-end with new (lsae_1285) data.
5. Feature additions: malaria suitability curve dimension, vaccination
   scenario, variable importance methods beyond "hold at 2022".

## Orientation
Pipeline infrastructure (stages 01–08, versioning, model registry) is built and
largely run. Several active fronts on lsae_1285:

**Malaria hybrid-SDG goalkeepers incidence deliverable (2026-07-20, current front).** The
Goalkeepers/SDG age-sex incidence-rate-per-1000-by-country deliverable
(`05_aggregation/hybrid_sdg_malaria_incidence_rate.py`; count-space, raked to external GBD2025 MAL
@2025). Formalized + run end-to-end twice this session.
- Fit registry now records `inc_count_threshold`/`pfpr_threshold` (backfilled all 7 prior entries);
  the "hybrid" model = the 2026_06_03 formulation refit with NO thresholds.
- The hybrid reads a SPECIFIC forecast dir via `--forecast_run_date <key>` (never touches `current`);
  sensitivity gained an `old_delivered` overlay + `--rake_years`.
- Stage-04 forecast is now the jobmon orchestrator (`01_forecast_malaria_admin_2s_orchestrator.py`,
  Python → a CC runs it, submits R rocket tasks); `--model-run-date` = registry key = output dir.
- DAH correction: the first delivery used erroneous DAH (future-only error in the Goalkeepers-2026
  drop); reloaded from FGH_2026_July → `2026_07_20_hybrid_fghjul` → wf 603553 → `20260720`. KEY
  PRINCIPLE: a DAH correction confined to FUTURE years leaves the FITTED model unchanged (fit is on
  past) — verified identical; only the forecast moves.
- Products → `hybrid_deliverable/lsae_1285/<RUNDATE>/`; the client deliverable at
  `/ihme/forecasting/data/37/future/incidence/<DATE>_malaria_incidence_goalkeepers/` is a Bobby-run
  cp (a CC can't write /ihme outside /ihme/homes).

**Malaria PfPR model-selection (current focus).** idd-tools **jobmon MANIFEST** workflow
(replaced the old param_map). Three files, one seam:
- `build_malaria_neighborhood_specs.r` — **single source of truth for formulas**. FE-only
  neighborhood (lags dropped); writes `neighborhood_specs.rds` + `spec_table.parquet`
  (spec_index, n_smooths, **n_scams**, formula_text) → current dir **20260710_efs** (1620 specs).
- `fit_malaria_models_orchestrator.py` — idd-tools cells/partition/**manifest**. Fans each spec
  into **1 IS cell + 10 temporal OOS windows** (max_lag=0), partitions per (cell, n_smooths,
  n_scams) into serial tasks; the worker reads its cell list from the saved `manifest.json` by
  `--task-id`. Resources still **FLAT** (`MAX_PER_TASK` 4/3, flat `MEM`, `max(5,)` floor) — NOT
  yet wired with the per-tier idd-tools recommendation.
- `select_malaria_models_rocket.r` — **three-engine** worker keyed on n_scams: `n_scams>0 → scam`,
  `n_scams==0 & n_smooths>0 → gam(REML)`, else `lm`. `arrow::set_io_thread_count(2L)`.

**599278 = the CENSUS** — the completed prior full run (5,837 valid tasks @ conc 5000), the
authoritative per-task cost source (real-scale contention; we OVER-allocate: runtime 2-13× actual,
mem ~2× = the intended floor). The post-`fin` teardown segfault (`memory not mapped`) is a **benign**
arrow/jsonlite DLL crash — output is written first; **io=2 does NOT fix it** (verified); ~occasional
jobmon retries; real fix = output-gating (deferred). This session **dogfooded** the idd-tools
calibrate/run_registry path → **10 gaps filed to idd-tools/inbox**. Judge on temporal OOS (10
windows) + parsimony, NOT in-sample AIC.

**Dengue modeling (separate workstream) — now PROTOTYPING the forecast pipeline.**
The active dengue work moved from planning to building a Python/pyGAM forecast prototype in
`reports/03_modeling/pygam_dengue_models_explore.ipynb` (fit → anchor-shift → age/sex →
count-space aggregate → validate → forecast to 2100), which DRIVES the R-pipeline plan in
`.claude/FORMALIZE_DENGUE_FORECAST_PIPELINE_PROMPT.md` (fit+registry+forecaster = R reading the
Python-built 08b nc; finishing = Python). Built this session: dengue forecast-inputs (`08b`;
malaria `08`→`08a`; ran `07c` → 29,109 locs), age/sex CFR via the fitted `as_id` (fit at fhs,
additive offset, rake to observed age/sex, no invented deaths — verified super-region CFR tracks
observed), and a grain-generalized forecast (each formulation at its own grain; fhs via a verified
pop-weighted 08b roll-up). See memory.md Dengue + DECISIONS/DEAD_ENDS 2026-07-10.

**Malaria PfPR forecasting — the FE is moot under the 2023 shift (2026-07-07).** A Python
pyGAM sandbox (`reports/03_modeling/pygam_model_selection.ipynb` + `lib/modeling/`
{data,specs,fit,metrics,shift,forecast}) recreates the scam PfPR path and adds a compounded
rolling-forecast harness (`forecast.py`). Key finding: under the per-location 2023-anchor
shift (`shift.apply_shift`), the country fixed effect (and intercept) CANCEL from the forecast
trajectory — shifted(y) = obs_2023 + [smooths(y) − smooths(2023)] — so the FE is inert in
forecasts and base+FE shift-anchored IS the forecaster. "Replace the FE with a lagged admin-0
PfPR covariate" was therefore a non-problem and is dropped for malaria. The correctly-framed
open question (does a lag *add* skill on top of base+FE, scored on *shifted* preds) is DEFERRED
to dengue; the harness is built and reusable. See memory Dengue for the Python how-to.

**Malaria final-model comparison + forecast (2026-07-08).** GDP switched to the V5 income
forecast's `reference` scenario applied to all RCPs (decoupled from climate); GDP rebuilt +
stage-02 past-inputs (→20260707) and stage-08 forecast-inputs (→20260708) re-run.
`02_fit_final_malaria_models.r` is now a FORMULATIONS-list multi-fit (each →
`{run_date}_{id}_malaria_models.RData`, registered under key `{run_date}_{id}` — the key is any
non-empty string, so many coexist). Six candidate PfPR specs f1–f6 (suitability form: linear
`logit_malaria_suitability` vs `s(malaria_suit, mpi)`; ± `mean_low_temperature` linear or s())
fit on the subsetted (inc_count≥1 & pfpr≥1e-4) data, then forecast to 2100 × 3 SSPs via the
launcher (now MODEL_IDS-parameterised, per-formulation output dirs, no auto-finalize; rocket +
stage-08 gained `malaria_suit` + single-realization `mean_low_temperature`). Now comparing the
formulations and deciding a single winner vs an ensemble (matched per-draw weighted blend).

## Recent steps
- 2026-07-20: **Malaria hybrid-SDG deliverable re-run on CORRECTED DAH.** The first delivery used
  erroneous DAH (Goalkeepers-2026 20260713 drop; error was FUTURE-only — 2000–2024 identical, 2025+
  ~5–7% too high). Reloaded `make_dah_df` from **FGH_2026_July** (pa file, sum of 11 `mal_*` PAs ≡
  `hfa='mal'`); re-ran 05 (past inputs, unchanged) + 08a (forecast inputs, corrected future DAH;
  sbatch, 106 GB MaxRSS). Re-fit → **`2026_07_20_hybrid_fghjul`** (id bumped from "hybrid") — VERIFIED
  numerically identical to `2026_07_14_hybrid` (same iters/convergence/formulas), because a future-only
  DAH error can't move a past-fit model. Forecast ssp245 via the jobmon orchestrator (wf 603553,
  100% finite); re-aggregated to `hybrid_deliverable/lsae_1285/20260720/` (deliverable + plotdata +
  sensitivity; `--rake_years 2025`, `old_delivered`=20260527, pin23/24 dropped) + 7 plots at
  `07-figures/20260720/`. Corrected delivered sits a few % BELOW the original 20260527 (rake-to-2025
  anchoring inverts the raw admin-2 direction). Delivery cp to /ihme is Bobby's. See DECISIONS 2026-07-20.
- 2026-07-14: **Malaria hybrid-SDG deliverable formalized + first run.** Added inc_count/pfpr threshold
  tracking to the fit registry (fit-script run-level knobs + backfilled all 7 entries: 2026_06_03=0/0,
  f1-f6=1/1e-4). Registered `2026_07_14_hybrid` (2026_06_03 formulation, no thresholds; formulas
  verified vs the registry). Parameterized the hybrid (`--forecast_run_date` reads a specific forecast
  dir, never `current`) + added the `old_delivered` sensitivity overlay. Forecast via the new jobmon
  orchestrator (`01_..._orchestrator.py`; probe wf 601175 → full run by Bobby); deliverable + plotdata +
  sensitivity + 7 plots at `20260714`. (Superseded by the 20260720 corrected re-run.)
- 2026-07-10: **Malaria selection — idd-tools dogfood + fresh spec build; full run NOT launched.**
  (1) Established 599278 as the CENSUS (5837 valid tasks @ conc 5000); OVER-allocation confirmed
  (runtime asked 2-13× actual; mem ~2× = intended floor). (2) Segfault: benign post-`fin`
  arrow/jsonlite teardown crash (output written first); applied `set_io_thread_count(2L)` but
  VERIFIED it does NOT fix it (probe still segfaulted at io=2) — arrow io<2 was a red herring; real
  fix = output-gating (deferred). (3) idd-tools: added `ResourcePredictor(target_transform="log")`
  default (+112 tests pass, ruff/mypy clean); ran an 85-task calibration probe → seeded run_registry
  (census+probe) → `fit_from_probe_history` per tier. Surfaced **10 idd-tools gaps** →
  `idd-tools/inbox/2026-07-10_..._resource-calibration-gaps.md` (+ `final_run_setup/idd_tools_findings.md`).
  (4) Built fresh spec dir **20260710_efs** (1620 specs). (5) Full-run command handed to Bobby via
  `idd-jobmon-launch` but NOT run (0 log/tasks/outputs); orchestrator still FLAT. See DECISIONS/DEAD_ENDS 2026-07-10.
- 2026-07-10: **Dengue forecast pipeline — planning prompt + built the Python/pyGAM notebook
  prototype.** (1) Wrote/iterated `.claude/FORMALIZE_DENGUE_FORECAST_PIPELINE_PROMPT.md` (registry →
  R forecaster → finishing; configurable finest grain; shared-lib mandate; dated 2026-07-10 Update
  block folding in this session's build). (2) Built + ran `08b_build_dengue_forecast_inputs.py`
  (near-copy of malaria's `08`, which was RENAMED `08`→`08a` incl. test + 5 refs); registered
  `dengue_suitability` (climate_draw) + `DEFAULT_DENGUE_FORECAST_COVARIATES` in covariate_registry;
  added `DEN_FORECAST_INPUTS_{READ,WRITE}_PATH`; ran `07c` → 29,109 prediction locs; 3 forecast-input
  ncs (admin-2, dims loc×year×draw). (3) Rebuilt `pygam_dengue_models_explore.ipynb`: added age/sex
  **CFR/mortality** — fitted `as_id`, fit at fhs grain, additive base+offset, rake to observed
  age/sex, cfr=0 = no invented deaths (VERIFIED super-region CFR tracks observed within ~10–20%,
  fixing a 30×/200× flat-CFR error); added a **forecast-to-2100** section — each formulation at its
  own grain (fhs via verified pop-weighted 08b roll-up ~1e-16), cached nc read, `w_inc`/`mort_weight`
  cheap wins (both documented alongside the faithful route). (4) Diagnosed High-income CFR/inc
  jaggedness = the `06b` presence-filter (in-sample only; forecast on the consistent 08b set is
  clean); fixed the `D_fhs` fhs-grain forecast crash. Kernel crashed at the end — likely OOM from the
  lsae age/sex-CFR 29M-row frames. See DECISIONS/DEAD_ENDS 2026-07-10 + memory.md Dengue.
- 2026-07-08: **Malaria: V5 GDP + 6-formulation fit → forecast → compare.** (1) GDP → V5
  `reference` scenario replicated across all RCP labels (make_gdppc_df.py); rebuilt GDP + re-ran
  05 past-inputs (current→20260707) + 08 forecast-inputs (current→20260708) on V5 GDP. (2)
  `02_fit_final_malaria_models.r` refactored to a FORMULATIONS list (per-id `.RData` + registry
  key `{date}_{id}`, best=FALSE); 6 PfPR specs f1–f6 fit on subsetted (inc_count≥1 & pfpr≥1e-4;
  inc_count reconstructed = inc_rate × population; A0_af built AFTER the subset) — all converged
  (keys 2026_07_08_f1..f6). Bobby runs the R (won't execute in a Python CC). (3) Launcher
  MODEL_IDS-parameterised (per-formulation dirs, no auto-finalize); rocket + 08 read `malaria_suit`
  (raw suitability, matches fit) + `mean_low_temperature` (new covariate_registry `climate_mean`
  kind = ensemble mean over draws, loc×year). Forecast 6×3 SSP (array 8951, 18/18 COMPLETED, peak
  56.7/60 GB, kept=9918). (4) Comparison plots in
  `07-figures/20260708/malaria_formulation_comparison/` (super-region/global obs-vs-6+ensemble,
  rate+count; `by_formulation_rcp/` = 28 per-formulation+ensemble panels, 3 RCPs + 95% draw UI).
  Aggregate-level RATES use the FULL level population from the population df, never summed-endemic.
  See DECISIONS/DEAD_ENDS 2026-07-08.
- 2026-07-07: **pyGAM PfPR rolling-forecast harness + the FE-cancels-in-shift finding
  (malaria forecasting resolved).** Built `lib/modeling/forecast.py` (compounded rolling OOS:
  `LagCovariate`, `rolling_forecast`, `compare_modes`) on `fit.fit_gam`/`predict_gam`, +
  `metrics.score_by_depth`; tests incl. depth-0≡observed gate + hand-computed depth-1 feedback
  (`tests/lib/modeling/test_forecast.py`). Generalized `data.add_a0_lag` (added `var`,
  log/logit transform, NaN-safe pop-weighted avg) + tests. Ran a full compounded grid (8
  windows × {mpi,smooth} × lags[1,2,3,5] + per-window FE; driver `~/roll_grid_run.py`, outputs
  `~/roll_forecast_grid/20260707/{scores,preds}.parquet`). KEY: the 2023-anchor shift cancels
  the FE from forecasts → base+FE shift-anchored is the forecaster, lag-as-FE-replacement is
  moot for malaria. (Also: under rolling, the 8 gap×test windows collapse to 5 configs by
  train_end.) See DECISIONS/DEAD_ENDS 2026-07-07.
- 2026-07-01: **Malaria model-selection migrated to jobmon + run end-to-end.**
  (1) Made prep the single formula authority — deleted the worker's
  `build_term`/`build_formula`/`K_DEFAULT`; it now fits
  `spec_table.parquet$formula_text` via `as.formula`. (2) Per-(var,form) K:
  `var_forms` re-encoded as named vectors (form→K, NA linear) + added an
  unconstrained `smooth` form. (3) n_smooths-aware bundling:
  `CALIB[(template,n_smooths)]→(bundle,runtime)` + per-template `MEM`; cells bundle
  by (cell, n_smooths); param_map one row per (task_id, spec_index); bundle-aware
  `spec_done`; worker summaries re-keyed by spec_index. Two adversarial contract
  reviews (both clean). (4) Ran full bundled run (wf **595991**, 506 tasks / 1080
  cells). Vet: mem fine (max_rss ≤4.4 GiB, frac ≤0.79), 0 errored fits, but several
  TIMEOUTs — CALIB (from a 1-arbitrary-spec probe) too tight; jobmon auto-retried
  (scale_up_on_retry, 3 attempts). (5) Results: temporal-OOS neighborhood FLAT (top
  specs within <0.001 `oos_pfpr_r`); simple specs (n_smooths 2–3) win/tie on all
  OOS; in-sample AIC favors complex (ΔAIC ~7000) but forecasts worse.
- 2026-06-08/09: **Dengue past-inputs pipeline built.** New dengue location
  helpers in lib/processing/locations.py (dengue_fit_location_ids,
  dengue_prediction_location_ids); constants gained
  dengue_{fit,pred}_{mort,inc}_threshold (=0.0) + _A03_DEN_FIT_LOCATIONS /
  _A04_DEN_FORECAST_LOCATIONS. New 06a_build_dengue_fit_locations.py;
  06b_build_dengue_past_inputs.py (git mv from 06, nc→parquet: dense AS within
  included loc-years, draw-000, store rate+population (drop counts),
  A0/region/super_region); 07c_build_dengue_prediction_locations.py (mirrors
  07b). Ran 06a (29,177 fit locs) → 06b (29.6M rows / 652 MB, current→20260527).
  07c not yet run.
- 2026-06-09: **Dengue AS-disaggregation pinned + explore harness.** AS inc =
  pop × exp(base_log_rate) × rr_inc_as (disaggregation.py / as_dengue_shifts.py);
  rr_inc_as = rate ÷ reference age3/sex1 (cause_map['dengue']) at year 2022
  (make_as_md_gbd_dengue_df.ipynb); NOT a base+rest regression. Built out
  03_modeling/fit_dengue_models_explore.r (base inc lm → rr AS rates → aggregate
  to gbd/super-region → obs-vs-pred plots). httpgd/radian plotting confirmed in VS Code.
- 2026-06-03: **Stage-04 malaria forecaster rewritten + run end-to-end.** Built
  forecast_malaria_admin_2s_rocket.r (sourceable arg-based helpers + --file
  main-guard; order-safe apply_shift; mclapply over draws; reads 08 netCDF +
  raked-AA 2023 anchors; predicts+rakes pfpr→inc/mort; writes one float32 netCDF
  per (ssp,dah) [dims location_id,year_id,draw] + location-status sidecar), plus
  01_..._launcher.r and finalize_malaria_forecast.r — canonical names, _draft/_OLD
  kept. All 8 test points pass (offline 1–7 + staged cluster probe). 3 bugs caught
  by real runs: apply_shift named-vector → unname(); stale image 4222 lacks tidync
  → pinned 4523; ..dah_scenario scoping → renamed param. setkey(suit_dt,draw) =
  −10GB peak, time-neutral. Model resolved via registry (run_date 2026_06_02).
- 2026-06-03: **Real forecast run COMPLETED.** Array 47891353 (-a 1-3): 3/3
  COMPLETED, finalize on **afterok** (open question resolved — afterok works
  here), current→20260602. Outputs: 3× malaria_forecast_{ssp}_Baseline.nc +
  _location_status.parquet. Actuals ~29–31 min, peak 47.5 GB (per-scenario
  43.4/47.5/41.9), kept=9918. Resourcing tuned from data: -c10/--mem=60G/-t45
  (time ≤2× rule). New record_resources.sh + forecast_resource_log.csv.
- 2026-06-01: Full 02-stage rebuild on the new pop vintage. Renamed
  year-range constants (MODELING_YEARS / FORECAST_YEARS / FUTURE_YEARS /
  ALL_YEARS / EXTENDED_YEARS), bound 03/04 outputs to MODELING_YEARS.
  Split upstream run dates into LSAE_POP_RUN_DATE /
  MALARIA_SUITABILITY_RUN_DATE / FLOODING_RUN_DATE
  (CLIMATE_COVARIATE_RUN_DATE stays 2026_01_12). Fixed the float32-vs-
  float64 RCP-scenario filter bug by switching mbpc.ssp_scenarios + the
  three economic parquets to string RCP labels
  ("rcp26"/"rcp45"/"rcp85"); removed the temporary cast band-aid.
  Refactored read_shared_covariates with optional `variables=`; 08 and
  07b pass through what they need. Created
  00_prep_economic_variables.py orchestrator + extracted main() from
  the four make_*_df.py scripts. Renamed 07 → 07a (+ test) for
  symmetry with 07b. Successfully ran 02a → 02b → 03 → 04 → 07b → 08
  (malaria only, urban check disabled in 07b for the incidence-only
  test pass); three SSP forecast-input netCDFs (~0.6 GB each) at
  _A04_MAL_FORECAST_INPUTS/20260527/.
- 2026-06-01 (parallel CC session): Built a single-source-of-truth
  malaria model registry at
  `03-modeling_data/malaria_model_registry.json` (flat singleton, not
  versioned). One JSON array of run records, exactly one carries
  `best: true`. R writes via new `lib/model_registry.R`
  (`upsert_malaria_model_run`, atomic write, single-best invariant);
  Python reads via new `constants.MALARIA_MODEL_REGISTRY` +
  `read_malaria_model_registry()` +
  `get_malaria_model_run_date(best=True, run_date=None)`. Format JSON
  (jsonlite in cluster R image + json stdlib — zero new deps).
  `03_modeling/02_fit_final_malaria_models.r` now requires
  `RUN_DESCRIPTION` (stop()s if blank) + `FLAG_AS_BEST` toggle; upserts
  the run with full provenance (rdata_file, models, pfpr_formula,
  parquet_path, suit_variant). 8 pytest tests pass; R end-to-end smoke
  passed. Registry currently has one entry (2026_06_01, best=False) —
  Bobby is running the model now and will re-run with FLAG_AS_BEST=TRUE
  after verification. Open follow-ups (other CC owns): record
  resolved/dated `parquet_path` instead of `current/`-symlinked path;
  wire `forecast_malaria_admin_2s_rocket.r` to read the registry (it
  hardcodes model_date today — deferred since the forecast script is
  being rewritten). Dengue will get a sibling
  `dengue_model_registry.json` + helpers when its fit script lands.
- 2026-05-06: Built xarray/netCDF past-inputs infrastructure
  (`array_builders.py`, script 05 netCDF version). Retroactive versioning
  gap fix (`finalize_artifact`, `assert_artifact_ready`).
- 2026-05-07: Pivoted to flat parquet for past inputs (292,109 rows × 29 cols).
  Fixed row-filter refactor bug. Built 73,710-spec model-selection scaffold
  (210-spec neighborhood). Cache layer + BLAS-tamed mclapply. Created
  `lib/netcdf_helpers.R`.
- 2026-05-11: Rewrote `fit_malaria_models_rocket.r` with 4 CV strategies
  (none/random/country/country_no_fe). Launcher passes CV_STRATEGY via
  `--export`. Runtime auto-adjusts (30 min none, 180 min otherwise).
- 2026-05-12: Fixed PfPR metrics: renamed `_pfpr_r_sq` → `_pfpr_r` throughout
  rocket (metric is Pearson correlation, not R²).
- 2026-05-12: Launched 210-spec SLURM array (job 35135444, `20260512_v2`
  output dir). Later cancelled.
- 2026-05-12: Confirmed scam `summary()` fails on deserialized objects
  (fundamental limitation — model frame environment lost). Direct field
  access (`$deviance`, `coef()`, `predict()`) still works.
- 2026-05-12: Created `reports/03_modeling/explore_malaria_fit_metrics.qmd`
  for interactive model comparison.
- 2026-05-12: Promoted "prefer attached/current file content over earlier
  reads" rule to `~/.claude/CLAUDE.md`.
- 2026-05-13: Built model-selection ranking infrastructure across three
  notebooks (`reports/03_modeling/rank_malaria_models{,_loop,_w_urban}.ipynb`).
  Integrated 4 MCDM methods (Borda, TOPSIS, Pareto frontier, pairwise
  dominance) with a `build_winners_summary_row` helper for cross-config
  aggregation. Loop notebook iterates over all
  (oos × pfpr × mae × urban × gates) combinations; gates restructured as
  `{metric: {cutoff, direction}}`.
- 2026-05-13: Sensitivity-analysis viz parameterized by `CONFIG` (METHOD,
  FOCUS_TASK, SUBSET_URBAN): one-way stacked bars, two-way heatmaps,
  three-way interaction tables, wins-by-task bar chart with urban T/F
  stacking and "← flood" annotation for `people_flood_days` models.
- 2026-05-13: Characterised the 12-metric ranking set as ~6 correlated
  pairs (τ 0.5–0.9 within). General-fit and PfPR-fit halves anti-correlate
  IS-side (τ ≈ −0.4 to −0.7), soften to mild positive OOS. Bloc structure
  makes TOPSIS least reliable for this set; 3-of-4 method convergence is
  the stronger signal.
- 2026-05-14: Diagnosed silent EFS non-convergence in scam fits (133 of
  1388 fits hit `maxit = 300`; convergence flag also unreliable on the
  remaining iter-1 fits because EFS stores `$conv` as a list, not a
  logical). Verified BFGS converges on the failing specs (task 413: 4
  outer iters in 715 s vs EFS 300 iters in 22 s without convergence).
  Wrote `fit_malaria_models_rocket_bfgs.r` with bfgs + per-fit
  iter/converged + per-fold OOS metrics + optimizer/maxit/scam_version
  metadata. Updated launcher to `-c 8 --mem=32G -t 240` with
  `OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8` (without those, OpenBLAS in
  the singularity image silently runs single-threaded). Launched 1400-spec
  batch to `20260514_v2/`.
- 2026-05-27: Stage-01 layout investigation; settled on "stage 01 stops
  writing pop, reads upstream `LSAE_POP_PATH`" (no separate POP artifact,
  no launcher-flag single-writer). Earlier `.claude/DECISIONS_draft.md`
  iterations superseded by the simpler "use upstream pop" decision once
  rapidresponse's canonical aggregated `population.parquet` was confirmed.
- 2026-05-27: Diagnosed the 20 NaN locs in urban/suitability outputs as
  zero-pop in the upstream gridded pop (verified loc 93390: zero pop
  1950-2014, real monotone growth 2015-2100). Not a stage-01 bug.
  Rewrote `02_data_prep/07b_build_malaria_prediction_locations.py` with
  pop_zero-first drop policy + explicit `drop_reason` audit column.
- 2026-05-28 morning: MAP 202508 release adopted. Built
  `00_pull_raw_data/pull_map_rasters.py` (WCS downloader using
  `https://data.malariaatlas.org/geoserver/Malaria/ows`); downloaded 7
  Pf/Pv covariates × ~25 years = 175 GeoTIFFs into
  `02-processed-data/{subdir}/202508/`. `COVARIATE_DICT.yaml` bumped:
  year_end 2022 → 2024 + paths → 202508/ for 5 Pf covariates.
- 2026-05-28 morning: Block-skip filter for stage-01. Built
  `01_map_to_admin_2/block_utils.py` with extracted `load_raking_shapes`
  + new `blocks_with_shapefile_intersections(hierarchy)`. Filter applied
  to `pixel_hierarchy.py`, `pixel_urban_hierarchy.py`,
  `02_pixel_main_parallel.py`, `04_pixel_urban_main_parallel.py`.
  Verified 528/784 blocks for lsae_1285 (~33% task reduction). All 4
  workers passed `--help` smoke test.
- 2026-05-28 afternoon: `.claude/` tracking → strict hybrid (only
  STATUS.md / DECISIONS.md / DEAD_ENDS.md tracked; rest gitignored).
  Chore commit `f855cf1`.
- **2026-05-28 afternoon: `git filter-repo --force` on dirty working
  tree wiped 67 modified tracked files / 4,745 line-changes.** Recovered
  via IHME NFS hourly snapshot (rsync; `.git/` and `.gitignore`
  excluded). Recovery commit `da683cc`. Hard rule landed in
  `~/.claude/CLAUDE.md § ⛔ ABSOLUTE BLOCK: destructive git ops require
  a clean working tree` + `~/.claude/STANDARDS.md` callout via
  `/formalize-rule --immediate`. Post-mortem at
  `/mnt/share/homes/bcreiner/2026-05-28-claude-filter-repo-incident.md`.
  22 `Co-Authored-By: Claude` trailers stripped from history (rewrite
  worked).
- 2026-05-28 evening: `feature/refactor-shared-lib` pushed to
  `origin/feature/refactor-shared-lib` via SSH for the first time.
  Branch had ~2 months of work local-only; local main also stale vs
  remote main (last local main commit 2026-02-25).
- 2026-05-29: Built malaria forecast-input chain. New scripts:
  `02_data_prep/07b_build_malaria_prediction_locations.py` (pop-zero-first
  drop policy + `drop_reason` audit, check window 2023+) and
  `02_data_prep/08a_build_malaria_forecast_inputs.py` (per-SSP netCDF
  with dims `location/year/draw/dah_scenario`, 6-covariate default set).
  New modules: `lib/io/covariate_registry.py` (17-entry registry with
  paths + descriptions), `lib/processing/locations.py` (fit + prediction
  location helpers; `05_build_malaria_past_inputs.py` migrated to
  import), `lib/processing/dah_scenarios.py` (array-shaped
  Baseline+Constant via `build_dah_array`). Constants additions:
  `_A04_MAL_FORECAST_LOCATIONS`, `_A04_MAL_FORECAST_INPUTS` + WRITE/READ
  paths. 15-test regression module at
  `tests/02_data_prep/test_08a_build_malaria_forecast_inputs.py`. First
  end-to-end run surfaced 689 NaN-coverage locations now correctly
  dropped by 07b.
- 2026-05-29: Added `NearestResampler` (cached source→destination pixel
  map) to `01_map_to_admin_2/block_utils.py`. `pixel_main.py` inner loop
  switched from `to_raster(...).resample_to(...)` to lazy-build +
  `apply()`. 6 unit tests at
  `tests/01_map_to_admin_2/test_block_utils.py`. Expected ~25–30%
  per-task wall-time reduction; bit-for-bit equivalence vs the legacy
  chain not yet verified on real data (Azure outage hides jobmon).
- 2026-05-29: Stage-01 output-layout fix. New constants helpers
  `pixel_artifact_root` / `pixel_write_path` / `pixel_read_path` produce
  `02-processed_data/GBD2023/<hierarchy>/<RUN_DATE>/...`. Both per-block
  scratch (pixel_main) and per-hierarchy aggregates (pixel_hierarchy)
  now use this versioned root. `02_pixel_main_parallel.py` and
  `03_pixel_hierarchy_parallel.py` call `finalize_artifact` after
  successful `workflow.run`. Group A downstream readers updated:
  `constants.py:malaria_variables`/`dengue_variables` dicts redirected;
  `02_data_prep/03_rake_aa_A2_to_GBD.py` default `lsae_input_path`
  points at `pixel_read_path(LSAE_HIERARCHY)`. Urban side untouched
  (separate concern — urban is not GBD-release-tagged).
- 2026-05-29: Reviewed `03_modeling/02_fit_final_malaria_models.r`
  draft. Caught: `REPO_DIR` undefined, save line wired to 5 model
  objects when the new design uses 3, `past_inputs_nc/` naming
  carryover from when past inputs were NC instead of parquet.

## Next steps
**Active — malaria hybrid-SDG goalkeepers deliverable (2026-07-20):**
- **DELIVER:** Bobby runs `mkdir -p /ihme/forecasting/data/37/future/incidence/20260720_malaria_incidence_goalkeepers`
  then `cp hybrid_deliverable/lsae_1285/20260720/malaria.parquet` into it (a CC can't write /ihme
  non-homes). OPEN: the sibling in that hierarchy is `malaria.nc` — confirm whether the forecasting
  team wants netCDF (add an nc writer to the deliverable) vs the current parquet.
- **Deferred stage-05 finalize module** (`FORMALIZE_MALARIA_FORECAST_FINISHING_PROMPT.md`): pure cores
  + 13 tests exist (`lib/processing/finalize_forecast.py`); the CLI/driver/`_A04_MAL_FINALIZED`
  versioning/real-data probe are NOT built. Prompt is STALE (run-set → hybrid model not the 18 f1-f6;
  launcher → jobmon orchestrator w/ opt-in finalize).

**Active — malaria formulation comparison / ensemble (2026-07-08):**
- Compare the 6 forecast formulations (keys 2026_07_08_f{1..6}) + the ensemble at
  super-region/global; decide single formulation vs ensemble. Ensemble = matched PER-DRAW
  weighted blend (Σ wₘ·Xₘ per loc/year/draw), NOT draw pooling (pooling disagreeing models →
  bimodal/absurd or spread-reducing; the blend keeps the climate draw spread).
- On a chosen winner: set FLAG_BEST_ID in the fit script (or edit the registry) to mark it best,
  and finalize its forecast-output `current` by hand (the launcher no longer auto-finalizes for
  the multi-formulation exploratory run).
- Bump the launcher `--mem` from 60 GB to ~70 GB for headroom (56.7 GB peak observed); Bobby said
  he'll fix the launcher elsewhere.
- The 4 quick meeting plots (`malaria_{inc,mort}_{rate,cnt}_ssp245_by_superregion.png`) still use
  the old summed-endemic denominator (+ a 2023 splice jump); regenerate with the full-population
  denominator if kept.
- To restore draw-varying `mean_low_temperature` (currently single-realization): delete the
  `climate_mean` override in `lib/io/covariate_registry.py`, re-run 08, and bump the rocket mem.

**Closed 2026-06-01** (see Recent steps): stage-01 pixel pipeline launch,
Python data-prep 02a → 02b → 03 → 04 → 07b → 08 rebuild, reproject-cache
landed/exercised, 07b → 08 forecast-input chain produced 3 netCDFs.

**Active — malaria model-selection (2026-07-10; supersedes the 2026-07-01 param_map/CALIB block):**
- **LAUNCH THE FULL RUN.** `idd-jobmon-launch --out-dir <20260710_efs> python
  fit_malaria_models_orchestrator.py --spec-table <..>/spec_table.parquet --output-dir <20260710_efs>
  --worker select_malaria_models_rocket.r --r-image ihme_rstudio_4524.img --r-shell execRscript.sh
  --full --optimizer efs --maxit 30 --cores 16 --max-concurrent 5000` (full command in
  `final_run_setup/`; `conda activate idd-forecast-mbp` first). Orchestrator is still FLAT →
  this = **599278 redux** (valid data, over-allocated, ~segfault churn). OPTIONAL first: wire the
  idd-tools per-tier bundles/resources + output-gating into `resources()`/partition for a right-sized run.
- **Once the run is done: run model-selection FROM THE TOP** — `finalize_selection_run.py --run-dir
  <20260710_efs>` → `selection_summary.parquet` → the ranking notebook
  (`reports/03_modeling/malaria_pfpr_model_selection_report.ipynb` / `aggregate_selection_run.ipynb`).
  `finalize` MUST key off `manifest.json` (or filter the current `_s` naming) — the 599278 dir had 138
  pre-migration files that inflated the join to 58,936 rows (should be 1,620).
- idd-tools: commit/push `inbox/2026-07-10_..._resource-calibration-gaps.md`; the 10 gaps (+
  `final_run_setup/idd_tools_findings.md`) drive a future idd-tools fixing session.

**Active — dengue forecast prototype + R pipeline (2026-07-10):**
- Re-run the notebook forecast section (`pygam_dengue_models_explore.ipynb`) — the kernel crashed,
  **likely OOM** from the lsae age/sex-CFR 29M-row frames (`cfr_as_insample("lsae")` cache +
  broadcast + `past_base`). Mitigate: fewer draws / decade window / coarser CFR grain, or
  chunk/stream. Restart & Run All from the spec cell (the config cell resets the slim forecast
  defaults). Bobby will retry the notebook later.
- Vet the forecast trajectories (grain-generalized; `D_fhs` verified sane + continuous at 2023);
  scale `FORECAST_SSPS` / `N_FORECAST_DRAWS` / `FORECAST_YEARS_FC` once the crash is resolved.
- Build the real R pipeline per `.claude/FORMALIZE_DENGUE_FORECAST_PIPELINE_PROMPT.md`
  (fit+register → R forecaster reading the 08b nc → finishing). `08b` already built + ran; the
  prompt's dated 2026-07-10 Update block carries the full state + open questions (incl. "question the
  split fit→shift→rake organization").
- Known caveat (not a bug): in-sample aggregate CFR/inc saw-tooths in sparse regions (High-income)
  from the `06b` presence-filter's year-varying membership; the forecast (consistent 08b set) is
  clean. Finishing should aggregate over a consistent (all-modeled-loc × all-year) set.
- (07c is now RUN — 29,109 prediction locations.) Older TRACKING-goal iteration of
  `fit_dengue_models_explore.r` is superseded by the notebook prototype; the tracking-slope metric
  idea (memory `dengue-model-tracking-goal`) still applies to vetting.

1. **Stage 04 forecasting — DONE 2026-06-03** (see Recent steps). Outputs at
   `_A04_MAL_FORECAST_OUTPUTS/20260602/` (current). Forward from here:
   - Constant dah alongside Baseline (DAH_SCENARIOS 1→2; no code change, but
     re-measure mem — the launcher's REVISIT trigger).
   - Wire stage-05 aggregation to read the new forecast-output netCDFs
     (dims location_id,year_id,draw; vars log_malaria_{inc,mort}_rate_pred).
   - Cleanup: remove stage-04 `*_draft.r`/`*_OLD.r` reference files once the
     new chain is trusted; remove the 9 leftover login-node /tmp offline-test
     artifacts (Bobby's call).
2. **Bring urban back into 07b's coverage check** once the incidence-
   only test pass is verified end-to-end. Uncomment the
   `"weighted_1km_urban_threshold_300.0_simple_mean"` line in
   `SHARED_VARS_TO_CHECK` at
   `02_data_prep/07b_build_malaria_prediction_locations.py:51`, re-run
   07b → 08.
3. **Optional 05 re-run for vintage tidiness.** Current
   `malaria_past_inputs.parquet` (Jun 1 13:06) used ssp245 / rcp 4.5
   which was exactly representable as float, so the values are valid.
   Re-run only if you want the new string-RCP parquets reflected in
   the past-inputs timestamps.
4. **Sync local main with remote.** Local main is at `88d5215`
   (2026-02-25); remote main is at `0365b0b`. `git fetch && git merge
   origin/main` (or rebase) before any merge of
   `feature/refactor-shared-lib` back to main.
5. **Finalize rocket** — remove debug `save()` line at L102 of
   `fit_malaria_models_rocket_bfgs.r`; decide .rds vs .RData format.
6. **When `20260514_v2` lands**: filter summaries by
   `is_converged == TRUE & cv_n_converged == 5` before ranking.
   Rankings from the earlier EFS-fit batches are not trustworthy.
7. **Re-run MCDM ranking** (`rank_malaria_models_loop.ipynb`) against
   the BFGS-fit pool. Compare top-of-rankings against the EFS-fit
   rankings. Favour 3-of-4 method convergence (DECISIONS 2026-05-13).
8. Build `06_build_dengue_past_inputs.py` (same pattern as 05; watch
   for row-filter bug — see DEAD_ENDS.md). NOTE: dengue 03/04 already
   work; 06 is the dengue-side analog of 05.
9. **Scale to 73,710** via SLURM array job — same cache dir; previous
   fits dedupe automatically.
10. **Repeat scaffold for mortality and incidence** — same group
    structure, different response.
11. Design forecasted input/output file structure (draw dimension
    needed, netCDF appropriate here).
12. **[VALIDATION GAP]** Stage 06 (upload): not verified against
    goldens.
13. **[Open Q from stage-01 design]** Urban-threshold consistency with
    the per-covariate artifact pattern. Currently `URBAN_WRITE_PATH`
    is a single root for both thresholds; climate covariates each get
    their own. Decide whether to split urban thresholds into per-
    threshold artifacts or leave the asymmetry.
14. **[Independent backfill]** Loc 93390 has zero pop 1950-2014 then
    real growth 2015-2100. The 19 other NaN locs are permanently
    uninhabited. 07b now correctly handles both via the pop_zero-
    first drop policy. No action needed unless we ever want to fill
    93390's pre-2015 climate aggregates for the historical model
    window.
15. **08 regression test refresh.** Existing
    `tests/02_data_prep/test_08a_build_malaria_forecast_inputs.py`
    runs 08 standalone — needs to run 07b first or be told where to
    read the kept-loc parquet from. Current fixture predates the 07b
    pop-zero-first design.

## Parking lot
- **[2026-07-21] Relocate model-selection code out of `reports/03_modeling`** — Bobby: "reports/03_modeling is a horrible place for the only place where model selection code lives." The selection/ranking logic (esp. the report notebook) shouldn't have its only home under `reports/`; move into proper versioned modules (e.g. `src/idd_forecast_mbp/select/`) later. Note-for-later, not yet actioned.
- `read_grid_map()` in `netcdf_helpers.R` is a placeholder; canonical fix
  is for the upstream Python writer to embed a `grid_map` global attribute.
- Population weighting was discussed and rejected (high-pop ≠ high-malaria);
  revisit only if downstream use case changes.
- Group annotations in the qmd may evolve once AICs come in (some
  "concave-down" hypotheses for temperature variables may not pan out).
- Delete legacy modules (`parquet_functions`, `xarray_functions`,
  `helper_functions`, `hd5_functions`) after refactor fully verified.
- Delete old standalone test scripts in `src/04_forecasting/`.
- `combine_as_draws.py` dead-end — revisit when draw combining becomes a
  bottleneck. See DEAD_ENDS.md.
- Dengue revision (after malaria paper accepted).
- Feature: vaccination scenario dimension.
- Feature: variable importance methods beyond "hold at 2022".
- When suitability variant files become available (14 curves), re-enable
  all variants in script 05.
- dev.expl discrepancy (0.8189 SLURM vs 0.8191 interactive) — same data,
  unknown cause. Not blocking.
- The `pull_map_rasters.py` script downloads from MAP WCS for 7 covariates
  but `COVARIATE_DICT.yaml` only references the 5 Pf ones today. The 2 Pv
  files (`Pv_Incidence_Count`, `Pv_Incidence_Rate`) are on disk awaiting
  use; add YAML entries when they join the model.
- `03_pixel_hierarchy_parallel.py:17` and `pixel_hierarchy.py:31` still
  hardcode the modeling_frame path instead of using `mbpc.MODELING_FRAME_PATH`.
  Works today; future-proofing leftover.
- Feature: allow locations with 0 pfpr/inc/mort at the rake year (2023) to
  *emerge*. Currently `classify_zero_burden()` drops them, so a place that is
  0 in 2023 can never become nonzero in the forecast. Idea: seed them with a
  tiny offset, rake to that, forecast forward, then cull the ones that didn't
  grow over time. This is essentially the stubbed `zero_burden_policy='impute'`
  (option B) and pairs with the per-draw culling machinery. Parked 2026-07-07.
