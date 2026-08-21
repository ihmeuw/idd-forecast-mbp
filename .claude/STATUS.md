# Project status
Updated: 2026-08-21

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

**Formalization is the active malaria front (2026-08-04).** Stage-05 products and figures are
built and run; the work has shifted from *producing* figures to making the pipeline repeatable,
findable, tested and cause-shared. Plan: `.claude/FORMALIZATION_PLAN.md` (8 phases, with a
decisions-needed list). **Phase 0 is done** — 115 untracked files committed in 9 groups and
pushed; the entire stage-05 build, `lib/modeling/`, `lib/viz/` and all tests had been on one disk.
Two architectural commitments came out of a 3-round consult with the dengue workstream:
- **`CauseSpec` holds only what is true of a cause regardless of run** (`lib/cause_spec.py`, 10
  fields). Admission test: would this be the same for EVERY run of this cause? That excludes
  everything on the refit list — formulation, engine, anchor spec, reference cell, suitability
  variant, burden threshold — and all covariate trajectories (DAH scenarios, decay functions,
  holds), which live in a run manifest. Covariates are excluded twice over: which ones a run uses
  comes from its formulation, and covariate metadata already has a canonical owner in
  `lib/io/covariate_registry`. `as_draw_persist_grain` is the one genuinely cause-specific field,
  a grain not a boolean, independent of `fit_grain`.
- **The two causes share the product CONTRACT, never the producer** (`lib/processing/products.py`:
  schema + `validate_products` + a shared reader). `finish_run.py` is not in dengue's path — it
  calls `roll_up_hierarchy(start_level=ADMIN2_LEVEL)` and so cannot consume dengue's mixed-level
  473-leaf FHS set. The load-bearing check re-derives every rate from its count and that level's
  own population row. `anchor_diagnostic` compares predicted level to the ANCHOR TARGET, never to
  observed-at-anchor-year — hard pass/fail only for a point anchor.
Cause-sharing order is **stage 05 now, 04 next, 03 eventually** — NOT the earlier "03+ goes
cause-blind", which mistook stage 03's cause-specific science for shareable plumbing. Three
successive over-abstractions (`has_dah: bool` → `AxisSpec` → `EligibilityRule`) were each proposed
and retracted; see DEAD_ENDS 2026-08-04 so they are not re-derived.

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

**Dengue forecasting (2026-08-04, current dengue front).** The formulation-agnostic harness is
built and runs end-to-end in Python/pyGAM: `lib/modeling/{anchor,year_path,dengue_formulations,
dengue_pipeline,dengue_forecast}.py` + `lib/data/{dengue_inputs,dengue_forecast_covariates,
first_submission}.py` + `lib/processing/dengue_products.py`. A formulation declares outcome
structure (`inc_cfr` / `inc_mort` / `mort_then_inc` / `mort_cfr`), per-outcome response grain,
an `AnchorSpec`, a year term and an engine; the pipeline fits, anchors, fans out to age/sex,
aggregates in count space and summarizes. F4 (`GBD-esque_w_time`, mort_then_inc, all-age) has
been forecast 3 SSPs × 4 decays × 78 years × 100 draws →
`05-products/dengue/lsae_1285/20260804/f4_forecast_summary.parquet`.
- An R/mgcv sibling (`03_modeling/fit_dengue_formulations.r`) exists because **pyGAM has no
  factor-`by` smooth** — masking a column works for a LINEAR by-group term (out-of-group β·0
  carries no information) but not for a spline (`s(0) ≠ 0`, and 71–98% of rows pile at 0). Five
  formulations fit; output `07-figures/20260804/dengue_formulation_comparison/`. Bobby drives it
  interactively and is stripping the CLI scaffolding.
- **Anchor semantics govern how any of it reads.** F4 anchors to the 2014–2023 MEAN, not a 2023
  point. Its 2023 (20.3 M) is within 4% of the decade mean (19.60 M) and 45% below observed 2023
  (37.5 M) — correct, because 2023 was a record dengue year. **Compare predicted level to the
  ANCHOR TARGET, never to observed-at-anchor-year.** Incidence passes at +3.8%; mortality does
  NOT, at **+21%** (61.6 k vs a 50.9 k target) — unexplained, and not retransformation bias
  (summing per-location geometric means biases aggregates DOWN, wrong sign).
- **The past frame is now the COMPLETE FHS set (2026-08-21).** `dengue_past_inputs.parquet`
  (`20260821_v2`) holds all **473** FHS-most-detailed locations × 24 yr × 25 age × 2 sex =
  **567,600 rows**, dense, no NaN in any of 32 columns, zeros stored as zeros. Previously it held
  305: 06a's A0 count gate cost 91 locations and 06b's per-location-year `all-age cases > 0` rule
  cost 77 more. Both are off; 06a's gate survives as a `fit_eligible` COLUMN, so
  `fit_eligible & inc_rate > 0` reproduces the old frame from the new artifact (verified: exactly
  305, and bit-identical on all 27 shared columns over the 366,000 overlapping rows). **A location
  with zero cases every year is an observed zero, not missing data** — the raked source carries
  exact zeros with no NaN for all 50 of its cells. Grain selection keys on the `most_detailed_fhs`
  FLAG, never a level cut: the FHS set straddles levels 3 and 4.
- **The remaining zero-dropping is in the FIT path, not the artifact.**
  `load_dengue_inputs(grain="fhs")` still returns 305 locations, because `build_age_sex_rr` gates
  its RR universe on `inc_count > 0` at the reference cell and `attach_age_sex_rr` then filters
  `past` to it. That filter is unnecessary and slated for removal (DECISIONS 2026-08-21):
  `broadcast_to_cells` already `fillna(0)`s the RR, `rr_inc_as` has no consumer at all under the
  default per-cell-anchor formulations, and the one hazard that could justify it — a nonzero
  all-age prediction being silently zeroed — cannot occur, because the all-age anchor for those
  locations is itself zero. What must NOT be deleted along with it: `attach_age_sex_rr` also
  assigns `A0_af` and `as_id`, and the forecast must reuse the fit's exact `A0_af` codes.
- **The real constraint on the 168 is the ANCHOR, not any filter.** An anchor pinned to an observed
  zero can never produce a nonzero forecast, so those locations project zero to 2100 whatever their
  covariates do. That is the open question for climate-driven range expansion — same shape as the
  parked malaria `zero_burden_policy='impute'` idea.
- **The FHS forecast frame is DERIVED, not missing.** `lib/data/dengue_forecast_covariates.py`
  defaults to `grain="fhs"` and population-weights the admin-2 netCDF up at read time, reproducing
  observed past FHS covariates to ~1e-16. So "rebuild forecast inputs at FHS grain" is a
  packaging/cost change (5.1 GB → ~250 MB, no roll-up per call), never a capability gap.
- **Aggregation to any hierarchy node is now a lib primitive.**
  `aggregate_outcomes_to_ancestors` (counts summed, rates from that level's OWN population),
  `roll_up_covariates_to_ancestors` (population-weighted covariates to arbitrary ancestors), and
  `make_rate_from_count(join_cols=…)` for age/sex grain. The 473 leaves reach 513 nodes (the 40
  aggregates above the grain plus themselves). Note "Global" here means the sum of the 473, which
  is 99.923% of the hierarchy's stored `location_id == 1` population.
- **Cross-cause architecture settled and dengue's figure build is UNPAUSED (2026-08-04).** Both
  prerequisites landed and are pushed: `b5df5fe` (`CauseSpec`/`AnchorSpec`) and `53debbe` (product
  contract + `validate_products`). Handoff for the dengue session is `.claude/DENGUE_UNBLOCKED.md`,
  pointed to from memory.md's Dengue section. Two things dengue cannot infer from the consult
  files: `CauseSpec` is SMALLER than `DENGUE_CONSULT_ROUND3.md` describes (no `covariates`,
  `anchor`, `measures`, reference cell or burden threshold — so nothing is left for them to
  confirm), and the F4 summary must split into `all_age_summary_{ssp}_{trajectory}.parquet` with
  `level` + `population` added, because a single-valued `decay` column is rejected outright.
  Settled jointly: NO `AxisSpec` (DAH and decay are both covariates with alternative
  future trajectories, recorded in the run manifest, not a cause property); stage 03 is not
  permanently cause-specific (four outcome structures and two engines are exploration
  scaffolding); `as_draw_persist_grain` is a grain and independent of `fit_grain`; population and
  age-structure holds are `aggregation_transforms`, not covariate trajectories; refit-vs-
  re-evaluate is decided by whether an alternative changes the PAST or only the FUTURE. See
  `.claude/DENGUE_CONSULT_REPLY.md` (incl. its RETRACTION section) and `DENGUE_CONSULT_ROUND3.md`.
- **The endemic/outbreak framework bake-off moved out (2026-08-10/11)** to the new repo
  `idd-forecast-lab` (scaffolded, pushed; briefs in its `.claude/`): hhh4, robust outbreak
  labelling, R-INLA, glmmTMB/brms/mvgam, MR-BRT flat-tail priors, EVT tails — scored on
  non-circular decomposition, decay-to-flat projection, uncertainty propagation, and cost at
  ~305 locations. Rationale: this repo's venv-only scheme forbids mrtool's conda Python, and
  exploration-weight R deps (INLA, Stan, GitHub-only hhh4addon) don't belong in
  `environment-r.yml`. mbp consumes only the eventual winner, ported into `03_modeling`.
  **Candidate 1 (hhh4) is REJECTED as of 2026-08-11** (annual-frequency AR-vs-trend competition;
  DECISIONS 2026-08-11); the front-runner is now a per-location outbreak MIXTURE — EM
  soft-labelling for occurrence + EVT for magnitude.

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
- 2026-08-21: **Dengue — past inputs widened to the full FHS set; ancestor aggregation moved into
  lib.** (1) **`20260821_v2`: all 473 FHS most-detailed locations**, 567,600 rows, zero NaN across
  32 columns, zeros kept, and the old 305-location frame reproduced bit-identically — recoverable
  from the new artifact via the new `fit_eligible` flag. 06a untouched: its gate is now a flag, not
  a filter. Verified the source universe rather than assuming it (2000–2023 × 25 ages × 2 sexes is
  exactly what the raked AS frame carries, so nothing else was narrowed), and that covariates reach
  all 473 — the 88 gdppc holes at admin-2 (76 of them Russian subunits) close under the
  population-weighted roll-up. (2) **Two corrections to my own claims**: the FHS forecast frame is
  produced at read time (~1e-16), so that STATUS next-step was packaging not a gap; and a first
  build using `--max-level 4` was wrong — 3,688 locations, because the FHS grain straddles levels
  3 and 4 and no level cut can express it. Superseded dir `20260821` left in place. (3) **Traced
  the remaining drop** to `build_age_sex_rr`'s `base_location_ids` filter and established it is
  unnecessary — see Orientation and DECISIONS. (4) **Built the exploration notebook, then rebuilt
  it over lib** after it reimplemented `roll_up_to_ancestors` AND used summed-child populations as
  rate denominators — the defect `validate_products` exists to reject, worth 0.077% on the global
  rate. Added `aggregate_outcomes_to_ancestors`, `roll_up_covariates_to_ancestors`, `join_cols` on
  `make_rate_from_count`; 21 tests, one of which caught pandas' groupby-sum silently skipping NaN
  and reweighting the surviving children. (5) **Pre-commit is broken repo-wide**:
  `.pre-commit-config.yaml` still calls `poetry run ruff`/`mypy` after the venv-only migration, so
  three hooks fail with "Executable `poetry` not found" on any file. Commits `90d00d7`, `fa2e76f`,
  `4147332`. See DECISIONS/DEAD_ENDS 2026-08-21.
- 2026-08-11: **Dengue bake-off — candidate 1 (hhh4) REJECTED and ratified; front-runner reframed
  as a mixture.** At annual frequency hhh4's lag-1 epidemic term and the endemic trend explain the
  same secular growth, and the MLE prefers the AR (λ 1.11–1.16 super-critical, endemic share 1–3%,
  endemic intercepts on a flat ridge, robust to warm-starting from a saturated endemic optimum) —
  so the anti-circularity mechanism never engages AND no fitted time trend survives for the decay
  to act on. Kept from the arm: 2018/2020 confirmed model-side as the most-negative shared-residual
  years (COVID-era reporting; bears on terminal-slope trust for ANY winner), and endemic-only hhh4
  reproduces the gam's trend structure (parametric NB is a viable EM engine). New direction
  (Bobby's framing): per-location outbreak PROBABILITY + per-location MAGNITUDE = EM soft-labelling
  (candidate 2b, posteriors a first-class product) + EVT/GPD on the excess ratio (candidate 5); 2a
  (farringtonFlexible at frequency 1) is a mechanical test only. Lab output node declared:
  `/mnt/team/idd/pub/forecast-lab`. Also corrected an invented "throwaway evaluation envs / delete
  once the memo is written" policy in the lab's CLAUDE.md → cross-repo tool-env pattern (provision
  once at user level, document in `~/.claude/tools/`, NEVER delete); rule logged to persistent
  memory as `feedback_never_delete_envs.md`. See DECISIONS 2026-08-11.
- 2026-08-10/11: **Dengue — framework bake-off spun out to `idd-forecast-lab`.** New repo
  scaffolded via /scaffold-repo (pure-uv `.venv` + declared conda R env `idd-forecast-lab-r`),
  pushed to ihmeuw. The bake-off brief (+ the MR-Tool eval as candidate 6, + a copy of
  `dengue_formulations_handoff.md`) live in its `.claude/`. Readback surfaced four feasibility
  flags (annual data vs hhh4's design, non-integer counts vs `sts`, conda availability of
  hhh4addon/INLA, gap-free series) — recorded in the brief as verify-first items. Handoff prompt
  delivered via /handoff; the lab session passed its orientation quiz and is green-lit (first
  deliverable: the hhh4 control-list spec, text-only, hard stop before any code/env/install).
  idd-ssms deliberately kept separate (mechanistic vs statistical boundary, documented in the
  lab's README). In mbp: no code touched; `c29b24d` (venv migration, prior session) pushed at
  wrap. See DECISIONS 2026-08-10.
- 2026-08-04: **Malaria — formalization Phase 0, CauseSpec + product contract, O(n²) figure fix.**
  (1) **Phase 0 committed and pushed.** 115 untracked files in 9 disjoint groups (`37c15fd`..
  `05cec54`), then `b5df5fe`/`53debbe`; `06a01bc`→`53debbe` on the remote. Routed absolute paths
  in 4 active files through new `constants.py` entries (`previous_upload_path`,
  `FIRST_SUBMISSION_RUN_PATH`, `PREVIOUS_COVARIATE_NC`, `GDPPC_SOURCE_PATH`) + added
  `den_products_{root,write_path,read_path}`. Wrote `.claude/FORMALIZATION_PLAN.md` (8 phases).
  (2) **O(n²) bug in three netCDF loaders**: `[i for i in locs if i in set(int(x) for x in
  ds.location_id.values)]` rebuilds a 51k-element set per candidate location. Covariate figures
  went from 2-of-10 in 31 min to 9-of-10 in 150 s (~60×); the GDP-hold figure job had burned 30 min
  at 100% CPU without emitting a PNG. Regenerated all 5 figure sets (~400 figures, 0 errors) with
  `ts_bars`, `bars_values`, trimmed `differences_stack` titles and the 5 `__weightpair` 1×2s.
  (3) **`CauseSpec`/`AnchorSpec`** (`lib/cause_spec.py`, 42 tests) and the **product contract**
  (`lib/processing/products.py`, 29 tests) — the two items blocking dengue. Verified against a
  shipped product (malaria ssp245 Baseline, 7,800 rows, levels 0–3) which conforms unmodified.
  Two self-inflicted bugs found and fixed: `_check_intervals` unpacked `STATS` positionally so it
  compared mean against lower, and a test fixture with counts proportional to population made the
  wrong-denominator test **vacuous** (green while asserting nothing). (4) **`CauseSpec` shrank
  15→10 fields** after Bobby challenged its extensibility — see DECISIONS. (5) **Cumulative global
  totals by scenario** for both GDP arms: the ~11% mortality drop vs the 2025 run is present in
  BOTH arms at both horizons, so GDP coupling is ruled out and a different `mort_mod` vintage is
  the remaining hypothesis. Cumulative incidence is within ±4% and sign-inconsistent across
  scenarios. Tables at `…__gdpscen/current/tables/cumulative_global_totals{,_both_arms}.csv`.
  See DECISIONS/DEAD_ENDS 2026-08-04.
- 2026-08-04: **Dengue — cross-cause consult, R harness repaired, F4 anchor understood.**
  (1) **Consult with the malaria workstream** (`.claude/DENGUE_CONSULT.md` → `DENGUE_CONSULT_REPLY.md`
  → their `DENGUE_CONSULT_ROUND3.md`): answered 7 questions, then RETRACTED both original
  disagreements after Bobby corrected them — `AxisSpec` was `has_dah` one abstraction level up
  (DAH and decay are just covariates with alternative futures), and "stage 03 never goes
  cause-blind" mistook a testing harness's temporary state for the pipeline's permanent shape.
  Malaria side accepted both, plus `as_draw_persist_grain` as a grain not a boolean. Their
  amendments accepted in return: `aggregation_transforms` distinct from covariate trajectories,
  and refit-vs-re-evaluate keyed on past-vs-future. The `EligibilityRule` taxonomy that grew out
  of it was then dropped by both sides as over-built — the fit filter is idempotent because the
  cull already happened in the data. (2) **R script made to run**: fixed wrong stage-03 input
  paths (raked_as/raked_aa live in stage 02), and a fit-set divergence — it read
  `most_detailed_fhs` from `lsae_1285_to_fhs_table.parquet` (513 flagged) instead of the
  hierarchy (473), fitting **314** locations against Python's **305**. Also moved figure output
  to `07-figures/` after wrongly writing three runs into a home directory, and added a guard that
  refuses any path outside the figures node. (3) **F4 anchor**: the 2023 "shortfall" is the
  2014–2023 mean anchor working as specified (observed decade mean 19.60 M vs F4's 20.34 M).
  Ruled out Duan smearing despite `exp(σ²/2) × 20.34 = 37.4 M` matching observed 2023 almost
  exactly — recorded in DEAD_ENDS so it is not re-derived. Mortality's +21% over target is still
  open. (4) **Covariate audit**: past inputs already carry all 16 covariates; only the forecast
  side is subsetted to 7. At FHS grain all 16 fit in ~250 MB against today's 5.1 GB admin-2 build.
  All 16 source files verified present across 3 SSPs. See DECISIONS/DEAD_ENDS 2026-08-04.
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
**Active — malaria formalization (2026-08-04; `.claude/FORMALIZATION_PLAN.md` is the plan):**
1. **`tests/05_aggregation/` for malaria BEFORE rewiring anything.** The dir now exists but holds
   only dengue's tests. `plot_run_comparison.py` is 1,400+ lines with 32 DAH references and zero
   tests, and has already been hand-repaired once with no `git checkout` fallback. Pin current
   figure-data behaviour first so the refactor has something to violate.
2. **Make the trajectory selector generic** across stage-05's 41 DAH references (`--dah-scenario`
   is a covariate-trajectory selector with one covariate's name baked in), and swap
   `roll_up_hierarchy(start_level=ADMIN2_LEVEL)` → `roll_up_to_ancestors` — with a test proving the
   two agree exactly for uniform level-5 input, so the swap is provably behaviour-preserving.
3. **Run manifest (Phase 4).** Replaces string-concatenated arm dir names
   (`…__gdpscen__gdppc_hold2023`). Fields: `fit_inputs` (refit axes — model_id, suitability
   variant, engine, `year_center`), `covariate_trajectories` (re-evaluate axes), and
   `aggregation_transforms` (population / age-structure holds, which are NOT covariates — they
   appear in no fit formula). Also backfill `status: invalid` onto `__anchor_first_submission`,
   which matched 0 of 47,459 rows and is currently indistinguishable from a valid arm by name.
4. **Draw-level aggregates at levels 0–3 (Phase 5b)** — required for matched-draw sensitivity
   differences and rate-of-change-by-draw, both of which today's year-chunked `finish_run.py`
   structurally cannot do (it collapses the draw axis inside the year loop). ~175 MB/arm against
   80 MB now; admin-2 draws stay in stage 04. Draw alignment is already proven (RCP4.5 was
   bit-identical across the two GDP arms) — add a hard assertion so a future reseed fails loudly.
5. **14-variant suitability sweep (Phase 5) — GATED.** Grid is 14 variants × 3 SSPs × Baseline DAH,
   no sensitivities. **Measure one arm's stage-04 draw footprint before submitting**: products are
   80 MB/arm but `lsae_1209` totals 921 GB, and the draw number was never taken. Probe one variant
   end-to-end and extrapolate explicitly first.
6. **Fix pre-commit — it cannot execute at all (found 2026-08-21).** `.pre-commit-config.yaml`
   still uses `entry: poetry run ruff check …` / `poetry run mypy .`, and poetry is gone after the
   venv-only migration (`c29b24d`), so **ruff-format, ruff and mypy all fail with "Executable
   `poetry` not found" on any file, repo-wide**. The file-hygiene hooks (docstring-first, debug
   statements, EOF, whitespace, line endings) do pass. Fix the entries first (`uv run` / a direct
   `.venv/bin` path); only THEN is the older debt reachable — `pyproject.toml` sets
   `select = ["ALL"]`, which no committed file satisfies. Until both are done every commit needs
   `--no-verify`.
7. **Still unrun: the mortality-vs-incidence R diagnostic.** Compare
   `2025_10_10_malaria_models.RData` against the current `.RData` `mort_mod` smooths. Income is
   ruled out, and now so is GDP coupling (both arms drop ~11%). Deaths-per-case is already 5.7%
   lower at the 2023 anchor.
8. Lower priority, from the plan: single top-level `archive/` sweep (last, one pass, with a README
   noting each file's era); notebook tiering (`reports/<NN>_<stage>/` durable vs
   `notebooks/<NN>_<purpose>/` deliverable-facing vs `notebooks/scratch/`); cull the superseded
   `.claude` prompt docs into `_archive/`.

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

**Active — dengue (2026-08-04; supersedes the 2026-07-10 prototype block):**
- **Remove `build_age_sex_rr`'s location filter so the fit frame is all 473.** This is the one
  thing still shrinking the data (DECISIONS 2026-08-21): left-join the RR and leave it NaN
  off-base, keep `base_location_ids` on `DengueInputs` as a record rather than a gate. Do NOT
  remove `attach_age_sex_rr` itself — it assigns `A0_af`/`as_id`, and the forecast must reuse the
  fit's exact `A0_af` mapping. Then decide what a log-link fit does with the zeros: they still go
  to `log(0) = -inf` and get dropped by the finite-row filter, so consuming them needs a
  count-space/offset or hurdle component. Note 14,640 rows INSIDE the endemic 305 are zero cells
  too — the zero question is not only about the 168.
- **Repackage forecast inputs at FHS grain (optional, cost-only).** 08b writes admin-2 × 7
  covariates = 5.1 GB over 3 SSPs; the FHS frame is derived from it at read time already
  (~1e-16), so this saves ~20× the bytes and the per-call roll-up, not new capability. If done,
  add the 9 missing covariates (`days_over_30C`, `mean_temperature`, `mean_high_temperature`,
  `mean_low_temperature`, `precipitation_days`, `wind_speed`, `ldipc_mean`, `med_consumppc`,
  urban-1500 — all sources verified present × 3 SSPs), write parquet wide plus a draw-free mean
  collapse for R, and carry `population`/`super_region_id`/`region_id`/raw `year_id`. **Two live
  snags**: `forecast_inputs/current` → `20260527` (6 covariates) while the newer `20260803`
  (7, adds `total_precipitation`) is unlinked, and `DEN_FORECAST_INPUTS_WRITE_PATH` resolves to
  `RUN_DATE` = `20260527`, so a rebuild without `IDD_RUN_DATE` set writes into the OLD dated dir.
- **Widen the forecast/prediction location set if the fit goes to 473.** `07c` starts from the
  A0-gated `dengue_prediction_location_ids` (29,109 admin-2 of 47,459), so today you could fit on
  473 and still not predict on 473.
- **`fit_dengue_formulations.r` now reads a different frame.** `current` moved under it: 473
  locations, a new `fit_eligible` column, and no admin-2 rows. Bobby drives it interactively — do
  not edit without asking, but it needs checking against the new artifact.
- **Diagnose F4 mortality at +21% over its anchor target** (61.6 k predicted at 2023 vs a
  2014–2023 mean of 50.9 k; observed 2023 is 52.7 k). Incidence is fine at +3.8%. Not smearing —
  wrong sign. This is in the anchor/products path both engines feed, so it contaminates any
  formulation eventually chosen.
- **Blocked on the malaria side** (their two deliverables): `CauseSpec` + `AnchorSpec`, then the
  product schema + `validate_products`. When those land: split
  `f4_forecast_summary.parquet` into `all_age_summary_{ssp}_{trajectory}.parquet` (~1 h), then
  build the run-comparison figures — timeseries per location as 4 decay rows × 4 columns
  (inc/mort × count/rate), 3 scenarios solid current + dashed previous run, observed in black;
  global + 6 super-regions, no countries, SR 31 excluded (genuinely zero burden). SKIP the
  `differences`/`differences_stack` figures: decay spans 235→3,529 M at 2100 while scenario spans
  235→251 M, so scenario contrasts measure the wrong axis until decay is chosen.
- **Absent-vs-zero: RESOLVED for the past frame (2026-08-21)** — the 168 are present with explicit
  zeros, so the decision is now decidable downstream instead of being pre-empted by their absence.
  Still open on the PRODUCTS side: whether forecast products carry them as zeros or omit them.
- **Zero-incidence rows with nonzero mortality**: 42,723 rows across 40 locations (Guam, Shanghai,
  Bulgaria, Tibet, Qinghai, Heilongjiang, Bahrain, Jiangxi…), magnitudes 5e-9 to 1.1e-5 — dengue
  deaths where there are no cases. An upstream raked-AS inconsistency, newly visible because those
  locations used to be filtered out. CFR is NaN there by construction, but a mortality-first
  formulation would see a finite `log_dengue_mort_rate` at zero incidence.
- Open, lower priority: the R and Python F4 do not use the same anchor estimator (R `rake="median"`
  vs Python `Baseline(statistic="mean")` over 2014–2023) — a comparability divergence of the same
  class as the 314-vs-305 bug. And `fit_location_ids.parquet` flags 382 FHS locations while
  `past_inputs` keeps 305; the artifact named "fit locations" is not the one that decides the fit.
- **Framework bake-off is external** — runs in `idd-forecast-lab`; nothing to do here until it
  recommends an architecture, then port the winner into `03_modeling`.

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
