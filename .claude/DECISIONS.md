# Decisions log

<!-- Append-only. Never delete or overwrite entries. -->

## 2026-03-26: Monorepo refactor over separate repos
**Decision:** Refactor idd-forecast-mbp into lib/ + malaria/ + dengue/ structure rather than creating separate repos.
**Why:** Need shared infrastructure (variable importance, I/O, raking) for both diseases. Single repo = simpler git workflow, immediate availability of shared code changes. Dengue code can stay untouched while malaria is rebuilt.
**Revisit if:** Repos diverge so much that shared code becomes a burden, or if different teams need independent release cycles.

## 2026-03-26: Phased refactor approach
**Decision:** Phase 1 = file moves + import changes only (no logic). Phase 2 = verify. Phase 3+ = add features.
**Why:** Minimizes risk of breaking functionality. Can verify at each phase before proceeding.
**Revisit if:** Phase 1 takes too long or reveals deeper entanglement requiring logic changes.

## 2026-03-26: Constants import alias `mbpc`
**Decision:** Change `from idd_forecast_mbp import constants as rfc` to `as mbpc` across codebase.
**Why:** `rfc` was inherited from another repo. `mbpc` = "mbp constants", distinguishable from other repos (e.g., `cdc` for climate_data constants).
**Revisit if:** Never — this is just naming.

## 2026-04-15: Output validation status at end of lib/ refactor
**Decision:** Document what was and was not validated with `tests/compare_outputs.py` (exact value comparison, rtol=1e-5) at the end of the lib/ refactor (feature/refactor-shared-lib branch).

**Known validated (compare_outputs.py confirmed to pass after fixes):**
- Stage 05 (`05_aggregation/`): comparison run caught and fixed two bugs (extra gbd_location_id/aa_count columns, level variable leaking into output). Passed after fixes.

**Known NOT validated (compare_outputs.py not run):**
- Stage 06 (`06_upload/`): listed in STATUS.md as "still to do" as of 2026-04-07.

**Plausibility-checked only (slow pytest tests, not exact value comparison):**
- Stage 02 scripts 02a (`02a_fhs_population.py`) and 02b (`02b_full_population.py`): 19 slow tests passed verifying schema, row count, location IDs, and total population sum (rtol=1e-4), but NO row-by-row value comparison against the golden files.

**Unknown — no record of comparison runs:**
- Stage 01 (`01_map_to_admin_2/`)
- Stage 02 scripts 00, 01, 03–09 (all other data prep scripts)
- Stage 04 (`04_forecasting/`)

**Why this matters:** The refactor was planned as Phase 1 = file moves + import changes only (no logic). But without exact value comparison, we cannot confirm the refactor didn't accidentally change behavior.

**How to validate:** Run `python tests/compare_outputs.py --ref_dir <golden_flat_root> --test_dir <versioned_output_dir>` per stage. Golden files live at `/mnt/team/idd/pub/forecast-mbp/` stage roots. With versioning, test_dir = `{MODEL_ROOT}/{stage}/{RUN_DATE}/`.

**Revisit if:** A full end-to-end pipeline run completes — compare all stage outputs against the pre-refactor golden files at that point.

## 2026-04-16: Cell-by-cell regression test strategy for pipeline scripts
**Decision:** Regression tests for pipeline scripts (02a–07) call main() with lsae_1209 golden inputs and compare outputs cell-by-cell (`np.testing.assert_allclose`, rtol=1e-5 for parquet, rtol=1e-4 for NetCDF) against existing lsae_1209 golden files. For large files (>100M rows or >50K location×year combos), sample 5% of rows/locations.
**Why:** Sum checks and schema checks don't catch numerical regressions. The prior total-population-sum test approach was explicitly rejected (session 2026-04-16).
**Revisit if:** Golden files are regenerated for a reason (e.g., input data update), at which point the test baselines need to be re-established.

## 2026-04-15: No batch-reading before acting
**Decision:** Never read more than 3 files before writing something. Start with what you have; read more only as needed.
**Why:** I spent 20+ minutes reading 8+ source files trying to build a complete mental model before touching any test file. That is the wrong order. The cost is user time and context bloat with no benefit.
**Revisit if:** Never — this is a hard behavioral limit.

## 2026-03-27: Malaria disaggregation canonical method
**Decision:** `as_malaria_fractions.py` (normalized RR fractions) is canonical. `as_malaria_shifts.py` is superseded.
**Why:** Bobby confirmed. The fractions method normalizes so age-sex counts sum to the all-age total; shifts method does not.
**Revisit if:** Model refit changes the recommended disaggregation approach.

## 2026-03-27: level_filter() is canonical hierarchy filter
**Decision:** `level_filter()` in `helper_functions.py` is the intended canonical function. All inline copies in pipeline scripts are divergences to fix in Phase 4.
**Why:** Bobby confirmed it was always the intended canonical form; scripts diverged from it accidentally.
**Revisit if:** Never — will be unified during Phase 4.

## 2026-03-27: RH clip is a universal covariate rule
**Decision:** `UNIVERSAL_COVARIATE_CLIP_RULES = {"relative_humidity": (0.001, 99.999)}` is the default for all diseases in `load_covariates_for_draw`. Malaria was missing this clip historically; it will be applied after refactor.
**Why:** The clip is a data-quality rule for the covariate itself, not a disease-specific modeling choice. Disease-specific additions can be passed via `extra_clip_rules`.
**Revisit if:** Evidence that malaria model behavior is sensitive to RH values at the extremes and the clip is harmful.

## 2026-05-07: Past inputs format: flat parquet not netCDF
**Decision:** 05_build_malaria_past_inputs.py (and future 06_dengue) output a flat parquet with one row per valid (location_id, year_id), not a dimension-aware netCDF.
**Why:** netCDF squareness forces ~24% of cells to be NaN (locations that had zero pfpr in some years). This wastes space, requires explicit NaN handling in R, and makes merging awkward. Past inputs have no draw dimension, so netCDF buys nothing. NetCDF is still the right format for forecasted inputs/outputs where draw dimension is needed.
**Revisit if:** Past inputs acquire a draw dimension, or if R merging of multiple forecast files makes the flat format too unwieldy.

## 2026-05-07: DAH missingness in source = $0
**Decision:** NaN values in the DAH source file (dah_by_channel_hfa_recip_1990_2100.csv) for a given country-year mean DAH = $0, not missing data. Fill with 0.
**Why:** Confirmed by Joe (DAH data owner) directly during this session.
**Revisit if:** Joe clarifies there are cases where NaN means "data not collected" rather than "no DAH."

## 2026-05-07: GDP population-masked locations stay in pipeline as NaN
**Decision:** 309 admin-2 locations with NaN gdppc_mean (all <5K population, pop_masking_flag=1 in source) are retained in the pipeline output with NaN gdppc_mean. R model drops them via nan_toss().
**Why:** All NaN GDP rows are confirmed population-masked in source. Dropping them in Python would silently change the location set; better to let R handle them explicitly with a warning message.
**Revisit if:** Bianca provides a fill strategy or GDP estimates for these locations.

## 2026-05-06: flooding is a scalar (non-draw) covariate in past inputs
**Decision:** Flooding goes into read_shared_covariates (loc×year scalar), not into read_draw_climate.
**Why:** Flooding parquet uses _mean_r1i1p1f1 naming — already ensemble-averaged, no draw variance.
**Revisit if:** A draw-varying flooding product becomes available.

## 2026-05-06: dengue_suitability not in read_draw_climate defaults
**Decision:** read_draw_climate() does not include dengue_suitability. Dengue script adds it via extra_vars.
**Why:** Disease-specificity boundary — malaria uses Mordecai/Villena suitability variants instead.
**Revisit if:** Never.

## 2026-05-11: CV strategy = country_no_fe (preferred)
**Decision:** Use country_no_fe as the default CV strategy for model selection. This drops A0_af from the OOS formula entirely, fitting 7 models: IS with FE, IS without FE, 5-fold country-holdout without FE.
**Why:** Country-holdout without fixed effects isolates pure covariate signal — the fairest comparison across specs. The "country" strategy (mean-FE imputation) produces terrible OOS metrics because the mean FE is a bad predictor.
**Revisit if:** We find that FE contribute meaningful OOS signal via a better imputation strategy.

## 2026-05-12: PfPR-space metric is Pearson correlation, not R²
**Decision:** PfPR-space fit metric uses `cor(observed_pfpr, predicted_pfpr)` — Pearson correlation — not R². Column name is `_pfpr_r`.
**Why:** R² can be negative for poor OOS predictions, making ranking awkward. Correlation is always in [-1, 1] and more interpretable for model comparison. The R² naming was a misnomer from the original implementation.
**Revisit if:** Never.
**Decision:** `write_netcdf` will accept a `mkdir=True` parameter to create parent directories. `write_parquet` `use_atomic` default changes from False to True.
**Why:** Standardizes behavior: both functions now create dirs and use atomic writes by default. Eliminates footguns where caller forgets to pre-create dirs.
**Revisit if:** Atomic writes cause 2x disk usage problems on specific filesystems.

## 2026-03-30: Dengue modeling df — yn filter commented out, 2473 new locations included
**Decision:** Accept that `as_md`, `base_md`, and `rest_md` dengue modeling dataframes now include ~2,473 level-5 LSAE-most-detailed locations (e.g. US counties) that were previously excluded by the `yn` filter.
**Why:** The `yn` filter (`dengue_stage_2_df[dengue_stage_2_df["yn"] == 1]`) is commented out in `06_dengue_modeling_dataframe.py`. These locations have no dengue historically but may be intentionally brought into the model at a later stage.
**Revisit if:** Downstream modeling stages (04+) need to handle or explicitly exclude these locations. Track whether they cause issues in forecasting steps.

## 2026-03-30: Known hierarchy drift — 44858 and new LSAE locations
**Decision:** Accept that production `full_hierarchy_2023_lsae_1209.nc` is stale relative to the current LSAE 1209 source hierarchy. Do not try to match it in validation.
**Why:** Location 44858 was removed from the LSAE hierarchy and its population manually split across 60908, 95069, 94364 (see test_02). Five new locations [44850, 44851, 44934, 44939, 50559] were added to the source hierarchy after the production .nc was last generated. The production parquet was updated but the .nc was not. Our test .nc is correct; the production .nc is stale.
**Revisit if:** The production .nc is regenerated — at that point this comparison should pass.

## 2026-03-27: Test scripts mirror production directory structure
**Decision:** Test scripts write outputs to `test_output/02-processed_data/`, `test_output/03-modeling_data/`, `test_output/04-forecasting_data/` — matching the production directory layout — rather than a flat `test_output/stage_02/`.
**Why:** Comparison utility needs a single ref_dir/test_dir pair per subdirectory; flat layout caused MISSING_REF errors and would prevent chaining test scripts end-to-end.
**Revisit if:** We switch to a different testing strategy (e.g., pytest fixtures, dedicated test data).

## 2026-03-30: Stage 05 output must not include extra columns from input netCDF
**Decision:** After loading and renaming columns in `cause_as_aggregation_by_draw.py`, explicitly keep only `as_merge_variables + ['count_pred']` before the hierarchy aggregation loop. Do not rely on the drop-by-keyword pattern alone.
**Why:** The input forecast netCDF now contains `gbd_location_id` and `aa_malaria_mort_count` (which renames to `aa_count`). These don't match the 'rate'/'pop' drop filter and propagate into the output, producing extra data variables absent in production. The fix locks the output schema explicitly.
**Revisit if:** Input netCDF schema changes again — this explicit selection will need to be updated.

## 2026-03-30: Stage 05 output must not include `level` variable
**Decision:** Drop `level` column from the aggregated df before calling `convert_with_preset` in stage 05 aggregation scripts. Also removed `level` from `variable_dtypes` dict.
**Why:** Production output contains only `count_pred`. The `level` column was being merged in for the aggregation loop but was never intended as a final output variable.
**Revisit if:** We intentionally want `level` in the upload output.

## 2026-03-30: Stage 06 fhs_location_ids must be deduplicated
**Decision:** Use `sorted(set(...))` when constructing `fhs_location_ids` and `locations_to_filter` in `create_and_combine_as_and_aa_draws.py`.
**Why:** `swap_location_ids = [60908, 95069, 94364]` are already present in `fhs_hierarchy_df["location_id"]` (they are in the FHS hierarchy). Appending them without deduplication caused duplicate `location_id` coordinates, which crashes xarray's `reindex()` with "cannot reindex along dimension 'location_id' because the index has duplicate values."
**Revisit if:** Never — deduplication is always correct here.

## 2026-03-27: data/covariates.py covers all model-predictor loading
**Decision:** No `data/climate.py`. Single `data/covariates.py` module covers climate (draw-specific), income, urban, and the merge_dataframes helper.
**Why:** All three loading patterns (climate, income, urban) follow the same dict-of-paths→read→merge structure and serve the same purpose (loading model predictors). Splitting by data source would create artificial module boundaries.
**Revisit if:** Covariate loading patterns diverge substantially between climate and socioeconomic data.

## 2026-04-17: test_06 dengue modeling golden mismatch — accepted, not a refactor bug
**Decision:** Skip test_06_dengue_modeling_dataframe.py row-count assertion and accept the discrepancy between the current script output and the `pre_restructure` golden.
**Why:** The `pre_restructure` golden was generated using `yn==1` filtering on `dengue_stage_2_df`. That filter was changed to `A0_dengue_ids` filtering in commit 9aef0f2, which predates the refactor commit (ead6495). The current script produces ~2,473 extra locations (US counties + 36 countries). Values are identical for all locations present in both outputs. The refactor did not introduce this discrepancy.
**Revisit if:** lsae_1285 run generates new goldens — at that point establish a new baseline and decide whether the expanded location set is scientifically correct.

## 2026-04-17: Stage 04 forecasting_data versioning
**Decision:** Version `04-forecasting_data/` matching stages 01-03 structure: `{cause}/lsae_1209/{YYYYMMDD}/` + `current` + named symlinks. First versioned run date: `20250811` (first_submission). ~3,800 orphan/obsolete files deleted.
**Why:** Flat unversioned root made re-runs destructive and origin of each file unclear. Multiple partial runs (Jul 3, Jul 8, Jul 29, Jul 30, Aug 10/11, Nov 2/7) were interleaved.
**What was deleted:** 600 Increasing/Decreasing malaria parquets (never completed pipeline); 2,400 GK malaria files (GK_cut20, GK_reference); 200 better/reference malaria parquets (Nov 2 re-generation, pipeline not re-run); 190 Jul 29 orphans (old hold naming: logit_malaria_suitability, mal_DAH_total_per_capita, people_flood_days_per_capita superseded by short-name Jul 30 files); 2 dengue orphans; hierarchy_lsae_1209_full.parquet (unreferenced).
**Revisit if:** A second run completes — create `20YYYYMMDD/` + update `current` symlink.

## 2026-04-05: Implement node-level output versioning per STANDARDS.md
**Decision:** Implement STANDARDS.md versioning: all pipeline output goes to `{stage_dir}/{RUN_DATE}/`; `current/` symlink points to the active run. Controlled via `IDD_RUN_DATE` env var (default `"20260405"`).
**Why:** Pipeline was writing flat files into stage roots, making re-runs destructive with no recovery path. Versioning is required before running with new input data.
**How it works:**
- `constants.py` exposes `RUN_DATE`, versioned write paths (`PROCESSED_DATA_PATH = _PROCESSED_STAGE / RUN_DATE`), and read paths (`PROCESSED_DATA_READ_PATH` via `current/` or flat-root fallback).
- `lib/versioning.py` provides `finalize_stage()`, `finalize_all_stages()`, `tag_run()`, `list_runs()`, `list_unlinked_runs()`.
- Bootstrap: no `current/` symlink on first versioned run → `_read_path()` falls back to the stage root where the old flat outputs live.
- Within a single full run, all scripts read AND write from the same `RUN_DATE` dir; `*_READ_PATH` is for partial-pipeline re-runs reading from a prior session.
- After a successful run: call `finalize_all_stages()` to update `current/` symlinks.
- Pre-revision data: existing flat outputs remain intact at stage roots. They are accessible via the bootstrap fallback. Use `tag_run()` to create a named symlink once the first versioned run completes.
**Files changed:** `constants.py`, `lib/versioning.py`, all stage 01 and stage 02 scripts.
**Remaining bypass files (deferred — not needed for this run):**
- `05_aggregation/`: cause_as_aggregation_by_draw.py, cause_as_aggregation_by_draw_raked.py, create_as_dalys_by_draw_raked_parallel.py, make_population_hold_variables_by_draw.py
- `06_upload/`: fhs_upload_as_draws.py, create_and_combine_as_and_aa_draws.py, make_full_means_ds.py
- Legacy modules: helper_functions.py, covariate_functions.py, fhs_functions.py, counterfactual_functions.py
**Revisit if:** Partial-pipeline re-runs become common — the read/write path split may need a cleaner interface.

## 2026-04-16: Cell-by-cell regression test strategy — netCDF addendum
**Decision:** For large netCDF outputs (tens of millions of cells), use xarray `.sel()` slice-based checks rather than 5% row sampling. Select 3 meaningful (location_id, year_id) coordinate pairs covering early/mid/late years and zero/nonzero locations; compare the full age×sex grid at each (25 age groups × 2 sexes = 50 cells per slice, 150 per variable). Always include a global mean check via `xr.DataArray.mean()`. Never fewer than ~150 cells for any array-based output regardless of size.
**Why:** 5% row sampling is natural for parquet (row-oriented) but wasteful and awkward for netCDF (coordinate-indexed). Slice checks are cheaper to compute, easier to interpret on failure, and cover more meaningful coordinate combinations. The 150-cell floor ensures systematic errors (wrong merge key, dtype change, off-by-one in year filter) cannot slip through.
**Revisit if:** Never — applies to all repos and all refactors.

## 2026-04-20: Verification status at start of lsae_1285 runs
**Decision:** Document exact verification state before switching to lsae_1285 input data.

**Verified via passing pytest regression tests (cell-by-cell, rtol=1e-5, against lsae_1209 first_submission goldens):**
- Stage 02 scripts 01–09 (all Python data prep scripts): `tests/02_data_prep/`
- Stage 04 Python compute scripts: as_malaria_fractions.py, rake_dengue.py, as_dengue_shifts.py: `tests/04_forecasting/`

**Verified via compare_outputs.py (exact value comparison) but no pytest:**
- Stage 05 (aggregation): passed compare_outputs.py as of 2026-03-30 after two bug fixes

**Excluded by design (R scripts — no Python regression test possible):**
- Stage 01 (map_to_admin_2): all R/pixel processing
- Stage 03 (modeling): all R modeling scripts
- Stage 04 scripts 01–02: R forecast launchers (forecast_malaria/dengue_admin_2s)

**Never verified:**
- Stage 06 (upload): migrated to lib/ but compare_outputs.py never run and no regression tests written

**Why:** Starting lsae_1285 runs without this record makes it impossible to distinguish pipeline bugs from input-data-driven output changes.
**Revisit if:** Stage 06 verification is completed, or a full end-to-end lsae_1285 run produces new goldens — at that point re-establish regression test baselines.

## 2026-05-06: flooding is a scalar (non-draw) covariate in past inputs
**Decision:** Flooding goes into `read_shared_covariates` (loc×year scalar),
not into `read_draw_climate`.
**Why:** Flooding parquet uses `_mean_r1i1p1f1` naming — already
ensemble-averaged, no draw variance.
**Revisit if:** A draw-varying flooding product becomes available.

## 2026-05-06: dengue_suitability not in read_draw_climate defaults
**Decision:** `read_draw_climate()` does not include `dengue_suitability`.
Dengue script adds it via `extra_vars`.
**Why:** Disease-specificity boundary — malaria uses Mordecai/Villena
suitability variants instead.
**Revisit if:** Never.

## 2026-05-07: Past inputs format: flat parquet not netCDF
**Decision:** `05_build_malaria_past_inputs.py` (and future
`06_build_dengue_past_inputs.py`) output a flat parquet with one row per
valid (location_id, year_id), not a dimension-aware netCDF.
**Why:** netCDF squareness forces ~24 % of cells to be NaN (locations that
had zero pfpr in some years). Wastes space, requires explicit NaN handling
in R, makes merging awkward. Past inputs have no draw dimension, so netCDF
buys nothing. NetCDF is still the right format for forecasted inputs /
outputs where the draw dimension is needed.
**Revisit if:** Past inputs acquire a draw dimension, or if R merging of
multiple forecast files makes flat too unwieldy.

## 2026-05-07: DAH missingness in source = $0
**Decision:** NaN values in the DAH source file
(`dah_by_channel_hfa_recip_1990_2100.csv`) for a given country-year mean
DAH = $0, not missing data. Fill with 0.
**Why:** Confirmed by Joe (DAH data owner) directly.
**Revisit if:** Joe clarifies that there are cases where NaN means "data
not collected" rather than "no DAH."

## 2026-05-07: GDP population-masked locations stay in pipeline as NaN
**Decision:** 309 admin-2 locations with NaN `gdppc_mean` (all <5K
population, `pop_masking_flag=1` in source) are retained in the pipeline
output with NaN `gdppc_mean`. R model drops them via `nan_toss()`.
**Why:** All NaN GDP rows are confirmed population-masked in source.
Dropping them in Python would silently change the location set; better to
let R handle them explicitly with a warning message.
**Revisit if:** Bianca provides a fill strategy or GDP estimates for these
locations.

## 2026-05-06: Retroactive fix — versioning implementation gaps

**Decision:** Fix two gaps from the 2026-04-05 versioning rollout that were spec'd but never coded.

**Gap 1 — `finalize_artifact` not called by scripts.**
Every stage script that writes to a versioned artifact root must call `finalize_artifact(mbpc._AXX_*)` at the end of `main()`. None of the stage 02 scripts did this. Fixed in this session for all in-scope stage 02 scripts (01, 02a, 02b, 03, 04, 05, 06, 07, make_dah, make_gdppc, make_ldipc). Deferred (explicitly): stage 05 orchestrator, stage 06 scripts.

**Gap 2 — `_artifact_read` fallback never implemented.**
The original decision spec'd that if `current/` is missing, `_artifact_read` falls back to the most recent dated subdirectory. The implementation just returned `artifact_root / "current"` unconditionally. Fixed: `_artifact_read` now iterates dated subdirs and returns the most recent, with a `warnings.warn` so the gap is visible.

**New enforcements added:**
- `lib/versioning.assert_artifact_ready(artifact_root)` — call at top of `main()` for each artifact the script reads; raises `FileNotFoundError` with a clear message instead of a cryptic pandas error.
- `tests/test_versioning_completeness.py` — AST-level check that any script importing a `*_WRITE_PATH` constant also imports `finalize_artifact`. Fails CI if a new script is added without it. Known deferred scripts are listed explicitly in `DEFERRED` set.

**Why:** The missing fallback meant the pipeline silently broke whenever `current/` wasn't present. The missing `finalize_artifact` calls meant `current/` was never updated, so every downstream run was reading stale data or failing with confusing errors.
**Revisit if:** Stage 05 orchestrator and stage 06 scripts are fixed — remove them from `DEFERRED` in the completeness test.

## 2026-05-13: Model selection uses 4-method MCDM with convergence test
**Decision:** Rank candidate models with four methods in parallel — Borda
count, TOPSIS, Pareto frontier (filter, not ranker), and pairwise
dominance — and treat 3-of-4 method agreement as the robust pick. Do not
rely on any single method.
**Why:** Each method has different sensitivities to correlated metric
blocs. TOPSIS in particular is not invariant to bloc structure (a pair of
metrics with τ ≈ 0.9 effectively counts as one signal weighted ~2×).
Standard MCDM practice: use multiple methods, look for convergence, treat
disagreement as a diagnostic. Demonstrated on the 12-metric set this
session — 3 methods picked task 1, only TOPSIS picked task 22; the
disagreement aligned exactly with TOPSIS's known bloc-bias weakness.
**Revisit if:** A different MCDM method (e.g., VIKOR, weighted Borda)
shows clearly better convergence properties on this kind of metric set.

## 2026-05-14: BFGS over EFS for scam fits in this model-selection grid
**Decision:** Use `optimizer = "bfgs"` (not "efs") in scam for the malaria
PfPR neighborhood grid. Wall bumped to 240 min and threads to `-c 8`.
**Why:** EFS silently hits `maxit = 300` without converging on a
non-trivial fraction of specs in this grid (133 of 1388 finished EFS fits
had iter = 300; many more iter-1 fits may also be non-converged — the
convergence-check loop is suspect because EFS stores `$conv` as a list
rather than a logical, so `isTRUE(fit$conv)` always returns FALSE). BFGS
converges in ~4 outer iterations on the same specs. The trade is fewer
but heavier iterations: a single fit takes ~12 min instead of 22 s, but
the rate of unreliable fits drops to near zero. The wall + threading
bump absorbs the heavier-per-iter cost.
**Revisit if:** A scam release improves EFS convergence behavior, or a
different optimizer (e.g. `optim` with `method = "BFGS"`, or `"newton"`)
shows a better wall × convergence trade.

## 2026-05-14: BLAS threading must be set explicitly inside singularity
**Decision:** When submitting scam jobs from the singularity image, pass
`OPENBLAS_NUM_THREADS={n} OMP_NUM_THREADS={n}` via `sbatch --export` in
addition to slurm's `-c {n}`.
**Why:** Without the env vars, OpenBLAS inside the singularity container
falls back to 1 thread even when slurm has allocated more cores; matrix
solves run single-threaded and scam fits are 5–8× slower than they
should be. This compounded with EFS non-convergence to produce the
original "16 timed out" failures.
**Revisit if:** The singularity image is rebuilt with a different BLAS,
or upstream slurm/singularity integration changes behavior.
