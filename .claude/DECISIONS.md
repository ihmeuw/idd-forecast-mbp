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

## 2026-05-27: LSAE population read from upstream, not owned locally
**Decision:** Stage 01 (`pixel_hierarchy.py`, `pixel_urban_hierarchy.py`) does not write `population.parquet`. 02b reads from `mbpc.LSAE_POP_PATH = CLIMATE_AGGREGATES_PATH / LSAE_HIERARCHY / "population.parquet"` — the rapidresponse-team-published file. Pinned to `CLIMATE_COVARIATE_RUN_DATE` to match the rest of the climate-aggregates reads (not the `current/` symlink — silent updates break reproducibility).
**Why:** The rapidresponse team produces the canonical aggregate from the same gridded pop our pixel scripts use. Owning a duplicate is dead code and a staleness vector. Confirmed schema + coverage matches (location_id int64, year_id int64, population float32; 51,783 locs × 151 years; includes global).
**Revisit if:** Upstream stops publishing pop alongside climate aggregates, or schema diverges.

## 2026-05-28: Strict-hybrid .claude/ tracking
**Decision:** `.gitignore` includes `.claude/*` with `!STATUS.md`, `!DECISIONS.md`, `!DEAD_ENDS.md` as the only un-ignored files. Drafts, memory.md, audits, refactor prompts, common-patterns notes are local-only. Eight previously-tracked files removed from index via `git rm --cached`; files remain on disk.
**Why:** Append-only history and current project state warrant sharing with collaborators (and survival across machines); session-state and drafts don't. Keeps PRs clean of personal-workspace clutter without losing the documentation that does matter.
**Revisit if:** A specific other .claude/ file (e.g., a long-lived design doc) earns a permanent place — add a `!filename.md` line to .gitignore and `git add` it.

## 2026-05-28: 07b NaN policy uses pop_zero-first drop + real-NaN audit
**Decision:** `07b_build_malaria_prediction_locations.py` reads `mbpc.LSAE_POP_PATH`, builds a `pop_zero_mask`, and counts "real NaN" only where `pop > 0`. Locations with pop==0 across all check-window years are dropped with `drop_reason="pop_zero_all_years_in_window"`; locations with NaN in covariate values where pop > 0 are dropped with `drop_reason="real_nan"`. The `dropped_locations.parquet` audit log carries the `drop_reason` column.
**Why:** Previous "drop any loc with any NaN in window" was over-dropping. Loc 93390 has zero pop pre-2015 → NaN per-capita math → was being dropped despite having real data 2015+. New policy: NaN explained by pop==0 is expected (the upstream gridded pop says "no people lived there"), not an actionable coverage gap.
**Revisit if:** A covariate emerges that produces real NaN unrelated to pop (e.g., from a different upstream pipeline) — would need a third drop_reason category.

## 2026-05-28: Stage-01 block-skip filter (lossless task reduction)
**Decision:** New `01_map_to_admin_2/block_utils.py` provides `blocks_with_shapefile_intersections(hierarchy)` — computes the set of modeling-frame blocks whose footprint intersects ≥1 polygon in the hierarchy's raking shapefile. Applied at launcher time (`02_pixel_main_parallel.py`, `04_pixel_urban_main_parallel.py`) and at hierarchy-aggregation read time (`pixel_hierarchy.py`, `pixel_urban_hierarchy.py`). For lsae_1285: 528/784 blocks intersect (~33% Slurm task reduction). The helper also extracts `load_raking_shapes` from the worker scripts into the shared module (dedup).
**Why:** Open-ocean / Antarctic-interior blocks produce zero-row outputs at the per-block level. Skipping them at the launcher costs nothing scientifically (lossless under sum-then-divide aggregation) and saves ~33% of Slurm tasks per workflow. Same helper used at both ends ensures launcher's skip set matches hierarchy reader's skip set.
**Revisit if:** Hierarchy shapefile or modeling frame structure changes such that the intersection set becomes non-deterministic between launcher time and hierarchy time.

## 2026-05-28: MAP 202508 release adopted for all Pf covariates
**Decision:** All five Pf covariates (PfPR, incidence rate/count, mortality rate/count) read from `/mnt/team/rapidresponse/pub/malaria-denv/data/02-processed-data/malaria-{pfpr,pf-incidence-rate,pf-incidence-count,pf-mortality-rate,pf-mortality-count}/202508/202508_Global_Pf_*_{year}.tif`. `year_end` bumped from 2022 to 2024 for all five in `COVARIATE_DICT.yaml`. Files downloaded directly from MAP's WCS endpoint (`https://data.malariaatlas.org/geoserver/Malaria/ows`, coverage_ids `Malaria__202508_Global_*`) via `00_pull_raw_data/pull_map_rasters.py`.
**Why:** MAP 202508 (Aug 2025 release) extends coverage from 2022 to 2024 and supersedes the 202406 vintage. Bobby downloads from MAP directly; no IHME-side reconciliation step exists (the `GBD2023` in old filenames was MAP's own labeling).
**Revisit if:** MAP ships a newer vintage. Run `python src/idd_forecast_mbp/00_pull_raw_data/pull_map_rasters.py --release <new>` and bump paths + year_end in COVARIATE_DICT.yaml accordingly.

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

## 2026-05-29: Malaria forecast-input netCDF schema (locked)
**Decision:** One netCDF per (cause, ssp_scenario) at
`MAL_FORECAST_INPUTS_WRITE_PATH/malaria_forecast_inputs_{ssp}.nc`. Dims:
`(location_id, year_id 2000–2100, draw 0–99, dah_scenario)`. DAH
scenarios: Baseline + Constant only (Increasing/Decreasing supported by
`build_dah_array` but not in the default set). No covariate-variant
sensitivity dim — sensitivity is a re-run with swapped inputs, not a
schema slot. Transforms (logit, log) NOT applied here; R derives at
predict time. Locations come from 07b's kept set, not computed inline.
**Why:** Avoids the per-(ssp, draw, dah) parquet explosion (~10,000
files for malaria) while preserving R's ability to predict across draws
and DAH trajectories. ~6 GB total for malaria across 3 SSPs.
**Revisit if:** A new prediction-time dimension is needed (e.g.,
vaccination scenarios), or sensitivity-variant analysis grows enough to
warrant a `covariate_variant` dim.

## 2026-05-29: Stage-01 pixel output layout
**Decision:** Both `pixel_main` (per-block scratch) and
`pixel_hierarchy` (per-hierarchy aggregates) write under
`02-processed_data/GBD2023/<hierarchy>/<RUN_DATE>/...` via new
`pixel_artifact_root` / `pixel_write_path` / `pixel_read_path` helpers
in `constants.py`. Each launcher calls `finalize_artifact` after a
clean `workflow.run`. Urban scripts (`pixel_urban_main.py`,
`pixel_urban_hierarchy.py`) and urban readers are NOT affected — urban
is not GBD-release-dependent.
**Why:** Previous layout had two problems: (a) `02-processed_data/GBD2023/`
was used for the per-block scratch but the aggregate outputs lived
flat at `02-processed_data/<hierarchy>/` — asymmetric. (b) No RUN_DATE
versioning anywhere — re-runs silently clobbered. New layout is
symmetric across per-block and aggregate, versioned, and preserves the
GBD-release prefix that's semantically meaningful (MAP TIFFs are tuned
to GBD2023).
**Revisit if:** Source TIFFs are recalibrated against a future GBD
release (write to a sibling `GBD2024/` tree). Or if urban-side
versioning gets the same treatment (currently a separate open
question).

## 2026-05-29: 07b drop policy — pop-zero-first then real-NaN check
**Decision:** Locations are dropped from the malaria forecast prediction
set if EITHER (a) upstream gridded pop is 0 (or NaN) every year of the
forecast check window, OR (b) any forecast covariate has NaN where
pop > 0 in the window. Check window is 2023..2100 (forecast horizon).
Dropped locs are logged to `dropped_locations.parquet` with a
`drop_reason` column (`real_nan` vs `pop_zero_all_years_in_window`).
**Why:** Earlier "any NaN in window" policy over-dropped locations like
loc 93390 that are uninhabited 1950–2014 then populate from 2015 on —
those have no real coverage gap once pop-zero rows are masked. The new
policy distinguishes "covariate is missing where people live" (a real
upstream bug) from "covariate is NaN where no one lives" (expected).
**Revisit if:** Upstream `01_map_to_admin_2` NaN gap (~20 admin-2 locs
in urban + suitability) gets fixed; then `FORECAST_CHECK_START_YEAR`
can drop to 2000.

## 2026-05-29: Reproject cache via NearestResampler
**Decision:** `pixel_main.py` builds a `NearestResampler` once per task
(in the first inner-loop iteration), caches the source→destination
pixel-index map, and applies it via `numpy.take` + `numpy.putmask` for
all subsequent (scenario, year) iterations. Replaces the per-iteration
`to_raster(...).resample_to(pop_raster, "nearest")._ndarray` chain.
**Why:** The geometric reprojection is constant for a given (block,
covariate) — re-computing it every iteration was wasted work.
climate-data's port of the same class shows ~5–8× per-call speedup and
bit-for-bit equivalence on a 5-block multi-geography A/B. Expected
idd-forecast-mbp impact: ~25–30% per-task wall-time reduction (smaller
inner loop here than in climate-data).
**Revisit if:** Real-data A/B verification ever shows divergence vs the
legacy path (deferred — Azure outage hides jobmon at session close), or
if the destination raster geometry stops being constant across the
inner loop.

## 2026-05-29: Malaria final fit produces 3 model objects, not 5
**Decision:** `02_fit_final_malaria_models.r` saves `malaria_pfpr_mod`,
`mortality_scam_mod`, `incidence_scam_mod` only. The old
`mortality_base_scam_mod` / `incidence_base_scam_mod` (reference-age-
group base-rate variants) are dropped. The R forecaster reads exactly
these three names.
**Why:** Bobby's call. Base-rate variants were not used downstream in
the new forecast chain.
**Revisit if:** Base-rate-conditional model variants become useful
again (e.g., for AS disaggregation that doesn't go through
`as_malaria_fractions`).

## 2026-06-01: String RCP labels in mbpc.ssp_scenarios + parquets
**Decision:** Store `scenario` as string ("rcp26"/"rcp45"/"rcp85") in
gdppc_mean.parquet, ldipc_mean.parquet, med_consumppc_mean.parquet, and
mbpc.ssp_scenarios[ssp]['rcp_scenario']. Replaces previous float values
(2.6/4.5/8.5) everywhere in the codebase.
**Why:** Float-storage caused silent 0-row matches under pyarrow
`('scenario', '==', 2.6)` filters because 2.6 is not exactly
representable in float32 — the column value and the comparand diverge.
4.5 and 8.5 ARE exactly representable, so only ssp126 was affected,
which presented as 07b dropping every location for "real_nan" gdppc at
ssp126 while ssp245/ssp585 dropped only ~650 each. Equality on string
labels is always exact.
**Revisit if:** A future upstream insists on numeric RCP values; the
fix would be the float32 round-trip cast we briefly used (see
DEAD_ENDS 2026-06-01), but only with a clear comment about why.

## 2026-06-01: Decouple upstream run-dates into separate constants
**Decision:** Introduce `LSAE_POP_RUN_DATE`, `MALARIA_SUITABILITY_RUN_DATE`,
`FLOODING_RUN_DATE` as their own constants, separate from
`CLIMATE_COVARIATE_RUN_DATE`. Each is consumed by exactly one path
(`LSAE_POP_PATH`, `get_malaria_suitability_path`, the flooding probe in
05/06/07a/07b/08).
**Why:** The four artifact families live under different upstream
pipelines that publish on different cadences. Coupling them via
`CLIMATE_COVARIATE_RUN_DATE` meant bumping one accidentally moved all,
which broke `get_malaria_suitability_path` when the upstream filename
pattern changed in the 14-variant publish (the un-prefixed
`malaria_suitability_<ssp>.parquet` no longer exists at the bumped
date).
**Revisit if:** The upstream pipelines align cadences again, or if the
proliferation of `*_RUN_DATE` constants grows past ~5 separate ones.

## 2026-06-01: read_shared_covariates gains optional `variables=`
**Decision:** `read_shared_covariates` accepts a `variables: list[str]
| None = None` keyword arg. Default None preserves legacy behavior
(load every source whose path is provided). When a list is passed,
only the sources whose declared variables intersect the request are
read from disk. 08 passes `by_kind["shared_scalar"] +
by_kind["flooding"]`; 07b passes its `SHARED_VARS_TO_CHECK`. Public
mapping `SHARED_COVARIATE_SOURCES` declares which variables each
source produces; `ALL_SHARED_VARIABLES` is the union.
**Why:** Without this, 08 unconditionally loaded med_consumppc even
when it wasn't in `--covariates`, causing the float-vs-string filter
mismatch to bite even users who didn't ask for med_consumppc.
"Load what's needed" is the cleaner long-term contract.
**Revisit if:** A source needs to fan out into multiple variables
non-trivially (the current `SHARED_COVARIATE_SOURCES` mapping is 1:1).

## 2026-06-01: 07 → 07a rename + 00_prep_economic_variables.py orchestrator
**Decision:** Rename
`07_forecasted_dataframes_non_draw_part.py` →
`07a_forecasted_dataframes_non_draw_part.py` for naming symmetry with
`07b_build_malaria_prediction_locations.py`. The four un-numbered
`make_*_df.py` scripts (gdppc, ldipc, med_consumppc, dah) now expose
`main()` and are orchestrated by `00_prep_economic_variables.py
--variables {gdppc,ldipc,med_consumppc,dah}` (defaults to all four).
Each helper still runs standalone via `python make_*_df.py`.
**Why:** "07b" next to bare "07" reads as parent/child, but they're
independent siblings (07a builds non-draw covariate panels for all
locs; 07b builds the malaria prediction-location filter list).
Numbered orchestrator + helper-with-main matches the project's
"necessary scripts get numbers" convention.
**Revisit if:** A genuine "07c" / "07d" appears; consider whether
07a/07b/07c should consolidate.

## 2026-06-01: Single JSON malaria model registry, R/Python parity
**Decision:** Flat singleton JSON at
`03-modeling_data/malaria_model_registry.json` is the source of truth
for which malaria-model run is "best." One array of run records; one
carries `best: true`. R writes via
`lib/model_registry.R::upsert_malaria_model_run` (atomic write +
single-best invariant); Python reads via
`constants.read_malaria_model_registry()` and
`constants.get_malaria_model_run_date(best=True, run_date=None)`.
Each record includes provenance: rdata_file, models, pfpr/mort/inc
formulas (deparse1'd), suit_variant, and `parquet_path` resolved
through any `current/` symlinks to the dated path at upsert time so
the entry pins the actual data the model was fit against.
Format: JSON (jsonlite in cluster R image + json in Python stdlib —
zero new deps).
**Why:** Hardcoded `model_date` in forecast scripts was brittle; the
registry lets R record canonical state and any consumer read it
without re-deriving. JSON over YAML because both R and Python read
JSON without adding a dep.
**Revisit if:** Other artifact families need similar indexing —
current pattern is one registry per cause (dengue will get a sibling
`dengue_model_registry.json` + helpers when its fit script lands).
If that multiplies, consider a single multi-cause registry with a
`cause` discriminator.

## 2026-06-01: Fit-verify-flag-best workflow for malaria model selection
**Decision:** New runs go into the registry with `best: false` by
default (`FLAG_AS_BEST <- FALSE` in `02_fit_final_malaria_models.r`).
Marking a run as best is a deliberate second step: re-run the fit
with `FLAG_AS_BEST <- TRUE` after verifying the model. Upsert demotes
whatever was previously best. The forecaster reads the registry via
`get_malaria_model_run_date(best=TRUE)` and will fail loudly if no run
is flagged — that's intentional, surfaces unverified runs rather than
silently picking one.
**Why:** Separates fitting from canonicalizing. A run is recorded for
provenance without being trusted as the production input. The
verification check (concordance, sanity) happens outside the script.
**Revisit if:** Verification becomes scripted (e.g., automated
concordance gates) such that "best" can be determined inside the fit
script itself.

## 2026-06-02: Stage-04 rocket built as sourceable helpers + main-guard
**Decision:** `forecast_malaria_admin_2s_rocket.r` is structured as small
argument-based helpers (`pull_grid`, `read_forecast_inputs`,
`read_rake_year_observed`, `classify_zero_burden`, `fallback_level`,
`make_A0_af`, `apply_shift`, `make_predict_one_draw` factory,
`assemble_arrays`, `write_forecast_netcdf`) plus a `main()` that is only run
when the file is the Rscript `--file` entrypoint (R's `if __name__ ==
"__main__"`). No module-level pipeline globals.
**Why:** Lets a scratch harness `source()` the rocket and exercise every
function at tiny scale (the 8 test points in `.claude/FORECAST_04_DRAFT_NOTES.md`)
without running the full task. Caught two real bugs offline/at-probe that a
monolithic script would have surfaced only at full scale.
**Revisit if:** Helpers are needed by another script — promote them to a
`lib/` module rather than re-sourcing the rocket.

## 2026-06-02: apply_shift is order-safe by construction (no non-equi join)
**Decision:** The rake shift aligns the per-location shift back onto the
prediction frame by `match(dt$location_id, ...)`, never by row position, and
selects each location's rake-year row via a boolean mask
(`dt$year_id == unname(rake_years_vec[...])`) rather than a data.table
non-equi/rename join. Prediction runs over `read_years` (⊇ rake year) then
trims to `forecast_years`, so the rake-year row is always present.
**Why:** The draft's `dt[rk, on=.(location_id, year_id==rake_year)]` join was
the single most fragile piece (flagged in the draft notes). Test 4a proves the
new form is exact under per-location rake years AND row shuffling; test 4b
proves end-to-end inc/mort at the rake year equal the observed anchors to 0.
A latent bug was found here: `rake_years_vec[as.character(...)]` returns a
*named* vector, so `dt[year_id == ry]` mis-read the names as columns — fixed
with `unname()`.
**Revisit if:** Never — positional alignment after a merge/sort is the bug
this guards against.

## 2026-06-02: Pin singularity image ihme_rstudio_4523, not latest.img
**Decision:** The launcher pins `-i /mnt/share/singularity-images/rstudio/
ihme_rstudio_4523.img` (the image `latest.img` resolved to on 2026-06-02), not
`latest.img` itself. The previously-inherited `4222` (from the drafts + the
dengue launcher) is stale and lacks `tidync`.
**Why:** `latest.img` silently repoints when ops bumps it, so a re-run could
land on a different R/package set and change forecast results — the exact
version-drift trap. Pinning the resolved image makes a committed launcher
reproducible (same rationale as pinning `model_run_date`). The stale 4222
caused the first probe to die at `library(tidync)`; host `Rscript` had tidync
so offline tests didn't catch it — only a real cluster submit did.
**Revisit if:** A needed package is missing from the pinned image (resolve via
`R_LIBS_USER=~/packages` or bump the pin deliberately), or a deliberate image
upgrade is wanted — bump the pin and note it here.

## 2026-06-02: Stage-04 forecast resourcing from staged-draw probe
**Decision:** Per-task Slurm ask = `-c 10 --mem=80G -t 120`. `suit_dt` is
`setkey`'d on `draw` once in the parent before mclapply forks.
**Why:** A staged probe (full universe + full year window, 5/10/20 draws) gave:
read ≈ 125s fixed, predict ≈ 130s per 10-fork wave, full 100-draw task ≈ 25 min
and ≈ 42 GB peak RSS (kept=9918 locs after zero-burden drop). `--mem=80G` ≈ 1.9×
peak; `-t 120` ≈ 5× runtime (buffer for NFS / co-scheduled tasks). `setkey`
was time-neutral but dropped peak RSS ~8–10 GB (it stopped each of 10 forks
from independently building a secondary index on the 209.85M-row table).
predict is memory-bandwidth-bound (5→10 forks: 50→135s), so >10 cores won't
scale linearly — parked as a future optimization (fewer cores / chunked
per-draw reads) rather than chased.
**Revisit if:** The kept-location count, year window, or draw count changes
materially, or the predict step is re-profiled with a different core count.

## 2026-06-03: Keep stage-04 forecast at --mem=60G (measured against the real run)
**Decision:** Leave the launcher at `--mem=60G`. The 2026-06-03 real run
(array 47891353, 100 draws) finished 3/3 with ACTUAL per-scenario peak RSS:
ssp126 43.4 GB, ssp245 47.5 GB, ssp585 41.9 GB (min 41.9 / median 43.4 /
mean 44.3 / max 47.5), wall 29–31 min. 60G clears the 47.5 GB peak with ~12 GB
(1.26×) headroom and consistently sits under 50 GB.
**Why:** Bobby's call — the usage is consistently <50 GB, so the deliberately
tight 60G ask is justified rather than padding to 64G+. The per-scenario peaks
are recorded (here + in the launcher comment + resourcing memory) so the
headroom is visible at the point of change.
**Revisit if:** The rocket is changed to do MORE per task — more draws, running
Constant dah alongside Baseline, or adding outcome variables — any of which
would raise the per-task peak toward/over 60G. Re-measure and bump then.

## 2026-06-08: Dengue past-inputs split into 06a / 06b / 07c
**Decision:** Dengue past-inputs are built in three scripts:
`06a_build_dengue_fit_locations.py` (fit location_ids via `dengue_fit_location_ids`),
`06b_build_dengue_past_inputs.py` (the past-inputs parquet; renamed from `06`,
rewritten nc→parquet), `07c_build_dengue_prediction_locations.py` (prediction
locations via `dengue_prediction_location_ids`, mirrors malaria 07b). 06b is dense
over age/sex within included loc-years (not valid-rows-only), draw-000 only, stores
RATES + population (counts dropped — recoverable as rate×pop), and carries
A0/region/super_region ids. Skips DAH + med_consumppc; single suitability.
Endemic-A0 thresholds (`dengue_{fit,pred}_{mort,inc}_threshold`) live in constants,
all 0.0 for now (exploratory). Fit = A0's 2023 all-age deaths>thr AND cases>thr;
predict = A0 any-year deaths>thr AND cases>thr.
**Why:** Fit locations are needed BEFORE building past-inputs, and 07c
coverage-checks the covariates 06b defines → dependency 06a → 06b → 07c. Dense AS
+ rate/pop matches the AS grain dengue models need while staying flat-parquet (no
draw dim — past draws identical).
**Revisit if:** thresholds tighten, or the dengue fit is wired to read
`dengue_past_inputs.parquet` directly (would fold transforms into that chain).

## 2026-06-09: Dengue AS disaggregation = base rate × GBD age-sex RR (no "rest" model)
**Decision:** Age-sex dengue incidence = `population × exp(base_log_inc_rate) ×
rr_inc_as`, where `base_log_inc_rate` is a regression on a SINGLE reference age/sex
group and `rr_inc_as` is the GBD age-sex relative risk (= rate ÷ reference group's
rate). Reference = `cause_map['dengue']` (age_group_id 3, sex_id 1); RR built from
year 2022 at most-detailed-GBD locations by `make_as_md_gbd_dengue_df.ipynb`.
Canonical impl: `lib/processing/disaggregation.py::disaggregate_age_sex_dengue`
(used by `as_dengue_shifts.py`). There is NO second / "rest" regression.
**Why:** The age/sex shape is an empirical GBD pattern, not separately modeled.
Confirmed by reading the production worker + the RR builder.
**Revisit if:** the dengue model moves to predicting age/sex directly.

## 2026-06-09: Dengue fits judged on TRACKING, not correlation
**Decision:** Select dengue models on whether super-region predicted rates *track*
observed — operationalized as slope(predicted vs a heavily-smoothed lowess through
observed) ≈ 1 (intercept ≈ 0) — not on correlation. Data are stochastic, so
concordance with the raw observed is neither expected nor wanted. More tracking
metrics may follow.
**Why:** Current fits show high correlation but flat/level-shifted predictions that
don't follow the observed trend (super-region obs-vs-pred plots, 2026-06-09).
**Revisit if:** a better tracking metric emerges. See memory `dengue-model-tracking-goal`.

## 2026-07-01: Malaria model-selection runs on a jobmon (idd-tools) workflow, not SLURM arrays
**Decision:** The malaria PfPR model-selection grid is orchestrated by
`fit_malaria_models_orchestrator.py` (idd-tools jobmon) driving the R worker
`select_malaria_models_rocket.r`, replacing the `01a_fit_prelim_..._launcher.r`
sbatch-array approach. Each spec fans into 4 cells (IS + within/tempA/tempB OOS).
**Why:** jobmon gives native per-task resources, retries with resource-scaling, and
dependency/DAG support the array launcher lacked. The two-cell design (IS computed
once per spec; OOS experiments as separate cells) stops redundant IS refits.
**Revisit if:** the grid shrinks enough that a plain array is simpler again.

## 2026-07-01: Prep is the single source of truth for spec formulas
**Decision:** `build_malaria_neighborhood_specs.r` bakes each spec's full formula
(per-term k / bs codes) into `spec_table.parquet$formula_text`; the worker fits that
string via `as.formula` and has NO `build_term`/`build_formula`/`K_DEFAULT` of its own.
**Why:** Two parallel formula builders (prep + worker) had to be hand-synced ("keep
K_DEFAULT in sync with prep" was a live hazard). One authority removes that whole
drift class. The worker always fits FE-present and `formula_text` already includes
A0_af, so nothing else needs the spec object for the formula.
**Revisit if:** the worker needs a formula variant prep doesn't emit — then emit it
from prep, don't rebuild in the worker.

## 2026-07-01: Per-(var, form) K, encoded in var_forms
**Decision:** `var_forms` values are named vectors mapping form-code → basis dimension
K (NA for linear); `K_DEFAULT` is only a fallback. K is per-(covariate, form), so a var
offered as both `mpi` and `smooth` can carry different K. Added an unconstrained
`smooth` form (`s(var, k=K)`, no `bs=`).
**Why:** K was a single global; basis dim is a per-term property. Named-vector encoding
keeps K next to the form (can't drift from a separate list) and leaves `n_smooths` /
`matches` unchanged (the spec still stores plain form strings).
**Revisit if:** never — this is the natural home for K.

## 2026-07-01: n_smooths-aware bundling of selection cells
**Decision:** The orchestrator bundles cells into serial tasks by (cell, n_smooths),
with bundle size + runtime from a `CALIB[(template, n_smooths)]` table and memory from
per-template `MEM`. No special-casing of oos_within — it's a per-combination value
(most within = 1; within n_smooths=2 = 2). The worker loads data once per task and
loops its bundle's specs.
**Why:** Per-cell cost swings ~4× across n_smooths, so a FIXED bundle size would blow
high-n_smooths cells into 20-min mega-tasks while under-packing low ones. n_smooths-aware
sizing normalizes task time and amortizes the one-time data load. CALIB is regenerated
from a probe (currently wf 595991).
**Revisit if:** the fit cost profile changes (new covariates, R image, or optimizer) —
regenerate CALIB from a fresh probe (≥2 specs/level).

## 2026-07-01: Select the malaria spec on temporal OOS + parsimony, NOT in-sample AIC
**Decision:** Rank the PfPR spec on temporal-OOS PfPR-space skill (tempA/tempB
`oos_pfpr_r`) and pick the most parsimonious spec in the (flat) top cluster. Do NOT use
in-sample AIC.
**Why:** On wf 595991, in-sample AIC favors the complex specs (n_smooths 5–7) by ΔAIC
~7000, but those forecast WORSE on all three OOS experiments — AIC rewards the in-sample
overfit. The temporal-OOS neighborhood is flat (top specs within <0.001 `oos_pfpr_r`), so
parsimony is the tie-breaker. within-country OOS (~0.85) is easier than temporal (~0.79);
temporal is the honest forecasting number.
**Revisit if:** a spec's temporal OOS clearly separates from the cluster, or the goal
shifts from forecasting to in-sample description.

## 2026-07-01: EFS (not BFGS) for the scam selection grid — reverses 2026-05-14
**Decision:** Use `optimizer = "efs"` for the model-selection grid. This reverses the
2026-05-14 "BFGS over EFS" decision.
**Why:** With the modeled-rows data filter (`malaria_inc_count >= 1 & malaria_pfpr >=
0.0001`) and the current grid, efs and bfgs agree to 5–6 sig figs on the fits and efs is
~2× faster. The EFS non-convergence that drove the BFGS decision was a property of the
*unfiltered* 313k-row data; on the filtered data that pathology is gone.
**Revisit if:** EFS non-convergence reappears (new covariate/grid) — bfgs remains the fallback.

## 2026-07-01: Keep scale_up_on_retry=True for the bundle worker
**Decision:** `submit_with_manifest(..., scale_up_on_retry=True)` (the default) for this
workflow, despite it being a bundle worker with atomic per-task writes.
**Why:** The idd-tools docstring suggests `False` for `subtask_skip` workers (a retry has
less work). Ours does NOT checkpoint within a bundle — a retry re-runs the whole bundle —
and the observed failures are genuine runtime under-provisioning, so the retry needs MORE
time. `False` would re-timeout at the same limit and burn all 3 attempts into dead tasks.
**Revisit if:** the worker gains within-bundle `subtask_skip` (then `False` becomes correct).

## 2026-07-07: Malaria PfPR forecast = base covariates + country FE, 2023-shift-anchored (FE is moot in the forecast)
**Decision:** Fit malaria PfPR WITH the country fixed effect and forecast by shifting each
location's prediction to its observed 2023 anchor. Do NOT pursue a lagged admin-0 PfPR
covariate as an FE replacement for malaria.
**Why:** Under the per-location 2023-anchor shift (logit space), the FE and global intercept
are additive constants that cancel: shifted(y) = raw(y) − raw(2023) + obs_2023 = obs_2023 +
[smooths(y) − smooths(2023)]. The FE contributes nothing to the forecast TRAJECTORY (only a
raw level the shift discards), so it was never a forecasting obstacle and never needed
replacing. It still earns its place in the FIT (absorbs between-country means → clean
within-country smooths). The lag covariate was motivated by a non-problem.
**Revisit if:** working on DENGUE (re-derive whether the shift-cancellation holds for the
dengue forecast's anchoring — this is malaria-specific), or if a forecast drops the 2023-anchor
shift. The correctly-framed test (base+FE vs base+FE+lag, scored on SHIFTED preds) is built and
deferred, not answered.

## 2026-07-08: Malaria GDP = V5 `reference` scenario applied to all RCPs
**Decision:** `make_gdppc_df.py` reads the V5 income-forecast CSV, keeps ONLY the `reference`
scenario, and emits it under all three RCP labels (rcp26/45/85). GDP is decoupled from the
climate scenario for now.
**Why:** Bobby wanted one GDP trajectory for every SSP. Replicating `reference` under all RCP
labels means the downstream per-RCP selection returns reference GDP with ZERO downstream change.
**Revisit if:** GDP should vary by scenario again — emit each source scenario under its mapped RCP
label (GDPPC_SCENARIO_MAP retained for exactly this).

## 2026-07-08: Multi-formulation malaria fit/forecast keyed by an arbitrary registry string
**Decision:** `02_fit_final_malaria_models.r` fits a FORMULATIONS list, saving each as
`{run_date}_{id}_malaria_models.RData` and registering under run_date `{run_date}_{id}`
(best=FALSE). The 04 launcher forecasts a `MODEL_IDS` list — one output dir per formulation, no
auto-finalize.
**Why:** `upsert_malaria_model_run` validates run_date only as a non-empty string, so many
same-day formulations coexist and each is forecastable via `MODEL_RUN_DATE`. Lets us compare
candidate specs (and an ensemble) before committing a single best.
**Revisit if:** We settle on one production model — flag it best + finalize its `current` normally.

## 2026-07-08: `mean_low_temperature` as a single-realization (`climate_mean`) forecast covariate
**Decision:** New `climate_mean` covariate kind (loc×year): stage-08 reads the climate draws and
stores the ensemble MEAN. `mean_low_temperature` uses it (overrides its `climate_draw` default).
**Why:** Consistent with every other non-suitability forecast covariate (gdppc/DAH/flood are
single-trajectory; only suitability carries draws), matches the fit (one temp value per loc-year),
and avoids adding a second ~20 GB draw-varying array near the rocket's 60 GB budget.
**Revisit if:** Temperature should carry climate-draw uncertainty — delete the override (reverts to
`climate_draw`), re-run 08, bump rocket mem + re-probe.

## 2026-07-08: Aggregate-level rates use the FULL-population denominator, never aggregation
**Decision:** Aggregating admin-2 malaria forecasts to super-region/global, the rate is malaria
count (summed) ÷ the TRUE full population of the level, read straight from the population df's own
level row — never ÷ a summed endemic-admin-2 subset.
**Why:** Same principle as the hybrid SDG deliverable. A partial (summed-endemic) denominator both
inflates rates and breaks observed→forecast continuity at the 2023 rake year (worst where many
locs are zero, e.g. North Africa). Full population is already in the population artifact at every
level.
**Revisit if:** Never — this is the correct denominator.

## 2026-07-10: Forecast aggregates stay internally consistent (admin-2 roll-up); never reconciled to GBD's inconsistent region+ rows
**Decision:** Every aggregate in our finalized forecast products (admin-1 → global) is the pure
count-space sum of its admin-2 children at every level (stage-05 `finalize_forecast` cores roll up
via `roll_up_hierarchy`). We NEVER read the observed file's stored region/super-region/global rows
into our products or the observed overlay — always roll up from admin-2, the internally-consistent
base the rocket anchors each forecast to.
**Why:** The raw GBD 2023 extract (`gbd_2023_malaria_aa.csv`, and therefore
`aa_full_malaria_df.parquet`) is NOT internally consistent above the country level. Verified 2026-07-10
(Bobby traced it to the raw GBD data, not our processing): 7 of 21 regions have a region `val`
ABOVE the sum of their constituent countries — Oceania largest at +5.7% (~127k cases/yr, every year
2000–2023). By level, totals are self-consistent within {admin-2, admin-1, country} = 242.93M (yr
2000) and within {region, super-region, global} = 243.13M, but the two blocks disagree by ~0.08% at
the country→region seam. Reading GBD's stored region+ rows would (a) inherit that inconsistency and
(b) create a 2023 splice jump between our pure-sum forecast and the higher observed aggregate.
**Consequence (intentional):** Our region/super-region/global forecast numbers sit ~0.08% (up to
5.7% for Oceania) below GBD's observed region rows. Do not "reconcile" this — it means we're
self-consistent and GBD isn't. The observed overlay for plots is likewise rolled up from admin-2
(as `quick_gbd2023_timeseries` / `_gen_formulation_rcp_plots` already do), so the 2023 splice stays
continuous.
**Revisit if:** GBD publishes internally-consistent aggregates, or a deliverable explicitly requires
matching GBD's region envelope — that raking belongs in a one-off (like the SDG hybrid), never the
standard finalize.

## 2026-07-10: Dengue age/sex CFR = fitted `as_id`, fit at fhs, raked to observed age/sex; never invent deaths
**Decision:** Dengue CFR is age/sex-specific via the fitted `as_id` term (`logit_cfr ~ log_gdppc +
as_id + A0`, mirroring `mod_cfr_all`). Because age/sex CFR at admin-2 is memory-infeasible (~29M-row
design matrix), fit at the **fhs age/sex grain** and use the additive model to get age/sex =
`base_pred + as_id_offset` (predict the base group only + a ~50-value offset). Rake each
(loc, age, sex) to the OBSERVED age/sex CFR over the anchor window; where no observed CFR in (0,1)
over the window → cfr := 0 → mort := 0.
**Why:** A single base-group CFR broadcast flat across ages is catastrophically wrong at the
aggregate (super-region CFR ~30× high in SE Asia, ~200× low in Sub-Saharan Africa) — CFR varies
enormously by age. Fitting WITH `as_id` also de-biases the gdppc/A0 coefficients (the de-biased
gdppc drives the forecast CFR trend, since the rake absorbs the as_id/A0 level). No-invented-deaths
is Bobby's explicit constraint. Verified: post-fix super-region CFR tracks observed within ~10–20%.
**Revisit if:** an `as_id`×covariate interaction is added (breaks the additive base+offset shortcut →
revert to a full age/sex predict), or CFR can be fit at admin-2 with adequate memory.

## 2026-07-10: Dengue fit + registry + forecaster are R; 08b inputs + finishing are Python
**Decision:** The dengue pipeline's fit, model registry, and forecaster are **R** (the forecaster
reads the 08b nc, exactly like malaria's R rocket); the 08b forecast inputs and the finishing are
Python. Resolves open decision #7 in `FORMALIZE_DENGUE_FORECAST_PIPELINE_PROMPT.md`.
**Why:** Malaria's proven split is R-forecaster → Python-`05_aggregation`, with netCDF the
language-agnostic handoff. The notebook's Python/pyGAM prototype is exploration; only the
language-agnostic pieces cross to the pipeline (anchor-shift math, count-space aggregation, the 08b
input contract), not the pyGAM-specific bits.
**Revisit if:** the team standardizes the whole pipeline on one language.

## 2026-07-10: Each dengue formulation forecasts at its OWN grain; coarser grains roll up 08b pop-weighted
**Decision:** Fit-grain == forecast-grain. Since 08b covariate inputs are admin-2 (lsae), a
coarser-grain (e.g. fhs) formulation forecasts by first rolling the admin-2 08b covariates up to its
grain **pop-weighted**, then predicting/shifting/aggregating there.
**Why:** Applying a model at a grain other than its fit grain is wrong; the pop-weighted roll-up
reproduces the past (`06b`) fhs covariates to ~1e-16 (verified), so it's consistent. This is the
"configurable finest location set" the pipeline wants.
**Revisit if:** we decide to standardize all forecasts at one grain for apples-to-apples comparison.

## 2026-07-10: Size the final malaria selection run from the 599278 census, not a fresh probe
**Decision:** Use the completed full run 599278 (5,837 tasks @ conc 5000) as the authoritative per-task cost source for sizing the next full run. A dedicated calibration probe is not needed and is worse: it runs at a different (lower) concurrency, so its NFS-load contention doesn't match the real run (isolated load 2.7s vs 42.6s at 85-way vs 5000-way real). The probe's value was as an idd-tools machinery TEST, not a sizing input.
**Why:** 599278 already measured real-scale contended cost — exactly what the next run experiences. It confirmed we OVER-allocate (runtime asked 2-13x actual; memory ~2x = the intended floor), not under.
**Revisit if:** the spec set or worker changes materially (599278 no longer like-for-like), or the run's concurrency changes enough to shift contention.

## 2026-07-10: Teardown segfault is benign; io=2 is NOT its fix
**Decision:** Treat the post-`fin` `memory not mapped` segfault as benign/tolerated (output always written before it; jobmon retries clear it). `arrow::set_io_thread_count(2L)` is kept (arrow's floor) but is NOT the fix — verified it still segfaults at io=2. Real cause-agnostic fix (deferred): gate task success on the output file existing.
**Why:** inherent arrow/jsonlite DLL-unload race under singularity, independent of thread count (io<2 warning was a red herring). Data never at risk; 599278 completed through the churn.
**Revisit if:** churn/FATAL rate becomes material at scale -> implement output-gating in the worker command.

## 2026-07-14: Malaria model registry records the inc_count/pfpr fit thresholds
**Decision:** `02_fit_final_malaria_models.r` exposes run-level `INC_COUNT_THRESHOLD` /
`PFPR_THRESHOLD` knobs (applied where the old hardcoded `inc_count>=1 & pfpr>=1e-4` filter was) and
records both on every registry record's `extra`. Backfilled all 7 prior entries (2026_06_03 = 0/0
full-data; 2026_07_08_f1..f6 = 1 / 1e-4) via a validated atomic JSON patch (one-time; R stays the
normal writer).
**Why:** The "hybrid" (2026_06_03) model was fit on FULL endemic data while f1-f6 used the filter, and
the registry couldn't tell which. Recording the thresholds makes each model self-documenting.
**Revisit if:** thresholds become per-formulation rather than per-run — move them into each FORMULATIONS entry.

## 2026-07-20: DAH corrected to FGH_2026_July; a future-only DAH error leaves the fitted model unchanged
**Decision:** The malaria DAH covariate source is
`/mnt/share/resource_tracking/forecasting/dah_channel_HFA/FGH_2026_July/dah_by_channel_pa_recip_1990_2100.csv`
(pa file, sum of the 11 `mal_*` program areas — provably ≡ the `hfa='mal'` total). It corrects an
erroneous prior drop (Goalkeepers-2026 20260713) whose error was FUTURE-only (2000–2024 identical;
2025+ ~5–7% too high). PRINCIPLE: because the malaria model is fit on PAST data (2000–2023) and DAH
enters as a covariate, a DAH correction confined to future years does NOT change the fitted model —
only the forecast. Verified: `2026_07_20_hybrid_fghjul` is numerically identical to `2026_07_14_hybrid`
(same iters/convergence/formulas).
**Why:** Determines the minimal correct re-run — the refit is redundant except for cutting a clean key;
the real change is 08a forecast-inputs → forecast → aggregate. (A new key is still cut so the corrected
forecast doesn't collide with the erroneous run's output dir.)
**Revisit if:** a DAH correction ever touches historical (≤ fit-year) DAH — then the model genuinely refits.

## 2026-07-14: Hybrid deliverable reads a specific forecast dir; delivery to /ihme is a Bobby-run cp
**Decision:** `hybrid_sdg_malaria_incidence_rate.py` reads a chosen forecast via
`--forecast_run_date <model_key>` (default `current`) so a new run never repoints the shared `current`
symlink; sensitivity takes `--old_delivered_plotdata` for the `old_delivered` overlay and `--rake_years`
to limit the pin variants. Products are written to a dated project dir
`hybrid_deliverable/lsae_1285/<RUNDATE>/`; the client-facing deliverable at
`/ihme/forecasting/data/37/future/incidence/<DATE>_malaria_incidence_goalkeepers/` is placed by BOBBY
(a CC cannot write under /ihme outside /ihme/homes per org file-safety policy — it stages the file and
hands over the mkdir+cp).
**Why:** Non-destructive (old deliveries preserved, `current` untouched) and org-compliant.
**Revisit if:** the org file-safety policy changes, or the deliverable's canonical home moves.

## 2026-07-31: 2025_08_28 is the first-submission forecast run
**Decision:** The comparison baseline for all "old run" / "previous run" figures and tables is
`05-upload_data/upload_folders/2025_08_28/`. That is the run behind the FIRST SUBMISSION, and it
is the run the manuscript figure notebooks in `notebooks/10_manuscript/02_figures/` read (pinned
as the string `2025_08_28` in `counterfactual_functions.py` and again at each notebook call site).
**Why:** `upload_folders/` holds eight dated runs plus a `GK_2025_11_02` Goalkeepers run that is
chronologically LATER but is not the submission run — picking by newest date gives the wrong
comparison. Recording the run date explicitly removes the ambiguity.
**Contents:** 21,210 arm directories, each `draws.nc` with dims (location_id 51590, year_id
2022-2100, draw_id 100), variable `val`, all-age COUNTS at every hierarchy level (so global and
super-region are already present, no re-aggregation needed). Arms cover cause x measure
(incidence/mortality/yll/yld/daly) x metric x ssp x dah, plus five holds encoded as an ssp-token
suffix: `_hold_as_structure`, `_hold_gdppc`, `_hold_population`, `_hold_suitability`, `_hold_urban`.
**Caveats:** counts only, so rates must be derived with the population artifact; starts at 2022
while the current run starts at 2023; and its 2023 global mortality (706,112) differs from the
current run (669,712), so it is anchored to a different observed vintage -- an old-vs-new gap
includes that baseline shift, not just a model difference.
**Revisit if:** a later submission supersedes it, at which point record the new run date here
rather than changing what "old run" means implicitly.

## 2026-07-31: the four sensitivity definitions, recovered from the OLD hold chain
**Decision:** Reuse the OLD chain arithmetic verbatim (from
`05_aggregation/OLD_make_population_hold_variables_by_draw.py:97-161`), with the reference year
moved from 2022 to 2023 to match the current anchor. Both population and age-structure holds are
pure multiplicative rescalings of AGE/SEX counts:
```
as_ref_frac = as_pop[ref] / as_pop[year]      # per (loc, age, sex, year)
aa_ref_frac = aa_pop[ref] / aa_pop[year]      # per (loc, year)
hold_population   = count * as_ref_frac
hold_as_structure = count * as_ref_frac / aa_ref_frac
```
**Why the age-structure one is not a redistribution:** dividing the two ratios cancels total
population and leaves `as_share[ref]/as_share[year]` -- each age/sex group reweighted by the change
in its SHARE of the population. Total population still evolves; only the structure is frozen. The
per-age weights differ, so the age groups do NOT sum back to the original all-age total: the old run
shows +5.31% at global 2100. An earlier claim in this session that it leaves all-age untouched was
WRONG.
**Population hold is accounting-only, confirmed empirically:** the finishing-side hold (freeze the
population frame that supplies both the rate->count multiplier and the aggregate denominator)
reproduces -67.1% at global 2100 against the old run -68.1%. No re-forecast, no per-capita covariate
rebuild needed. The earlier claim that -68% was too large to be an accounting effect was WRONG.
**GDP and DAH DO need a forecast**, because they are model covariates: GDP enters the pfpr model as
`gdppc_mean` and inc/mort as `log_gdppc_mean`, so freezing it is a rocket-side `--hold-covariate
gdppc`. Verified +66% on the global 2100 mortality RATE, with 2023 bit-identical.
**DAH: two DIFFERENT experiments, do not conflate.** `--dah-scenario Constant` (holds DAH TOTAL at
2023, so per-capita falls as population grows) gives +3.0% at global 2100 -- and reverses sign around
2065, averting ~27k deaths/yr near 2030 before costing ~30k/yr by 2080. The OLD run`s `_hold_DAH` arm
(-18.9%) is NOT this: it froze per-capita DAH, available now as `--hold-covariate dah`. The old
`dah_scenario=Constant` arm was -1.6%; the sign flip vs our +3.0% follows from the FGH July DAH
correction changing the future baseline trajectory.
**Implementation note for age-structure:** the factor must be formed at admin-2 and carried through
the existing count-space roll-up, NOT applied at an aggregate level, because the old chain applied it
before aggregating. Under our normalized disaggregation the all-age factor per admin-2 is
`sum_as( rr_share[a,s] * as_share[ref][a,s] / as_share[y][a,s] )`, computable from the observed
age/sex risk shares plus the `as_population_fraction` column already in
`as_2023_full_population_df.parquet` -- so it needs no age/sex product and no forecast.
**Revisit if:** the anchor year moves off 2023, or the disaggregation stops being normalized to the
all-age total (which is what makes the scalar-factor shortcut exact).

## 2026-08-03: Dengue runs at FHS most-detailed and up; admin-2 buys no information
**Decision:** The default location grain for dengue fitting AND prediction is **FHS
most-detailed and up** (513 locations: 204 countries with no subnational + 280
subnationals + 29 aggregate rows). Admin-2 (LSAE most-detailed, 47,450) stays a
supported parameter — the machinery is kept alive — but is no longer the default and
is not required by any deliverable. This **closes open decision #1** in
`FORMALIZE_DENGUE_FORECAST_PIPELINE_PROMPT.md`, which still lists finest-grain as
unresolved. Contrast malaria, which has genuine admin-2 data and where admin-2 IS the
natural grain; this decision is dengue-specific and must not be generalized to malaria.

**Why:** Dengue has no real admin-2 data. Admin-2 dengue was distributed DOWN from
national data using suitability, so descending to admin-2 and re-aggregating cannot
recover information that was never there. Three supports, of differing strength — the
first two are general, the third is scoped:

1. **Covariate identity (exact, general).** The full admin-2 `dengue_suitability`
   aggregates *exactly* to the FHS national value (max diff ~1e-14). Admin-2 is a
   re-aggregation of the same national number, so for anything ultimately aggregated to
   FHS the admin-2 covariate carries literally zero additional signal. This is an
   identity, not a statistical finding.
2. **Circularity (general).** The admin-2 dengue that defines the fit filter was itself
   distributed down from national data via suitability, so the surviving high-suitability
   admin-2 units "predict" the dengue that selected them. Self-fulfilling, not skill.
3. **Empirical skill (scoped).** Apparent LSAE spike-timing skill (r ~ 0.72 High-income,
   ~0.9 Argentina) collapses to r ~ 0.13 / 0.25 under production-faithful aggregation —
   within noise of FHS (~0.01 / 0.19). The high r appears ONLY when aggregating over the
   dengue-filtered fit set, whose admin-2 membership varies year to year: a shrinking
   numerator over a full-population denominator deflates the rate exactly when observed
   is low, faking spike-tracking. Scoped to model A and the High-income super-region,
   n = 24 years.

**Sizing consequence:** the hard deliverable (FHS-hierarchy age/sex/year/location/draw
cases and deaths) is ~200M cells per measure per ssp at FHS = ~1.6 GB float32 for cases
+ deaths — an ordinary file. The same object at admin-2 would be ~74 GB per measure.
The "must we avoid materializing age/sex x draw?" problem is therefore an admin-2-only
problem, not a dengue-pipeline problem.

**Provenance / why this is being written now:** the analysis was done and documented in
`reports/03_modeling/high_income_spike_diagnostics.ipynb` (header, section 2 findings,
and Synthesis) but was never promoted to DECISIONS, STATUS, or memory. The dengue prompt
demotes that notebook to "the earlier aggregation-artifact investigation" and still
presents finest-grain as an open question, so a later session (2026-08-03) re-derived the
grain question from scratch and got it wrong. Documenting a conclusion inside the
notebook that produced it is not sufficient; conclusions must be promoted to the durable
record. Related: on 2026-07-31 Bobby stated "the saved age/sex/draw product is just FHS",
also never promoted.

**Revisit if:** real admin-2 dengue data becomes available that is NOT derived from
national data via suitability — that is the only thing that would make a finer-resolution
test meaningful. Also revisit if a deliverable ever genuinely requires admin-2 output, in
which case the grain parameter is already there; only the age/sex writer needs chunking.

## 2026-08-03: The prediction frame is decoupled from the fit design matrix
**Decision:** What we predict ON is chosen by what survives the anchor shift — never by
what appears in the fit formula. A model fit WITH `as_id` (or any age/sex term additive
in link space and constant in time) is predicted with that term pinned at a single level,
on a `(location, year, draw)` frame. It is never expanded to
`(location, year, age_group_id, sex_id, draw)`. Age/sex enters exactly once, at the end,
from the observed anchor. Whether `as_id` is in the fit stays a one-line formula toggle,
but it must NOT drive the prediction frame. Bobby named this the key concept for the
dengue rebuild.

**Why:** under a per-`(location, age, sex)` anchor shift, any link-space-additive,
time-constant term cancels exactly:

    pred(l,y,a,s)  = base(l,y) + Δ(a,s)
    shift(l,a,s)   = obs(l,anchor,a,s) − base(l,anchor) − Δ(a,s)
    final(l,y,a,s) = obs(l,anchor,a,s) + [base(l,y) − base(l,anchor)]      # Δ cancels

Expanding the prediction frame to age/sex builds a ~50x larger design matrix whose entire
extra contribution is then subtracted away. This is the same algebra as the malaria
country-FE cancellation (DECISIONS 2026-07-07, which explicitly said to re-derive it for
dengue — this is that derivation), and it explains a concrete finding: the `as_id`
adjustment in `04_forecasting/OLD_rake_dengue.py` had **no effect** on the delivered
first-submission numbers. The OLD chain already predicted on the reduced frame, then added
Δ back, then raked it away; only the Δ step was waste.

The cancellation needs additivity in **link** space, not multiplicativity in level space,
so it holds for log AND logit links. The late fan-out is multiplicative for log-link rates
and additive-in-logit for CFR — `expit(logit(cfr_obs) + δ)` — and both are exact. An
earlier claim in this session that CFR could not be fanned out at the end was WRONG; it
came from assuming a late fan-out must be a multiplication, and from applying a
"share of the total" framing to CFR, which is a per-cell ratio that does not decompose
into shares at all.

**Consequence for `as_id`:** it has exactly one surviving effect — it de-biases the
coefficients on the time-varying covariates, which does move `base(l,y) − base(l,anchor)`.
So "with or without `as_id`" is a fit-quality axis, not an age/sex-mechanism axis. The
age/sex pattern is the observed anchor's either way.

**Two conditions on the reduced frame:**
1. Every age/sex term must be additive and time-constant. An `as_id` main effect
   qualifies; an `as_id` x covariate interaction does NOT — it is genuinely time-varying,
   does not cancel, and forces a real age/sex prediction. The predictor must inspect the
   formula and expand only when it must, rather than always or never.
2. It holds only where an observed anchor exists. In a `(location, age, sex)` cell with no
   anchor there is nothing to cancel against, the model value survives, and the `as_id`
   choice does change the answer. Same cell set as the no-invented-deaths rule.

**Revisit if:** a formulation introduces an age/sex x covariate interaction, or the anchor
stops being per-`(location, age, sex)` (e.g. an all-age-only anchor) — either breaks the
cancellation and reinstates a genuine age/sex prediction.

## 2026-08-03: GDP-decoupled vs GDP-coupled is a SENSITIVITY — keep both, never delete
**Decision:** The two figure generations are not old-and-new, they are two arms of a sensitivity on
whether income is coupled to the climate scenario. Both stay on disk permanently. Bobby: "Do not
delete any of the old ones! They are VERY VERY useful! ... They are an entire sensitivity I didn't
know we needed."
**The two arms:**
  - DECOUPLED (product nodes WITHOUT `__gdpscen`): `make_gdppc_df.py` replicated the `reference`
    income trajectory across all three RCP labels, so income did not vary by scenario. The RCP
    contrast is then climate-only.
  - COUPLED (`__gdpscen` nodes): each income scenario under its mapped RCP label per
    GDPPC_SCENARIO_MAP (better->rcp26, reference->rcp45, worse->rcp85), confirmed with the data
    producer as the intended pairing.
**Why it is a real sensitivity:** the two runs differ ONLY in the GDP covariate, so the difference
between them measures how much of the RCP spread is income rather than climate. Global 2100 cases:
decoupled 414.7 / 400.5 / 353.6 (range 61.1M, RCP2.6 worst) vs coupled 382.9 / 400.5 / 372.0
(range 28.5M, RCP4.5 worst). Coupling income roughly HALVES the apparent climate signal and flips
the ordering. RCP4.5 is bit-identical across both arms because it maps to `reference`, which is what
all three previously used -- that identity proves the comparison is clean.
**The hazard, and the real fix:** the arms are distinguishable only by `__gdpscen` in the path, and
that has already caused one misread (an old differences panel showing RCP4.5-RCP2.6 negative when
the coupled run has it positive). Fix by stamping provenance INTO each figure -- a footer naming the
arm and input vintage -- so a figure is self-identifying whichever file is opened. NOT by deleting.
**Still to do:** give the decoupled arm the same figure treatment as the other sensitivities
(sens_full / effect-comparison), and rename the arms so they read as a sensitivity pair rather than
as vintages.
**Revisit if:** never delete. If space becomes an issue, compress rather than remove.

## 2026-08-03: a single GDP "elasticity" is not a well-defined quantity in this model
**Decision:** Never characterise GDP leverage with one percent-per-percent number, and never derive
a small-perturbation response from a large-cut response or vice versa. Investigation handed off in
`.claude/GDP_LEVERAGE_INVESTIGATION.md`.
**Why:** GDP enters the fitted model TWICE with different functional forms. `log_gdppc_mean` is
linear in the log-rate inc/mort equations -- that term alone is a constant elasticity. But GDP also
enters PfPR as `s(gdppc_mean, bs="mpd")`, a monotone-decreasing spline on the RAW dollar scale with a
logit response, and that shifted logit-PfPR then passes through `s(logit_malaria_pfpr, bs="mpi")` to
reach log-rate. The indirect path is a composition of two splines across raw -> logit -> log space,
so the total derivative depends on where each location sits on both curves. An average over a large
change is a SECANT; a small perturbation is a TANGENT. They are different quantities.
**The observation that exposed it:** coupling income to the climate scenario moved global 2100
incidence -7.67% (RCP2.6) and +5.20% (RCP8.5) off an apparent ~1% income change, an implied
elasticity 3-6x the model response to a 38.5% income cut (+66% on the 2100 mortality rate).
**Leading explanation is NOT curvature.** An in-session claim that the smooth is simply steeper
locally is unverified and probably wrong. The ~1% was measured UNWEIGHTED over all admin-2, most
with no malaria; the endemic locations carrying the burden sit at the steep low-income end. So the
burden-relevant income change is likely several times larger and the puzzle is measurement, not
response. Re-measure burden-weighted before drawing any conclusion.
**Consequence:** every covariate-comparison number produced by `compare_covariates.py` is
population-weighted over all admin-2 and understates changes concentrated in endemic areas. Add
burden weighting before quoting any of them.
**Revisit if:** the fitted formulas change such that GDP enters only linearly.

## 2026-08-04: No `AxisSpec` — DAH and decay are covariates with alternative futures
**Decision:** The shared `CauseSpec` gets NO `has_dah` boolean and NO `secondary_axis`/`AxisSpec`
field. Malaria's DAH scenarios and dengue's time-decay functions are the same generic thing: *a
covariate may have alternative future trajectories, and a run may be executed across them.* A hold
is that mechanism with a trivial trajectory. The variation is recorded in the RUN MANIFEST
(which already carries holds and couplings), never in the cause spec.
**Why:** `has_dah: bool` encoded "malaria has a thing dengue lacks", which is exactly the
malaria-shaped thinking the cross-cause consult existed to catch. The dengue side's proposed
`AxisSpec` was the same error one abstraction level up — it only looked more general, and it still
treated a covariate property as a property of the cause. Bobby's correction: DAH is a covariate
that has several supplied future trajectories; decay is the *time* covariate with a transformation
applied. Neither is structural. Whether a trajectory ships as a deliverable arm (all DAH scenarios
are delivered and compared) or is a single choice plus a sensitivity (decay) is per-run config.
**Two amendments that survive from the malaria side:** (a) population and age-structure holds are
`aggregation_transforms`, NOT covariate trajectories — neither appears in any fit formula for
either cause, so a hold changes the rate→count denominator and post-hoc fan-out but does not mean
the forecast was evaluated on a different future; (b) refit-vs-re-evaluation is decided by whether
an alternative changes the PAST (suitability variant → coefficients move → refit, new run dir) or
only the FUTURE (DAH, decay → fit untouched → extra arms in one dir). That test is mechanical and
checkable before running anything.
**Also settled:** stage 03 is NOT permanently cause-specific — dengue's four outcome structures and
two fitting engines are exploration scaffolding, and one of each will be chosen; `fit_grain`
remains a real difference but as a PARAMETER (malaria admin-2, dengue FHS-most-detailed), which is
an argument for parameterising `roll_up_hierarchy`'s hardcoded `start_level=ADMIN2_LEVEL`, not for
forking; `as_draw_persist_grain` is a grain and independent of `fit_grain`, because a boolean named
for admin-2 makes dengue's real requirement (age/sex draws at FHS) unnameable.
**Revisit if:** a cause acquires variation that genuinely cannot be expressed as a covariate
trajectory or an aggregation transform.

## 2026-08-04: Dengue fit grain reads `most_detailed_fhs` from the hierarchy, not the mapping table
**Decision:** `fit_dengue_formulations.r` takes the FHS-most-detailed flag from
`full_hierarchy_2023_lsae_1285.parquet`, matching `lib/data/dengue_inputs.py`. Never from
`lsae_1285_to_fhs_table.parquet`.
**Why:** the mapping table flags **513** locations against the hierarchy's **473** — a superset with
40 extras, nothing missing the other way. Reading it there inflated the anchor-eligible set from
305 to 338 and left the R script fitting **314** locations where the Python fits **305**. The whole
purpose of the R script is a like-for-like mgcv/scam comparison against pyGAM, so any divergence in
which rows enter the fit makes the comparison meaningless. Related, still open: the two engines also
disagree on the anchor estimator (R `rake="median"` vs Python `Baseline(statistic="mean")` over
2014–2023).
**Revisit if:** the mapping table becomes the authoritative source for FHS membership, in which case
`dengue_inputs.py` moves too — they must not diverge again.

## 2026-08-04: CauseSpec holds only run-invariant facts; everything else is a run property
**Decision:** A field belongs in `CauseSpec` (`lib/cause_spec.py`) only if it would be the same for
EVERY run of that cause. That admits ten fields: name, cause_id, `fit_grain`, `burden_column`,
`absent_means_zero`, the three artifact paths, `products_read_path`, and `as_draw_persist_grain`.
It EXCLUDES covariates, `anchor`, `measures`, `reference_age_group_id`/`reference_sex_id` and the
burden threshold — all of which move to the run spec. `AnchorSpec`, `MeasureStructure` and
`BurdenFilter` survive as types that a run instantiates.
**Why:** anything on the refit list varies per run by definition, since changing it is what forces
a refit — and the dengue side had itself classified formulation, engine, anchor spec and reference
cell as refit axes, so putting three of them in `CauseSpec` contradicted the agreed test.
Covariates are excluded twice over: which ones a run uses comes from its formulation (malaria's
model selection exists precisely to vary that), and covariate metadata already has a canonical
owner in `lib/io/covariate_registry.COVARIATE_REGISTRY`, so a second copy would go stale.
Enumerating 14 suitability variants in a frozen tuple is self-evidently wrong. The test is also
what keeps the class extensible: adding a covariate or a variant never requires editing it. Two
guardrail tests enforce this by asserting no field name matches a covariate-future or a refit axis.
`as_draw_persist_grain` is a grain rather than a boolean because `persist_as_draws_admin2: bool`
made admin-2 the implicit default and left dengue's real requirement (age/sex draws at FHS grain)
unnameable — the honest value would have been `False`, reading as "dengue needs no age/sex draws".
**Revisit if:** a cause acquires variation expressible neither as a covariate trajectory, an
aggregation transform, nor a run-spec field.

## 2026-08-04: Share the product CONTRACT, not the producer
**Decision:** malaria's `finish_run.py` is NOT in dengue's path. The shared artifacts are the
product schema, `validate_products` (called by BOTH producers) and a shared reader, in
`lib/processing/products.py`. Product filename is
`all_age_summary_{ssp}_{trajectory}.parquet`, with `level` and `population` required columns.
**Why:** the fork risk was always `plot_run_comparison.py`, and a shared schema plus reader removes
it without either side adopting the other's 780 lines. `finish_run.py` also literally cannot
consume dengue's leaf set — it calls `roll_up_hierarchy(start_level=ADMIN2_LEVEL)`, hardwired to a
uniform single level, while dengue's 473 FHS leaves span levels 3 and 4 (which is what
`roll_up_to_ancestors` exists for). A contract with no enforcement drifts, hence a shared validator
rather than a schema document. Its load-bearing check re-derives every rate from its count and that
level's own population row, catching the whole family of denominator errors — summed children's
populations, or a population-weighted average of children's rates — that leave a file looking
correct. The `{trajectory}` token replaced `{axis_value}` when `AxisSpec` was withdrawn; the
one-file-per-arm layout is unchanged and is required by the house rule against storing a redundant
constant column.
**Revisit if:** the two producers' outputs converge enough that one implementation is genuinely
cheaper than two conforming ones.

## 2026-08-04: The anchor check compares predicted level to the anchor TARGET
**Decision:** `anchor_diagnostic` compares the prediction at `max(anchor.years)` against the
anchor's own target — observed at that year for a point anchor, the window mean for a window-mean
anchor. Hard pass/fail applies only when `AnchorSpec.reproduces_observed` is True; otherwise the
artifact is a diagnostic carrying a `flagged` threshold (default 10% aggregate).
**Why:** "does the run reproduce observed at the anchor year" is meaningful only for a point
anchor. Dengue's F4 sits within 4% of its 2014–2023 observed mean and 45% below observed 2023,
because 2023 was an epidemic spike a decade mean deliberately does not chase. Checked the wrong
way a correct run is condemned — and, worse, a genuinely broken one can pass. Malaria keeps the
hard check, since its anchor is a point anchor and any deviation there IS a bug. F4's incidence
passes at +4% and its mortality fails at +21%; anchor-year equality would have flagged both as
catastrophic and distinguished nothing.
**Revisit if:** an anchor kind appears that is neither a point nor a window mean.
