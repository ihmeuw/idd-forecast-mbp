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

## 2026-03-27: write_netcdf gains mkdir; write_parquet atomic default changes
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
