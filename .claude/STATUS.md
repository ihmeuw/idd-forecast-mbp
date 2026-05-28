# Project status
Updated: 2026-05-14

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

## Recent steps
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

## Next steps
1. **Rerun Python data-prep pipeline** (02a → 02b → 03 → 04 → 05) with new
   population data (conda env `idd-forecast-mbp`).
2. **Finalize rocket** — remove debug `save()` line at L102, decide .rds vs
   .RData format.
3. **When `20260514_v2` lands**: filter summaries by
   `is_converged == TRUE & cv_n_converged == 5` before ranking. Rankings
   from the earlier EFS-fit batches are not trustworthy — a non-trivial
   fraction of those "finished" fits silently didn't converge.
4. **Re-run MCDM ranking** (`rank_malaria_models_loop.ipynb`) against the
   BFGS-fit pool. Compare top-of-rankings against the EFS-fit rankings to
   see how much the silent non-convergence shifted the picture. Favour
   3-of-4 method convergence over any single method's pick
   (DECISIONS 2026-05-13).
5. Build `06_build_dengue_past_inputs.py` (same pattern as 05; watch for
   row-filter bug — see DEAD_ENDS.md).
6. **Scale to 73,710** via SLURM array job — same cache dir; previous fits
   dedupe automatically.
7. **Repeat scaffold for mortality and incidence** — same group structure,
   different response.
8. Design forecasted input/output file structure (draw dimension needed,
   netCDF appropriate here).
9. **[VALIDATION GAP]** Stage 06 (upload): not verified against goldens.
10. **[NEXT SESSION — stage 01 investigation]** Two distinct issues
    surfaced during 07b/08 work that should be fixed upstream rather
    than tolerated:

    a. **NaN in urban + malaria_suitability outputs — root cause
       identified 2026-05-27.** 20 admin-2 locations have NaN in
       `weighted_1km_urban_threshold_*` and `malaria_suitability`
       (and people_flood_days_per_capita, but that's rapidresponse
       not this repo). **Root cause: zero population in the upstream
       rapidresponse gridded pop product** at
       `/mnt/team/rapidresponse/pub/climate-aggregates/current/results/lsae_1285/population.parquet`.
       The per-capita math at `pixel_hierarchy.py:94` is
       `value = weighted_climate / population.replace(0, np.nan)`,
       so any (loc, year) with zero pop yields NaN value by
       construction. 19 locs have zero pop across all 101 years
       (uninhabited / water polygons). Loc 93390 has zero pop
       1950-2014 and real, monotonically growing pop 2015-2100
       (verified by reading the upstream parquet 2026-05-27).
       **Not a stage-01 bug.** No fix needed in `pixel_hierarchy.py`
       or `pixel_urban_hierarchy.py`.

       The earlier "new admin-2 boundary added after historical
       climate aggregate" hypothesis was inconsistent with the
       aggregator code (which can't silently invent partial-year
       NaN for a location absent from per-block inputs — it would
       KeyError or include the location with whatever data was
       there). The hypothesis is replaced by the zero-pop finding.

       **Consumer-side fix (07b drop policy).** Current 07b drops
       affected locs across the entire forecast window 2023+. That's
       over-dropping — 93390 has valid pop and covariate data from
       2015 onward and should not be removed from forecasts. New
       policy for 07b:
       1. Drop (loc, year) rows where population == 0 first. These
          aren't real data; no per-capita covariate can exist for
          them. The 19 uninhabited locs come out naturally here
          (every year-row drops, location effectively gone).
       2. Check remaining rows for NaN in any covariate.
       3. Report what was found (counts, location_ids, year ranges)
          — surface, don't silently drop.
       4. Apply context-aware drop: if a loc still has NaN across
          its entire model window after step 1, drop the whole loc;
          if only some year-rows, drop just those rows.
       Action: rewrite the NaN-handling block in
       `src/idd_forecast_mbp/02_data_prep/07b_build_malaria_prediction_locations.py`
       per this policy.

    b. **Output-layout / versioning problems in pixel_hierarchy.py.**
       (From a separate audit.) Two related bugs in stage 01:
       - `pixel_hierarchy.py:228-236` writes climate aggregates
         to a NON-versioned flat path (`02-processed_data/{hierarchy}/
         {summary_covariate}_{scenario}.parquet`); re-running silently
         overwrites the previous run. Urban side uses URBAN_WRITE_PATH
         which IS versioned; they disagree.
       - Both aggregators write `population.parquet` only via an
         `if not exists` guard
         (`pixel_hierarchy.py:240-252`, `pixel_urban_hierarchy.py:212-214`).
         Re-running stage 01 with the SAME hierarchy but a NEW gridded
         population leaves the on-disk pop file STALE while everything
         else updates — downstream stage-02 reads stale data with no
         warning.
       - A layout-convention fork exists already: urban writes to
         `<artifact>/<hierarchy>/<RUN_DATE>/`; the 02b reader expects
         `<RUN_DATE>/<hierarchy>/`. 02b currently hardcodes "20260405".
         Picking either convention for the pixel_hierarchy fix touches
         files outside stage 01.

    Status (2026-05-27): design settled in conversation, with a major
    simplification at the end. Draft at `.claude/DECISIONS_draft.md`.
    Final shape:
    - **Stage 01 stops writing `population.parquet` entirely.** The
      rapidresponse team already publishes the canonical version at
      `/mnt/team/rapidresponse/pub/climate-aggregates/current/results/lsae_1285/population.parquet`,
      versioned via `CLIMATE_COVARIATE_RUN_DATE`. Same dir tree we
      already read every other climate covariate from. Verified schema
      matches (location_id int64, year_id int64, population float32),
      coverage matches (51,783 locs × 151 years, includes global).
      Delete pop-write blocks from `pixel_hierarchy.py:238-252` and
      `pixel_urban_hierarchy.py:207-214`; both still derive pop
      in-memory for `post_process`'s per-capita math.
    - **02b reads upstream.** Add `LSAE_POP_PATH = CLIMATE_AGGREGATES_PATH
      / LSAE_HIERARCHY / "population.parquet"` to `constants.py`.
      Delete the hardcoded `_LSAE_POP_PARQUET_ROOT = ... / "20260405"`
      in `02b_full_population.py`; replace line 86 with `mbpc.LSAE_POP_PATH`.
    - **Climate aggregates** (per-covariate, per-scenario parquets at
      `pixel_hierarchy.py:228-236`) move onto registry-derived helpers
      `covariate_write_path/read_path(name)` in constants.py — one
      helper, not N hand-written constants. Paths derived from
      `COVARIATE_DICT.yaml`. Per-covariate artifact root at
      `02-processed_data/covariates/<name>/<hierarchy>/<RUN_DATE>/`.
    - **`malaria_variables` / `dengue_variables`** dicts at
      `constants.py:230-236` become registry-derived too (they're
      reads of the climate-aggregate output we're now versioning).
    Earlier proposals (launcher-flag single-writer, `_A02_PIXEL_POP`
    artifact, separate POP_RUN_DATE) are kept in the draft's
    "Options considered" section for the audit trail — they were the
    right answer to "how do we own pop correctly" before Bobby
    surfaced that we don't need to own it.
    One open Q remaining: urban-threshold consistency with the
    per-covariate artifact pattern (independent of the pop change).
    Action for next session: (1) confirm or reject the urban-threshold
    open Q, (2) promote the draft to `.claude/DECISIONS.md`,
    (3) implement. Item (a) — backfilling the loc 93390 climate gap
    — remains independent and separable.

## Parking lot
- `read_grid_map()` helper in netcdf_helpers.R is a placeholder — canonical
  fix is for upstream Python writer to embed a `grid_map` global attribute.
- Population weighting rejected (high-pop ≠ high-malaria). Equal weighting kept.
- Delete legacy modules (parquet_functions, xarray_functions, helper_functions,
  hd5_functions).
- Delete old standalone test scripts in `src/04_forecasting/`.
- combine_as_draws.py dead-end.
- Dengue revision (after malaria paper accepted).
- Feature: vaccination scenario dimension.
- Feature: variable importance methods beyond "hold at 2022".
- When suitability variant files become available (14 curves), re-enable all
  variants in script 05.
- dev.expl discrepancy (0.8189 SLURM vs 0.8191 interactive) — same data,
  unknown cause. Not blocking.

## Parking lot
- `read_grid_map()` in `netcdf_helpers.R` is a placeholder; canonical fix
  is for the upstream Python writer to embed a `grid_map` global attribute.
- Population weighting was discussed and rejected (high-pop ≠ high-malaria);
  revisit only if downstream use case changes.
- Group annotations in the qmd may evolve once AICs come in (some
  "concave-down" hypotheses for temperature variables may not pan out).
- Delete legacy modules (`parquet_functions`, `xarray_functions`,
  `helper_functions`, `hd5_functions`) after refactor fully verified.
- `combine_as_draws.py` dead-end — revisit when draw combining becomes a
  bottleneck. See DEAD_ENDS.md.
- Dengue revision (after malaria paper accepted).
- Feature: vaccination scenario dimension.
- Feature: variable importance methods beyond "hold at 2022".
- When suitability variant files become available (14 curves), re-enable
  all variants in script 05.
