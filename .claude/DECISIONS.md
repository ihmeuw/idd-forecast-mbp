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

## 2026-03-27: data/covariates.py covers all model-predictor loading
**Decision:** No `data/climate.py`. Single `data/covariates.py` module covers climate (draw-specific), income, urban, and the merge_dataframes helper.
**Why:** All three loading patterns (climate, income, urban) follow the same dict-of-paths→read→merge structure and serve the same purpose (loading model predictors). Splitting by data source would create artificial module boundaries.
**Revisit if:** Covariate loading patterns diverge substantially between climate and socioeconomic data.
