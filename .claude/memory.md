# Session memory
Updated: 2026-03-27

## Current task
Phase 2 complete. LIB_DESIGN.md written. Waiting for Bobby to review before Phase 3.

## Context / why
Refactoring shared infrastructure for malaria and dengue pipelines. Phase 2 produced the blueprint (function signatures) that Phase 3 will implement.

## Where we are
- Branch: `feature/refactor-shared-lib`
- Phase 1 (audit): complete — PIPELINE_AUDIT.md + COMMON_PATTERNS.md committed
- Phase 2 (design): complete — LIB_DESIGN.md written to .claude/ (not committed yet)
- All 5 open questions from Phase 1 answered by Bobby

## All decisions resolved
1. **M2 canonical**: `as_malaria_fractions.py` (normalized RR fractions, not shifts)
2. **H1 canonical**: `level_filter()` in `helper_functions.py` — use it, don't inline
3. **H2 RH clip**: universal covariate rule — `UNIVERSAL_COVARIATE_CLIP_RULES` default in `load_covariates_for_draw`; malaria was missing it historically, will be applied after refactor
4. **H3 mkdir**: `write_netcdf()` gains `mkdir=True` parameter (matches write_parquet)
5. **H3 atomic**: `write_parquet(use_atomic=True)` new default (was False)
6. **Covariates**: `data/covariates.py` (not `climate.py`) — covers climate + income + urban
7. **Disaggregation**: separate module from aggregation; two named functions (_malaria, _dengue)
8. **Spatial raking**: deferred — review `05_aggregation/create_raked_outcomes.py` before extracting

## LIB_DESIGN.md covers (9 modules, ~30 function signatures)
- `lib/io/parquet.py` — 6 functions (read, write, filter, sort, cast helpers)
- `lib/io/netcdf.py` — 8 functions (read, write, convert, filter)
- `lib/io/hdf5.py` — 4 functions (write, create_structure, write_draw, read_metadata)
- `lib/data/hierarchy.py` — 4 functions (load, level_filter, get_ids, make_filter)
- `lib/data/covariates.py` — 4 functions (load_for_draw, income, urban, merge)
- `lib/processing/raking.py` — 3 functions (rake_level, rake_aa_lsae_to_gbd, logit_shift_rake)
- `lib/processing/aggregation.py` — 4 functions (aggregate_level, aa_count, aa_rate, to_parent)
- `lib/processing/disaggregation.py` — 2 functions (_malaria, _dengue)
- `lib/processing/scenarios.py` — 1 function (generate_dah_scenarios)
- `lib/utils/transforms.py` — 2 functions (logit, expit)

## Next steps
1. Bobby reviews LIB_DESIGN.md (pay attention to raking/disaggregation/covariates signatures)
2. Commit LIB_DESIGN.md + session log updates
3. Phase 3, Task 3.1: create lib/ directory structure + __init__.py files
4. Phase 3, Task 3.2: implement lib/io/ (parquet → netcdf → hdf5)
5. STOP after Task 3.2 for review

## Resume prompt
Phase 2 is complete. LIB_DESIGN.md is in .claude/ with signatures for all 9 lib/ modules. All design decisions are resolved (see memory.md "All decisions resolved" section). Next: Bobby approves LIB_DESIGN.md, then Phase 3 begins with Task 3.1 (create lib/ directories and __init__.py) followed by Task 3.2 (implement lib/io/). Commit LIB_DESIGN.md + SESSION_LOG before starting Phase 3.
