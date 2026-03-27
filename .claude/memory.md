# Session memory
Updated: 2026-03-27

## Current task
Phase 3 in progress. Tasks 3.1 and 3.2 complete (lib/ structure + lib/io/). Ready for Task 3.3 (lib/data/).

## Context / why
Refactoring shared infrastructure for malaria and dengue pipelines. Phase 3 creates lib/ with shared functions, keeping originals intact.

## Where we are
- Branch: `feature/refactor-shared-lib`
- Commit 6ddd448: Phase 3 Tasks 3.1 + 3.2 committed
- lib/io/parquet.py — 48 tests passing
- lib/io/netcdf.py  — 48 tests passing (combined run)
- lib/io/hdf5.py    — 48 tests passing (combined run)
- Original source files untouched

## Key implementation notes
- write_parquet: use_atomic=True default; 'full'/'sample' validation dropped (OOM at scale)
- write_netcdf: mkdir=True added; validation reads file back but only metadata (not data)
- .gitignore: added negation rules for src/.../lib/ and tests/lib/ (bare `lib/` was gitignored)

## Next steps (Phase 3 remaining)
1. Task 3.3: lib/data/hierarchy.py + lib/data/covariates.py + tests
2. Task 3.4: lib/processing/ (raking, aggregation, disaggregation, scenarios) + tests
3. lib/utils/transforms.py (logit/expit) + tests
4. STOP after 3.3 for review before 3.4

## Resume prompt
Phase 3 Tasks 3.1 and 3.2 complete (lib/ directories + all 3 io modules with 48 passing tests). Next: Task 3.3 — implement lib/data/hierarchy.py (load_hierarchy, level_filter, get_location_ids, make_location_filter) and lib/data/covariates.py (load_covariates_for_draw with UNIVERSAL_COVARIATE_CLIP_RULES, read_income_paths, read_urban_paths, merge_dataframes). Write tests for each. Stop after 3.3 for review.
