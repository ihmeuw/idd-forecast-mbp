# Idiom-migration note — porting `02_data_prep` to the idd-aedes-spread idiom

**Date:** 2026-07-20 · **By:** Claude (for Bobby) · **Context:** idd-aedes-spread borrows the
hierarchy + population prep from this repo. Doubles as a **template for gradually migrating
idd-forecast-mbp itself** to the shared idd-* idiom.

## What was ported (into idd-aedes-spread)
- **`02a_fhs_population.py` → done + validated.** All-age FHS population (past GBD pull +
  future FHS netCDF). Reproduces this repo's golden `aa_2023_fhs_population_df.parquet`
  **exactly** (51,813 rows, max abs diff 0.0).
- **`01_make_full_hierarchy.py` → done + validated.** Ported pandas-internal (small,
  intricate, order-dependent). Structurally exact vs the golden `full_hierarchy_2023_lsae_1285`:
  all id/level/flag/`sort_order`/`path_to_top_parent`/lookup/`A0` columns identical across all
  51,176 rows. Only diffs are benign representation (missing → proper `null` vs golden's `''`)
  and 2 New-Zealand names where ours preserves the `Māori` macron vs golden's ASCII.
- **`02b_full_population.py` (all-age) → done + validated.** Ported to **polars** (raking +
  future projection). Reproduces the golden `aa_2023_full_population_df` **exactly** — all
  5,168,776 rows, full coverage, max relative diff 1.06e-15 (machine epsilon). The
  hierarchy in-place `in_gbd_not_lsae` writeback is intentionally skipped (annotation only;
  doesn't affect population values).

Age/sex-specific outputs are intentionally dropped for aedes (the spread model is not
age-structured). `02b`'s raking + future-projection logic is kept.

**All three ported, validated against this repo's golden outputs, committed to
idd-aedes-spread `main`.**

## The idiom translation (what "our idiom" means, as a migration checklist)
| forecast-mbp today | target idiom |
|---|---|
| script runs at import (`01`) or a bare `main()` | typed functions in `lib/`; thin **Click** `main()` CLI in the stage dir; logic is import-safe + testable |
| pandas throughout | **polars** for frame ops (`scan_parquet` + pushdown for big reads); xarray only at the netCDF boundary, converted straight to polars |
| `read_parquet_with_integer_ids` / `write_parquet` | `lib/io.write_parquet_atomic` (metadata-validated, atomic); **explicit dtypes**; **drop redundant constant columns** (e.g. `age_group_id=22`, `sex_id=3`) |
| `finalize_artifact` (dated dir + `current` symlink) | `lib/versioning.versioned_output_dir` — same dated-dir + `current` pattern |
| absolute paths in `constants.py` | **`paths.yaml`** (gitignored) for absolute roots + `constants.py` for relative fragments/values; **no absolute paths committed** (pre-commit guard enforces) |
| few/no unit tests; correctness by eyeball | every function unit-tested; **each ported stage validated cell-by-cell against this repo's golden output** (the rewrite-validation rule) |

## Worked example: `02a`
- `lib/population.py`: `combine_fhs_population_aa(past, future)` (pure, unit-tested) +
  `load_fhs_past_aa` / `load_fhs_future_aa` (I/O). ~60 lines vs the original ~185.
- `stage_00_data_acquisition/run_fhs_population.py`: Click CLI, paths from `paths.yaml`.
- Validation: joined my output to the golden on `(location_id, year_id)` → full coverage,
  0.0 max diff.

## Notes for migrating forecast-mbp itself
- `01_make_full_hierarchy.py` is the hardest: it runs at import and uses `iterrows` grafting
  + `path_to_top_parent` string parsing. Porting to polars needs care; validate against the
  golden `full_hierarchy_2023_lsae_1285.parquet`.
- The `02b` raking comments say "level 4 = admin-2 / level 5 = admin-3", but for the LBD
  standard shapefiles admin-2 = **level 5** (admin-1 = level 4, national = level 3). The
  raking logic is level-consistent internally; just don't trust the comment labels — verify
  levels from the hierarchy data.
- `finalize_artifact`/`_artifact_write` and `versioned_output_dir` are the same concept —
  a shared idd-tools versioning module would let both repos drop their local copy.

*This note is untracked in `inbox/`. It will be updated as `01`/`02b` land.*
