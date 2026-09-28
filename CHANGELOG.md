# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- Raking to external parent targets in `lib/processing/raking.py`: `rake_children_to_parent_targets` (count space,
  explicit `leave` zero rule, factor frame at parent grain, excluded cells reported), `apply_raking_factors` (reference
  factors applied to any arm), `check_raked_matches_targets` and `check_sum_identity` (2026-09-28).
- `lib/processing/derived_measures.py`: the FHS measure table (incidence, death, yll, yld raked from our incidence and
  mortality; daly = yll + yld) as data, with lookups and `sum_components` (2026-09-28).
- `lib/data/fhs_returns.py`: reader for the FHS returns (round map from YAML, file-contract validation, one-draw reads
  subset to our age groups, sexes and years) (2026-09-28).
- `select/gate.py` + `reports/model_selection/malaria_selection_gate.ipynb`: the two-button selection gate
  (re-rank with the overridable `rank:` keys, Record pick, Flag best via idd-tools freeze/promote) as tested functions
  with an import-only ipywidgets notebook (2026-09-28).
- `constants.MAL_SELECTION_NODE`: the malaria model-selection runs node; `build_malaria_spec_design.py` writes there
  instead of a hardcoded absolute path (2026-09-28).
- `--forecast-run-dir` on `vaccine_impact_scenarios.py` and `run_malaria_vaccine_pipeline.py`; the impact summary's
  `model_run` column is the resolved forecast snapshot name (2026-09-28).
### Changed
- Coverage: `--no-cov-on-fail` removed, so a shortfall fails the suite; `fail_under` set to the measured fast-suite
  floor with the untested modules listed in `pyproject.toml` (2026-09-28).
- `select/model_selection.py` moved to `03_modeling/archive/superseded/` (imported only by archived notebooks) (2026-09-28).
### Fixed
- `lib/processing/vaccine_impact.py` no longer carries a hardcoded run label that went stale when the forecast node's
  `current` moved to the gdpscen arm (2026-09-28).
- Nine `06_upload` scripts read the `lsae_1285` hierarchy from the hierarchy node via constants instead of the
  retired `lsae_1209` file (2026-09-28).
