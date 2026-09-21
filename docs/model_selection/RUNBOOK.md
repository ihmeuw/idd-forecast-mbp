# Malaria model selection and versioned outputs: runbook

Written 2026-09-16. How to redo the malaria model selection from the top, and how every stage of
this repo writes and freezes output since the idd-tools versioning conversion of 2026-09-16.

## Output versioning (every stage)

Every output node follows `idd_tools.versions` (STANDARDS, Output management):

- A run writes into `<node>/working/`; a test run into `<node>/scratch/<label>/`.
- Nothing is frozen unless the launch asked. A click launcher takes `--current --description "why"`
  (freeze and promote on success), `--freeze` (freeze only), `--label NAME`, `--scratch LABEL`.
  A stage script without a CLI takes the same through the environment:

      IDD_VERSIONS_CURRENT=1 IDD_VERSIONS_DESCRIPTION="stage 02, 2026 covariates" \
          .venv/bin/python src/idd_forecast_mbp/02_data_prep/01_make_full_hierarchy.py

  (`IDD_VERSIONS_FREEZE`, `IDD_VERSIONS_LABEL`, `IDD_VERSIONS_SCRATCH`, `IDD_VERSIONS_TAG` likewise.)
- Readers follow `<node>/current/`. A chained rebuild therefore launches each stage with
  `--current` so the next stage sees it; without it the run sits in `working/` until
  `idd-versions <node> freeze "why" [--current]`.
- Two-script stages share one slot and finish once: 02a then 02b (population), pixel_main then
  pixel_hierarchy. Launch the first half without `--current`; the second half finishes.
- Inspect a node: `.venv/bin/idd-versions <node> status`. Snapshots, `current`, labels and
  unregistered directories are all listed there.
- One run per slot: finish a run before launching the next into the same node (file names in
  a slot carry no run identity; the label does).

## Redo the malaria model selection from the top

Paths below are relative to the repo; `<image>` and `<shell>` are the IHME singularity R image
and `execRscript.sh` wrapper passed to every R launcher.

1. **Specs.** `.venv/bin/python src/idd_forecast_mbp/03_modeling/build_malaria_spec_design.py`
   writes `spec_table.parquet` (1,620 specs) into a new run dir under
   `<modeling stage>/malaria/scam_prelim/<hierarchy>/`.
2. **Fits.** `.venv/bin/python src/idd_forecast_mbp/03_modeling/fit_malaria_models_orchestrator.py
   --spec-table <run_dir>/spec_table.parquet --output-dir <run_dir> --worker
   src/idd_forecast_mbp/03_modeling/select_malaria_models_rocket.r --r-image <image> --r-shell <shell>
   --full` (probe first with `--probe`). Fit-time judgments (windows, thresholds, optimizer) are
   recorded in `reports/model_selection/malaria_selection_config.yaml` under `fit:`.
3. **Summary.** `.venv/bin/python src/idd_forecast_mbp/03_modeling/finalize_selection_run.py --run-dir <run_dir>`
   writes `selection_summary.parquet`.
4. **Rank and report.** `.venv/bin/python src/idd_forecast_mbp/03_modeling/rank_selection_run.py
   --config reports/model_selection/malaria_selection_config.yaml [--run-dir <run_dir>]`
   applies the `rank:` parameters, writes `selection_result.json` (status tentative),
   `ranking.parquet`, `candidates.parquet` and renders `report.html` into the run dir. Read the report.
   (The gate notebook with parameter toggles and the Record-pick / Flag-best buttons is the next step
   of `.claude/SELECTION_PIPELINE_PLAN.md` and is not built yet; until then steps 5 and 6 are the gate.)
5. **Fit the selected model.** `.venv/bin/python src/idd_forecast_mbp/03_modeling/fit_selected_malaria_model.py
   --config reports/model_selection/malaria_selection_config.yaml --r-image <image> --r-shell <shell>
   [--run-dir <run_dir>] [--freeze --description "why" --label <name>]` runs
   `fit_selected_malaria_model.r` with the config's `final_fit:` settings into the models node
   (`<modeling stage>/malaria/models/<hierarchy>/working/`) and writes `malaria_models.RData` +
   `run.json`. With `--freeze` it becomes a snapshot; with `--current` it is promoted at once.
   Run it in an interactive Slurm session: three scam fits on the full past inputs.
6. **Flag best.** `.venv/bin/idd-versions <models node> promote <snapshot>` (and
   `label <snapshot> <name>`). The forecast uses `current`.
7. **Forecast.** `.venv/bin/python src/idd_forecast_mbp/04_forecasting/01_forecast_malaria_admin_2s_orchestrator.py
   --worker src/idd_forecast_mbp/04_forecasting/forecast_malaria_admin_2s_rocket.r --r-image <image>
   --r-shell <shell> [--model-version <snapshot or label>] --current --description "why" --label <run name>`.
   The model dir is resolved from the models node and passed to the workers; output goes to the
   forecast_outputs node's `working/` and is frozen under the label on success. A hold run
   (`--hold-covariate`) is a separate launch with its own label after the baseline is finished.
   Then `src/idd_forecast_mbp/05_aggregation/finish_run.py --forecast-run-dir <snapshot dir> ...
   --current --description "why"` for the products.

## Where the record lives

- The pick: `<run_dir>/selection_result.json` (parameters, anchor, pick, candidates, code commit,
  spec-table fingerprint) and `report.html` beside it.
- The fitted model: `<models node>/<snapshot>/run.json` carries formulas, thresholds, convergence
  and a `selection` block pointing at the run dir and spec.
- The registers: `registry.json` on every node; `idd-versions <node> status` reads them.
- The pre-2026-09-16 flat model registry (`malaria_model_registry.json`) is read-only history.

## Refits from another repo (the importable fitter, 2026-09-18)

The body of `fit_malaria_models_orchestrator.py` is
`idd_forecast_mbp.lib.modeling.malaria_fit_run.submit_malaria_fit_run(spec_table, output_dir, *, r_image, r_shell,
past_inputs, ...)`. It writes only under `output_dir` (any directory; it refuses a finished selection run, a registered
snapshot, a `current` target, or anything under the models node), fans the spec table into one in-sample cell plus the
out-of-sample cells, bundles them, submits through `idd_tools.jobmon`, and returns a `MalariaFitRun`. Arguments beyond the
orchestrator's options: `past_inputs` (required; nothing follows `current`), `prep_script` (an R file defining
`prepare_malaria_fit_frame(parquet_path, inc_count_min, pfpr_min, suit_variant)`; default the package's
`lib/malaria_fit_frame.R`), `inc_count_min` / `pfpr_min` (1 and 0.0001 give the selection frame, 167,649 rows on the
`20260527` past inputs; 0 and 0 give the registered final model's 319,073 rows), `oos_windows` (default the ten windows of
the 2026-07 run, built by `temporal_windows()`), `save_fits` and `save_predictions` (both off by default).

- `spec_table` needs `spec_index`, `n_smooths`, `n_scams`, `formula_text`; an optional `suit_variant` column picks the
  suitability variant per spec (absent: `mordecai_0_0`).
- `save_fits=True` writes `fits/<cell>/spec_<i>.rds` (predict-stripped, a few MB) with `spec_<i>.json` beside it: the
  spec, the cell and its years, the thresholds, the past-inputs and prep-script paths with sha256, the rows and country
  levels the fit used, versions and timing. `save_predictions=True` writes `predictions/<cell>/spec_<i>.parquet`
  (`location_id`, `year_id`, `observed`, `predicted_lp`, `predicted_response`, `observed_response`).
- The worker (`select_malaria_models_rocket.r`) takes the same things as flags: `--past-inputs`, `--prep-script`,
  `--inc-count-min`, `--pfpr-min`, `--save-fits`, `--save-predictions`, `--strip-fits`.
- `fit_selected_malaria_model.{py,r}` use the same preparation (`--prep-script`, same default).
- Consumer pin: `"idd-forecast-mbp"` in `dependencies` and
  `idd-forecast-mbp = { git = "ssh://git@github.com/ihmeuw/idd-forecast-mbp.git", rev = "<commit>" }` in
  `[tool.uv.sources]`; `branch = "main"` once the work is on `main`. The consumer declares `idd-tools` and the IHME
  index itself; `climate-data` is not needed (it is the `climate` extra).
- Tests: `tests/lib/modeling/test_malaria_fit_run.py` (pytest) and `tests/testthat/` (run
  `execRscript.sh -i <image> -s tests/testthat.R` from the repo root; set `MBP_PAST_INPUTS` to the parquet for the
  two real row-count checks).
