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
