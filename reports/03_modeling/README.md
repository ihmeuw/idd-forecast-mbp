# reports/03_modeling

Exploration notebooks for the modeling stage, sorted by status (2026-09-16):

- `archive/`: everything as it stood before the sort. History, not process. Nothing here is
  the record of a decision; the record is `.claude/DECISIONS.md` and, for model selection,
  the written result beside each run (`selection_result.json`, `report.html`).
- `malaria/`, `dengue/`: notebooks in active use, promoted from `archive/` on demand.
  Today: the two dengue notebooks Bobby edits (`dengue_formulation_lab.ipynb`,
  `dengue_past_data_explore.ipynb`).
- The canonical malaria model-selection chain does not live here. Its config and report
  template are in `reports/model_selection/`; its code is `src/idd_forecast_mbp/select/`.

A notebook moves out of `archive/` when someone needs it again, and back when it is done.
Git history follows each file (`git log --follow`).
