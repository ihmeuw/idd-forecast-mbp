# Dead ends

<!-- Append-only. Never delete or overwrite entries. -->

## YYYY-MM-DD: [approach name]
**What I tried:** One or two sentences.
**Why I stopped:** What failed or made it not worth continuing.
**Refs:** File paths or commit hashes if useful.

## 2026-03-30: combine_as_draws.py — parallel draw combining
**What I tried:** `06_upload/combine_as_draws.py` was an attempt to combine all draws into a single upload file more efficiently. `01_combine_as_draws_parallel.py` submits it as a cluster job.
**Why I stopped:** Script is broken mid-refactor — `list_of_dataarrays` and `draw_var_name` are used but never defined. Never got it working and moved on.
**Revisit when:** Working on "faster/more efficient" phase. The goal was presumably to replace or complement `create_and_combine_as_and_aa_draws.py` with a faster approach. Worth revisiting if draw combining becomes a bottleneck.
**Refs:** `src/idd_forecast_mbp/06_upload/combine_as_draws.py`, `src/idd_forecast_mbp/06_upload/01_combine_as_draws_parallel.py`

## 2026-05-07: refactor bug — row filter dropped when extracting _endemic_location_ids
**What I tried:** Extracted endemic location selection into `_endemic_location_ids()`. The function applies the row-level filter (pfpr > 0, mort_count > 0, inc_count >= 0) internally to identify valid locations, but the caller then rebuilt `aa_sub` from the unfiltered `aa_df`, silently restoring 90K zero-pfpr rows (~23% of total). The logit on those rows produces -inf.
**Why I stopped:** Bug caught when the divide-by-zero warning appeared on re-run. Fixed by applying the same row filter to `aa_sub` in the caller.
**Watch for:** The same bug will exist in `06_build_dengue_past_inputs.py` when we build it — apply the row filter to the dengue equivalent of `aa_sub`, not just to the location ID selection step.
**Refs:** `src/idd_forecast_mbp/02_data_prep/05_build_malaria_past_inputs.py`

## 2026-05-12: scam summary() on deserialized objects
**What I tried:** Used `saveRDS()`/`readRDS()` and `save()`/`load()` to serialize scam model objects, then called `summary()` on the restored object.
**Why I stopped:** Fails with "data must be of a vector type, was NULL" — the model frame environment doesn't survive serialization. This is a fundamental limitation of scam/mgcv, not a bug in our code. Direct field access (`fit$deviance`, `coef(fit)`, `predict()`) still works fine. The rocket computes all needed metrics before saving.
**Refs:** Tested in interactive R session with fit from `20260512_v2` output.

## 2026-05-14: "Drop the 12 failed specs" recommendation
**What I tried:** When 12 specs timed out even after the threading fix, I
recommended dropping them as <1% of the grid that wouldn't change the
top-of-rankings anyway.
**Why I stopped:** Bobby correctly pushed back: this is a single-winner
model-selection exercise, so any spec without an OOS metric is a spec
that can't compete — "the failed ones might be the best." The fix is to
evaluate them, not exclude them. Pivoted to diagnosing the optimizer and
ultimately switching to BFGS (DECISIONS 2026-05-14).
**Refs:** Session 2026-05-14, `delete_once_model_selected.rmd`.

## 2026-05-14: "4+ shape-constrained smooths" as the failure fingerprint
**What I tried:** Hypothesized that specs with 4+ shape-constrained
smooths (`mpd`/`mpi`/`cv`) were dying because the EFS optimizer was
fighting the constraints. The 12 visibly-lost specs all had 4–5 such
smooths.
**Why I stopped:** Cross-tabulation of `neighborhood_specs` showed all 48
specs in the *6*-smooth bucket finished cleanly with EFS. If
constraint-count drove failure, the 6-bucket should fail most, not 0%.
Disproven by data. Real cause was EFS non-convergence on specific
covariate × constraint × data interactions, not constraint count.
**Refs:** Cross-tab in `delete_once_model_selected.rmd`, session 2026-05-14.
