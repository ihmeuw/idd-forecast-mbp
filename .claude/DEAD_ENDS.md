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
