# Malaria vaccine pipeline — runbook

How to produce vaccine impact results, and how to produce them again under
different assumptions. Written 2026-08-24.

## One command

```bash
.venv/bin/python src/idd_forecast_mbp/04_forecasting/run_malaria_vaccine_pipeline.py \
    --ve-variant loglinear_severe0 \
    --product-scenario projected \
    --age-reference start_of_year \
    --prelag backcast \
    --out-dir /mnt/team/idd/pub/forecast-mbp/07-figures/<RUN>/
```

~53s for the data chain, ~2 min with figures. Every stage runs in-process, so a
failure stops the chain rather than leaving a half-updated output set. The
assumption set is echoed at the top of every run.

Useful flags: `--skip-curves` (reuse the built VE curves), `--skip-figures`.

## The assumption set

| flag | values | default | what it changes |
|---|---|---|---|
| `--ve-variant` | 4 factorial cells | `loglinear_severe0` | VE curve shape; case curves identical between the two `loglinear` cells, only deaths differ |
| `--product-scenario` | `projected`, `all_r21` | `projected` | which product each location uses; dose *counts* are unchanged |
| `--age-reference` | `start_of_year`, `mid_year`, `end_of_year` | `start_of_year` | where in the reporting year age is evaluated — moves the 50/50 blend between dose 3 and dose 4. **PARKED decision**, see DECISIONS.md 2026-08-24 |
| `--prelag` | `backcast`, `zero`, `none` | `backcast` | dose_4 whose dose-3 antecedent predates the coverage series |

Non-default `--age-reference` keys its own output filename, so it cannot
overwrite the default run.

## Stages, if you need to run them individually

| # | script | reads | writes | time |
|---|---|---|---|---|
| 1 | `02_data_prep/09_build_vaccine_efficacy_curves.py` | `VE_ANCHORS.yaml` | `02-processed_data/malaria_vaccine_efficacy/<RUN_DATE>/` + `current` | <1s |
| 2 | `04_forecasting/apply_vaccine_coverage_to_population.py` | VE curves, coverage CSV, age-sex population | `04-forecasting_data/malaria/vaccine_cohorts/<hier>/<RUN_DATE>/` | ~18s |
| 3 | `04_forecasting/vaccine_impact_scenarios.py` | cohort protection, forecast netCDF, as_rr inputs | `<out-dir>/vaccine_impact_{summary,draws}_*.parquet` | ~35s |
| 4 | `08_visualization/plot_vaccine_{impact,coverage,doses}.py` | the parquets above | PNGs + `vaccine_impact_box_table.{parquet,csv}` | ~60s |

## Inputs and where they come from

| input | path | notes |
|---|---|---|
| VE anchors | `src/idd_forecast_mbp/VE_ANCHORS.yaml` | in-repo, human-edited, provenance in comments |
| coverage | `01-raw_data/malaria_vaccine_coverage/current/` | received; an LME **projection** to 2100, not observed |
| age-sex population | `02-processed_data/population/<hier>/current/as_2023_full_population_df.parquet` | 2.97 GB — always read with predicate pushdown |
| age metadata | `02-processed_data/age_specific_fhs/age_metadata.parquet` | bin bounds; never hardcode them |
| malaria forecast | `04-forecasting_data/malaria/forecast_outputs/<hier>/current/` | model `2026_07_31_full_model_selection_results`, DAH **Baseline** |
| age/sex pattern | `02-processed_data/malaria/raked_as/<hier>/current/` | historical only, 2000–2023 |

## Gotchas that have already cost time

- **netCDF reads must be contiguous.** `.to_numpy()` then subset in numpy.
  `.sel(location_id=[...])` on 60 scattered locations takes 36s versus 3s for
  reading the whole 317 MB variable; on 1,986 it did not finish in 600s.
- **Renaming outputs leaves the old files behind.** Stale PNGs and parquets have
  twice been mistaken for current ones. `plot_vaccine_impact.py` now warns about
  PNGs it did not write; the cohort and impact nodes have no such guard.
- **Dose counts use the RAW coverage file, protection uses the back-cast one.**
  Both are correct: the 2024–25 boosters were really administered, but the
  protection calculation has no dose 3 to attribute them to.
- **Population reaches back to 2022**, which `doses_4(2024)` needs via
  `surviving_infants(2022)`. Verified; a missing lagged cohort yields NaN, never
  a silent zero.

## Verifying a change did not alter results

Take content hashes before, re-run, compare:

```python
h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
```

Every refactor in Phase A was accepted on this basis. The VE curves additionally
have sha256s pinned in `tests/lib/processing/test_vaccine_efficacy.py`, so a
change in anchors or construction fails loudly.
