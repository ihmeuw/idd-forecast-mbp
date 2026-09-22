# Request from idd-forecast-malaria: what the second submission needs from the new model run

**Date:** 2026-09-21 · **From:** idd-forecast-malaria (Claude, for Bobby) · **Context:** number-plugging the malaria second submission from `2026_07_31_full_model_selection_results`. Bracketed items and the "decisions" section are Bobby's to fill in; everything else is read off the node and this repo's code.

Working inventory for the request to mbp. Updated: 2026-09-21 16:11. Facts from the node and this repo's code; decisions marked as Bobby's.

## 0. The new model's covariates define the holds

Registry record of `2026_07_31_full_model_selection_results` (models node, `run.json`):

- PfPR: logit suitability + s(GDP per capita, mpd) + s(DAH per capita, mpd) + country effect
- incidence, mortality: s(logit PfPR, mpi) + log GDP per capita + country effect

So the sensitivities that exist for this model are DAH, GDP per capita and
suitability (covariate holds or scenarios), plus the two demographic holds
applied to the burden arithmetic, population and age structure. Flooding
and urbanization are not in the model: the first submission's flooding
counterfactual and any urban hold no longer exist as sensitivities.

## 1. Sensitivities, new model

Every arm is needed for ssp126, ssp245 and ssp585, for incidence, mortality
and DALYs, as all-age draw-level series at the FHS locations (global,
super-region, region, country).

| # | arm | DAH scenario | hold | in the draft today | new run 2026_07_31 |
|---|---|---|---|---|---|
| 1 | reference | Baseline | none | every stats section; global_burden, super_region_burden, daly_maps, amenable_burden, mortality_rate_* maps; burden_by_super_region, cumulative_amenable_*, malaria_*_by_location (6) | forecast + product (base and `__gdpscen`), summary only |
| 2 | DAH constant | Constant | none | summary, discussion, results_counterfactuals; counterfactual_series_dah, counterfactual_dah_{dalys,deaths,cases} (+3 tables) | forecast + product (base and `__gdpscen`), summary only |
| 3 | GDP per capita held | Baseline | gdppc | summary, discussion, results_counterfactuals; counterfactual_series_gdppc, counterfactual_gdppc_* (+3 tables) | forecast `__gdppc_hold2023` (base and gdpscen); product under `__gdpscen__gdppc_hold2023`, summary only |
| 4 | suitability held | Baseline | suitability | not in the draft (V1 upload had the arm) | none; whether the paper wants it is Bobby's |
| 5 | population held | Baseline | population | results_counterfactuals (2100); counterfactual_series_population, counterfactual_population_* (+3 tables) | product `__denom_hold2023` (base and gdpscen), summary only; applied at finish |
| 6 | age structure held | Baseline | as_structure | results_counterfactuals; counterfactual_series_age_structure, counterfactual_age_structure_* (+3 tables) | product `__asstruct_hold2023` (gdpscen), summary only; applied at finish |
| 7 | suitability sensitivity | | | a run mbp must do for the paper (Bobby 2026-09-21); specification to come from Bobby: **[Bobby: what varies, which arms, which outputs]** | none |

Dropped with the covariates (in the draft today, no longer sensitivities):
the flooding hold, used by discussion (ssp585), results_counterfactuals
(ssp126, ssp585), counterfactual_series_flooding and
counterfactual_flooding_{dalys,deaths,cases} (+3 tables); and the main
figure `flooding_gdp` maps a covariate the model no longer has. What
replaces them in the paper is Bobby's.

Held in V1 and never consumed: `hold=DAH`. YLD and YLL arms existed in V1
for every combination and are not read directly.

Counterfactual figures and tables run over all three ssps (panels A to C)
and all three measures.

## 2. Measures

| measure | V1 | new run |
|---|---|---|
| incidence (cases) | yes | yes (log rate, admin 2, draws) |
| mortality (deaths) | yes | yes (log rate, admin 2, draws) |
| DALYs | yes | **no** |
| YLL, YLD | yes | **no** |

Bobby 2026-09-21: DALYs, YLLs and YLDs are impossible until other files
arrive; state this in the request.

## 3. Form of the outputs

| what | V1 (upload folders) | new run (05-products) | needed |
|---|---|---|---|
| level | FHS locations, all-age | FHS locations levels 0 to 3, all-age | FHS locations, all-age |
| draws | `draws.nc` + `mean.nc` per arm | mean/lower/upper only; draws collapsed inside `finish_run.py` per year (the rolled draw frame exists there) | draw level, by arm |
| holds | separate forecast arms | population and age-structure holds applied at finish; GDP hold a forecast run | either, as long as every arm in section 1 exists |

Bobby 2026-09-21: all-age draws at FHS level are the easy ask; age- and
sex-specific draws at admin 2 are the hard ask. Both go in the request.

## 4. Other inputs the paper reads

| input | V1 | new run | state |
|---|---|---|---|
| covariates by location, year, ssp | `cov_ds_Baseline.nc` (GDP pc, DAH pc, suitability, flooding pc, urbanization), one file, ssp dim, no draws, lsae_1209 admin 2 | `forecast_inputs/lsae_1285/20260803/malaria_forecast_inputs_{ssp}.nc`: gdppc_mean, mal_DAH_total_per_capita, malaria_suitability (+ flooding, urban, mean_low_temperature, A0 id, unused by the model), per ssp, draw and dah_scenario dims | present; readers need the rename and layout |
| hierarchy | lsae_1209 | lsae_1285 | present; maps need lsae_1285 geometry |
| population | lsae_1209 | lsae_1285 | present |
| DAH | `dah_df_2025_07_08.parquet` | `covariates/dah/20260527/dah_df.parquet` | present |
| GBD 2023 pull | 20260713 | same | present, already pinned |
| suitability draws (external, rapidresponse) | lsae_1209 | lsae_1285 path to confirm | to check when the pins move |
| flooding (external) | read for the flooding_gdp figure | not a model covariate | drop follows Bobby's call on that figure |

## 5. Not a request item

- Anchor-2023 check fails on every product run (58/400 cells): GBD is
  internally inconsistent across levels (Bobby 2026-09-21); expected.

## 6. Decisions that are Bobby's

- Which forecast run is the paper's reference: base
  `2026_07_31_full_model_selection_results` or `__gdpscen` (`current` points
  at gdpscen).
- The specification of the suitability sensitivity (row 7), and whether the
  plain suitability hold (row 4) is wanted as well.
- What replaces the flooding counterfactual (5 SI figures, 3 SI tables, two
  stats sections) and the `flooding_gdp` main figure.
