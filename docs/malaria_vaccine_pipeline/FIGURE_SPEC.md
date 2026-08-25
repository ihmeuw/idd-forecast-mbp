# Malaria vaccine figures — requested set

Assembled 2026-08-24 from the working thread. Until now there was no written
list and figures were being built from recollection; this is the correction.
Status is honest, not aspirational.

## Axes that multiply through most figures

| axis | values | notes |
|---|---|---|
| VE variant | `loglinear_severe0`, `loglinear_severeSmooth` | case curves identical; only deaths differ |
| product scenario | `projected`, `all_r21` | dose counts unchanged by the swap |
| geography | `global`, `either`, `rtss`, `r21` | "either" = the 37 vaccine countries |
| weighting | population, deaths-2023-fixed, deaths-yearly | 2023 = observed/raked; yearly = forecast-derived |
| age domain | all 0–20, excluding never-coverable (2/3/388) | the second is "among children old enough to be vaccinated" |
| ssp | ssp126, ssp245, ssp585 | DAH Baseline only |

## Figures

| # | figure | status | notes |
|---|---|---|---|
| 1 | Overview 2×4 (levels, averted, cumulative boxes) per variant × product | **built** | 4 PNGs |
| 2 | Per-measure 4-row (annual/cumulative levels + averted) | **built** | 8 PNGs |
| 3 | VE variant comparison (severe0 vs severeSmooth) | **built** | 2 PNGs |
| 4 | Product comparison (projected vs all-R21) | **built** | 2 PNGs — NOT yet restricted to RTS,S countries |
| 5 | Coverage, 9 lines/panel × 3 weightings × 2 age domains | **built** | 4 PNGs, one per geography |
| 6 | Averted by age (x = age) — among the vaccinated | **NOT BUILT** | lines: RTS,S / R21 / either |
| 7 | Averted by age (x = age) — among everyone in vaccine places | **NOT BUILT** | denominator differs from #6 |
| 8 | Deaths averted per dose — global | **NOT BUILT** | numbers exist as a table only |
| 9 | Deaths averted per dose — by super-region | **NOT BUILT** | `super_region_map` now exists |
| 10 | Dose delivery time series (dose 3, dose 4, total) | **NOT BUILT** | `dose_counts` exists and is tested |
| 11 | Country choropleths: amount + % averted at 2050 / 2100 / cum-2050 / cum-2100 | **NOT BUILT** | `lib/viz/maps.py` exists to reuse |
| 12 | Product comparison restricted to RTS,S countries | **NOT BUILT** | current version dilutes ~4x by including R21 locations |
| 13 | Super-region cuts of the above | **NOT BUILT** | near-degenerate: 32/37 countries are Sub-Saharan Africa |

## Open questions

1. **"Out of all deaths that can be vaccinated"** — is that the existing
   "excluding never-coverable ages" row (deaths in ages ≥6 months within the
   chosen geography), or deaths in vaccinatable ages *globally* as the
   denominator? These differ and the second is not built.
2. **Panels vs lines** for #6–#11 — #5 settled at 9 lines in one panel; the
   others are unspecified.
3. **How much of the factorial** to render. The full cross is
   2 × 2 × 4 × 3 × 2 = 96 panels per figure family; the built set fixes
   variant/product and varies geography.

## Known caveats to carry onto any figure

- Coverage is an LME **projection** to 2100, not observed — hence `projected`,
  not `observed`.
- RTS,S = 71 admin1s in 12 countries; R21 = 302 in 25. No country has both, so
  "either" is a weighted average across disjoint sets and always lies between.
- Ages 2/3/388 are 2.5% of population but 18.7% of deaths and can never be
  covered — the whole population-vs-death weighting gap.
- Deaths-averted numbers depend on placeholder-free VE curves but on a single
  model run (`2026_07_31_full_model_selection_results`, DAH Baseline).
