# Vaccine-efficacy module — handoff

Drop-in module for the malaria forecast's post-disaggregation VE layer. Built to
the repo's conventions as inspected: function-not-topic layout, YAML anchors with
provenance, construction in `02_data_prep/`, raw-anchors → processed-curves split.

## Read this first (in order)

1. `docs/KNOWLEDGE.md` — the epidemiology and the *why* (paper Methods seed).
2. `docs/DERIVATIONS.md` — every derived value + assumption, **with alternatives**.
3. `docs/VE_report.md` — fuller narrative with figures; `docs/ve_anchors.csv` —
   the flat provenance record.
4. `docs/QUIZ.md` — comprehension check; `docs/QUIZ_KEY.md` — key. Take the quiz
   before trusting downstream use of these curves.

## What's here

```
src/idd_forecast_mbp/VE_ANCHORS.yaml         anchors + assumption axes, provenance in comments
lib/processing/vaccine_efficacy.py           curve construction + VECurve + load_ve_curve
02_data_prep/09_build_vaccine_efficacy_curves.py   stage script (YAML -> processed CSVs)
docs/                                        KNOWLEDGE, DERIVATIONS, VE_report, ve_anchors.csv, QUIZ(+KEY)
outputs/                                      example run output (regenerable)
figures_deck/                                 figure/table/deck scripts + the built deck
```

## Scope (deliberately narrow — confirmed against the repo)

This module does **only**: build monthly VE curves from anchors, serve
continuous-age lookups (`VECurve.ve(vacc, column, age_years)`), and load/validate
against the forecast's contract. It does **not** touch:

- birth-cohort C3/C4 resolution, trigger gating, dose-4 lag, bin averaging →
  `lib/processing/vaccine_cohort_fractions.py` (already built, 111 tests).
- age_group_id binning / `age_metadata.parquet` → the stage script.
- application to counts (burden-weighted collapse) → already implemented.

The VE module never imports age groups. Its only age contract is
`VECurve.ve(vacc_name, column, age_years) -> float` (continuous age, linear
interpolation between whole months, 0 at/beyond the last month).

## Integration notes (wire to the real repo)

- **Anchors**: `VE_ANCHORS.yaml` sits at package root like `COVARIATE_DICT.yaml`;
  read via `lib/io/yaml.py` (resolves path from `constants.REPO_ROOT`).
- **Paths / RUN_DATE**: add to `constants.py` following the `*_READ_PATH` /
  `*_WRITE_PATH` / `RUN_DATE` pattern. The stage script currently takes them via
  env/relative paths as placeholders — repoint to `constants`.
- **Raw vs processed**: anchors are raw (never written by the pipeline). Curves
  write to `02-processed_data/malaria_vaccine_efficacy/<RUN_DATE>/` with a
  `current` symlink and a `DEFAULT_CELL.txt`.
- **load_ve_curve**: this module's `load_ve_curve` is the public loader; if the
  stage script already had one, replace it with this (same contract).

## Loader contract (enforced by `validate_ve_frame`)

- columns exactly `vaccine, age_months, ve_case_d3, ve_case_d34, ve_death_d3, ve_death_d34`
- `age_months` a gapless run from 0 per product
- values in [0,1], no nulls
- case channel encodes dose 3 (first month `ve_case_d3>0`) and booster (first
  month `ve_case_d3 != ve_case_d34`); raises unless 6 and 24
- every coverage product present (currently `rtss`, `r21`)
- each curve reaches 0 by the last month (currently month 1200)

## Acceptance test (run before swapping over)

Regenerate the four cells from `VE_ANCHORS.yaml` and diff against the delivered
CSVs in `figures_deck/` — must be **byte-identical**. Verified here:

```
python 02_data_prep/09_build_vaccine_efficacy_curves.py   # writes to outputs/
# then diff outputs/.../ve_<cell>.csv against the delivered ve_<cell>.csv
```

All four cells passed byte-identical at build time.

## The four cells (2×2 factorial)

|                | severe→0 at booster | severe smooth past booster |
|----------------|---------------------|----------------------------|
| **linear**     | `linear_severe0`    | `linear_severeSmooth`      |
| **log-linear** | `loglinear_severe0` | `loglinear_severeSmooth`   |

Default for the forecast: `loglinear_severe0` (in `VE_ANCHORS.yaml:build.default_cell`).

## Regenerating figures / tables / deck

`figures_deck/` holds the scripts (`plot_factorial.py`, `annotate_factorial.py`,
`tables_factorial.py`, `makedeck_full.js`) and the built `malaria_ve_deck.pptx`
(10 slides: 4 cells × table+plot, plus derived-values and assumptions slides for
the default cell). These are presentation/paper artifacts, not part of the
forecast runtime.

## Two future extensions flagged in the docs

1. **Reset method as a third axis** — `max_observed` and `shape_repeat` are
   documented in `DERIVATIONS.md` and stubbed in `VE_ANCHORS.yaml:reset_method`.
2. **Transmission-dependent VE** — the biggest modelling gap (KNOWLEDGE §6): make
   VE a function of local PfPR, not just age.
