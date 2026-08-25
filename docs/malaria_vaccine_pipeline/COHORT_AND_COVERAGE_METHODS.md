# Cohort tracking and coverage — methods and assumptions

How delivered dose-3/dose-4 coverage becomes protection by age group, and every
choice made along the way. Companion to `docs/vaccine_efficacy/KNOWLEDGE.md` and
`DERIVATIONS.md`, which cover the VE curves. Written 2026-08-24.

## 1. The problem

Vaccination happens at fixed **ages** — dose 3 at 6 months, dose 4 at 24 months —
while coverage is reported by **calendar year**, and population comes in bins
that are not uniform width (sub-annual under 1, one year at 12–23 months, three
to five years above 2). Protection also depends on time since the dose.

Tracking by birth cohort resolves all three: for a cohort, two numbers are fixed
once assigned and never change again.

## 2. `dose_4` is unconditional — the load-bearing fact

`d3_ever` is the coverage reported for the year the cohort turned 6 months.
`d4_ever` is the `dose_4` value for the year they turned 2. **`dose_4` already
has the dropout applied**, so it is never multiplied by `d3_ever`.

Verified on the delivered Kebbi series: `dose_4(t)/dose_3(t-2)` settles to a
constant by 2029–2033. Double-applying would understate boosted coverage by a
factor of roughly `dose_3` — about 3× at plateau.

**Precision note.** That constant is **0.640–0.642**, not the 0.633 (= 1 − 0.367)
the handoff states — roughly 1pp apart. Per-location median implied ratios span
0.6332–0.6366 across all 373 locations. Do not hardcode 0.633 in a tolerance.

A separate figure, **0.619**, appears if you divide total dose-4 by total dose-3
across 2024–2100. That is *not* a dropout rate: summing a two-year-lagged series
over a finite window drops the two highest-coverage years and picks up two
near-zero ones. Cohort-matching recovers 0.6367.

## 3. Gating

Both fractions are 0 until the cohort has actually reached the trigger age, and
permanently 0 if the trigger year precedes the coverage series. This is an easy
off-by-one: a dose must not be reported before it was given.

## 4. Age reference — the one genuinely open choice

A child with `age_int` completed years at time `t + phi` was born in
`(t + phi - age_int - 1, t + phi - age_int]`, so **phi decides which calendar
years each trigger falls in**:

| convention | dose 3 | dose 4 |
|---|---|---|
| `start_of_year` (phi=0, **current default**) | 50/50 across two years | single year |
| `mid_year` (phi=0.5) | single year | 50/50 across two years |
| `end_of_year` (phi=1) | 50/50, shifted +1 | single year, shifted +1 |

The 50/50 blend is **not a property of dose 3** — it moves between the doses with
the convention. Under `start_of_year` a cohort's 6-month trigger straddles two
calendar years; under `mid_year` it does not, and the booster does instead.

**Status: PARKED.** Measured difference on real coverage is up to **6.7
percentage points** during ramp-up years (age 3 in 2029: 0.195 vs 0.262),
converging to ~0.4pp at plateau. Population estimates are commonly mid-year,
which would argue for `mid_year`.

**The regression test cannot settle it**: the synthetic harness evaluates ages at
`float(y)` — the same start-of-year convention — so it validates the code against
its own assumption. Changing the convention requires parameterising the harness
too, or the test is vacuous.

It is now `--age-reference`, so revisiting is a flag, not surgery. The default
reproduces every historical number bit-for-bit (verified by content hash).

## 5. Bin averaging

Bins ≤1 year wide and entirely below age 2 use a single representative age at the
bin midpoint — validated exact to numerical noise. Bins from age 1 upward
(12–23 months and wider) compute each single year of age and average.

The average is **uniform**, because no single-year-of-age population exists
anywhere in this pipeline — checked against `age_metadata.parquet`, where 238
(12–23 months) is the only bin at or above age 1 that is ≤1 year wide. Measured
peak error well under 1% of the bin during the fastest ramp years, vanishing at
plateau. If single-year population becomes available upstream, this approximation
and the dose-4 flow proxy both disappear.

Protection is averaged, **not VE**: coverage varies by birth cohort within a bin,
so VE cannot be pre-averaged independently of coverage. Averaging protection is
exact because protection is linear in `d3_ever`/`d4_ever` at fixed age.

## 6. Pre-series dose_4

38 locations report `dose_4 > 0` in 2024–25 while `dose_3` starts in 2024 —
boosters whose dose-3 antecedent predates the file (early RTS,S rollout). Two
treatments, both explicit:

- **`backcast`** (default): invert each location's own implied dropout ratio to
  recover the pre-series `dose_3`. Recovers 31.7M dose-3 and 19.0M dose-4
  person-years versus discarding them. No implied value exceeds 1; 37 of 38 give
  a monotone ramp into their 2024 value.
- **`zero`**: set those `dose_4` to 0. Discards real vaccinated children.

**Limit**: only 2022 and 2023 are recoverable — the years whose boosters appear
in the file. Real pilot dose 3 from 2019–2021 leaves no trace, so cohorts born
then still read 0.

## 7. Mortality invariance

Fraction-vaccinated is exact regardless of real-world mortality, **provided
mortality does not depend on vaccination status** — it thins vaccinated and
unvaccinated members of a cohort at the same rate. This is what lets the module
carry no population or mortality logic at all.

If the vaccine's mortality benefit were fed back into the age structure —
vaccinated children preferentially surviving and becoming over-represented in
older groups — that requires a hazard-based multi-state model and is out of
scope. Note this is **unchanged** by the arrival of `ve_death_*`: those columns
describe efficacy against death, they do not feed back into who is alive.

This matters more now than it did: we quote ~5M deaths averted, computed against
a population that assumes those deaths still occur.

## 8. Application to the forecast

The malaria forecast is age-less `(location, year, draw)`. Age enters post-hoc
via `disaggregation.py`'s as_rr → fractions, and protection is applied as

```
R(loc, yr) = SUM_as f(loc, yr, as) * protection(loc, yr, as)
vaccine    = novacc * (1 - R)
```

Exact, not an approximation: the fractions sum to 1 within each (location, year),
so this equals cell-wise multiplication and re-aggregation, without materialising
the age-specific array.

Protection is reported at admin1 and broadcast to admin2 children — also exact,
since the fractions carry no population weighting.

## 9. Doses

Deliberately **not** routed through any of the above. A dose count is

```
doses_3(t) = dose_3(t) x surviving_infants(t)
doses_4(t) = dose_4(t) x surviving_infants(t - 2)
```

one multiplication per location-year. "Surviving infants" is age groups
2+3+388+389 summed, spanning exactly one year; there is no births file in this
pipeline. Both proxies slightly *under*count, since mortality removes recipients
before they reach the bin.

Dose counts use the **raw** coverage file while protection uses the back-cast
one. Both are correct for their purpose: those 2024–25 boosters were genuinely
administered; only the protection calculation has no dose 3 to attribute them to.

## 10. Assumptions a reviewer will probe, in priority order

1. Age-reference convention (§4) — parked, up to 6.7pp in ramp years.
2. Uniform bin averaging (§5) — no single-year population exists.
3. Pre-series back-cast (§6) — and the unrecoverable 2019–21 pilot cohorts.
4. Mortality invariance (§7) — increasingly load-bearing as deaths-averted
   becomes the headline.
5. Dose-flow proxies (§9) — stock standing in for flow.
