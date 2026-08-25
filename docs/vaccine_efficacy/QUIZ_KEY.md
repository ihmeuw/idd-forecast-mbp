# Quiz answer key

**Q1.** Any two of: (a) dose-1-anchored ITT, not post-dose-3 per-protocol;
(b) mixed estimands (first-or-only pooled with all-episodes); (c) same-cohort
double-counting (2011 and 2014 are the same phase-3 cohort); (d) no severe/
mortality VE and no booster time-resolution.

**Q2.** "Agnandji 2011" and "Agnandji 2014" are the *same* phase-3 cohort
(NCT00866619) reported at successive follow-ups — the 2014 data include and
extend the 2011 data (same children, longer window). Treating them as independent
double-counts the person-time, which shrinks the CI and biases the pooled weight.

**Q3.** First-or-only-episode VE is a time-to-first-event (Cox/hazard) quantity;
all-episodes VE is a recurrent-event rate (negative-binomial/incidence-rate-ratio)
quantity. They differ because malaria is recurrent and protection against repeat
infections wanes differently. For a model applied to an incidence **rate**, the
**all-episodes** (rate) estimand is correct.

**Q4.** No — it is not evidence of no effect. The null was driven by the trial's
high standard of care suppressing malaria-specific mortality (only ~6.6% of trial
deaths were malaria-specific), leaving the study underpowered to detect a
mortality effect. Hence mortality is routed through severe malaria instead.

**Q5.** Assumption: severe VE is a constant fraction of clinical VE (they wane in
lockstep). Pinned pre-booster by 33.9% (severe, [18] Table 2) ÷ 45.1% (disease,
[18] Table 1) = ratio ≈ 0.75.

**Q6.** All severe/death curves change (`ve_death_d3`, `ve_death_d34` for both
products — the C/D/J/K points), because they are ratio × disease with a
constant ratio; a more-durable severe channel breaks the constant-ratio
assumption. All **case** curves (`ve_case_*`) are untouched — they are measured
directly and independent of the ratio.

**Q7.** Before the booster the two regimens are the same children (booster not yet
given), so protection is identical. The loader enforces it implicitly: it defines
the booster trigger as the first month where `ve_case_d3 != ve_case_d34`, and
requires that to be month 24 — so the columns must be equal before month 24.

**Q8.** The trial's first post-booster measurement is at ~age 30, not at the
booster age (24), so the reset height (the instantaneous post-boost peak) was
never measured. Active method **back_extrapolate**: extrapolate the two
post-booster anchors (age 30, 44) *back* to the booster age using the cell's form
(linear or log-linear).

**Q9.** **max_observed** → reset to the max observed dose-3 VE (primary-series
peak); generally **higher** than back-extrapolation. **shape_repeat** → replicate
the pre-booster decay shape, shifted to fit the post-booster anchors; can be
higher *or* lower (may exceed the primary peak), so not fixed — acceptable answer
notes it is indeterminate in direction.

**Q10.** R21 has its own phase-3 *disease* data (Datoo 2024) — peak, waning,
booster response — so its disease curve stands independently (linkage E). R21 has
**no** standalone *severe*-malaria VE time series, so the severe channel must
borrow the RTS,S severe:disease ratios applied to R21's disease curve (linkage F).
Asymmetry = disease measured for R21, severe not.

**Q11.** R21's booster-restoration level (~74%) was measured with a 12-month
dose3→booster gap. The deployment schedule puts the booster at age 24 = 18 months
after dose 3. Applying the measured restoration level at the longer interval is
the booster-interval assumption (reasonable, untested).

**Q12.** A single regression slope imposes one constant fractional decay across
the whole curve, but (a) the curve is not monotone — the booster is an upward
*reset*, so a single decay line can't pass through an up-jump; and (b) the data
contradict constant-slope log-linear decay even within the pre-booster window
(68→41→26 is not a straight line in log space). Piecewise interpolation (two
anchors per segment) respects both.

**Q13.** Log-linear decay asymptotes toward 0 but never reaches it, so it needs a
threshold ε to produce a genuine 0 (required by the loader). Linear extrapolation
hits 0 at a finite age exactly, so no threshold is needed.

**Q14.** The dose-3-only severe curve post-booster has essentially one usable
region and then heads to 0 — it lacks two positive post-booster points to define a
log-space slope (and log(0) is undefined). So `smooth`+`loglinear` borrows the
better-defined dose-3+4 severe slope.

**Q15.** VE is applied **after disaggregation**, on the age-specific counts
(the forecast itself is age-less; age enters post-hoc via RR→fractions). It
multiplies the age-specific incidence/mortality (in code, via the burden-weighted
collapse on all-age counts).

**Q16.** Coverage (C3/C4) varies by **birth cohort within the bin**, and VE
depends on coverage. So you must compose protection per single year of age from
that cohort's own C3/C4 and VE, then average the **protection** values — not the
VE values. Pre-averaging VE over the bin independently of coverage is wrong.
(Protection is linear in C3/C4 at fixed age, which makes averaging protection
exact.)

**Q17.** A child aged 3 in 2030 was born ~2027, got dose 3 at age 6 mo (2027) and
the booster at age 24 mo (2029). C3 is a blend because a single year of age spans
two calendar birth-years (≈50/50), each with its own dose-3 coverage; C4 is the
dose-4 value lagged appropriately from birth.

**Q18.** The age/sex fractions sum to 1 within each (location, year), so
`Σ_as f·protection` is a weighted average of protection, and applying `(1−R)` to
the all-age total equals applying protection cell-wise and re-aggregating. It buys
avoiding a large materialized age-specific array.

**Q19.** It inspects the **case** columns only: `first_dose3` = first month where
`ve_case_d3 > 0`; `first_boost` = first month where `ve_case_d3 != ve_case_d34`.
It raises unless `first_dose3 == 6` and `first_boost == 24`. It is **not** a
discontinuity/inflection/smoothness test — just the first month the two case
columns differ, so smooth-severe cells pass fine.

**Q20.** Byte-identical proves the YAML port reproduces the delivered curves
exactly — the anchors and construction logic moved without drift. A non-trivial
diff would indicate a real change (a mis-typed anchor, a float-representation
difference, a changed rounding or construction detail) that must be explained
before swapping over, rather than silently accepted.
