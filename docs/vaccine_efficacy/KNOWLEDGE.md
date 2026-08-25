# Malaria vaccine efficacy — knowledge base

This is the *why* behind the VE curves the module builds. It is written to be
read start-to-finish before touching the code, and to seed the Methods/Supplement
of the forecasting paper. Companion files: `DERIVATIONS.md` (every number's
provenance + alternatives), `VE_report.md` (the fuller narrative + figures),
`QUIZ.md` (self-check).

## 1. What we are estimating

Vaccine efficacy (VE) against malaria, by **age in months**, for a child
vaccinated in the WHO-recommended **5–17 month** window, for the two licensed
vaccines **RTS,S/AS01** and **R21/Matrix-M**. Two outcome channels — clinical
malaria ("case") and severe malaria used as a mortality proxy ("death") — and
two dose regimens — primary series only ("dose 3") and primary series plus
booster ("dose 3+4").

"Matrix-M" is R21's saponin-based adjuvant (Novavax); "AS01" is RTS,S's adjuvant
(GSK). Both vaccines target the *P. falciparum* circumsporozoite protein; the
adjuvant difference is part of why their efficacy/durability profiles differ.

## 2. Why the meta-analysis is not a usable source

The natural starting point — Asjad et al. (2026), the RTS,S/R21 systematic
review and meta-analysis — cannot parameterize a waning cohort model. Four
independent reasons, all established by tracing its numbers back to the primary
trials:

1. **Dose-1-anchored, intention-to-treat.** Its pooled event/person-year data
   are months-after-**dose-1** ITT figures, not post-**dose-3** per-protocol.
   They do not isolate protection conditional on completing the primary series.
2. **Mixed estimands.** Its 5–17 mo pooled RR combines *first-or-only-episode*
   VE (Cox/hazard) with *all-episodes* VE (negative-binomial/rate) in one number.
   These are different quantities and are not exchangeable, especially at high
   transmission.
3. **Same-cohort double-counting.** Two of its three 5–17 mo rows ("Agnandji
   2011" and "Agnandji 2014") are the *same* phase-3 cohort (NCT00866619)
   reported at successive follow-ups — the 2014 data include and extend the 2011
   data. Pooling them as independent double-counts the person-time. The third
   row (Olotu 2013) is a *different* trial (NCT00872963, single-site Kilifi).
4. **No severe/mortality VE and no booster time-resolution.** It carries no
   severe-malaria VE or mortality VE by arm, and mentions the booster only as
   narrative citations of cumulative figures.

Conclusion: every number in this module comes from the **primary trials**, not
the meta-analysis.

## 3. Primary sources

| Tag | Citation | Role |
|---|---|---|
| **[18]** | RTS,S Clinical Trials Partnership, *Lancet* 2015;386:31–45 (final phase-3, with booster) | Disease VE by disjoint period, R3C vs R3R; severe-malaria VE by arm |
| **PLoS Med 2014** | RTS,S Clinical Trials Partnership, *PLoS Med* 2014;11(7):e1001685 | Disease VE by 6-month period after dose 3 (fine pre-booster waning) |
| **Olotu 2013** | Olotu et al., *NEJM* 2013;368:1111–20 (NCT00872963, Kilifi) | Dose-3-only long tail; transmission-dependence of VE |
| **2011 NEJM** | RTS,S Clinical Trials Partnership, *NEJM* 2011;365:1863–75 | Earliest post-dose-3 anchor (consistency) |
| **Datoo 2024** | Datoo et al., *Lancet* 2024;403:533–44 | R21 disease VE, 5–17 mo, with booster 12 mo after dose 3 |

## 4. The two channels

**Case (clinical malaria).** Directly measured, age-resolved. RTS,S: PLoS Med
2014 gives 6-month-period VE after dose 3 (68% / 41% / 26%); [18] gives the
post-booster periods split into R3C (no booster: 16.1%, 2.9%) and R3R (booster:
37.4%, 12.3%). R21: Datoo 2024 gives ~78% at 12 months with a slower decay.

**Death (severe malaria as mortality proxy).** *No trial found significant
direct mortality VE* — attributed to trial-provided high standard of care
suppressing malaria-specific deaths (only ~6.6% of trial deaths were
malaria-specific). So mortality is routed through **severe malaria**, which is
the direct antecedent of malaria death. The severe channel is not measured as a
curve; it is constructed as (ratio × disease curve) — see `DERIVATIONS.md`.

## 5. Dose 3 vs dose 3+4

Before the booster, the two regimens are **identical** — same children, booster
not yet given — so there is one curve until the booster age. The booster
*resets* protection upward, then it wanes again. Without the booster, dose-3-only
disease VE collapses toward zero by ~4 years and dose-3-only severe VE is
essentially null ([18] R3C cumulative severe 1.1%, ns). The booster's benefit is
the gap between the two curves after the booster age.

## 6. The single most important caveat: transmission-dependence

These are **average-setting** curves. Olotu 2013 shows RTS,S VE is strongly
transmission-dependent — ~45% at low exposure vs ~16% at high exposure, with
year-4 VE flipping toward null in high-transmission children. Applied over an
admin-2 PfPR surface, VE ideally would be a function of both age **and** local
PfPR. The current curves do not encode this. It is the first thing to revisit if
the forecast needs setting-specific VE.

## 7. How VE enters the forecast (for context — not built in this module)

The malaria forecast is age-less: `(location, year, draw)`. Age enters
post-hoc via disaggregation (RR → fractions). VE is applied **after
disaggregation**, as a multiplier on age-specific counts, using birth-cohort-
resolved dose-3 / dose-4 coverage. In code this is the burden-weighted collapse
`R(loc,year) = Σ_as f · protection`, applying `(1 − R)` to all-age counts —
algebraically identical to cell-wise multiplication because the fractions sum to
1. The cohort resolution of coverage (a child aged 3 in 2030 got dose 3 in 2027,
booster in 2029) and the age-group binning live in
`vaccine_cohort_fractions.py` and the stage script — **not** in this module.
This module only builds the monthly curve and serves continuous-age lookups.
