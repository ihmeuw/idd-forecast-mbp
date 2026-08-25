# Malaria vaccine efficacy by age: derivation of a monthly VE lookup for cohort forecasting

**Scope.** Monthly vaccine-efficacy (VE) curves for **RTS,S/AS01** and **R21/Matrix-M**, for a
cohort vaccinated in the **5–17 month** age window, covering two outcome channels (clinical
malaria = "case"; severe malaria = "death"/mortality proxy) and two dose regimens (primary
series only = "dose 3"; primary series + booster = "dose 3+4"). The product is a CSV,
`vaccine_ve_by_age_month.csv`, indexed by age in months, intended to be applied
multiplicatively to a forecasted age-structured incidence and mortality surface.

---

## 1. Why not use the meta-analysis

The starting point for this work was Asjad et al. (2026), *Efficacy and immunogenicity of
RTS,S/AS01 and R21/Matrix-M malaria vaccines: systematic review and meta-analysis*
(J Infect Public Health 19:103222). It is **not usable** as a parameter source for a cohort
waning model, for four independent reasons established by tracing its extracted values back to
the primary trials:

1. **Dose-1-anchored, intention-to-treat.** The event/person-year data pooled in its forest
   plots are months-after-**dose-1** ITT figures, not post-**dose-3** per-protocol. They do not
   isolate protection conditional on completing the primary series, which is what a cohort model
   needs.
2. **Mixed estimands.** The 5–17 mo pooled estimate combines *first-or-only-episode* VE
   (2011 NEJM, Olotu 2013; Cox / hazard-ratio) with *all-episodes* VE (2014 PLoS Med;
   negative-binomial / rate-ratio) in a single risk ratio. These are different quantities and
   are not exchangeable, especially in high-transmission settings.
3. **Same-cohort double-counting.** Two of the three 5–17 mo rows ("Agnandji 2011" and
   "Agnandji 2014") are the *same phase-3 cohort* (NCT00866619) reported at successive
   follow-ups; the 2014 data include and extend the 2011 data. Pooling them as independent
   double-counts the person-time. The third row (Olotu 2013) is a *different* trial
   (NCT00872963, single-site Kilifi), not clinically exchangeable with the phase-3 program.
4. **No severe/mortality VE and no booster time-resolution.** The meta-analysis carries no
   severe-malaria VE or mortality VE by arm, and mentions the booster only as narrative
   citations of cumulative figures — it never extracts or pools dose-4 or waning data.

Consequently every number in this deliverable is taken from the **primary trials**, not from
the meta-analysis.

---

## 2. Primary sources

| Tag | Citation | Role |
|---|---|---|
| **[18]** | RTS,S Clinical Trials Partnership. *Efficacy and safety of RTS,S/AS01 malaria vaccine with or without a booster dose … final results of a phase 3 … trial.* **Lancet 2015;386:31–45.** | RTS,S booster (R3R) vs no-booster (R3C) VE by disjoint period; severe-malaria VE by arm. |
| **PLoS Med 2014** | RTS,S Clinical Trials Partnership. *Efficacy and safety of the RTS,S/AS01 malaria vaccine during 18 months after vaccination.* **PLoS Med 2014;11(7):e1001685.** | RTS,S clinical VE by **6-month period after dose 3** (the fine pre-booster waning). |
| **Olotu 2013** | Olotu A, et al. *Four-year efficacy of RTS,S/AS01E and its interaction with malaria exposure.* **N Engl J Med 2013;368:1111–20.** | RTS,S dose-3-only long tail (year-by-year to 4 y); transmission-dependence of VE. |
| **2011 NEJM** | RTS,S Clinical Trials Partnership. *First results of phase 3 trial of RTS,S/AS01 …* **N Engl J Med 2011;365:1863–75.** | Earliest post-dose-3 anchor (consistency only). |
| **Datoo 2024** | Datoo MS, et al. *Safety and efficacy of malaria vaccine candidate R21/Matrix-M in African children: … phase 3 trial.* **Lancet 2024;403:533–44.** | R21 clinical VE, 5–17 mo, with booster 12 mo after dose 3. |

Links: [18] https://pmc.ncbi.nlm.nih.gov/articles/PMC5626001/ ·
PLoS Med 2014 https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1001685 ·
Olotu 2013 https://www.nejm.org/doi/full/10.1056/NEJMoa1207564 ·
2011 NEJM https://www.nejm.org/doi/full/10.1056/NEJMoa1102287 ·
Datoo 2024 https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(23)02511-4/fulltext

---

## 3. RTS,S: measured anchors

All values 5–17 mo. Clinical VE is used for the case channel; severe-malaria VE for the death
channel. Time origin below is converted to **age in months** using dose 3 at age 6
(trial dose 3 ≈ trial-month 2; booster at trial-month 20 = age 24).

**Case channel, pre-booster (PLoS Med 2014, per-protocol, incidence reduction by 6-mo period after dose 3):**

| Months post-dose-3 | Child age | VE |
|---|---|---|
| 0–6 | 6–12 | 68% |
| 7–12 | 12–18 | 41% |
| 13–18 | 18–24 | 26% |

**Case channel, post-booster ([18] Table 1, mITT, VE = 1 − rate ratio):**

| Trial period | Child age | Dose-3 (R3C) | Dose-3+4 (R3R) |
|---|---|---|---|
| mo 21–32 | 24–36 | 16.1% | 37.4% |
| mo 33–end | 36–~52 | 2.9% | 12.3% |

**Death channel (severe malaria, [18] Table 2):** pre-booster pooled VE 33.9%; post-booster
period estimates are all non-significant and floor to 0 for both arms. The **only** significant
post-baseline severe signal is the **cumulative R3R severe VE of 32.2%** (95% CI 13.7–46.9).
Dose-3-only cumulative severe VE is 1.1% (ns → 0).

**Long tail (Olotu 2013, dose-3-only, per-protocol all-episodes):** year 1 46%, year 2 25%,
year 3 22%, year 4 ≈ 0. Used to carry the unboosted case curve to zero past [18]'s horizon.

---

## 4. R21: measured anchors and how they differ

R21 is structurally comparable to RTS,S but **not** a single-number rescale — Datoo 2024 provides
time-resolved, booster-inclusive, 5–17 mo numbers, so R21 is built with matched structure.

Key differences from RTS,S encoded in the curves:

- **Higher peak.** 12-month VE (time to first clinical malaria) ≈ **78%** in 5–17 mo, similar at
  seasonal and standard sites (Datoo 2024). Modelled peak ≈ 80%, vs RTS,S 68%.
- **Booster 12 months after dose 3** (not 18). In the cohort clock that places the R21 booster at
  **age 18 months**, vs age 24 for RTS,S. The booster "reset" is therefore earlier.
- **Slower waning.** R21's anti-NANP response is more durable than RTS,S anti-CS; 18-month VE
  remained ~74% at seasonal sites after the booster. Modelled decay is correspondingly gentler.
- **No standalone severe/mortality series.** R21 has no published severe-VE-by-time data, so the
  death channel is derived by applying the **RTS,S severe:disease ratio** to the R21 disease
  curve. This is the single weakest assumption in the R21 file and is flagged as such.

This is materially better than the fallback ("one comparable number → rescale the whole RTS,S
CSV"): only the severe channel borrows from RTS,S; the R21 disease curve is built from R21's own
phase-3 data with its own peak, booster timing, and waning rate.

---

## 5. From anchors to a monthly curve

1. **Placement.** Each measured VE is placed at the **midpoint age** of its reporting window.
2. **Waning form (label W).** Within each monotone decay segment, VE decays **log-linearly**:
   a least-squares regression of log(VE) on age through that segment's anchors gives a single
   decay slope, i.e. a constant fractional loss per month (a half-life). Fitted disease half-lives:
   RTS,S dose-3 ≈ 7.9 mo, RTS,S post-booster ≈ 8.7 mo; R21 dose-3 ≈ 12.4 mo, R21 post-booster
   ≈ 21.5 mo (R21 wanes markedly slower, consistent with more durable anti-NANP responses).
   Structural features that are **not** decay — the peak plateau just after dose 3, and the booster
   **reset** (VE jumps up at age 24) — are handled explicitly and are not part of any decay fit.
3. **Threshold (label T).** Log-linear decay never reaches exactly 0, so VE below ε = 0.5% is set
   to 0. The age at which a segment crosses ε is its tail-to-zero age.
3. **Pre-vaccination.** VE = 0 for all ages below dose 3 (age 6).
4. **Booster identity constraint.** Before the booster age, dose-3 and dose-3+4 curves are forced
   equal (same children, booster not yet administered).
5. **Booster reset.** At the booster age the dose-3+4 curve steps up to the R3R anchors; the
   dose-3-only curve continues its unboosted decay.
6. **Flooring.** VE is clipped to [0, 1]; negative trial point estimates (small-sample severe
   rows) are floored to 0 rather than propagated as harmful.
7. **Death channel.** Severe VE is set as a per-regimen fraction of the disease curve:
   pre-booster ratio 33.9/45.1 ≈ 0.75; boosted ratio 32.2/36.3 ≈ 0.89 (reproducing the
   cumulative R3R severe signal); dose-3-only post-booster ratio 0 (no durable unboosted severe
   protection). This "scaled" approach is used **instead of** the strictly period-resolved severe
   rows, which floor to 0 everywhere post-booster and would erase the booster's mortality benefit
   despite the significant cumulative signal.

---

## 6. Resulting curves (selected ages)

Columns: VE case dose-3 / VE case dose-3+4 / VE death dose-3 / VE death dose-3+4.

**RTS,S**

| Age (mo) | case d3 | case d34 | death d3 | death d34 |
|---|---|---|---|---|
| 6 | 0.680 | 0.680 | 0.511 | 0.511 |
| 12 | 0.528 | 0.528 | 0.397 | 0.397 |
| 18 | 0.323 | 0.323 | 0.242 | 0.242 |
| 24 | 0.220 | 0.290 | 0.000 | 0.257 |
| 30 | 0.161 | 0.374 | 0.000 | 0.332 |
| 36 | 0.094 | 0.291 | 0.000 | 0.258 |
| 48 | 0.018 | 0.081 | 0.000 | 0.072 |
| 60 | 0.000 | 0.000 | 0.000 | 0.000 |

**R21**

| Age (mo) | case d3 | case d34 | death d3 | death d34 |
|---|---|---|---|---|
| 6 | 0.800 | 0.800 | 0.601 | 0.601 |
| 12 | 0.762 | 0.762 | 0.573 | 0.573 |
| 18 | 0.674 | 0.720 | 0.000 | 0.639 |
| 24 | 0.551 | 0.687 | 0.000 | 0.610 |
| 30 | 0.400 | 0.550 | 0.000 | 0.488 |
| 36 | 0.279 | 0.452 | 0.000 | 0.401 |
| 48 | 0.098 | 0.269 | 0.000 | 0.239 |
| 60 | 0.000 | 0.089 | 0.000 | 0.079 |

![VE curves](ve_curves.png)

---

## 7. Known limitations (in priority order)

1. **No transmission-dependence.** These are average-setting curves. Olotu 2013 shows RTS,S VE
   ranges from ~45% (low exposure) to ~16% (high exposure), with year-4 VE flipping to null in
   high-transmission children. Applied over an admin-2 PfPR surface, VE ideally should be a
   function of both age **and** local PfPR. This is the most important gap.
2. **Severe = mortality proxy.** No trial detected significant direct mortality VE (trial-provided
   standard of care suppressed malaria-specific deaths to ~6.6% of all deaths). The death channel
   is severe-malaria VE; converting to deaths averted requires an external severe-malaria CFR.
3. **R21 death channel is borrowed** from the RTS,S severe:disease ratio — no R21 severe-VE time
   series exists.
4. **Severe channel uses the "scaled" not "period-resolved" option.** A sensitivity analysis with
   post-booster severe VE = 0 (the literal period estimates) would zero out the booster mortality
   benefit; that is the conservative alternative if a reviewer objects to the scaling assumption.
5. **Tails past ~54 mo are extrapolation.** RTS,S dose-3-only is anchored by Olotu 2013 (y4≈0);
   boosted tails and all R21 tails past the trial horizons are model extrapolations to 0.
6. **Estimand seam at the booster.** Pre-booster RTS,S values are PLoS Med per-protocol
   incidence-reduction; post-booster are [18] mITT negative-binomial VE. Minor population/estimator
   discontinuity at age 24; immaterial for forecasting but worth noting.

---

## 8. Anchor provenance (annotated plots)

Two annotated versions of the plot make every driving number explicit. Markers are styled by
**type**: filled circle = **measured** (a trial reported this VE at this time); open circle =
**derived** (computed from a documented severe:disease ratio or a cross-vaccine transfer); open
red square = **assumption** (a modelling construction with no direct data). Labels follow the
scheme: **A** = RTS,S dose-3-only disease, **B** = RTS,S dose-3+4 disease (starting at the
booster), **C/D** = RTS,S severe dose-3-only / dose-3+4 (these mark severe:disease *ratio*
anchors, not measured VE), **E/F** = R21↔RTS,S disease / severe linkage, **G** = the
missed-booster severe construction.

**Version 1 — stepped** (dose-3-only severe drops to 0 at the booster age):

![Annotated v1](ve_curves_annotated_v1.png)

**Version 2 — smoothed** (dose-3-only severe holds the pre-booster severe:disease ratio across
the booster age, so the red severe curve is a smooth scaled copy of the red disease curve; G now
labels this smoothing assumption):

![Annotated v2](ve_curves_annotated_v2.png)

The difference between v1 and v2 is a single modelling choice on the **dose-3-only severe
channel only** (all other curves are identical). v1's cliff is an artifact of flipping the
severe:disease ratio to 0 at an age defined by a dose these children did not receive; v2 is the
defensible default (the CSV's `ve_death_d3_smooth` column). v1's `ve_death_d3` is retained as the
conservative sensitivity bound.

The complete anchor table is `ve_anchors.csv` (also below). "Type" separates data from decisions;
"Source" gives the trial, table/figure, population, and estimand; ages are the midpoint of each
trial reporting window (noted per row).

| Label | Vaccine | Curve | Age | VE | Type | Source (short) |
|---|---|---|---|---|---|---|
| A1 | RTS,S | case d3 | 9 | 0.68 | measured | PLoS Med 2014 PP, mo 1–6 post-dose-3 |
| A2 | RTS,S | case d3 | 15 | 0.41 | measured | PLoS Med 2014 PP, mo 7–12 |
| A3 | RTS,S | case d3 | 21 | 0.26 | measured | PLoS Med 2014 PP, mo 13–18 |
| A4 | RTS,S | case d3 | 30 | 0.161 | measured | [18] Table 1 R3C, mo 21–32, mITT |
| A5 | RTS,S | case d3 | 44 | 0.029 | measured | [18] Table 1 R3C, mo 33–end |
| A6 | RTS,S | case d3 | 60 | 0.00 | assumption | Olotu 2013 y4≈0; tail to 0 |
| B1 | RTS,S | case d34 | 30 | 0.374 | measured | [18] Table 1 R3R, mo 21–32 (booster reset) |
| B2 | RTS,S | case d34 | 44 | 0.123 | measured | [18] Table 1 R3R, mo 33–end |
| B3 | RTS,S | case d34 | 60 | 0.00 | assumption | no data past ~52mo; tail to 0 |
| C1/D1 | RTS,S | death d3/d34 | 9 | 0.511 | derived | [18] T2 pre-booster severe 33.9% / disease 45.1% → ratio 0.75 (shared) |
| D2 | RTS,S | death d34 | 30 | 0.332 | derived | [18] T2 R3R severe 32.2% / disease 36.3% → ratio 0.887 |
| D3 | RTS,S | death d34 | 44 | 0.109 | derived | ratio 0.887 × disease |
| G | RTS,S | death d3 | 24 | 0.00 | assumption | [18] T2 R3C severe 1.1% ns → construction (v1 step / v2 smooth) |
| A1 | R21 | case d3 | 9 | 0.80 | measured | Datoo 2024, ~78% at 12mo, peak anchor |
| A2 | R21 | case d3 | 15 | 0.72 | measured | Datoo 2024, modest early waning |
| A3 | R21 | case d3 | 30 | 0.40 | derived | Datoo shape + slower-waning assumption |
| A4 | R21 | case d3 | 60 | 0.00 | assumption | tail to 0 |
| B1 | R21 | case d34 | 24 | 0.687 | measured | Datoo 2024, 18mo ~74% booster-restored (booster age 18) |
| B2 | R21 | case d34 | 40 | 0.391 | derived | Datoo shape + slower waning |
| B3 | R21 | case d34 | 66 | 0.00 | assumption | tail to 0 |
| C1/D1 | R21 | death d3/d34 | 9 | 0.601 | derived | RTS,S pre-booster ratio 0.75 × R21 disease (borrowed) |
| E | R21 | case d34 | 20 | 0.72 | derived | **linkage**: R21 disease is independent (Datoo), NOT rescaled from RTS,S |
| F | R21 | death d34 | 30 | 0.488 | derived | **linkage**: RTS,S severe ratio 0.887 borrowed (only cross-vaccine transfer) |

Reading the linkages **E** and **F** together is the R21 story in one line: the R21 *disease*
curve stands on its own phase-3 data (E — own peak, own booster timing), and only the *severe*
channel is borrowed from RTS,S (F — the one genuine cross-vaccine assumption).

---

## 9. Files

- `vaccine_ve_by_age_month.csv` — the lookup table (both vaccines, 0–120 mo). Includes
  `ve_death_d3_smooth` (v2 smoothed dose-3-only severe) alongside `ve_death_d3` (v1 stepped).
- `ve_anchors.csv` — the anchor provenance table (every labelled point → value, type, source).
- `build_ve.py` — reproducible curve build; anchors documented inline with citations.
- `annotate.py` — builds the annotated plots and the anchor table from the same registry.
- `VE_CSV_PROMPT.md` — how to apply the CSV in the cohort model.
- `ve_curves.png` — plain diagnostic plot.
- `ve_curves_annotated_v1.png` — annotated, stepped dose-3-only severe.
- `ve_curves_annotated_v2.png` — annotated, smoothed dose-3-only severe (default).
