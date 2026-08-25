# Derivations and assumptions — with alternatives

Every value in the VE curves is one of three kinds: **measured** (a trial
reported it), **derived** (computed from a documented ratio/relationship), or
**assumption** (a modelling construction with no direct data). This file records
the derived and assumption points, *why* each is what it is, and *what else we
could do*. The alternatives matter: they are the reviewer's questions and the
paper's sensitivity analyses.

The construction has **two independent binary axes** (a 2×2 factorial), plus a
booster-reset method that is currently tied to one axis but could become a third.

---

## Axis 1 — interpolation / extrapolation form  (`axes.interpolation`)

How VE moves *between* observed anchors and *beyond* the last anchor.

- **linear** (`linear`): piecewise straight lines in VE space between consecutive
  anchors; the curve passes through every anchor exactly. Beyond the last anchor,
  extrapolate the last two anchors' slope straight down to 0.
- **log-linear** (`loglinear`): piecewise straight in log(VE) — constant
  *fractional* decay per segment (a local half-life). Beyond the last anchor,
  extrapolate the log-slope and cut to 0 at the threshold ε (label **T**).

**Why offered both.** Log-linear is more biologically motivated (antibody decay
is roughly exponential) and matches the `0.5^(τ/H)` half-life form used elsewhere.
Linear is simpler, reaches 0 without a threshold, and makes no mechanistic claim.
Neither uses regression — interpolation is strictly piecewise (two anchors per
segment), so a single global decay slope is never imposed (the data contradict a
constant-slope log-linear decay across the whole curve).

---

## Axis 2 — dose-3-only severe after the booster  (`axes.severe_post_booster`)

- **zero** (`zero`): dose-3-only severe VE drops to 0 at the booster age. Basis:
  [18] R3C cumulative severe VE 1.1% (ns) — unboosted severe protection is
  essentially gone by then (label **G**).
- **smooth** (`smooth`): dose-3-only severe keeps decaying past the booster age.
  Under log-linear it **borrows the dose-3+4 severe slope** (its own points can't
  define a log slope); under linear it scales the disease curve.

**Why offered both.** `zero` is the literal measured-data reading; `smooth`
avoids an abrupt cliff at an age defined by a dose the child did not receive.
The choice only affects the dose-3-only death channel.

---

## The booster reset  (label **R**, `reset_method`)

The dose-3+4 curve jumps up at the booster age. The **reset height is not
measured** — the trial's first post-booster measurement is at ~age 30, not at the
booster (age 24). Current method:

- **back_extrapolate** (active): extrapolate the two post-booster anchors (age 30,
  44) *back* to the booster age using the cell's form (linear or log-linear).
  Result: RTS,S reset ≈ 0.48 (linear) / 0.60 (log); R21 ≈ 0.76 / 0.80.

**Documented alternatives** (not yet active; would make reset a third axis):

- **max_observed**: reset to the maximum observed dose-3 VE (the primary-series
  peak — RTS,S 68%, R21 80%). "The booster puts you back where you started."
  Generally a *higher* reset than back-extrapolation.
- **shape_repeat**: replicate the pre-booster decay *shape* post-booster, shifted
  (in VE or log-VE space) to best-fit the post-booster anchors; the reset height
  falls out of the fit and may land above or below the primary peak.

**Why it matters.** The reset height is the tallest point of the booster bump and
it is a construction. Different methods move it materially, and the mortality
benefit of boosting scales with it.

---

## The severe:disease ratio  (labels **C, D, J, K**)

The entire severe/death channel is (ratio × disease curve). The ratio is pinned
by the trial's severe-vs-clinical summary numbers:

- **pre-booster ratio 0.75** = 33.9% (severe, [18] T2) ÷ 45.1% (disease, [18] T1),
  matched window. So pre-booster severe protection ≈ 75% of clinical.
- **boosted ratio 0.887** = 32.2% (R3R severe, the one significant post-baseline
  severe signal) ÷ 36.3% (R3R disease), cumulative.
- **dose-3-only post-booster ratio 0** = [18] R3C cumulative severe 1.1% (ns).

**LOAD-BEARING ASSUMPTION.** The whole death channel assumes severe VE is a
constant fraction of clinical VE, i.e. severe and clinical protection wane in
lockstep. They might not — severe protection could be more durable. If a reviewer
rejects this, every C/D/J/K point moves. This is the single biggest assumption to
probe, and the natural first sensitivity analysis.

---

## R21-specific derivations

- **E — disease independence** (linkage): the R21 disease curve is built from
  Datoo 2024's own numbers (peak ~80%, own booster timing), **not** rescaled from
  RTS,S. This is deliberately *not* the fallback "one number → rescale everything"
  approach; R21 disease stands on its own data.
- **F — severe borrowed** (linkage): R21 has **no** standalone severe-VE time
  series, so the R21 severe channel applies the RTS,S severe:disease ratios
  (0.75 / 0.887) to the R21 disease curve. This is the one genuine cross-vaccine
  transfer and the weakest link in the R21 file.
- **Booster-interval assumption**: R21's booster-restoration level (~74%) was
  measured with a 12-month dose3→booster gap; the deployment schedule applies it
  at a longer interval (booster at age 24 = 18 months after dose 3). Assuming the
  restoration level holds across the longer interval is reasonable but untested.

---

## The threshold  (label **T**, `constants.eps_threshold = 0.005`)

Log-linear decay never reaches exactly 0, so VE below ε = 0.5% is set to 0. The
age at which a curve crosses ε is its tail-to-zero age (RTS,S disease ~58 mo, R21
~86 mo). Linear cells reach 0 exactly and need no threshold. The loader requires
every curve to reach 0 by the last month, so ε (or the linear zero-crossing)
guarantees the contract is met.

**Alternative**: remove the threshold and let log-linear curves asymptote (a
tiny-but-positive tail out to 100 years). Rejected as default because the loader
wants a genuine 0, and sub-0.5% VE is not meaningfully protective.

---

## Age placement of anchors

Each measured period-VE is placed at the **midpoint age** of its reporting
window (e.g. PLoS Med's mo 1–6 → age 9; [18]'s mo 21–32 → age 30). This is an
assumption (period-average VE treated as instantaneous VE at the midpoint), minor
relative to the ratio and reset assumptions, but noted for completeness.

---

## Summary: the assumptions a reviewer will probe, in priority order

1. **Severe = constant fraction of clinical VE** (the whole death channel).
2. **Booster reset height is back-extrapolated** (not measured; method-dependent).
3. **R21 severe borrowed from RTS,S** (no R21 severe data).
4. **No transmission-dependence** (average-setting curves; see KNOWLEDGE §6).
5. **Interpolation form** (linear vs log-linear) and **severe-post-booster** (0 vs
   smooth) — the two factorial axes, offered explicitly so both can be shown.
6. R21 booster-interval, threshold ε, midpoint age placement (minor).
