# VE module comprehension quiz

Answer after reading `KNOWLEDGE.md` and `DERIVATIONS.md`. These target the
understanding that matters for using and defending the curves — not trivia.
Write answers inline or in a scratch file; check against `QUIZ_KEY.md`.

## Sources & estimands

**Q1.** The Asjad meta-analysis reports a pooled RR for RTS,S. Give two distinct
reasons it cannot be used to parameterize this waning cohort model. (There are
four in total.)

**Q2.** Two of the meta-analysis's three 5–17 mo forest-plot rows are not
independent studies. What is the relationship between them, and why does treating
them as independent bias the pooled estimate?

**Q3.** "First-or-only-episode VE" and "all-episodes VE" appear in the same
pooled number. Why are these different quantities, and which one is appropriate
for a model applied to an incidence *rate*?

## Channels

**Q4.** No trial found significant direct mortality VE. Is that evidence the
vaccine doesn't prevent malaria deaths? Explain what actually drove the null.

**Q5.** The entire severe/death channel rests on a single assumption. State it,
and name the two trial numbers whose ratio pins it for the pre-booster period.

**Q6.** If a reviewer argues severe-malaria protection is *more durable* than
clinical protection, which curves in the CSV change and which are untouched?

## Dose regimens & the booster

**Q7.** Before the booster age, `ve_case_d3` and `ve_case_d34` are identical in
the file. Why must that be true, and what enforces it in the loader?

**Q8.** The booster reset height (the top of the bump) is flagged as an
assumption, not a measured value. Why can't it be measured directly from the
trial, and what does the *active* method do to produce it?

**Q9.** Name the two documented alternative reset methods, and for each say
whether it would generally produce a *higher* or *lower* reset than the active
back-extrapolation.

## R21

**Q10.** The R21 disease curve is built from Datoo 2024, but the R21 severe curve
is borrowed from RTS,S. Explain the asymmetry — why is one independent and the
other borrowed?

**Q11.** What is the "booster-interval assumption" for R21, and why does the
deployment schedule (booster at age 24) create it?

## Construction mechanics

**Q12.** The interpolation is described as "piecewise, not regression." What would
go wrong if a single log-linear slope were fit by regression across a whole curve
(pre-booster through post-booster)?

**Q13.** The linear cells need no threshold (ε) but the log-linear cells do. Why?

**Q14.** In the `severeSmooth` + `loglinear` cell, the dose-3-only severe curve
"borrows the dose-3+4 severe slope" past the booster. Why can't it use its own
points to define that slope?

## Integration (the seam)

**Q15.** The forecast is age-less `(location, year, draw)`. At what point in the
pipeline does VE get applied, and to what does it get multiplied?

**Q16.** Why is it wrong to pre-average the monthly VE curve over an age group's
months to make a "VE per age_group_id" table? (Hint: what varies within the bin?)

**Q17.** Coverage C3 and C4 are "cohort-resolved, not calendar-year." For a child
aged 3 in 2030, when did they get dose 3 and the booster, and why is C3 a blend?

**Q18.** The application code computes `R = Σ f·protection` and applies `(1−R)` to
all-age counts instead of materializing age-specific counts. Why is that
algebraically identical, and what does it buy?

## Contract & reproducibility

**Q19.** The loader derives the dose-3 and booster trigger months from the file
and raises if they disagree with the cohort model. Which columns does it inspect,
and what exactly does it look for? (Not a smoothness test — be precise.)

**Q20.** The acceptance test for the YAML port is "byte-identical to the delivered
CSVs." Why byte-identical rather than "close enough," and what would a non-trivial
diff indicate?
