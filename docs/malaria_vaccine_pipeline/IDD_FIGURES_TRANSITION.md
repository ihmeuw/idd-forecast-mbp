# Proposal — transitioning the vaccine figures to idd-figures

Written 2026-08-24. **A proposal, not a change.** Nothing here has been done.

## Why bother

The vaccine figures were built with raw matplotlib because idd-figures is not a
dependency of this repo. That was a deliberate deferral, but it has already cost
real bugs that a tested library would not have had:

| bug hit this session | what already exists |
|---|---|
| tick labels collided (`5M, 5M, 5M`) on a narrow axis | `idd_figures.lib.numbers.count_scale` / `smart_ui_format` |
| zero-span axis rendered `0, 0, -0, -0` | same |
| hand-rolled band drawing (`_band`) | `lib/viz/lines.timeseries_panel` — **already in this repo** |
| hand-rolled shared y-limits per row | `lib/viz/forecast_timeseries._shared_count_scale` |
| hand-rolled figure saving | `idd_figures.lib.io.save_figure` (does thumbnails too) |

Two of those bugs were in code that already existed, tested, twenty metres away.

## What is genuinely ours vs duplicated

**Duplicated** (should be deleted, not ported): `_band`, `_millions`,
`_lighten`, figure saving, colour lookup. All have equivalents.

**Genuinely ours** (keep, and it is the subject of the inbox note already filed
at `idd-figures/inbox/2026-08-24_idd-forecast-mbp_paired-arm-boxes-shared-frame.md`):

- paired arm boxes within a category, where the pairing axis is an arbitrary
  two-level scenario arm rather than two years
- five-number boxes as an alternative mark to lo/hi/med/mean bars
- the shared-frame layout — two panels sharing an edge, y-axes on the outside,
  for quantities on different scales that belong visually together

## Proposed sequence

**Step 1 — make it a dependency.** Add `idd-figures` as a path dep alongside
`climate-data` and `idd-tools` in `pyproject.toml`. Verify `uv sync --inexact`
resolves and that nothing in the existing suite breaks. This is the only step
with any packaging risk and it should land alone.

**Step 2 — adopt `numbers.py` first.** Replace `_millions` with `count_scale` /
`smart_ui_format`. Smallest surface, highest value: it is where both tick bugs
were, and it touches one function. Regenerate all figures and diff the PNGs —
tick labels will change, nothing else should.

**Step 3 — adopt `lib/viz/lines.timeseries_panel`** (this repo's, not
idd-figures') for the band drawing, deleting `_band`. Same regenerate-and-diff
check. Note this is an *internal* consolidation and does not depend on step 1.

**Step 4 — adopt `io.save_figure`** so figures get consistent naming and
thumbnails.

**Step 5 — decide on the box painter.** Either idd-figures generalises
`range_bars_panel` per the inbox note, in which case adopt it; or they decline,
in which case ours stays and is documented as intentionally local. Do not block
steps 1–4 on this.

## What I would NOT do

- Port everything at once. The figures are currently correct and reviewed;
  a big-bang port risks silently changing numbers on figures we have already
  discussed.
- Rebuild the layout logic. The 2×4 overview and the shared-frame column are the
  product of several rounds of your feedback; they should survive the port
  unchanged in appearance.
- Start before Phase C. Delivery threading is worth more than figure aesthetics.

## Acceptance test

The same one used for every Phase A refactor: regenerate every figure before and
after each step and compare. For figures, "unchanged" means the **box table**
(`vaccine_impact_box_table.parquet`, 144 rows of statistics generated from the
same arrays the boxes are drawn from) is bit-identical. Appearance may change;
numbers must not.

## Effort

Steps 1–4 are small — roughly a day, mostly regenerate-and-compare. Step 5
depends on idd-figures. The risk is not the work, it is doing it before the
pipeline is finished and thereby re-opening figures that are currently settled.
