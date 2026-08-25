# Phase C — threading vaccination into delivery

Scoped 2026-08-24. The **application layer is implemented** (see "What is
done"); what remains is wiring it to the live path, which depends on
`finalize_forecast` going live.

## What the delivery path actually consumes

`06_upload/create_and_combine_as_draws.py` and its siblings read

```
{UPLOAD_DATA_PATH}/upload_folders/{run_date}/
    full_as_{cause}_measure_{measure}_ssp_scenario_{ssp}{dah_text}_draw_{draw}_with_predictions{hold_text}.nc
```

i.e. **age-specific, draw-level** products. Optional dimensions are carried as
appended filename segments (`dah_text`, `hold_text`) rather than as extra path
components — that is the idiom a vaccine dimension should follow.

## The mismatch with what the vaccine work currently produces

The impact pipeline built in Phase A operates on the **all-age** forecast and
emits **aggregated totals** for figures. Delivery needs the reduction applied to
per-`(location, year, age_group, sex, draw)` products.

That is not a hard problem — the reduction factor `R(loc, year)` is draw-free and
already computed by `burden_weighted_reduction` — but it has to be applied where
the age-specific draws are built, which is not where the current code sits.

## What is done

`finalize_age_sex_draws` now takes an optional `protection` frame and applies it
**cell-wise** to the disaggregated admin-2 counts before roll-up, so every
aggregate inherits it. `None` is the no-vaccine product -- no separate path.

The cell-wise detail is the point. Scaling all-age counts by the burden-weighted
`R` and then disaggregating gives the **same total** -- the fractions sum to 1 --
but spreads the reduction uniformly across ages. For an age-specific delivery
product that is wrong: it shows a reduction among 15-19 year olds who have no
protection and understates it among the under-fives who have most of it. A test
pins both halves of that: totals agree, age distributions do not.

`apply_protection_to_age_sex` lives in `lib/processing/vaccine_impact.py`,
6 tests, module at 100%.

## The right integration point

`lib/processing/finalize_forecast.py::finalize_age_sex_draws` — it already
produces `[location_id, year_id, age_group_id, sex_id, draw, {m}_count]`, which
is exactly the grain the reduction multiplies into. Applying it there means:

- one multiplication, no new machinery
- every downstream product (all-age rollup, summaries, upload folders) inherits
  it automatically
- the no-vaccine case is `vaccinated=False`, i.e. a reduction of 0 — no separate
  code path

## Why it is blocked

**Nothing live calls `finalize_forecast.py` yet.** It is the formalized
product-building layer from the 2026-08-04 formalization front; the live path
still runs through the `06_upload` scripts, which build products themselves.

So there are two options and only one of them is right:

1. **Wait for `finalize_forecast` to be wired in**, then add the reduction there.
   One multiplication, inherits everywhere, matches Bobby's instruction that
   malaria is the template dengue conforms to.
2. **Bolt the reduction onto the `06_upload` scripts now.** Faster, but those are
   precisely the scripts the formalization front is replacing, and it would spread
   vaccine logic across six files that each rebuild products their own way.

Recommendation: option 1. Phase C should not start until `finalize_forecast` is
on the live path.

## What to add when it is unblocked

| dimension | values | filename segment |
|---|---|---|
| `vaccinated` | `False`, `True` | `""` / `_vaccinated` |
| `ve_variant` | 4 factorial cells | `_ve_{variant}` (only when vaccinated) |
| `product_scenario` | `projected`, `all_r21` | `_{products}` (only when vaccinated) |

Follow `dah_text`/`hold_text`: an empty string when the dimension is off, so
existing paths are unchanged.

## Adjacent findings

- `06_upload/create_and_combine_as_draws.py:82` reads
  `full_hierarchy_lsae_1209.parquet` — the retired hierarchy. Third stale 1209
  reference found this session.
- Dengue's `vaccinate` flag is **already vestigial**: commented out in
  `create_and_combine_aa_draws.py:44`, live only in `OLD_*` files and a notebook.
  So there is no working convention to copy, which supports building it here
  first.
