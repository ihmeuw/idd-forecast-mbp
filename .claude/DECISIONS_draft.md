<!-- DRAFT — generated 2026-05-06 17:30 without review -->
<!-- Claude notes:
  - Three real decisions were made this session
  - The dimensional structure decision is the most important and should be preserved verbatim
  - The dengue_suitability placement decision is a concrete implementation choice
  - The flooding-as-scalar decision was implicit but real — Bobby confirmed flooding doesn't vary by draw
-->

## 2026-05-06: Replace parquet-per-draw storage with dimension-aware netCDF
**Decision:** Replace ~1,500 parquet files (one per draw/scenario/cause) with a small number of xarray netCDF files where each variable carries only the dimensions it actually varies over.

**Past input structure (malaria):**
- `malaria_past_inputs.nc`: loc×year (AA outcomes, DAH, gdppc, ldipc, urban, flooding), loc×year×draw (8 climate vars), loc×year×draw×suit_variant (malaria_suitability, 14 variants)

**Past input structure (dengue):**
- `dengue_past_inputs.nc`: loc×year×age×sex (AS outcomes), loc×year (gdppc, ldipc, urban, flooding), loc×year×draw (8 climate vars + dengue_suitability)

**Why:** Current storage is ~5TB due to duplicating all-age outcomes and non-draw covariates across 100 draw files × 3 SSP scenarios. xarray lets each variable carry only the dimensions it actually varies over, eliminating the redundancy.

**Revisit if:** File sizes become unmanageable (estimated ~750GB total for past+future), or if R/downstream tools cannot read netCDF efficiently.

## 2026-05-06: dengue_suitability belongs in dengue past inputs only, via extra_vars
**Decision:** `read_draw_climate()` in `lib/io/array_builders.py` does NOT include `dengue_suitability` by default. The dengue past inputs script adds it via `extra_vars={'dengue_suitability': ...}`. The malaria past inputs script does not include it at all.

**Why:** dengue_suitability is a dengue-specific predictor. Including it in the malaria file wastes space and is scientifically wrong. Malaria uses the 14 Mordecai/Villena suitability variants instead.

**Revisit if:** Never — this is a clear disease-specificity boundary.

## 2026-05-06: flooding is a scalar (non-draw) covariate in past inputs
**Decision:** Flooding goes into `read_shared_covariates` (loc×year scalar), not into `read_draw_climate` (loc×year×draw). The flooding parquet uses `_mean_r1i1p1f1` naming, confirming it is already ensemble-averaged.

**Why:** Bobby confirmed flooding does not vary by draw. Including it in the draw dimension would inflate file size by 100x for no information gain.

**Revisit if:** A draw-varying flooding product becomes available.
