# Climate Pipeline Documentation

## Overview

This document describes how custom malaria suitability curves are run through the `climate-data` pipeline to produce population-weighted, admin-level results with 100 draws.

## Curve Definitions

**3 Methods × 7 Shifts = 21 Curves**

| Methods | Shifts |
|---------|--------|
| mordecai | 0_0 (no shift) |
| symons | m0_25 (-0.25°C) |
| villena | m0_5 (-0.5°C) |
| | m1_0 (-1.0°C) |
| | p0_25 (+0.25°C) |
| | p0_5 (+0.5°C) |
| | p1_0 (+1.0°C) |

## Variable Naming Convention

```
malaria_{method}_{shift}_suitability
```

Examples:
- `malaria_mordecai_0_0_suitability`
- `malaria_symons_m0_5_suitability`
- `malaria_villena_p1_0_suitability`

## Pipeline Steps

### Step 1: Generate Annual Suitability Rasters

```bash
cdrun generate scenario_annual --run-malaria-only
```

**What it does:**
- Reads daily temperature rasters
- Applies each curve's temperature→suitability mapping
- Aggregates to annual mean suitability
- Produces one raster per curve × scenario × GCM × year

**Input:** Daily temperature NetCDF files
**Output:** Annual suitability rasters at `MODEL_ROOT/results/scenario_annual/{variable}/{scenario}/{gcm}/{year}.nc`

### Step 2: Generate Draws

```bash
cdrun generate draws --run-malaria-only
```

**What it does:**
- Takes the 22 GCM outputs from step 1
- Creates 100 draws by sampling GCMs with replacement
- Saves draw→GCM mapping as parquet

**Input:** GCM-specific annual rasters
**Output:** 
- Draw rasters at `MODEL_ROOT/results/draws/{variable}/{year}.nc`
- Draw mapping at `MODEL_ROOT/results/draws/{variable}/draw_mapping_{variable}.parquet`

### Step 3: Pixel Aggregation

```bash
cdrun aggregate pixel --hierarchy lsae_1209 --run-malaria-only
```

**What it does:**
- Reads draw rasters + population weights
- Computes population-weighted mean for each admin unit
- Produces parquet files per draw

**Input:** Draw rasters + population model
**Output:** `AGGREGATE_ROOT/{hierarchy}/pixel/{measure}/{scenario}/{draw}.parquet`

### Step 4: Hierarchy Aggregation

```bash
cdrun aggregate hierarchy --hierarchy lsae_1209 --run-malaria-only
```

**What it does:**
- Takes pixel-level admin results
- Rolls up to full location hierarchy (parents from children)
- Produces final output files

**Input:** Pixel parquet files
**Output:** `AGGREGATE_ROOT/{hierarchy}/hierarchy/{measure}/{scenario}.parquet`

## File Locations

| Type | Path |
|------|------|
| Curve parquet files | `climate-data/src/climate_data/generate/supplementary_data/malaria_suitability_method_{method}_shift_{shift}.parquet` |
| MODEL_ROOT | `/mnt/share/erf/climate_downscale/` |
| AGGREGATE_ROOT | `/mnt/team/rapidresponse/pub/climate-aggregates/` |
| POPULATION_MODEL_ROOT | `/mnt/team/rapidresponse/pub/population-model/` |

## Implementation Details

### Single Source of Truth

All 21 curve names are defined once in `climate-data/src/climate_data/constants.py`:

```python
MALARIA_SUITABILITY_METHODS = ["mordecai", "symons", "villena"]
MALARIA_SUITABILITY_SHIFTS = ["0_0", "m0_25", "m0_5", "m1_0", "p0_25", "p0_5", "p1_0"]

CUSTOM_MALARIA_SUITABILITY_MEASURES = [
    f"malaria_{method}_{shift}_suitability"
    for method in MALARIA_SUITABILITY_METHODS
    for shift in MALARIA_SUITABILITY_SHIFTS
]
```

### The `--run-malaria-only` Flag

Added to all 4 pipeline commands. When set:
- **scenario-annual**: Only runs transforms for the 21 custom curves
- **draws**: Only processes the 21 custom curve variables
- **pixel**: Only aggregates the 21 custom measures
- **hierarchy**: Only rolls up the 21 custom measures

Without the flag, the full pipeline runs all standard climate variables.

## Output Schema

Final parquet files contain:

| Column | Description |
|--------|-------------|
| location_id | LSAE location identifier |
| year | Year (1950-2100) |
| value | Population-weighted suitability (0-1 scale) |

Draw dimension is encoded in filename (draw 000-099).

## Timeline Coverage

- **Historical:** 1950-2023 (ERA5 reanalysis)
- **Forecast:** 2024-2100 (CMIP6 SSP projections)
- **Scenarios:** ssp126, ssp245, ssp585

## Curve File Format

Each curve parquet file must have:

```
temperature: float  # Temperature in °C
suitability: float  # Suitability index (0-1)
```

The pipeline interpolates between temperature values to map any daily temperature to its suitability.
