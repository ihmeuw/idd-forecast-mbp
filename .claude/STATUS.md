# Project status
Updated: 2026-03-26

## Goals
Infectious disease forecasting pipeline for malaria and dengue, projecting outcomes to 2100 under SSP climate scenarios + DAH funding scenarios. Currently undergoing major refactor to:
1. Add 14 different malaria suitability curves (for paper revision)
2. Add vaccination scenario dimension
3. Add proper variable importance methods beyond "hold at 2022" decomposition
4. Clean architecture so malaria/dengue share common infrastructure

## Recent steps
- 2026-03-26: Decided on monorepo refactor approach (vs separate repos)
- 2026-03-26: Created .claude/ tracking files
- 2026-03-26: Established phased refactor plan (reorganize → verify → add features)

## Next steps
1. Create `feature/malaria-curves` branch
2. Reorganize into `lib/` + `malaria/` + `dengue/` structure (move only, no logic changes)
3. Verify existing functionality still works
4. Add suitability curve dimension to malaria pipeline
5. Build variable importance module in shared lib/

## Parking lot
- Efficiency improvements to hot paths (after refactor verified)
- Dengue revision (after malaria paper accepted)
- Better computational efficiency / data types throughout
