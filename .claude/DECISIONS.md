# Decisions log

<!-- Append-only. Never delete or overwrite entries. -->

## 2026-03-26: Monorepo refactor over separate repos
**Decision:** Refactor idd-forecast-mbp into lib/ + malaria/ + dengue/ structure rather than creating separate repos.
**Why:** Need shared infrastructure (variable importance, I/O, raking) for both diseases. Single repo = simpler git workflow, immediate availability of shared code changes. Dengue code can stay untouched while malaria is rebuilt.
**Revisit if:** Repos diverge so much that shared code becomes a burden, or if different teams need independent release cycles.

## 2026-03-26: Phased refactor approach
**Decision:** Phase 1 = file moves + import changes only (no logic). Phase 2 = verify. Phase 3+ = add features.
**Why:** Minimizes risk of breaking functionality. Can verify at each phase before proceeding.
**Revisit if:** Phase 1 takes too long or reveals deeper entanglement requiring logic changes.

## 2026-03-26: Constants import alias `mbpc`
**Decision:** Change `from idd_forecast_mbp import constants as rfc` to `as mbpc` across codebase.
**Why:** `rfc` was inherited from another repo. `mbpc` = "mbp constants", distinguishable from other repos (e.g., `cdc` for climate_data constants).
**Revisit if:** Never — this is just naming.
