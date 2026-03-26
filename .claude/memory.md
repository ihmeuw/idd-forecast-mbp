# Session memory
Updated: 2026-03-26

## Current task
Ready for Claude Code to begin Phase 1: Audit and Map

## Context / why
Paper revision requires: (1) testing 14 suitability curves for malaria, (2) adding vaccination scenarios, (3) proper variable importance. Dengue revision will follow, so shared infrastructure needed. Refactoring to extract shared functions into `lib/`.

## Where we are
- REFACTOR_PROMPT.md is complete with detailed instructions
- SESSION_LOG.md created for history tracking
- Ready to create feature branch and begin audit
- Starting from commit: [check with `git log -1 --oneline`]

## What was decided
- Monorepo refactor approach
- Constants alias: `rfc` → `mbpc`
- Unit tests per function as we extract
- Integration tests after each phase
- Notebooks only when demonstrating complex features (always with autoreload)

## Next steps
1. Create branch: `git checkout -b feature/refactor-shared-lib`
2. Read REFACTOR_PROMPT.md thoroughly
3. Begin Task 1.1: Catalog all pipeline scripts
4. Create PIPELINE_AUDIT.md
5. STOP and show Bobby the audit

## Resume prompt
This is a refactor of idd-forecast-mbp to extract shared functions into `lib/`. Read `.claude/REFACTOR_PROMPT.md` for detailed instructions. The prompt includes critical constraints, phased tasks with mandatory stops, testing requirements, and documentation protocols. Start by creating the feature branch, then begin Phase 1 (Audit and Map). Stop after each subtask and show me the output before proceeding. Key files: REFACTOR_PROMPT.md (instructions), SESSION_LOG.md (history), this file (current state).
