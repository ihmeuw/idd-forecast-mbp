# Claude Code Refactor Instructions: idd-forecast-mbp

## BRANCH AND VERSION CONTROL

**All work happens on a new branch.** Before starting any work:
```bash
git checkout -b feature/refactor-shared-lib
```

### Commit Strategy
- **Commit after each completed subtask** (e.g., after creating `lib/io/parquet.py`)
- Use descriptive commit messages: `refactor: extract parquet IO functions to lib/io/parquet.py`
- Never leave uncommitted work when stopping for any reason
- If interrupted, commit with message: `WIP: [description of current state]`

---

## TOKEN LIMITS AND SESSION MANAGEMENT — CRITICAL

**Reality**: This refactor will take more tokens than available in one session. Plan for interruptions.

### Graceful Degradation Protocol
When you notice:
- The conversation is getting long
- You've been working for a while without a break
- You're about to start a large new task

**STOP and enter documentation mode:**
1. Commit all current work (even if WIP)
2. Update `.claude/memory.md` with detailed current state
3. Update `.claude/SESSION_LOG.md` with what was done this session
4. List exactly what the next steps are
5. Say: "📍 CHECKPOINT: I'm pausing here. See memory.md for resume instructions."

### Resume Instructions
When starting a new session after interruption:
1. Read `.claude/memory.md` first
2. Read `.claude/SESSION_LOG.md` for history
3. Verify git status — are there uncommitted changes?
4. Ask me: "Last session ended at [X]. Ready to continue with [Y]?"

---

## SUBAGENT USAGE — TOKEN CONSERVATION

Subagents (especially Explore) can parallelize work, but they still consume tokens. Use them strategically.

### When to USE Explore subagents:
- Phase 1 audit tasks (read-only discovery across many files)
- Searching for pattern occurrences across the codebase
- Quick lookups while you're mid-task ("find where X is defined")
- Parallel exploration of independent directories

### When NOT to use subagents:
- Any task involving file edits or git operations
- Simple single-file reads (just read it directly)
- When you already have the answer in context
- Phases 3-5 (actual refactoring work)

### Cost/benefit rule:
Ask yourself: "Will this save more tokens than it costs?" Subagents have overhead. If a direct file read would suffice, do that instead.

### How to invoke:
Say: "Use Explore subagent to [task]" with thoroughness level (quick/medium/thorough).

---

## DOCUMENTATION REQUIREMENTS — ONGOING

Documentation is not optional. Update these files continuously:

### `.claude/memory.md` — Current State (overwrite each update)
- What task is in progress
- What was just completed
- What comes next
- Any pending decisions
- Resume prompt for next session

### `.claude/SESSION_LOG.md` — Running History (append only)
```
## Session YYYY-MM-DD HH:MM
### Completed
- [list of completed tasks]
### Files Created
- [list]
### Files Modified  
- [list]
### Decisions Made
- [list]
### Stopped Because
- [reason: token limit / user request / end of phase / etc.]
### Next Steps
- [ordered list]
```

### Function Documentation
Every function in `lib/` MUST have:
- Docstring with description, parameters, return type
- Comment noting original source: `# Extracted from: file.py:line`
- Type hints

---

## NOTEBOOKS FOR DEMONSTRATION

Create notebooks ONLY when they add value (not for every function). Good uses:
- Demonstrating a complex data loading pattern
- Showing how raking/aggregation works visually
- Debugging a tricky transformation

### Notebook Requirements
**EVERY notebook MUST start with these cells:**

```python
# Cell 1: Autoreload (REQUIRED)
%load_ext autoreload
%autoreload 2
```

```python
# Cell 2: Standard imports
import pandas as pd
import numpy as np
from pathlib import Path
# ... other imports
```

Save notebooks in `notebooks/lib_demos/` with descriptive names like `hierarchy_loading_demo.ipynb`.

---

## CRITICAL CONSTRAINTS — READ FIRST

### You MUST NOT:
- Remove any functionality, even if it seems unused or redundant
- Skip edge cases or error handling, even if they seem unnecessary
- Assume code is dead/unused without explicit confirmation from me
- Make architectural decisions without presenting options and waiting for my choice
- Change any logic while moving files — Phase 1 is MOVE ONLY
- Delete any files — only move, copy, or create new ones
- Modify R scripts without explicit permission (they work, don't touch them yet)

### You MUST:
- Preserve ALL existing behavior exactly
- Ask before every decision that has multiple valid approaches
- Show me diffs/changes before applying them when modifying existing code
- Run verification after each phase before proceeding
- Document every function you extract with its original location
- Keep a running log of changes in `.claude/memory.md`

---

## PIPELINE DEBUGGING PRINCIPLES — CRITICAL

This is a **sequential pipeline**. Step N uses output from Step N-1 and creates input for Step N+1.

### Principle 1: Errors are upstream until proven otherwise
When debugging, the **DEFAULT ASSUMPTION** is that something went wrong in an earlier step. Before adding fixes or guards to Step 5, verify that Step 4 actually produced correct output. Trace errors backward, not forward.

### Principle 2: No defensive coding for internal pipeline data
**DO NOT** add validation checks for data that only ever comes from our own pipeline steps.

❌ BAD — unnecessary defensive code:
```python
def process_forecast(df):
    if 'location_id' not in df.columns:
        raise ValueError("Missing location_id")
    if 'year_id' not in df.columns:
        raise ValueError("Missing year_id")
    # ... do work
```

✅ GOOD — let it fail fast:
```python
def process_forecast(df):
    # location_id and year_id come from upstream pipeline step
    # if missing, the error will be obvious and immediate
    result = df.groupby(['location_id', 'year_id']).sum()
```

**Rationale**: If a column is missing, the code will fail immediately with a clear error. We don't need explicit checks for things that "can never happen" in normal pipeline operation. Adding them creates code bloat and hides the real problem (which is always upstream).

### When validation IS appropriate:
- First time reading external input (GBD data, FHS data, user-provided files)
- Reading files that might not exist yet
- Parsing command-line arguments
- Anywhere the data source is outside our control

### When validation is NOT needed:
- Processing output from our own pipeline step
- Passing data between functions within the same script
- Any intermediate pipeline data

---

## Project Context

This is an infectious disease forecasting pipeline for malaria and dengue. It projects disease burden to 2100 under different climate (SSP) and funding (DAH) scenarios.

**Current state**: Working but messy. Functions are duplicated across malaria/dengue scripts. Intermediate files explode storage (100 draws × 3 SSP × 4 DAH × multiple stages = ~20k files per run).

**Goal**: Refactor so that:
1. Shared functions live in `lib/`
2. Malaria and dengue use the same lib functions
3. We can add a `curve_name` dimension for 14 suitability curves
4. Pipeline still produces identical outputs

---

## Phase 1: Audit and Map

**Objective**: Understand what exists before changing anything.

### Task 1.1: Catalog all pipeline scripts

**Use Explore subagents here** — this is read-only discovery across many directories.

Launch parallel Explore subagents (medium thoroughness) for each directory:
- `src/idd_forecast_mbp/01_map_to_admin_2/`
- `src/idd_forecast_mbp/02_data_prep/`
- `src/idd_forecast_mbp/03_modeling/`
- `src/idd_forecast_mbp/04_forecasting/`
- `src/idd_forecast_mbp/05_aggregation/`
- `src/idd_forecast_mbp/06_upload/`

For each file, have the subagent document:
- What it does (1-2 sentences)
- What it inputs (file paths)
- What it outputs (file paths)
- Key functions it defines
- Key functions it imports from other modules

Synthesize subagent results into `.claude/PIPELINE_AUDIT.md`.

**Output**: Create `.claude/PIPELINE_AUDIT.md` with this catalog.

**STOP and show me this audit before proceeding.**

### Task 1.2: Catalog utility modules
Read every utility module at `src/idd_forecast_mbp/*.py` (not in subdirectories). Document:
- Each function and what it does
- Which pipeline scripts import it
- Any obvious duplication

**Output**: Add to `.claude/PIPELINE_AUDIT.md`.

**STOP and show me this before proceeding.**

### Task 1.3: Identify common patterns

**Use Explore subagents here** — searching for pattern occurrences across the codebase.

Launch Explore subagents (thorough) to search for these patterns:
- Loading hierarchy data
- Loading climate/covariate data by draw
- Merging dataframes with standard ID columns
- Writing parquet/netcdf with retry logic
- DAH scenario generation
- Age/sex aggregation
- Raking to hierarchy levels
- Aggregating to hierarchy levels

For each pattern, compile:
- Where it appears (file:line)
- How the implementations differ (if at all)
- Proposed shared function signature

#### Example patterns to look for (calibration examples):

**Example 1: Load hierarchy and filter by level**
```python
# Appears in: forecasted_draw_specific_malaria_dataframes.py, 
#             forecasted_draw_specific_dengue_dataframes.py,
#             multiple aggregation scripts
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)
filtered = df[df['location_id'].isin(hierarchy_df[hierarchy_df['level'] >= 3]['location_id'])]
```

**Example 2: Load climate covariate by draw**
```python
# Appears in both malaria and dengue data prep
for key, path_template in cc_sensitive_paths.items():
    path = path_template.format(CLIMATE_DATA_PATH=CLIMATE_DATA_PATH, ssp_scenario=ssp_scenario)
    columns_to_read = ["location_id", "year_id", draw]
    df = read_parquet_with_integer_ids(path, columns=columns_to_read)
    df = df.rename(columns={draw: key})
    forecast_df = pd.merge(forecast_df, df, on=["location_id", "year_id"], how="left")
```

**Example 3: DAH scenario generation**
```python
# The generate_dah_scenarios() function in malaria scripts
# Should be shared since dengue will need similar scenario logic eventually
```

These examples show the level of abstraction we want: common data loading, merging, and scenario generation patterns.

#### Prioritization scheme for pattern review:

Categorize patterns by frequency:
- **HIGH priority** (used 5+ times across malaria/dengue): Review in detail with me
- **MEDIUM priority** (used 3-4 times): Quick approval unless something looks wrong
- **LOW priority** (used 1-2 times): Can wait or be bundled

For HIGH priority patterns, we will discuss:
1. Which implementation is canonical?
2. What parameters should be exposed?
3. What should the function be named?

**Output**: Create `.claude/COMMON_PATTERNS.md` with patterns categorized by priority.

**STOP and show me this before proceeding.**

---

## Phase 2: Design Shared Library

**Objective**: Define what goes in `lib/` before writing code.

### Task 2.1: Propose lib structure
Based on the audit, propose a structure like:
```
src/idd_forecast_mbp/lib/
├── io/
│   ├── parquet.py      # read/write parquet
│   ├── netcdf.py       # read/write netcdf/xarray
│   └── hdf5.py         # read/write hdf5
├── data/
│   ├── hierarchy.py    # load/filter hierarchy
│   ├── climate.py      # load climate covariates by draw
│   ├── population.py   # load population data
│   └── gbd.py          # load GBD outcomes
├── processing/
│   ├── raking.py       # rake to hierarchy levels
│   ├── aggregation.py  # age/sex aggregation
│   └── scenarios.py    # DAH scenario generation
└── utils/
    ├── transforms.py   # log transforms, clipping, etc.
    └── validation.py   # data validation helpers
```

**STOP and ask me**: Does this structure make sense? What would you change?

### Task 2.2: Define function signatures
For each proposed shared function, write:
- Function name and signature
- Docstring with parameters and return type
- Which existing code it replaces (file:line references)

Do NOT implement yet. Just signatures.

**Output**: Create `.claude/LIB_DESIGN.md`.

**STOP and show me this before proceeding.**

---

## Phase 3: Extract Shared Functions

**Objective**: Create lib/ with shared functions, keeping originals intact.

### CRITICAL RULES FOR THIS PHASE:
1. Create NEW files in `lib/` — do not modify original scripts yet
2. Each function must have a docstring noting where it came from
3. Each function must handle all edge cases from ALL original implementations
4. If implementations differ, ASK ME which behavior to keep

### Task 3.1: Create lib structure
Create the directory structure and `__init__.py` files.

### Task 3.2: Implement IO functions
Extract `parquet_functions.py`, `xarray_functions.py`, `hd5_functions.py` content into `lib/io/`.

**STOP and show me the new files before proceeding.**

### Task 3.3: Implement data loading functions
Extract common data loading patterns.

**STOP and show me before proceeding.**

### Task 3.4: Implement processing functions
Extract raking, aggregation, scenario generation.

**STOP and show me before proceeding.**

---

## Phase 4: Update Scripts to Use lib/

**Objective**: Replace duplicated code with lib imports.

### CRITICAL RULES:
1. Change ONE script at a time
2. After each script, verify it still works (I'll tell you how)
3. Do not change any logic — only replace inline code with lib calls
4. Keep the original code as comments initially (we'll remove later)

### Task 4.1: Update constants alias
Replace all `import constants as rfc` with `import constants as mbpc`.

This is a safe global find-replace. Do it across all files.

**STOP and show me the diff before applying.**

### Task 4.2: Update data_prep scripts
For each script in `02_data_prep/`:
1. Show me the current imports and which lib functions will replace them
2. Show me the diff
3. Wait for my approval
4. Apply the change
5. Move to next script

### Task 4.3-4.6: Repeat for other stages
Same process for `04_forecasting/`, `05_aggregation/`, `06_upload/`.

---

## Phase 5: Reorganize into malaria/dengue

**Objective**: Move disease-specific scripts to disease folders.

### Task 5.1: Identify disease-specific scripts
List scripts that are:
- Malaria-only
- Dengue-only  
- Shared (work for both)

**STOP and show me this list.**

### Task 5.2: Create folder structure
```
src/idd_forecast_mbp/
├── lib/           # shared (already done)
├── malaria/       # malaria-specific
├── dengue/        # dengue-specific
├── pipeline/      # shared pipeline steps
└── config/        # constants, etc.
```

### Task 5.3: Move files
Use `git mv` to preserve history. Update imports.

**STOP after each move and confirm imports still resolve.**

---

## Phase 6: Verification

**Objective**: Confirm nothing broke.

### Task 6.1: Import check
Run `python -c "from idd_forecast_mbp import lib"` and similar for all modules.

### Task 6.2: Type check
Run `mypy src/idd_forecast_mbp/` and fix any new errors.

### Task 6.3: Integration test
I will run a small test case through the pipeline. You help me compare outputs.

---

## Communication Protocol

### When you encounter a decision point:
```
🔀 DECISION NEEDED:
Context: [what you're trying to do]
Options:
  A) [option A and its implications]
  B) [option B and its implications]
  C) [other options]
My recommendation: [your suggestion and why]
Waiting for your choice before proceeding.
```

### When you complete a phase:
```
✅ PHASE X COMPLETE
Summary: [what was done]
Files changed: [list]
Files created: [list]
Next phase: [what comes next]
Ready to proceed? [Y/N]
```

### When you find something unexpected:
```
⚠️ UNEXPECTED:
Found: [what you found]
Location: [file:line]
This matters because: [why]
Options: [what we can do about it]
```

### When considering a subagent:
Only suggest a subagent if the task is:
1. Read-only (no edits needed)
2. Spans multiple files/directories
3. Would take longer doing sequentially

```
🔍 SUBAGENT SUGGESTED:
Task: [what needs exploring]
Why subagent: [saves tokens because X / parallelizes Y]
Thoroughness: quick/medium/thorough
Proceed? [Y/N]
```

Do NOT use subagents for:
- Single file reads
- Tasks in Phases 3-5 (actual refactoring)
- When the answer is already in context

### When you hit a blocker you cannot resolve:
```
🛑 BLOCKED:
What I was trying to do: [task]
What went wrong: [error/issue]
What I tried: [list of attempts]
What I need from you: [specific ask]
Current state: [is code in a broken state? committed?]
```
Do NOT try to work around blockers silently. Stop and ask.

### When approaching token/context limits:
```
📍 CHECKPOINT:
Completed: [what was done this session]
In progress: [what was mid-flight]
Next steps: [ordered list]
Files updated: memory.md, SESSION_LOG.md
Git status: [committed / WIP commit made]
Ready to resume in next session.
```

---

## Files to Read First

Before starting, read these to understand current state:
1. `.claude/REFACTOR_PROMPT.md` — these instructions (you're reading it)
2. `.claude/memory.md` — current session context and resume prompt
3. `.claude/SESSION_LOG.md` — history of what was done in previous sessions
4. `.claude/STATUS.md` — project goals
5. `.claude/DECISIONS.md` — decisions already made
6. `src/idd_forecast_mbp/constants.py` — all path constants
7. `src/idd_forecast_mbp/outline.ipynb` — pipeline documentation

---

## Success Criteria

The refactor is complete when:
1. All shared functions are in `lib/`
2. No function is duplicated between malaria and dengue scripts
3. **Tests exist for each shared lib function** (we currently have ZERO tests — this is a problem we fix during refactor)
4. Running the existing pipeline produces byte-identical outputs
5. The codebase can accept a `curve_name` parameter (even if not yet implemented)
6. `.claude/PIPELINE_AUDIT.md` documents the final structure

### Testing Requirements (NEW)

**Framework**: pytest (already in dev dependencies)

We have ZERO tests currently. We fix this during the refactor with two types of tests:

#### Unit Tests (as we extract each function)
When creating each lib module, immediately create corresponding test file:
- `lib/io/parquet.py` → `tests/lib/io/test_parquet.py`
- `lib/data/hierarchy.py` → `tests/lib/data/test_hierarchy.py`
- etc.

Unit test requirements:
1. At least one test per function that verifies basic behavior
2. For IO functions, test round-trip (write then read, verify data matches)
3. For processing functions, test with a minimal synthetic example
4. Tests should be FAST — use small synthetic data, not real pipeline data
5. Test edge cases that exist in the original implementations

#### Integration Tests (after each phase)
After completing each phase, run integration verification:
- **After Phase 3**: Verify lib functions can be imported, basic smoke test
- **After Phase 4**: Run a single draw through the pipeline, compare output to baseline
- **After Phase 5**: Full pipeline run for one scenario, compare output to baseline

I will provide baseline outputs for comparison. You help set up the comparison.

Tests are how we verify the refactor didn't break anything. They're not optional.

---

## Questions You Should Ask Me

At minimum, stop and ask about:
1. Any function that exists in slightly different forms — which version is correct?
2. Any commented-out code — keep or remove?
3. Any hardcoded paths or magic numbers — how to handle?
4. Any imports that seem unused — confirm before removing
5. The hierarchy loading logic — it's complex and has history
6. Anything in R — don't touch without asking

