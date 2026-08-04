# 03_modeling jobmon migration — frozen interface contract (DRAFT)

**Status:** DRAFT, pending ratification. Three sessions build against this:
R-prep (Bobby), orchestrator (idd-forecast-mbp CC), idd-tools API (idd-tools CC).
**Rule:** seam-touching code waits for ratification; seam-independent internals
may start now. Keep this doc to tables — its only job is to freeze names, dtypes,
and signatures.

---

## 0. Pipeline + the bundling-mode gate

```
R prep (spec table) → orchestrator `materialize` (partition → param_map)
   → [inspect param_map / dry-run / hand-run a worker] → orchestrator `submit`
```

`materialize` is a step **separate from** `submit` (write param_map + render
commands without burning cluster time). This is the inspect/debug boundary.

**The bundle/trivial decision is made PER `n_smooths` LEVEL — there is no global regime.**
Per-cell cost (full cell = 1 IS + 5 OOS fits, FE) = f(`n_smooths`) and crosses the
5-min floor along that axis, so one workflow holds both regimes at once:

| Per `n_smooths` level | Per-cell cost | Partition | Probe |
|---|---|---|---|
| Low-smooth (sub-floor) | < ~5 min (0 smooths ≈ seconds) | bundle (size > 1) to clear the floor | size-sweep → `calibrate_bundle_size` (that level) |
| High-smooth (≥ floor) | ≥ ~5 min (high smooths ≈ 10+ min) | `trivial_partition` (1 spec/task) | single representative cell (no sweep) |

Always homogeneous by `n_smooths`. One `rectangular_partition(fix=["n_smooths"],
max_per_task=<dict>)` sets each level's bundle size (1 for high-smooth, N for
low-smooth). `param_map` schema (§1b) is identical across both — only rows-per-`task_id` differ.

---

## 1. Data + command seams — *idd-forecast-mbp CC drafts; Bobby ratifies science*

### 1a. Spec table  (R prep → orchestrator)
Emitted by R prep alongside `neighborhood_specs.rds`. R prep owns the grid
(`var_forms`/`groups`/`expand_models`/`matches()`/`MAX_SMOOTHS`); it does **not** chunk.

| column | dtype | meaning |
|---|---|---|
| `spec_index` | int32 | stable spec identity; indexes into `neighborhood_specs.rds` |
| `n_smooths` | int32 | # smooth terms — cost driver, partition fix-axis, task_feature |
| `formula_text` | str | FE-present formula (human inspection + echoed downstream) |

### 1b. param_map  (orchestrator `materialize` → worker) — serves probe AND full run
Written by the orchestrator (NOT R). One row per (task, spec). Format: **parquet,
explicit dtypes, integer IDs** (no float IDs).

| column | dtype | meaning |
|---|---|---|
| `task_id` | str | stable task identity = jobmon `task_id` = log/`select_summary` suffix |
| `spec_index` | int32 | a spec covered by this task (bundle = multiple rows, same `task_id`) — **the single join key** |
| `n_smooths` | int32 | materialized partition fix-axis value; **homogeneous within a task**; orchestrator → `task_features` |

Worker needs only `(task_id, spec_index)`. `n_smooths` is the materialized partition
record. `bundle_size` (= rows per `task_id`) and `formula_text` (= rds lookup by
`spec_index`) are derivable and omitted — no redundant columns.

### 1c. Worker CLI  (command_template → `select_malaria_models_rocket.r`)
Worker selects its `spec_index` rows from `param_map` by `--task-id`; keeps its
existing per-spec loop + atomic per-task write. Run-config are **node_args**
(per-task-capable, not hardcoded constant).

| flag | type | jobmon role |
|---|---|---|
| `--task-id` | str | node_arg |
| `--output-dir` | path | task_arg |
| `--param-map` | path | task_arg — passed explicitly (materialize writes it to a known path) |
| `--cv-strategy` | str | node_arg |
| `--optimizer` | str | node_arg |
| `--maxit` | int | node_arg |
| `--k` | int | node_arg |
| `--fit-is-fe` | str: `TRUE`/`FALSE` | node_arg (valued substitution — not a bare flag) |

`command_template` ≈ `<execRscript.sh -i <img>> -s select_malaria_models_rocket.r --task-id {task_id} --output-dir {output_dir} --cv-strategy {cv_strategy} --optimizer {optimizer} --maxit {maxit} --k {k} --fit-is-fe {fit_is_fe}`

### 1d. task_features (orchestrator manifest)
| key | dtype | use |
|---|---|---|
| `n_smooths` | int | partition fix-axis · calibrate `shared_axes` · resource feature |

### 1e. Conventions
- **completion check** (`filter_already_done`): `output_dir / f"select_summary_{task_id}.parquet"` exists.
- **version_tag**: Bobby-asserted (not git SHA), declared in `.jobmon_versions.toml`; required by `calibrate_bundle_size` + `submit_with_manifest`. Suggested: `malaria_screen_<YYYYMMDD>`.

### 1f. Science assertions — *Bobby-ratified; gates the partition/bundle decision*
| assertion | status |
|---|---|
| Per-cell cost (full cell = 1 IS + 5 OOS fits, all FE) is a **strong function of `n_smooths`**: 0 smooths ≈ seconds; high smooths ≈ 10+ min. There is **no single global regime** — cost crosses the 5-min floor along the `n_smooths` axis. | **RATIFIED** |
| ⇒ Partition **homogeneous by `n_smooths`** and set bundle size **per level**: low-smooth levels are sub-floor → **bundle** (size > 1) + small resources; high-smooth levels ≥ floor → **one spec/task** (size = 1) + large resources. We don't get everything for free. | **RATIFIED** |
| This is exactly why we scope/probe **by `n_smooths`**: each level sits at a different point vs. the floor, so bundle size and resources are per-level, never global. | **RATIFIED** |

---

## 2. Orchestrator ↔ idd-tools API signatures — *idd-tools CC fills + ratifies*

> STUBs below are the orchestrator's assumptions; idd-tools session pins exact
> signatures here so the orchestrator isn't built against imaginary shapes.

| function | assumed signature | status |
|---|---|---|
| `stratified_scope` | `(manifest, n_per_level, *, by, random=False, seed=None) -> TaskManifest` | **STUB — ratify** |
| `rectangular_partition` | `(..., fix=[...], max_per_task: int \| dict \| Callable) -> ...` | **EXTEND** `max_per_task` to dict/callable keyed by fix-group — ratify |
| dry-run renderer | `render_commands(manifest, templates, ...) -> list[str]` (name TBD) | **STUB — ratify** |
| `trivial_partition` | (existing) one cell → one task | reference |
| `calibrate_bundle_size` | `(manifest, *, version_tag, shared_axes, templates, sizes, output_dir, target_runtime_s=300.0, memory_safety_factor=1.6, runtime_safety_factor=1.6) -> CalibrationResult` | CONFIRMED. Default `memory_safety_factor=1.6`; **pass ≈2.0 explicitly** for the cgroup floor. Fits one model per call → **one call per `n_smooths` level**. |
| `submit_with_manifest` | (existing) `resources` may be `Callable[[Task], dict]` | reference |

**Known landmine (do not design around it yet):** `run_registry` buckets probes by
`shared_axes` *names*, not values — so `fit_from_probe_history` would pool all
`n_smooths` levels. Use each per-level `calibrate_bundle_size` **return** as the
source of truth; do not build cross-run history refitting until idd-tools carries
the fixed *value*. (This is the blocker on the deferred `ResourcePredictor`-from-history path.)

---

## 3. Ratification + sequencing

| Section | Owner ratifies |
|---|---|
| §1 mechanics (columns, flags, dtypes) | idd-forecast-mbp CC |
| §1 science (`n_smooths` semantics, spec-table meaning) | Bobby |
| §2 API signatures | idd-tools CC |

**Hold until ratified (seam-touching):** orchestrator param_map writer, worker
flag parser, R-prep output schema.
**May start now (seam-independent):** R grid logic, worker compute body,
orchestrator per-level control flow.
**Build the per-level path** (§1f): partition homogeneous by `n_smooths`; probe
each level (`stratified_scope`, single representative cell — sweep sizes only on
sub-floor levels) → `sacct` / `calibrate_bundle_size` → per-level
`{bundle_size, resources}`; feed one `rectangular_partition(fix=["n_smooths"],
max_per_task=<dict>)`. The **`max_per_task=dict` EXTEND is ON the critical path**
(high-smooth levels → `1`, low-smooth levels → `N`). Hold calls behind the
§2-ratified signatures.
