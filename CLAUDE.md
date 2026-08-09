# Project instructions — idd-forecast-mbp

## Environment scheme: venv-only (declared 2026-08-09)

This repo runs the **pure-uv scheme**: a project `.venv` at the repo root,
uv-managed interpreter, uv-managed dependencies. One scheme per repo, never
mixed.

- **Expected:** `./.venv` exists and is the only environment for this repo.
  Its interpreter is a uv-managed standalone CPython (`uv python install`),
  never a conda python. All code runs via `.venv/bin/python` — by absolute
  path in Slurm/jobmon task commands, no activation required.
- **Forbidden:** conda hosting of any kind for this repo — no conda env named
  after the repo, no `environment.yml`, no `UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX`
  idiom. Those belong to the retired conda+uv scheme. (The companion R env
  `idd-forecast-mbp-r` is conda-provisioned and stays — conda survives as the
  R provisioner only.)
- **Session start:** do NOT activate a conda env (this overrides the global
  "activate the matching conda environment" rule). Use `.venv/bin/python`
  directly.
- **Sync idiom:** `uv sync --inexact` day-to-day; plain `uv sync` (exact) for
  clean rebuilds, vetted first with `uv sync --dry-run`. `--all-extras` for
  the full surface including the `notebooks` extra.
- **jobmon:** `jobmon_installer_ihme==10.12.2` is a core dependency (cluster-only
  leaf pipeline); worker command templates invoke `.venv/bin/python` by
  absolute path.
- **Local path deps:** `climate-data` and `idd-tools` resolve editable from
  sibling clones (`../climate-data`, `../idd-tools`).
- Interpreter downloads are deliberate only: the user-level uv policy is
  `python-downloads = "manual"` — a plain `uv venv`/`uv sync` must never
  trigger a CPython download.

## Session memory is sectioned (do not blanket-overwrite)
`.claude/memory.md` is a SHARED, SECTIONED file — sections `General` / `Malaria` /
`Dengue`, used by multiple concurrent workstreams. The session-close / `/wrap`
protocol's "overwrite memory.md" means: rewrite the file with only the section(s)
your work touched updated and **every other section copied verbatim** — never
replace the whole file with one session's snapshot.
