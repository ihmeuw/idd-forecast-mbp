# idd-forecast-mbp

---

**Documentation**: [https://ihmeuw.github.io/idd-forecast-mbp](https://ihmeuw.github.io/idd-forecast-mbp)

**Source Code**: [https://github.com/ihmeuw/idd-forecast-mbp](https://github.com/ihmeuw/idd-forecast-mbp)

---

Word

## Installation

```sh
pip install idd-forecast-mbp
```

## Development

Local dev uses a project `.venv` with a uv-managed interpreter — no conda
anywhere in the Python story. uv owns the interpreter and every package
(including installing idd-forecast-mbp itself editable — no `PYTHONPATH=src`).
One-time setup:

```sh
uv venv --python 3.12        # .venv on the uv-managed CPython
uv sync --all-extras         # runtime + dev + notebooks extra
.venv/bin/pre-commit install
```

Resolution needs IHME artifactory access (`jobmon_installer_ihme` is a core
dependency of this cluster-only pipeline repo), plus sibling clones of the
local path dependencies `../climate-data` and `../idd-tools`.

Run code via `.venv/bin/python` by absolute path (Slurm/jobmon scripts do
exactly this — no activation of any kind); `source .venv/bin/activate` is
optional for interactive shells.

Day-to-day, re-sync after pulling or editing deps:

```sh
uv sync --inexact   # add/update declared deps, keep ad-hoc installs
uv sync             # exact clean rebuild (env == lockfile)
```

### Testing

```sh
pytest
```

### Documentation

The documentation is automatically generated from the content of the `docs` directory and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

### Releasing

Trigger the [Draft release workflow](https://github.com/bcreiner/idd-forecast-mbp/actions/workflows/draft_release.yml)
(press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the
[GitHub releases](https://github.com/bcreiner/idd-forecast-mbp/releases) and publish it. When
 a release is published, it'll trigger [release](https://github.com/bcreiner/idd-forecast-mbp/blob/master/.github/workflows/release.yml) workflow which creates PyPI
 release and deploys updated documentation.

### Pre-commit

Pre-commit hooks run all the auto-formatting (`ruff format`), linters (e.g. `ruff` and `mypy`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

---
