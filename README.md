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

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.10+
* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
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
