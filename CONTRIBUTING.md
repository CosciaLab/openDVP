# Contributing to openDVP

Thanks for considering a contribution — bug reports, documentation fixes and new features are all welcome.

This file is the quick reference.
The full step-by-step walkthrough, including how to fork and what makes a good test, lives in the [Contribution Guide](https://coscialab.github.io/openDVP/ContributionGuide.html).

By participating in this project you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Where to start

| I want to... | Go to |
| --- | --- |
| Report a bug | [Open an issue](https://github.com/CosciaLab/openDVP/issues/new/choose) |
| Request a feature | [Open an issue](https://github.com/CosciaLab/openDVP/issues/new/choose) |
| Ask how to do something | [Discussions](https://github.com/CosciaLab/openDVP/discussions) |
| Fix or add something | Read on |

## Setup

We use [uv](https://docs.astral.sh/uv/) for environments and packaging.

```bash
git clone https://github.com/YOUR_USERNAME/openDVP.git
cd openDVP
git remote add upstream https://github.com/CosciaLab/openDVP.git

uv sync                     # create the environment
uv run pre-commit install   # install the ruff hooks
```

Work on a branch, never on `main`:

```bash
git switch -c my-branch-name
```

## Commands

| Task | Command |
| --- | --- |
| Run all tests | `uv run pytest` |
| Run one test file | `uv run pytest tests/io/test_DIANN_to_adata.py` |
| Tests with coverage | `uv run pytest --cov` |
| Format | `uv run ruff format .` |
| Lint | `uv run ruff check .` |
| Build docs | `uv run --group docs sphinx-build -b html docs/ docs/_build/html/` |

Ruff settings live in `pyproject.toml`.
`pre-commit` runs the same checks on every commit, so CI rarely surprises you.

## Adding a new function

openDVP follows the [scverse](https://scverse.org/) layout: `io` for readers and writers, `pp` for preprocessing, `tl` for tools, `plotting` for figures, plus `imaging`, `metrics` and `utils`.

One public function per file, and the file is named after the function.
To add `my_function` to `tl`:

1. Write it in `src/opendvp/tl/my_function.py`.
2. Export it in `src/opendvp/tl/__init__.py` — both the import and the `__all__` entry.
3. Test it in `tests/tl/test_my_function.py`.
4. List it in the right section of `docs/api/tl.md`, or it will not appear in the rendered docs.
5. Run `uv run ruff format . && uv run ruff check . && uv run pytest`.

Functions that take an `AnnData` should say in their docstring exactly which `.obs`, `.var`, `.uns` or `.obsm` fields they write.

## Pull requests

Open your PR against the `main` branch of `CosciaLab/openDVP`.

- Keep it focused — one topic per PR is much easier to review than a large mixed change.
- Explain what you changed and why, and link any issue it closes.
- Make sure tests and linting pass locally first.
- CI runs the test suite on Linux, macOS and Windows across Python 3.11–3.13.

Versioning and releases are handled by the maintainers — you do not need to bump any version numbers.

## Questions

If anything here is unclear or out of date, that is a bug in this document.
Please [open an issue](https://github.com/CosciaLab/openDVP/issues/new/choose) or say so in [Discussions](https://github.com/CosciaLab/openDVP/discussions).
