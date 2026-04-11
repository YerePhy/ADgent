---
name: lint
description: Run the full linting suite (ruff check, ruff format, mypy) mirroring the CI pipeline. Use when the user wants to lint, check formatting, or type-check the codebase before committing or opening a PR.
---

Run the three linting steps that mirror the CI `lint` job, in order:

1. **`uv run ruff check .`** — catch lint errors and undefined names
2. **`uv run ruff format --check .`** — check formatting without rewriting files
3. **`uv run mypy backend/ scripts/`** — static type-checking

Run each command and report the output.

If `ruff format --check` finds files that would be reformatted, offer to fix them by running `uv run ruff format .`.

If `ruff check` reports fixable violations, offer to auto-fix them with `uv run ruff check --fix .`.

If `mypy` reports errors, show each error with its file and line number as a clickable markdown link (e.g. `[backend/tools.py:42](backend/tools.py#L42)`) and suggest a fix where the cause is clear.

Keep the output concise — group errors by file, skip noise, highlight what actually needs action.
