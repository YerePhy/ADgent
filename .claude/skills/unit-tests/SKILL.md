---
name: unit-tests
description: Examine new or modified code for unit-testable logic, then write and run the tests. Trigger when a new class or function is introduced, or when the user asks to test something.
---

Whenever a new class or function is introduced — or the user asks to write tests — follow this process:

**1. Assess testability before writing anything.**

For each new unit of code, ask:
- Does it have clear inputs and outputs with no side-effects, or are its side-effects easily faked?
- Can external dependencies (DB, filesystem, HTTP, LLM APIs) be replaced with in-memory fakes or `unittest.mock`?
- Is the logic non-trivial enough that a bug would not be immediately obvious?

If the answer to all three is yes, it warrants a unit test. Do not write tests for pure wiring/DI code, single-line pass-throughs, or abstract base classes with no logic.

**2. Choose the right isolation strategy.**

| Dependency type | Strategy |
|---|---|
| SQLite | Use `sqlite3.connect(":memory:")` — no mocking needed |
| Filesystem | Use pytest's `tmp_path` fixture |
| External APIs / LLMs | `unittest.mock.MagicMock(spec=ActualClass)` |
| Abstract interfaces | Write a minimal concrete stub — avoid `MagicMock` when the contract matters |

**3. Follow these conventions for this project.**

- Test files live in `tests/backend/` and are named `test_<module>.py`
- Use pytest; no test classes unless grouping genuinely aids clarity
- Test function names: `test_<what>_<expected_outcome>` (e.g. `test_save_upload_returns_record`)
- One assertion cluster per test — keep each test focused on a single behaviour
- Do not mock internals; mock at the boundary (the injected dependency, not its methods)
- No comment blocks (`# ---`) as section separators; use blank lines instead
- Private helper functions in test files are fine and preferred over repetitive boilerplate

**4. Cover the key paths.**

For every testable unit, cover at minimum:
- The happy path with representative inputs
- At least one boundary or edge case (empty list, zero, None, limit values)
- The primary error path (invalid input, missing resource, constraint violation)

**5. Run and verify.**

After writing tests, run them with `uv run pytest tests/` and confirm they pass. If a test fails, diagnose the root cause — do not adjust assertions to match wrong behaviour.
