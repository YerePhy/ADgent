---
name: test-integration-audit
description: Scan the repository and produce a structured report of all functions, classes, and components that are susceptible to unit or integration testing, along with a coverage gap analysis.
---

Scan every Python module under `backend/`, `frontend/`, and `scripts/`. For each, assess testability and produce the report below.

**How to classify each component:**

A component is a unit test candidate if:
- It has deterministic logic with clear inputs/outputs
- Its external dependencies (DB, filesystem, network, LLM) can be replaced with in-memory fakes or mocks without defeating the purpose of the test

A component is an integration test candidate if:
- It orchestrates multiple real subsystems (e.g. vectorstore + embeddings, LangGraph + SQLite memory)
- Correctness depends on the actual behaviour of an external dependency, not just its interface

A component is low priority / not worth testing if:
- It is pure wiring (DI, factory delegation with no logic)
- It is a thin pass-through with no branching or transformation

**Report format:**

Produce a structured markdown report with three sections:

### Unit Test Candidates
For each: module path, component name, why it is testable, recommended isolation strategy (mock / in-memory / tmp_path), and whether a test already exists.

### Integration Test Candidates
For each: module path, component name, what subsystems it exercises, and what test infrastructure it would need (e.g. a real Chroma instance, a seeded SQLite DB).

### Coverage Gaps
List components that are testable but currently have no test in `tests/`. Sort by estimated risk: components with complex branching or error paths that are untested should appear first.

**Rules:**
- Read each module before assessing it — do not guess from names alone
- Check `tests/backend/` for existing test files before marking something as untested
- Keep each entry concise: one or two sentences per component
- Flag any module approaching 500 lines as a split candidate in a separate note at the end
