---
name: commit
description: Stage and commit changes in conceptual chunks with human-readable messages. Use when the user wants to commit work, review staged changes, or split a diff into logical commits.
---

## Goal

Turn the current working-tree changes into one or more commits, each representing a single logical idea. Never bundle unrelated changes into one commit just because they happen to be in the same diff.

---

## Step 1 — Understand what changed

Run these in parallel:

- `git status` — identify modified, added, and deleted files
- `git diff` — read the actual content of unstaged changes
- `git diff --cached` — read already-staged changes
- `git log --oneline -10` — observe the existing commit style

Do not proceed until you have read and understood the full diff.

---

## Step 2 — Group changes into conceptual chunks

A conceptual chunk is a set of changes that share a single reason to exist. Ask yourself: *if this change were reverted, what specific behavior or structure would go back?* Each answer is a candidate commit.

**Split on these boundaries:**

| Separate these | Because |
|---|---|
| Bug fix vs. refactor | Different blame, different risk |
| New feature vs. plumbing that enables it | Plumbing is often backportable alone |
| Config / environment changes vs. logic | Ops vs. dev concerns |
| Test additions vs. production code | Tests should be reviewable independently |
| Docs / comments vs. code | Zero-risk change, reviewers scan faster |
| Unrelated file moves vs. edits | Move first, edit second — keeps diffs readable |

**Keep together:**

- Code and the tests that directly exercise it (when both are new)
- A data-model change and every call site that had to be updated as a direct consequence
- Formatting fixes that touch a file you're already committing for another reason

If the diff is already one coherent idea, a single commit is correct — do not split artificially.

---

## Step 3 — Write the commit message

### Format

```
<type>: <short imperative summary>          ← subject line, ≤ 72 chars

<optional body — wrap at 72 chars>
```

### Subject line rules

- Use the **imperative mood**: "Add", "Fix", "Remove", "Move", "Refactor", "Update", not "Added" or "Adding"
- No trailing period
- Start with a **type prefix** followed by a colon and space:

| Prefix | When to use |
|---|---|
| `feat` | New user-visible capability |
| `fix` | Corrects a defect |
| `refactor` | Internal restructure, no behavior change |
| `chore` | Tooling, build, config, dependencies |
| `docs` | Documentation or comments only |
| `test` | Test-only changes |
| `perf` | Performance improvement |
| `style` | Formatting, whitespace, naming — no logic change |

### Body (optional but encouraged for non-obvious changes)

- Explain **why**, not what — the diff already shows what
- Mention the context a reviewer would need: what was broken, what constraint drove the design, what alternative was considered and rejected
- Leave a blank line between subject and body

### Examples of good messages

```
feat: add NIfTI upload pipeline with tool registry

Gradio's file_types validation rejects .nii.gz unless the component is
given explicit mime-type overrides. The tool registry maps extension →
loader so new formats can be added without touching the upload handler.
```

```
fix: never raise gr.Error inside upload handler

Unhandled gr.Error propagated to the Gradio event loop and silently
dropped the user's file. Return a structured error dict instead.
```

```
refactor: decouple ingestion config from main Config dataclass

IngestionConfig was buried inside Config, making it impossible to
instantiate a pipeline without a full app config. Separate class allows
unit-testing the ingestion logic in isolation.
```

### Anti-patterns to avoid

- Vague subjects: `Fix bug`, `Update stuff`, `WIP`, `Big refactor`
- Describing the diff: `Change foo() to use bar() instead of baz()`
- Combining unrelated work: `Fix login and add dark mode and update deps`
- Noise words: `Some changes`, `Minor tweaks`, `Misc fixes`

---

## Step 4 — Stage and commit each chunk

For each conceptual chunk, in order:

1. Stage only the files (or hunks with `git add -p`) that belong to this chunk
2. Confirm the staged diff matches the intended chunk (`git diff --cached`)
3. Commit with the message composed in Step 3
4. Move to the next chunk

Use a HEREDOC to pass multi-line messages:

```bash
git commit -m "$(cat <<'EOF'
feat: add NIfTI upload pipeline with tool registry

Gradio's file_types validation rejects .nii.gz unless explicit mime-type
overrides are provided. Tool registry maps extension → loader.
EOF
)"
```

---

## Step 5 — Verify

After all commits, run `git log --oneline -10` and show the result. Confirm the log reads like a clear, human-navigable history of decisions — not a list of keystrokes.
