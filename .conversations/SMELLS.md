# app.py — Code Smells & Design Review

## Logic Flow

1. **Module-level init** (lines 34–65): DB, vectorstore, embeddings, agent, tools are all instantiated at import time as globals.
2. **Login** returns a `thread_id` (SHA256 of username+salt), swaps panels.
3. **load_history** replays checkpointed LangGraph messages back into Gradio format.
4. **respond** injects the system message only on first turn (to avoid duplication), invokes the agent synchronously, formats the reply, appends tool usage info.

---

## Smells

### 1. Module-level God Object (biggest smell)
Everything initializes at import time as implicit globals. This means:
- Untestable without side effects — you can't instantiate the app with different configs without monkey-patching
- Gradio's multi-worker mode can fork after import, leaving the SQLite connection in an inconsistent state
- Cold import builds the vectorstore, opens DB, calls the network — any failure crashes before the app even starts

**Fix:** wrap in a factory or a dataclass:
```python
@dataclasses.dataclass
class AppContext:
    agent: CompiledGraph
    system_message: BaseMessage
    ...

def build_app_context(config: Config) -> AppContext: ...
```

### 2. `check_same_thread=False` on SQLite (line 40)
Gradio is threaded. Sharing one `sqlite3.Connection` across threads without proper locking is a data-corruption risk. WAL helps reads but doesn't protect writes. Either use `threading.local()`, a connection pool, or switch to `SqliteSaver` with its own connection management.

### 3. Duplicate content-extraction logic
Lines 92–94 and 113–115 are identical — extracting text from multimodal LangChain content blocks. One helper, zero duplication:
```python
def extract_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if b.get("type") == "text")
    return content
```

### 4. `config` name collision (line 84)
`config: RunnableConfig` inside `load_history` shadows the module-level `config = load_config()`. If those two ever interact, the bug will be silent. Rename to `runnable_cfg` or `lc_config`.

### 5. Business logic in UI callbacks
`respond` does: system-message injection, agent invocation, content extraction, tool formatting, history management. That's 5 responsibilities. The "Tools used" annotation is presentation logic for something that should happen closer to the agent layer (or be a dedicated formatter).

### 6. Synchronous `agent.invoke` blocks the UI thread
For Phase 1+ (nibabel), Phase 2 (E2B sandbox), Phase 3 (AWS Batch), this will hang the UI. Gradio has `demo.queue()` + generator-based streaming (`yield`). This needs to be addressed before adding any slow tools.

### 7. `SESSION_SALT` default (minor)
A hardcoded fallback `"adgent-default-salt"` means two separate deployments without setting the env var will generate the same `thread_id` for the same username. Not a collision risk in a single-user local setup, but worth a loud warning log.

---

## Suggested Pattern

The file has three distinct responsibilities that should be split:

| File | Responsibility |
|---|---|
| `backend/app_context.py` | Build and wire all infrastructure (DB, agent, tools) |
| `backend/chat_handlers.py` | `login`, `load_history`, `respond` — pure functions taking context as arg |
| `backend/app.py` | Only Gradio block definition + wiring |

`app.py` drops to ~40 lines. Handlers become testable without spinning up Gradio. Infrastructure becomes mockable by swapping `AppContext`.

---

## Priority Order

1. **Streaming refactor** — most urgent before adding any slow tools (Phase 1+)
2. **`extract_text` helper** — trivial, eliminates the duplication now
3. **`AppContext` split** — unlocks testability, do before the codebase grows further
4. **SQLite threading** — fix when moving to production or multi-user deployment
