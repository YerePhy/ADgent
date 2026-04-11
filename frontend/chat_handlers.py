import hmac
import hashlib
import logging
from pathlib import Path

import gradio as gr
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from backend.app_context import AppContext

_ALLOWED_SUFFIXES = {".nii", ".nii.gz"}

logger = logging.getLogger(__name__)


def extract_text(content) -> str:
    """Extract plain text from a message content value.

    LangChain multi-modal messages can carry content as a list of typed blocks
    (e.g. ``[{"type": "text", "text": "..."}]``) or as a plain string.  This
    helper normalises both forms to a single string.

    Args:
        content: A raw message content value — either a string or a list of
            block dicts.

    Returns:
        The concatenated text blocks, or the original string if content is
        already a string.
    """
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if b.get("type") == "text")
    return content


def make_thread_id(session_salt: str, username: str) -> str:
    """Derive a deterministic, anonymised thread ID from a username.

    Uses HMAC-SHA256 so that the same username always maps to the same thread
    ID without exposing the raw name in storage.  The result is truncated to
    16 hex characters (64-bit) — enough to avoid collisions for typical user
    counts.

    Args:
        session_salt: HMAC key (from the ``SESSION_SALT`` env var).
        username: Raw username as entered by the user.

    Returns:
        A 16-character lowercase hex string.
    """
    return hmac.new(
        session_salt.encode(),
        username.strip().lower().encode(),
        hashlib.sha256,
    ).hexdigest()[:16]


def login(username: str, ctx: AppContext):
    """Handle the login button click and open a named session.

    Validates that the username is non-empty, derives a deterministic thread
    ID, and returns Gradio panel visibility updates to hide the login form and
    show the chat panel.

    Args:
        username: Username entered by the user.
        ctx: Application context supplying the session salt.

    Returns:
        A tuple of ``(thread_id, login_panel_update, chat_panel_update)``.

    Raises:
        gr.Error: If username is empty or blank.
    """
    if not username.strip():
        raise gr.Error("Username cannot be empty.")
    thread_id = make_thread_id(ctx.session_salt, username)
    logger.info("Session started for user=%r thread_id=%s", username.strip(), thread_id)
    return thread_id, gr.update(visible=False), gr.update(visible=True)


def load_history(thread_id: str, ctx: AppContext):
    """Restore prior conversation history for a thread from the checkpointer.

    Reads persisted LangGraph state and reconstructs the Gradio chat history
    list, filtering out tool messages (which are internal agent plumbing and
    should not be shown to the user).

    Args:
        thread_id: Active session thread ID.
        ctx: Application context supplying the compiled agent.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts compatible with
        ``gr.Chatbot``.
    """
    runnable_cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    prior = ctx.agent.get_state(runnable_cfg)
    history = []
    for msg in prior.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif not isinstance(msg, ToolMessage) and hasattr(msg, "content"):
            history.append({"role": "assistant", "content": extract_text(msg.content)})
    return history


def upload_file(
    file_path: str | None,
    history: list[dict],
    thread_id: str,
    ctx: AppContext,
):
    """Handle a NIfTI file upload from the Gradio file component.

    Validates the file extension server-side, saves bytes via FileStore,
    records the upload in UploadStore, and appends a confirmation message
    to the chat history.

    Args:
        file_path: Temporary path Gradio wrote the upload to (None if cleared).
        history: Current chat history.
        thread_id: Active session thread ID.
        ctx: Application context.

    Returns:
        Updated chat history.
    """
    if file_path is None:
        return history

    path = Path(file_path)
    filename = path.name

    # Server-side extension check (.nii.gz has two suffixes, so check the full name)
    suffix = "".join(path.suffixes)  # e.g. ".nii" or ".nii.gz"
    if suffix not in _ALLOWED_SUFFIXES:
        msg = f"Unsupported file type '{suffix}'. Only .nii and .nii.gz are accepted."
        return history + [{"role": "assistant", "content": f"Upload failed: {msg}"}]

    if not thread_id:
        msg = "No active session. Please log in before uploading."
        return history + [{"role": "assistant", "content": f"Upload failed: {msg}"}]

    try:
        data = path.read_bytes()
        store_key = ctx.file_store.save(data, filename, thread_id)
        ctx.upload_store.save_upload(thread_id, store_key, filename)
    except Exception:
        logger.exception("Failed to save upload: filename=%s thread_id=%s", filename, thread_id)
        return history + [
            {
                "role": "assistant",
                "content": f"Upload failed: could not save **{filename}**. Check the logs.",
            }
        ]

    logger.info(
        "File uploaded: filename=%s thread_id=%s store_key=%s", filename, thread_id, store_key
    )
    return history + [{"role": "assistant", "content": f"File uploaded: **{filename}**"}]


def respond(user_input: str, history: list[dict], thread_id: str, ctx: AppContext):
    """Stream the agent's reply to a user message and yield incremental updates.

    Prepends the system message on the first turn, then streams the agent
    graph token by token.  Tool usage is surfaced as a summary header prepended
    to the final assistant message.

    Args:
        user_input: The message typed by the user.
        history: Current Gradio chat history.
        thread_id: Active session thread ID used to route to the correct
            LangGraph checkpoint.
        ctx: Application context supplying the compiled agent and system message.

    Yields:
        Tuples of ``(updated_history, cleared_input)`` at each streaming step.
    """
    if not user_input.strip():
        yield history, ""
        return

    runnable_cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    prior = ctx.agent.get_state(runnable_cfg)
    has_prior = bool(prior.values.get("messages"))
    new_messages: list[BaseMessage] = [] if has_prior else [ctx.system_message]
    new_messages.append(HumanMessage(content=user_input))

    history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": ""},
    ]
    tool_names: list[str] = []

    for chunk, metadata in ctx.agent.stream(
        {"messages": new_messages, "llm_calls": 0},
        config=runnable_cfg,
        stream_mode="messages",
    ):
        if isinstance(chunk, ToolMessage):
            tool_names.append(chunk.name)
        elif metadata.get("langgraph_node") == "llm_call":
            text = extract_text(chunk.content) if chunk.content else ""
            if text:
                history[-1]["content"] += text
                yield history, ""

    if tool_names:
        tool_info = "*Tools used:*\n" + "\n".join(f"- {n}" for n in tool_names)
        history[-1]["content"] = tool_info + "\n\n" + history[-1]["content"]
        yield history, ""
