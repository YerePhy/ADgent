import hashlib
import logging

import gradio as gr
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from backend.app_context import AppContext

logger = logging.getLogger(__name__)


def extract_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if b.get("type") == "text")
    return content


def make_thread_id(session_salt: str, username: str) -> str:
    raw = f"{session_salt}:{username.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def login(username: str, ctx: AppContext):
    if not username.strip():
        raise gr.Error("Username cannot be empty.")
    thread_id = make_thread_id(ctx.session_salt, username)
    logger.info("Session started for user=%r thread_id=%s", username.strip(), thread_id)
    return thread_id, gr.update(visible=False), gr.update(visible=True)


def load_history(thread_id: str, ctx: AppContext):
    runnable_cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    prior = ctx.agent.get_state(runnable_cfg)
    history = []
    for msg in prior.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif not isinstance(msg, ToolMessage) and hasattr(msg, "content"):
            history.append({"role": "assistant", "content": extract_text(msg.content)})
    return history


def respond(user_input: str, history: list[dict], thread_id: str, ctx: AppContext):
    if not user_input.strip():
        yield history, ""
        return

    runnable_cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    prior = ctx.agent.get_state(runnable_cfg)
    has_prior = bool(prior.values.get("messages"))
    new_messages: list[BaseMessage] = [] if has_prior else [ctx.system_message]
    new_messages.append(HumanMessage(content=user_input))

    history = history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": ""}]
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
