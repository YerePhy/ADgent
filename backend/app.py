import hashlib
import logging.config
import os
import sqlite3
from pathlib import Path

import yaml
import gradio as gr
from pyprojroot import here
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

from backend.agent import build_agent
from backend.chat_model_factory import create_chat_model
from backend.config import load_config
from backend.dataloaders import LocalDataLoader
from backend.prompts import load_system_message
from backend.tools import make_query_table_tool, make_search_documents_tool

from langchain_chroma import Chroma
from backend.embeddings import create_embeddings

_logging_cfg = here("logging.yaml")
_logging_cfg.parent.joinpath("logs").mkdir(exist_ok=True)
with open(_logging_cfg) as _f:
    logging.config.dictConfig(yaml.safe_load(_f))

logger = logging.getLogger(__name__)

config = load_config()

db_path = Path(os.getenv("DATABASE_PATH", "./dbase"))
db_path.mkdir(exist_ok=True, parents=True)
db_path = db_path / os.getenv("DATABASE_NAME", "history.sqlite")

conn = sqlite3.connect(str(db_path), check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")
memory = SqliteSaver(conn)

data_loader = LocalDataLoader(
    data_dir=config.data_dir,
    registry=config.registry,
)
logger.info("DataLoader: %s (data_dir=%s)", type(data_loader).__name__, config.data_dir)
embeddings = create_embeddings(config.embeddings.provider, config.embeddings.model)
vectorstore = Chroma(
    persist_directory=str(config.data_dir / "vectorstore"),
    embedding_function=embeddings,
)
tools = [
    make_query_table_tool(data_loader),
    make_search_documents_tool(vectorstore),
]
chat = create_chat_model(
    provider=config.llm.provider,
    model_repo_id=config.llm.model_repo_id,
)
agent = build_agent(chat, tools, max_llm_calls=config.agent.max_llm_calls, checkpointer=memory)
system_message = load_system_message(data_loader, config.system_prompt)
logger.info("System prompt: %s", config.system_prompt)
logger.info("Agent built successfully (max_llm_calls=%d)", config.agent.max_llm_calls)

_SESSION_SALT = os.getenv("SESSION_SALT", "adgent-default-salt")


def make_thread_id(username: str) -> str:
    raw = f"{_SESSION_SALT}:{username.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def login(username: str):
    if not username.strip():
        raise gr.Error("Username cannot be empty.")
    thread_id = make_thread_id(username)
    logger.info("Session started for user=%r thread_id=%s", username.strip(), thread_id)
    return thread_id, gr.update(visible=False), gr.update(visible=True)


def load_history(thread_id: str):
    prior = agent.get_state({"configurable": {"thread_id": thread_id}})
    history = []
    for msg in prior.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif not isinstance(msg, ToolMessage) and hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, list):
                content = "\n".join(b["text"] for b in content if b.get("type") == "text")
            history.append({"role": "assistant", "content": content})
    return history


def respond(user_input: str, history: list[dict], thread_id: str):
    if not user_input.strip():
        return history, ""

    config = {"configurable": {"thread_id": thread_id}}

    # Include the system message only on the first invocation for this thread
    # so it is not duplicated across turns when the checkpointer appends messages.
    prior = agent.get_state(config)
    has_prior = bool(prior.values.get("messages"))
    new_messages = [] if has_prior else [system_message]
    new_messages.append(HumanMessage(content=user_input))

    state = agent.invoke({"messages": new_messages, "llm_calls": 0}, config=config)
    content = state["messages"][-1].content
    if isinstance(content, list):
        content = "\n".join(b["text"] for b in content if b.get("type") == "text")

    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if tool_msgs:
        tool_info = "\n".join(f"- {tm.name}" for tm in tool_msgs)
        content = f"*Tools used:*\n{tool_info}\n\n{content}"

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": content})
    return history, ""


with gr.Blocks(title="ADgent") as demo:
    thread_id_state = gr.State(value=None)

    with gr.Column(visible=True) as login_panel:
        gr.Markdown("## ADgent — Alzheimer Imaging Assistant")
        username_input = gr.Textbox(label="Username", placeholder="Enter your name to start")
        login_btn = gr.Button("Start session")

    with gr.Column(visible=False) as chat_panel:
        chatbot = gr.Chatbot(label="ADgent")
        msg_input = gr.Textbox(label="Message", placeholder="Ask something...", show_label=False)
        send_btn = gr.Button("Send")

    login_btn.click(
        fn=login,
        inputs=[username_input],
        outputs=[thread_id_state, login_panel, chat_panel],
    ).then(
        fn=load_history,
        inputs=[thread_id_state],
        outputs=[chatbot],
    )

    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot, thread_id_state],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot, thread_id_state],
        outputs=[chatbot, msg_input],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )
