import logging.config
import os
from pathlib import Path

import yaml
import gradio as gr
from pyprojroot import here
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
Path(_logging_cfg).parent.joinpath("logs").mkdir(exist_ok=True)
with open(_logging_cfg) as _f:
    logging.config.dictConfig(yaml.safe_load(_f))

logger = logging.getLogger(__name__)

config = load_config()

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
agent = build_agent(chat, tools, max_llm_calls=config.agent.max_llm_calls)
system_message = load_system_message(data_loader, config.system_prompt)
logger.info("System prompt: %s", config.system_prompt)
logger.info("Agent built successfully (max_llm_calls=%d)", config.agent.max_llm_calls)


def respond(user_input: str, history: list[dict]) -> str:
    messages = [system_message]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_input))

    state = agent.invoke({"messages": messages, "llm_calls": 0})
    content = state["messages"][-1].content
    if isinstance(content, list):
        content = "\n".join(
            block["text"] for block in content if block.get("type") == "text"
        )

    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if tool_msgs:
        tool_info = "\n".join(f"- {tm.name}" for tm in tool_msgs)
        content = f"*Tools used:*\n{tool_info}\n\n{content}"

    return content


demo = gr.ChatInterface(
    fn=respond,
    title="ADgent",
    description="Alzheimer Imaging Assistant",
)

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )