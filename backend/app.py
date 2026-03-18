import logging

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from backend.agent import build_agent  # noqa: E402
from backend.chat_model_factory import create_chat_model  # noqa: E402
from backend.config import load_config  # noqa: E402
from backend.dataloaders import LocalDataLoader  # noqa: E402
from backend.prompts import load_system_message  # noqa: E402
from backend.tools import make_query_table_tool, make_search_documents_tool  # noqa: E402

from langchain_community.vectorstores import Chroma  # noqa: E402
from scripts.ingest import create_embeddings  # noqa: E402

logging.getLogger().setLevel(logging.CRITICAL)

config = load_config()

data_loader = LocalDataLoader(
    data_dir=config.data_dir,
    registry=config.registry,
)
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


def respond(user_input: str, history: list[dict]) -> str:
    messages = [system_message]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_input))

    state = agent.invoke({"messages": messages, "llm_calls": 0})
    content = state["messages"][-1].content
    if isinstance(content, list):
        content = "\n".join(
            block["text"] for block in content if block.get("type") == "text"
        )
    return content


demo = gr.ChatInterface(
    fn=respond,
    title="ADgent",
    description="Research assistant for Tau PET imaging in Alzheimer's disease.",
)

if __name__ == "__main__":
    demo.launch()