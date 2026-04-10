import dataclasses
import logging
import os
import sqlite3
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from backend.agent import build_agent
from backend.chat_model_factory import create_chat_model
from backend.config import Config
from backend.dataloaders import LocalDataLoader
from backend.embeddings import create_embeddings
from backend.prompts import load_system_message
from backend.tools import make_query_table_tool, make_search_documents_tool

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AppContext:
    agent: CompiledStateGraph
    system_message: BaseMessage
    session_salt: str


def build_app_context(config: Config) -> AppContext:
    """Instantiate all application infrastructure from a Config.

    Nothing runs at import time — callers decide when and with which config
    to build the context, making this factory mockable in tests.

    Args:
        config: Fully populated application configuration.

    Returns:
        An AppContext holding the compiled agent, system message, and session salt.
    """
    db_path = Path(os.getenv("DATABASE_PATH", "./dbase"))
    db_path.mkdir(exist_ok=True, parents=True)
    db_path = db_path / os.getenv("DATABASE_NAME", "history.sqlite")

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
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

    agent = build_agent(
        chat, tools, max_llm_calls=config.agent.max_llm_calls, checkpointer=memory
    )
    system_message = load_system_message(data_loader, config.system_prompt)

    session_salt = os.getenv("SESSION_SALT", "adgent-default-salt")
    if session_salt == "adgent-default-salt":
        logger.warning(
            "SESSION_SALT is using the hardcoded default. "
            "Two deployments without this env var will generate identical thread_ids for the same username."
        )

    logger.info("System prompt: %s", config.system_prompt)
    logger.info("Agent built successfully (max_llm_calls=%d)", config.agent.max_llm_calls)

    return AppContext(
        agent=agent,
        system_message=system_message,
        session_salt=session_salt,
    )
