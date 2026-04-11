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
from backend.filestore import FileStore, create_file_store
from backend.infrastructure import Infrastructure
from backend.prompts import load_system_message
from backend.tools import build_tools
from backend.uploads import UploadStore, create_upload_store

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AppContext:
    """Runtime context consumed by request handlers.

    Attributes:
        agent: Compiled LangGraph state graph ready for invocation.
        system_message: Pre-built system prompt injected on the first turn.
        session_salt: HMAC key used to derive deterministic thread IDs.
        file_store: Upload storage backend (same instance held by Infrastructure).
    """

    agent: CompiledStateGraph
    system_message: BaseMessage
    session_salt: str
    file_store: FileStore
    upload_store: UploadStore


def build_infrastructure(config: Config) -> Infrastructure:
    """Set up all low-level infrastructure components from config.

    Handles SQLite connection (WAL mode), data loading, embeddings/vectorstore,
    and file storage backend. Nothing agent-specific lives here — this boundary
    is what makes build_app_context testable without real I/O.

    Args:
        config: Fully populated application configuration.

    Returns:
        An Infrastructure instance ready to be passed to build_app_context.
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

    fs_cfg = config.file_store
    file_store = create_file_store(fs_cfg.backend, base_dir=fs_cfg.base_dir, bucket=fs_cfg.bucket)
    logger.info("FileStore: backend=%s base_dir=%s", fs_cfg.backend, fs_cfg.base_dir)

    us_cfg = config.upload_store
    upload_store = create_upload_store(us_cfg.backend, conn=conn, dsn=us_cfg.dsn)
    logger.info("UploadStore: backend=%s", us_cfg.backend)

    return Infrastructure(
        memory=memory,
        data_loader=data_loader,
        vectorstore=vectorstore,
        file_store=file_store,
        upload_store=upload_store,
    )


def build_app_context(config: Config, infra: Infrastructure) -> AppContext:
    """Assemble the agent and runtime context from config and pre-built infrastructure.

    Intentionally separated from build_infrastructure so tests can inject a mock
    Infrastructure without triggering real DB or network calls.

    Args:
        config: Fully populated application configuration.
        infra: Pre-built infrastructure (from build_infrastructure or a test fixture).

    Returns:
        An AppContext ready to serve requests.
    """
    tools = build_tools(infra, config.agent.tools)

    chat = create_chat_model(
        provider=config.llm.provider,
        model_repo_id=config.llm.model_repo_id,
    )

    agent = build_agent(
        chat, tools, max_llm_calls=config.agent.max_llm_calls, checkpointer=infra.memory
    )
    system_message = load_system_message(infra.data_loader, config.system_prompt)

    session_salt = os.getenv("SESSION_SALT")
    if not session_salt:
        raise RuntimeError(
            "SESSION_SALT env var must be set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    logger.info("System prompt: %s", config.system_prompt)
    logger.info("Agent built successfully (max_llm_calls=%d)", config.agent.max_llm_calls)

    return AppContext(
        agent=agent,
        system_message=system_message,
        session_salt=session_salt,
        file_store=infra.file_store,
        upload_store=infra.upload_store,
    )
