"""Shared infrastructure container.

Kept in its own module to avoid circular imports between app_context
(which builds Infrastructure) and tools (which consumes it).
"""

import dataclasses

from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.sqlite import SqliteSaver

from backend.dataloaders import DataLoader
from backend.filestore import FileStore
from backend.uploads import UploadStore


@dataclasses.dataclass
class Infrastructure:
    """Low-level application plumbing: persistence, retrieval, and file storage.

    Constructed once at startup; can be injected in tests with mocks so that
    build_app_context never needs to touch real I/O during unit testing.

    Attributes:
        memory: LangGraph SQLite checkpointer for conversation state.
        data_loader: Provides access to registered data assets.
        vectorstore: Semantic search backend.
        file_store: Upload storage backend.
        upload_store: Upload metadata registry.
    """

    memory: SqliteSaver
    data_loader: DataLoader
    vectorstore: VectorStore
    file_store: FileStore
    upload_store: UploadStore
