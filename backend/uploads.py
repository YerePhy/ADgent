"""Upload record persistence.

Tracks every file saved through the FileStore so tools and handlers can
look up which files belong to a given thread.

Backends are registered in _BACKENDS as factory callables.
To add a new backend (e.g. Postgres), implement UploadStore and add one entry there.
"""

import dataclasses
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone


@dataclasses.dataclass
class UploadRecord:
    """A single row from the uploads table.

    Attributes:
        id: Auto-incremented row ID.
        thread_id: Thread that owns the file.
        store_key: Opaque key returned by FileStore.save — pass to FileStore.resolve.
        filename: Original filename as uploaded by the user.
        uploaded_at: ISO-8601 UTC timestamp.
    """

    id: int
    thread_id: str
    store_key: str
    filename: str
    uploaded_at: str


class UploadStore(ABC):
    """Abstract interface for persisting upload metadata."""

    @abstractmethod
    def save_upload(self, thread_id: str, store_key: str, filename: str) -> UploadRecord:
        """Record that a file was uploaded.

        Args:
            thread_id: Thread that owns the file.
            store_key: Value returned by FileStore.save.
            filename: Original filename.

        Returns:
            The persisted UploadRecord including its assigned id.
        """

    @abstractmethod
    def list_by_thread(self, thread_id: str) -> list[UploadRecord]:
        """Return all uploads for a thread, oldest first.

        Args:
            thread_id: Thread to query.

        Returns:
            List of UploadRecord instances (may be empty).
        """


class SqliteUploadStore(UploadStore):
    """UploadStore backed by SQLite.

    Designed to share the same connection (and WAL journal) as the
    LangGraph SqliteSaver so no second DB file is needed.

    Args:
        conn: An open sqlite3 connection (WAL mode expected).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT NOT NULL,
                store_key   TEXT NOT NULL UNIQUE,
                filename    TEXT NOT NULL,
                uploaded_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_uploads_thread ON uploads(thread_id)"
        )
        self._conn.commit()

    def save_upload(self, thread_id: str, store_key: str, filename: str) -> UploadRecord:
        uploaded_at = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO uploads (thread_id, store_key, filename, uploaded_at) "
            "VALUES (?, ?, ?, ?)",
            (thread_id, store_key, filename, uploaded_at),
        )
        self._conn.commit()
        return UploadRecord(
            id=cur.lastrowid,
            thread_id=thread_id,
            store_key=store_key,
            filename=filename,
            uploaded_at=uploaded_at,
        )

    def list_by_thread(self, thread_id: str) -> list[UploadRecord]:
        rows = self._conn.execute(
            "SELECT id, thread_id, store_key, filename, uploaded_at "
            "FROM uploads WHERE thread_id = ? ORDER BY uploaded_at",
            (thread_id,),
        ).fetchall()
        return [UploadRecord(*row) for row in rows]


# ---------------------------------------------------------------------------
# Backend registry
# Each entry is a callable(**kwargs) -> UploadStore.
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, Callable[..., UploadStore]] = {
    "sqlite": lambda **kw: SqliteUploadStore(conn=kw["conn"]),
    # "postgres": lambda **kw: PostgresUploadStore(dsn=kw["dsn"]),
}

_KNOWN_KWARGS: dict[str, set[str]] = {
    "sqlite": {"conn"},
    # "postgres": {"dsn"},
}


def create_upload_store(backend: str, **kwargs) -> UploadStore:
    """Instantiate an UploadStore for the given backend name.

    Args:
        backend: Registered backend name (e.g. ``"sqlite"``).
        **kwargs: Backend-specific keyword arguments.

    Returns:
        A ready-to-use ``UploadStore`` instance.

    Raises:
        ValueError: If ``backend`` is not in the registry.
    """
    factory = _BACKENDS.get(backend)
    if factory is None:
        raise ValueError(
            f"Unknown upload store backend {backend!r}. "
            f"Available: {', '.join(_BACKENDS)}"
        )
    allowed = _KNOWN_KWARGS.get(backend)
    filtered = {k: v for k, v in kwargs.items() if v is not None and (allowed is None or k in allowed)}
    return factory(**filtered)
