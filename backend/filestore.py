"""File storage abstraction for uploaded neuroimaging files.

Backends are registered in ``_BACKENDS`` as factory callables.
To add a new backend (e.g. S3), implement ``FileStore`` and add one entry there.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path


class FileStore(ABC):
    """Abstract interface for storing and resolving uploaded files."""

    @abstractmethod
    def save(self, data: bytes, filename: str, thread_id: str) -> str:
        """Persist bytes and return an opaque store_key.

        Args:
            data: Raw file bytes.
            filename: Original filename (used for extension/naming).
            thread_id: Per-user namespace.

        Returns:
            An opaque store_key — callers must not parse or construct this.
        """

    @abstractmethod
    def resolve(self, store_key: str) -> Path:
        """Return a local Path that nibabel (or any reader) can open.

        For remote backends this may download to a temporary file.

        Args:
            store_key: Value previously returned by ``save``.

        Returns:
            A local filesystem path to the file.
        """

    @abstractmethod
    def delete(self, store_key: str) -> None:
        """Remove the file from backing storage.

        Args:
            store_key: Value previously returned by ``save``.
        """


class LocalFileStore(FileStore):
    """FileStore backed by the local filesystem.

    Files are organised as ``<base_dir>/<thread_id>/<filename>``.

    Args:
        base_dir: Root directory for all uploads.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, filename: str, thread_id: str) -> str:
        dest = self._base / thread_id / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return str(dest)

    def resolve(self, store_key: str) -> Path:
        return Path(store_key)

    def delete(self, store_key: str) -> None:
        Path(store_key).unlink(missing_ok=True)


# Backend registry — each entry is a callable(**kwargs) -> FileStore.
# Lambdas document exactly which kwargs each backend requires.

_BACKENDS: dict[str, Callable[..., FileStore]] = {
    "local": lambda **kw: LocalFileStore(base_dir=Path(kw["base_dir"])),
}

_KNOWN_KWARGS: dict[str, set[str]] = {
    "local": {"base_dir"},
}


def create_file_store(backend: str, **kwargs) -> FileStore:
    """Instantiate a FileStore for the given backend name.

    Args:
        backend: Registered backend name (e.g. ``"local"``).
        **kwargs: Backend-specific keyword arguments.

    Returns:
        A ready-to-use ``FileStore`` instance.

    Raises:
        ValueError: If ``backend`` is not in the registry.
    """
    factory = _BACKENDS.get(backend)
    if factory is None:
        raise ValueError(
            f"Unknown file store backend {backend!r}. "
            f"Available: {', '.join(_BACKENDS)}"
        )
    allowed = _KNOWN_KWARGS.get(backend)
    filtered = {k: v for k, v in kwargs.items() if v is not None and (allowed is None or k in allowed)}
    return factory(**filtered)
