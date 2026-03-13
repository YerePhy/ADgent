import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TableSchema:
    """Schema metadata for a tabular dataset.

    Attributes:
        name: The table's registered name.
        columns: Ordered list of column names.
        dtypes: Corresponding pandas dtype strings for each column.
    """

    name: str
    columns: list[str]
    dtypes: list[str]


class DataLoader(ABC):
    """Abstract interface for loading research data assets."""

    @abstractmethod
    def list_papers(self) -> list[str]:
        """List available PDF papers by registered name."""
        ...

    @abstractmethod
    def list_tables(self) -> list[str]:
        """List registered table names."""
        ...

    @abstractmethod
    def load_table(self, name: Path | str) -> pd.DataFrame:
        """Load a table by its registered name.

        Args:
            name: Registered table name.

        Returns:
            The table as a pandas DataFrame.
        """
        ...

    @abstractmethod
    def get_table_schema(self, name: str) -> TableSchema:
        """Return the schema for a registered table.

        Args:
            name: Registered table name.

        Returns:
            A ``TableSchema`` with column names and dtype strings.
        """
        ...

    @abstractmethod
    def read_text(self, file_path: Path | str) -> str:
        """Read a registered unstructured text file.

        Args:
            file_path: Registered text file name.

        Returns:
            File contents as a string.
        """
        ...

    @abstractmethod
    def list_text_files(self) -> list[str]:
        """List registered text file names."""
        ...

    @abstractmethod
    def list_code_files(self) -> list[str]:
        """List registered code file names."""
        ...

    @abstractmethod
    def read_code(self, file_path: Path | str) -> str:
        """Read a registered code file.

        Args:
            file_path: Registered code file name.

        Returns:
            File contents as a string.
        """
        ...


class LocalDataLoader(DataLoader):
    """DataLoader implementation backed by a local directory and a JSON registry.

    Args:
        data_dir: Root directory containing all data assets.
        registry: Path to the JSON registry file mapping project names to asset paths.
    """

    def __init__(self, data_dir: Path | str, registry: Path | str) -> None:
        self._data_dir = Path(data_dir)
        self._registry = json.loads(Path(registry).read_text())

        self._table_registry: dict[str, Path] = {}
        self._text_registry: dict[str, Path] = {}
        self._code_registry: dict[str, Path] = {}
        self._paper_registry: dict[str, Path] = {}

        def _update_registry(
            registry: dict[str, Path], entries: list[dict], key: str
        ) -> None:
            for entry in entries:
                for path in entry.get(key, []):
                    p = Path(path)
                    registry[p.stem] = self._data_dir / p

        entries = (
            self._registry.values()
            if isinstance(self._registry, dict)
            else self._registry
        )
        _update_registry(self._table_registry, entries, "tables")
        _update_registry(self._text_registry, entries, "text")
        _update_registry(self._code_registry, entries, "code")
        _update_registry(self._paper_registry, entries, "papers")

    def list_papers(self) -> list[str]:
        """List available PDF papers by registered name."""
        return sorted(self._paper_registry.keys())

    def list_tables(self) -> list[str]:
        """List registered table names."""
        return list(self._table_registry.keys())

    def load_table(self, name: str) -> pd.DataFrame:
        """Load a table by its registered name.

        Args:
            name: Registered table name.

        Returns:
            The table as a pandas DataFrame.

        Raises:
            ValueError: If ``name`` is not found in the registry.
        """
        if name not in self._table_registry:
            raise ValueError(f"Table '{name}' not found in registry.")
        return pd.read_csv(self._table_registry[name])

    def get_table_schema(self, name: str) -> TableSchema:
        """Return the schema for a registered table.

        Args:
            name: Registered table name.

        Returns:
            A ``TableSchema`` with column names and dtype strings.

        Raises:
            ValueError: If ``name`` is not found in the registry.
        """
        if name not in self._table_registry:
            raise ValueError(f"Table '{name}' not found in registry.")
        df = pd.read_csv(self._table_registry[name], nrows=5)
        return TableSchema(
            name=name,
            columns=df.columns.tolist(),
            dtypes=df.dtypes.astype(str).tolist(),
        )

    def read_text(self, file_path: str) -> str:
        """Read a registered unstructured text file.

        Args:
            file_path: Registered text file name.

        Returns:
            File contents as a string.

        Raises:
            ValueError: If ``file_path`` is not found in the registry.
        """
        if file_path not in self._text_registry:
            raise ValueError(f"Text file '{file_path}' not found in registry.")
        return Path(self._text_registry[file_path]).read_text(encoding="utf-8")

    def list_text_files(self) -> list[str]:
        """List registered text file names."""
        return sorted(self._text_registry.keys())

    def list_code_files(self) -> list[str]:
        """List registered code file names."""
        return sorted(self._code_registry.keys())

    def read_code(self, file_path: str) -> str:
        """Read a registered code file.

        Args:
            file_path: Registered code file name.

        Returns:
            File contents as a string.

        Raises:
            ValueError: If ``file_path`` is not found in the registry.
        """
        if file_path not in self._code_registry:
            raise ValueError(f"Code file '{file_path}' not found in registry.")
        return Path(self._code_registry[file_path]).read_text(encoding="utf-8")