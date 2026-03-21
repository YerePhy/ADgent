"""Data loader abstractions for accessing research data assets."""

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


@dataclass
class PdfInfo:
    """Metadata for a registered PDF document (paper, thesis, etc.).

    Attributes:
        name: The file's stem name.
        title: The document's title.
        authors: The document's authors.
        type: Document type (e.g. ``"paper"``, ``"thesis"``).
    """

    name: str
    title: str
    authors: str
    type: str


@dataclass
class ProjectInfo:
    """Grouped view of all assets belonging to a single project.

    Attributes:
        name: The project's registered name.
        description: A short description of the project's scope and goals.
        tables: Table stem names belonging to this project.
        text_files: Text file stem names belonging to this project.
        code_files: Code file stem names belonging to this project.
        pdfs: PDF document metadata belonging to this project.
    """

    name: str
    description: str
    tables: list[str]
    text_files: list[str]
    code_files: list[str]
    pdfs: list[PdfInfo]


class DataLoader(ABC):
    """Abstract interface for loading research data assets."""

    @abstractmethod
    def list_pdfs(self) -> list[PdfInfo]:
        """List available PDF documents with their metadata."""
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
    def list_text_files(self) -> list[str]:
        """List registered text file names."""
        ...

    @abstractmethod
    def list_code_files(self) -> list[str]:
        """List registered code file names."""
        ...

    @abstractmethod
    def list_projects(self) -> list[ProjectInfo]:
        """List all projects with their grouped assets."""
        ...


class LocalDataLoader(DataLoader):
    """DataLoader implementation backed by a local directory and a JSON registry.

    Args:
        data_dir: Root directory containing all data assets.
        registry: Path to the JSON registry file mapping project names
            to asset paths.
    """

    def __init__(self, data_dir: Path | str, registry: Path | str) -> None:
        self._data_dir = Path(data_dir)
        self._registry = json.loads(Path(registry).read_text())

        self._table_registry: dict[str, Path] = {}
        self._text_registry: dict[str, Path] = {}
        self._code_registry: dict[str, Path] = {}
        self._pdf_registry: dict[str, PdfInfo] = {}
        self._project_registry: dict[str, ProjectInfo] = {}

        if isinstance(self._registry, dict):
            items = self._registry.items()
        else:
            items = enumerate(self._registry)

        for project_name, entry in items:
            project_tables: list[str] = []
            project_text: list[str] = []
            project_code: list[str] = []
            project_pdfs: list[PdfInfo] = []

            # Tables: list of dicts with at least a "path" key
            for table_entry in entry.get("tables", []):
                p = Path(table_entry["path"])
                self._table_registry[p.stem] = self._data_dir / p
                project_tables.append(p.stem)

            # Text and code: plain string paths
            for key, reg, project_list in (
                ("text", self._text_registry, project_text),
                ("code", self._code_registry, project_code),
            ):
                for path in entry.get(key, []):
                    p = Path(path)
                    reg[p.stem] = self._data_dir / p
                    project_list.append(p.stem)

            # PDFs: dicts with path, title, authors, type
            for pdf in entry.get("pdfs", []):
                if not pdf.get("path"):
                    continue
                p = Path(pdf["path"])
                info = PdfInfo(
                    name=p.stem,
                    title=pdf.get("title", ""),
                    authors=pdf.get("authors", ""),
                    type=pdf.get("type", "paper"),
                )
                self._pdf_registry[p.stem] = info
                project_pdfs.append(info)

            self._project_registry[str(project_name)] = ProjectInfo(
                name=str(project_name),
                description=entry.get("description", ""),
                tables=project_tables,
                text_files=project_text,
                code_files=project_code,
                pdfs=project_pdfs,
            )

    def list_pdfs(self) -> list[PdfInfo]:
        """List available PDF documents with their metadata."""
        return sorted(self._pdf_registry.values(), key=lambda p: p.name)

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

    def list_text_files(self) -> list[str]:
        """List registered text file names."""
        return sorted(self._text_registry.keys())

    def list_code_files(self) -> list[str]:
        """List registered code file names."""
        return sorted(self._code_registry.keys())

    def list_projects(self) -> list[ProjectInfo]:
        """List all projects with their grouped assets."""
        return sorted(self._project_registry.values(), key=lambda p: p.name)
