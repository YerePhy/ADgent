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
class PaperInfo:
    """Metadata for a registered paper.

    Attributes:
        name: The paper's registered stem name.
        title: The paper's title.
        authors: The paper's authors.
    """

    name: str
    title: str
    authors: str


@dataclass
class ProjectInfo:
    """Grouped view of all assets belonging to a single project.

    Attributes:
        name: The project's registered name.
        description: A short description of the project's scope and goals.
        tables: Table stem names belonging to this project.
        text_files: Text file stem names belonging to this project.
        code_files: Code file stem names belonging to this project.
        papers: Paper metadata belonging to this project.
    """

    name: str
    description: str
    tables: list[str]
    text_files: list[str]
    code_files: list[str]
    papers: list[PaperInfo]


class DataLoader(ABC):
    """Abstract interface for loading research data assets."""

    @abstractmethod
    def list_papers(self) -> list[PaperInfo]:
        """List available PDF papers with their metadata."""
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
        self._paper_registry: dict[str, PaperInfo] = {}
        self._project_registry: dict[str, ProjectInfo] = {}

        if isinstance(self._registry, dict):
            items = self._registry.items()
        else:
            items = enumerate(self._registry)

        for project_name, entry in items:
            project_tables: list[str] = []
            project_text: list[str] = []
            project_code: list[str] = []
            project_papers: list[PaperInfo] = []

            # Tables, text, code: plain string paths
            for key, reg, project_list in (
                ("tables", self._table_registry, project_tables),
                ("text", self._text_registry, project_text),
                ("code", self._code_registry, project_code),
            ):
                for path in entry.get(key, []):
                    p = Path(path)
                    reg[p.stem] = self._data_dir / p
                    project_list.append(p.stem)

            # Papers: dicts with path, title, authors
            for paper in entry.get("papers", []):
                if not paper.get("path"):
                    continue
                p = Path(paper["path"])
                info = PaperInfo(
                    name=p.stem,
                    title=paper.get("title", ""),
                    authors=paper.get("authors", ""),
                )
                self._paper_registry[p.stem] = info
                project_papers.append(info)

            self._project_registry[str(project_name)] = ProjectInfo(
                name=str(project_name),
                description=entry.get("description", ""),
                tables=project_tables,
                text_files=project_text,
                code_files=project_code,
                papers=project_papers,
            )

    def list_papers(self) -> list[PaperInfo]:
        """List available PDF papers with their metadata."""
        return sorted(self._paper_registry.values(), key=lambda p: p.name)

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
