from collections.abc import Callable
from typing import Any

from langchain_core.messages import SystemMessage

from backend.dataloaders import DataLoader, PdfInfo


def _render_table(table_name: str, data_loader: DataLoader) -> str:
    schema = data_loader.get_table_schema(table_name)
    columns_info = ", ".join(
        f"{col} ({dtype})" for col, dtype in zip(schema.columns, schema.dtypes)
    )
    return f"- {schema.name}: {columns_info}"


def _render_name(name: str, _data_loader: DataLoader) -> str:
    return f"- {name}"


def _render_pdf(pdf: PdfInfo, _data_loader: DataLoader) -> str:
    return f"- [{pdf.type}] {pdf.title} ({pdf.authors})"


_ASSET_SECTIONS: list[tuple[str, str, Callable[[Any, DataLoader], str]]] = [
    ("tables", "Tables", _render_table),
    ("text_files", "Text files", _render_name),
    ("code_files", "Code files", _render_name),
    ("pdfs", "PDFs", _render_pdf),
]


def load_system_message(data_loader: DataLoader, system_prompt: str) -> SystemMessage:
    """Build the system message by enriching the base prompt with available assets.

    Appends descriptions of all available tables, text files, code files, and
    papers sourced from ``data_loader`` to the provided base behavioural prompt.

    Args:
        data_loader: Provides access to available data assets and their metadata.
        system_prompt: Base behavioural prompt for the agent.

    Returns:
        A LangChain ``SystemMessage`` with the full system prompt.
    """
    project_sections = []

    for project in data_loader.list_projects():
        lines = [f"## Project: {project.name}"]

        if project.description:
            lines.append(project.description)

        for field, label, render in _ASSET_SECTIONS:
            items = getattr(project, field)
            if items:
                lines.append(f"{label}:")
                lines.extend(render(item, data_loader) for item in items)

        project_sections.append("\n".join(lines))

    content = system_prompt + "\n\n" + "\n\n".join(project_sections)

    return SystemMessage(content=content)
