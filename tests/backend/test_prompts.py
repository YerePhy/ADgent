"""Unit tests for backend.prompts — prompt rendering functions."""

from unittest.mock import MagicMock

from langchain_core.messages import SystemMessage

from backend.dataloaders import DataLoader, PdfInfo, ProjectInfo, TableSchema
from backend.prompts import _render_name, _render_pdf, _render_table, load_system_message


def _make_loader(
    *projects: ProjectInfo, schema_map: dict[str, TableSchema] | None = None
) -> DataLoader:
    loader = MagicMock(spec=DataLoader)
    loader.list_projects.return_value = list(projects)
    if schema_map:
        loader.get_table_schema.side_effect = lambda name: schema_map[name]
    return loader


def _project(
    name: str = "proj",
    description: str = "",
    tables: list[str] | None = None,
    text_files: list[str] | None = None,
    code_files: list[str] | None = None,
    pdfs: list[PdfInfo] | None = None,
) -> ProjectInfo:
    return ProjectInfo(
        name=name,
        description=description,
        tables=tables or [],
        text_files=text_files or [],
        code_files=code_files or [],
        pdfs=pdfs or [],
    )


def test_render_name_formats_bullet():
    assert _render_name("myfile", None) == "- myfile"  # type: ignore[arg-type]


def test_render_pdf_formats_bullet():
    pdf = PdfInfo(name="paper1", title="My Study", authors="Doe, J.", type="paper")
    assert _render_pdf(pdf, None) == "- [paper] My Study (Doe, J.)"  # type: ignore[arg-type]


def test_render_pdf_uses_type_field():
    pdf = PdfInfo(name="thesis1", title="A Thesis", authors="Smith, A.", type="thesis")
    assert _render_pdf(pdf, None) == "- [thesis] A Thesis (Smith, A.)"  # type: ignore[arg-type]


def test_render_table_formats_columns_and_dtypes():
    schema = TableSchema(name="patients", columns=["id", "age"], dtypes=["int64", "float64"])
    loader = MagicMock(spec=DataLoader)
    loader.get_table_schema.return_value = schema

    result = _render_table("patients", loader)

    assert result == "- patients: id (int64), age (float64)"
    loader.get_table_schema.assert_called_once_with("patients")


def test_render_table_single_column():
    schema = TableSchema(name="ids", columns=["subject_id"], dtypes=["object"])
    loader = MagicMock(spec=DataLoader)
    loader.get_table_schema.return_value = schema

    assert _render_table("ids", loader) == "- ids: subject_id (object)"


def test_load_system_message_returns_system_message_type():
    loader = _make_loader(_project())
    assert isinstance(load_system_message(loader, "base prompt"), SystemMessage)


def test_load_system_message_includes_base_prompt():
    loader = _make_loader(_project(name="p"))
    msg = load_system_message(loader, "You are a helpful agent.")
    assert "You are a helpful agent." in msg.content


def test_load_system_message_includes_project_header():
    loader = _make_loader(_project(name="AlzheimerStudy"))
    msg = load_system_message(loader, "")
    assert "## Project: AlzheimerStudy" in msg.content


def test_load_system_message_includes_project_description():
    loader = _make_loader(_project(name="p", description="Study on tau PET imaging."))
    msg = load_system_message(loader, "")
    assert "Study on tau PET imaging." in msg.content


def test_load_system_message_omits_empty_sections():
    loader = _make_loader(_project(name="p"))
    msg = load_system_message(loader, "")
    assert "Tables:" not in msg.content
    assert "PDFs:" not in msg.content
    assert "Text files:" not in msg.content
    assert "Code files:" not in msg.content


def test_load_system_message_renders_text_files():
    loader = _make_loader(_project(name="p", text_files=["notes", "readme"]))
    msg = load_system_message(loader, "")
    assert "Text files:" in msg.content
    assert "- notes" in msg.content
    assert "- readme" in msg.content


def test_load_system_message_renders_pdfs():
    pdf = PdfInfo(name="study", title="Tau PET Study", authors="Doe, J.", type="paper")
    loader = _make_loader(_project(name="p", pdfs=[pdf]))
    msg = load_system_message(loader, "")
    assert "PDFs:" in msg.content
    assert "- [paper] Tau PET Study (Doe, J.)" in msg.content


def test_load_system_message_renders_tables():
    schema = TableSchema(name="cohort", columns=["id", "group"], dtypes=["int64", "object"])
    loader = _make_loader(
        _project(name="p", tables=["cohort"]),
        schema_map={"cohort": schema},
    )
    msg = load_system_message(loader, "")
    assert "Tables:" in msg.content
    assert "- cohort: id (int64), group (object)" in msg.content


def test_load_system_message_multiple_projects():
    loader = _make_loader(_project(name="Alpha"), _project(name="Beta"))
    msg = load_system_message(loader, "")
    assert "## Project: Alpha" in msg.content
    assert "## Project: Beta" in msg.content
