from langchain_core.messages import SystemMessage

from backend.dataloaders import DataLoader


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
    sections = []

    tables = data_loader.list_tables()
    if tables:
        lines = []
        for table_name in tables:
            schema = data_loader.get_table_schema(table_name)
            columns_info = ", ".join(
                f"{col} ({dtype})" for col, dtype in zip(schema.columns, schema.dtypes)
            )
            lines.append(f"- {schema.name}: {columns_info}")
        sections.append("Available tables:\n" + "\n".join(lines))

    text_files = data_loader.list_text_files()
    if text_files:
        lines = [f"- {name}" for name in text_files]
        sections.append("Available text files:\n" + "\n".join(lines))

    code_files = data_loader.list_code_files()
    if code_files:
        lines = [f"- {name}" for name in code_files]
        sections.append("Available code files:\n" + "\n".join(lines))

    papers = data_loader.list_papers()
    if papers:
        lines = [f"- {name}" for name in papers]
        sections.append("Available papers:\n" + "\n".join(lines))

    content = system_prompt + "\n\n" + "\n\n".join(sections)

    return SystemMessage(content=content)
