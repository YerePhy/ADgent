"""System message construction for the agent."""

from langchain_core.messages import SystemMessage

from backend.dataloaders import DataLoader


def load_system_message(data_loader: DataLoader, system_prompt: str) -> SystemMessage:
    """Build the system message by enriching the base prompt with table schemas.

    Appends a description of all available tables (name, columns, and dtypes)
    sourced from ``data_loader`` to the provided base behavioural prompt.

    Args:
        data_loader: Provides access to available tables and their schemas.
        system_prompt: Base behavioural prompt for the agent.

    Returns:
        A LangChain ``SystemMessage`` with the full system prompt.
    """
    schemas = []
    for table_name in data_loader.list_tables():
        schema = data_loader.get_table_schema(table_name)
        columns_info = ", ".join(
            f"{col} ({dtype})" for col, dtype in zip(schema.columns, schema.dtypes)
        )
        schemas.append(f"- {schema.name}: {columns_info}")

    schema_section = "\n".join(schemas)
    content = f"{system_prompt}\n\nAvailable tables:\n{schema_section}"

    return SystemMessage(content=content)
