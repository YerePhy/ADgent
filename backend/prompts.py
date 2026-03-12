import os

from langchain_core.messages import SystemMessage

from backend.dataloaders import DataLoader, LocalDataLoader


def load_system_message(data_loader: DataLoader) -> SystemMessage:
    """Build the system message by enriching the base prompt with table schemas.

    Reads the ``SYSTEM_PROMPT`` environment variable as the base behavioural
    prompt, then appends a description of all available tables (name, columns,
    and dtypes) sourced from ``data_loader``.

    Args:
        data_loader: Provides access to available tables and their schemas.

    Returns:
        A LangChain ``SystemMessage`` with the full system prompt.
    """
    base_prompt = os.getenv("SYSTEM_PROMPT", "")

    schemas = []
    for table_name in data_loader.list_tables():
        schema = data_loader.get_table_schema(table_name)
        columns_info = ", ".join(
            f"{col} ({dtype})" for col, dtype in zip(schema.columns, schema.dtypes)
        )
        schemas.append(f"- {schema.name}: {columns_info}")

    schema_section = "\n".join(schemas)
    content = f"{base_prompt}\n\nAvailable tables:\n{schema_section}"

    return SystemMessage(content=content)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    data_loader = LocalDataLoader(data_dir="./data", registry="./registry.json")
    system_message = load_system_message(data_loader)
    print(system_message.content)
