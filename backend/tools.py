from langchain_core.tools import BaseTool, tool
from pandasql import sqldf

from backend.dataloaders import DataLoader


def make_query_table_tool(data_loader: DataLoader) -> BaseTool:
    """Factory that creates a query_table tool bound to a DataLoader.

    Args:
        data_loader: The DataLoader used to retrieve registered tables.

    Returns:
        A LangChain tool that executes SQL queries against a registered table.
    """

    @tool
    def query_table(table_name: str, sql: str) -> str:
        """Query a registered table using SQL syntax.

        Loads the requested table as a pandas DataFrame and executes the
        provided SQL query against it using pandasql. The table must be
        referenced in the FROM clause using its exact registered name.

        Args:
            table_name: The registered name of the table to query.
            sql: A valid SQL query. The FROM clause must reference the table
                by its registered name (i.e. the value passed as table_name).

        Returns:
            Query result as a formatted string.

        Raises:
            ValueError: If table_name is not found in the registry.
        """
        df = data_loader.load_table(table_name)
        result = sqldf(sql, {table_name: df})
        return result.to_string(index=False)

    return query_table