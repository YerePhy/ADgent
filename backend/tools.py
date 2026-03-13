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


def make_read_text_tool(data_loader: DataLoader) -> BaseTool:
    """Factory that creates a read_text_file tool bound to a DataLoader.

    Args:
        data_loader: The DataLoader used to retrieve registered text files.

    Returns:
        A LangChain tool that reads the content of a registered text file.
    """

    @tool
    def read_text_file(file_name: str) -> str:
        """Read the content of a registered text file.

        Args:
            file_name: The registered name of the text file to read.

        Returns:
            The full content of the text file as a string.

        Raises:
            ValueError: If file_name is not found in the registry.
        """
        return data_loader.read_text(file_name)

    return read_text_file


def make_read_code_tool(data_loader: DataLoader) -> BaseTool:
    """Factory that creates a read_code_file tool bound to a DataLoader.

    Args:
        data_loader: The DataLoader used to retrieve registered code files.

    Returns:
        A LangChain tool that reads the content of a registered code file.
    """

    @tool
    def read_code_file(file_name: str) -> str:
        """Read the content of a registered code file.

        Args:
            file_name: The registered name of the code file to read.

        Returns:
            The full source code as a string.

        Raises:
            ValueError: If file_name is not found in the registry.
        """
        return data_loader.read_code(file_name)

    return read_code_file