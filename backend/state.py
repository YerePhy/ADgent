import operator

from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AnyMessage


class MessageState(TypedDict):
    """LangGraph state schema for the chat agent.

    Attributes:
        messages: Accumulated list of messages; new messages are appended via operator.add.
        llm_calls: Running count of LLM invocations.
    """
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
