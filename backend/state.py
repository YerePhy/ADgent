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
    # No reducer annotation intentional: chat_handlers.py resets this to 0
    # on every turn by passing "llm_calls": 0 to agent.stream(). Adding a
    # reducer here (e.g. operator.add) would break the per-turn call guard.
    llm_calls: int
