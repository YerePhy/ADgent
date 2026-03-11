"""Graph state schema."""

import operator

from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AnyMessage


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
