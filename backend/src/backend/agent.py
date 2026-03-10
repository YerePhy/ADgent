"""Graph definition and compilation."""

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from backend.state import MessageState
from backend.nodes import make_llm_call


def build_agent(chat: BaseChatModel):
    """Build and compile the agent graph."""
    builder = StateGraph(MessageState)
    builder.add_node("llm_call", make_llm_call(chat))
    builder.add_edge(START, "llm_call")
    builder.add_edge("llm_call", END)
    return builder.compile()
