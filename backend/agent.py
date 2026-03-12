from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.nodes import make_llm_call
from backend.state import MessageState


def build_agent(chat: BaseChatModel) -> CompiledStateGraph:
    """Build and compile the agent graph.

    Args:
        chat: The chat model to bind to the LLM call node.

    Returns:
        A compiled LangGraph state graph ready for invocation.
    """
    builder = StateGraph(MessageState)
    builder.add_node("llm_call", make_llm_call(chat))
    builder.add_edge(START, "llm_call")
    builder.add_edge("llm_call", END)
    return builder.compile()
