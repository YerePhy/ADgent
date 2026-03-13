from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.nodes import make_llm_call, make_should_continue, make_tool_node
from backend.state import MessageState


def build_agent(
    chat: BaseChatModel,
    tools: list[BaseTool],
    max_llm_calls: int,
) -> CompiledStateGraph:
    """Build and compile the agent graph.

    Args:
        chat: The chat model to bind tools to and use for the LLM call node.
        tools: List of tools available to the agent.
        max_llm_calls: Maximum number of LLM invocations allowed per turn.

    Returns:
        A compiled LangGraph state graph ready for invocation.
    """
    chat_with_tools = chat.bind_tools(tools)

    builder = StateGraph(MessageState)
    builder.add_node("llm_call", make_llm_call(chat_with_tools))
    builder.add_node("tool_node", make_tool_node(tools))
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call", make_should_continue(max_llm_calls), ["tool_node", END]
    )
    builder.add_edge("tool_node", "llm_call")
    return builder.compile()
