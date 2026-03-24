import logging
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END

from backend.state import MessageState

logger = logging.getLogger(__name__)


def make_llm_call(chat: BaseChatModel) -> Callable[[MessageState], dict]:
    """Return an LLM call node bound to the given chat model.

    Args:
        chat: The chat model to invoke.

    Returns:
        A node function that invokes the chat model with the current state messages.
    """
    def llm_call(state: MessageState) -> dict:
        last = state["messages"][-1]
        logger.debug("LLM input: %s", last.content)
        response = chat.invoke(state["messages"])
        logger.debug("LLM output: %s", response.content)
        return {
            "messages": [response],
            "llm_calls": state["llm_calls"] + 1,
        }

    return llm_call


def make_should_continue(max_llm_calls: int) -> Callable[[MessageState], str]:
    """Return a routing function that enforces a maximum number of LLM calls.

    Args:
        max_llm_calls: Maximum number of LLM invocations allowed per turn.

    Returns:
        A routing function that returns ``"tool_node"`` if tool calls are
        pending and the call limit has not been reached, or ``END`` otherwise.
    """
    def should_continue(state: MessageState) -> str:
        if state["llm_calls"] >= max_llm_calls:
            return END
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END

    return should_continue


def make_tool_node(tools: list[BaseTool]) -> Callable[[MessageState], dict]:
    """Return a tool executor node bound to the given tools.

    Builds a name-to-tool lookup map at construction time. On each invocation,
    reads the tool calls from the last AIMessage, executes each matching tool,
    and returns the results as ToolMessages appended to state.

    Args:
        tools: List of tools available to the agent.

    Returns:
        A node function that executes requested tool calls and returns
        the results as ToolMessages.
    """
    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    def tool_node(state: MessageState) -> dict:
        last_message = state["messages"][-1]
        tool_messages = []
        for tool_call in last_message.tool_calls:
            logger.info("Tool call: %s(%s)", tool_call["name"], tool_call["args"])
            try:
                tool = tool_map.get(tool_call["name"])
                if tool is None:
                    raise ValueError(f"Unknown tool: {tool_call['name']}")
                content = str(tool.invoke(tool_call["args"]))
            except Exception as e:
                logger.error("Tool error: %s — %s", tool_call["name"], e)
                content = f"Tool error: {e}"
            tool_messages.append(
                ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": tool_messages}

    return tool_node
