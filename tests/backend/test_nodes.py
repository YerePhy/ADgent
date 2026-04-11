"""Unit tests for backend.nodes — LangGraph node factories."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END

from backend.nodes import make_llm_call, make_should_continue, make_tool_node
from backend.state import MessageState


def _state(messages: list, llm_calls: int = 0) -> MessageState:
    return {"messages": messages, "llm_calls": llm_calls}


def _ai_with_tools(tool_name: str = "my_tool", call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {"x": 1}, "id": call_id, "type": "tool_call"}],
    )


def _fake_tool(name: str, return_value: str = "ok") -> BaseTool:
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.invoke.return_value = return_value
    return tool


def test_should_continue_returns_end_when_limit_reached():
    fn = make_should_continue(max_llm_calls=3)
    state = _state([_ai_with_tools()], llm_calls=3)
    assert fn(state) == END


def test_should_continue_returns_tool_node_when_tool_calls_pending():
    fn = make_should_continue(max_llm_calls=10)
    state = _state([_ai_with_tools()], llm_calls=1)
    assert fn(state) == "tool_node"


def test_should_continue_returns_end_when_no_tool_calls():
    fn = make_should_continue(max_llm_calls=10)
    state = _state([AIMessage(content="Final answer.")], llm_calls=1)
    assert fn(state) == END


def test_should_continue_returns_end_for_non_ai_message():
    fn = make_should_continue(max_llm_calls=10)
    state = _state([HumanMessage(content="hello")], llm_calls=0)
    assert fn(state) == END


def test_should_continue_boundary_at_limit():
    fn = make_should_continue(max_llm_calls=1)
    assert fn(_state([_ai_with_tools()], llm_calls=0)) == "tool_node"
    assert fn(_state([_ai_with_tools()], llm_calls=1)) == END


def test_tool_node_executes_tool_and_returns_tool_message():
    tool = _fake_tool("my_tool", return_value="result_value")
    fn = make_tool_node([tool])
    state = _state([_ai_with_tools("my_tool", "call_1")])

    result = fn(state)

    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert msg.content == "result_value"
    assert msg.tool_call_id == "call_1"


def test_tool_node_unknown_tool_returns_error_message():
    fn = make_tool_node([])
    state = _state([_ai_with_tools("nonexistent", "call_x")])

    result = fn(state)

    msg = result["messages"][0]
    assert "Unknown tool" in msg.content
    assert msg.tool_call_id == "call_x"


def test_tool_node_tool_exception_returns_error_message():
    tool = _fake_tool("bad_tool")
    tool.invoke.side_effect = RuntimeError("boom")
    fn = make_tool_node([tool])
    state = _state([_ai_with_tools("bad_tool", "call_err")])

    result = fn(state)

    msg = result["messages"][0]
    assert "Tool error" in msg.content
    assert "boom" in msg.content


def test_tool_node_non_ai_message_returns_empty():
    fn = make_tool_node([_fake_tool("t")])
    state = _state([HumanMessage(content="hi")])
    assert fn(state) == {"messages": []}


def test_tool_node_multiple_calls_in_one_message():
    tool_a = _fake_tool("tool_a", "res_a")
    tool_b = _fake_tool("tool_b", "res_b")
    fn = make_tool_node([tool_a, tool_b])

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "tool_a", "args": {}, "id": "id_a", "type": "tool_call"},
            {"name": "tool_b", "args": {}, "id": "id_b", "type": "tool_call"},
        ],
    )
    result = fn(_state([ai_msg]))

    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "res_a"
    assert result["messages"][1].content == "res_b"


def test_llm_call_increments_counter():
    response = AIMessage(content="response")
    chat = MagicMock(spec=Runnable)
    chat.invoke.return_value = response

    fn = make_llm_call(chat)
    result = fn(_state([HumanMessage(content="hi")], llm_calls=2), config={})

    assert result["llm_calls"] == 3


def test_llm_call_appends_response_to_messages():
    response = AIMessage(content="the answer")
    chat = MagicMock(spec=Runnable)
    chat.invoke.return_value = response

    fn = make_llm_call(chat)
    result = fn(_state([HumanMessage(content="q")], llm_calls=0), config={})

    assert result["messages"] == [response]


def test_llm_call_passes_config_to_chat():
    chat = MagicMock(spec=Runnable)
    chat.invoke.return_value = AIMessage(content="ok")
    config = {"run_name": "test_run"}

    fn = make_llm_call(chat)
    fn(_state([HumanMessage(content="q")]), config=config)

    chat.invoke.assert_called_once()
    _, kwargs = chat.invoke.call_args
    assert kwargs["config"] == config
