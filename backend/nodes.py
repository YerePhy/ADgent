from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from backend.state import MessageState


def make_llm_call(chat: BaseChatModel) -> Callable[[MessageState], dict]:
    """Return an LLM call node bound to the given chat model.

    Args:
        chat: The chat model to invoke.

    Returns:
        A node function that invokes the chat model with the current state messages.
    """
    def llm_call(state: MessageState) -> dict:
        response = chat.invoke(state["messages"])
        return {
            "messages": [response],
            "llm_calls": state["llm_calls"] + 1,
        }

    return llm_call
