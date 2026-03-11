"""Graph node functions."""

from langchain_core.language_models.chat_models import BaseChatModel

from backend.state import MessageState


def make_llm_call(chat: BaseChatModel):
    """Return an llm_call node bound to the given chat model."""
    def llm_call(state: MessageState) -> dict:
        response = chat.invoke(state["messages"])
        return {
            "messages": [response],
            "llm_calls": state["llm_calls"] + 1,
        }
    return llm_call
