import os
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def _create_huggingface() -> BaseChatModel:
    """Instantiate a HuggingFace chat model from environment configuration.

    Returns:
        A LangChain-compatible chat model backed by a HuggingFace endpoint.
    """
    llm = HuggingFaceEndpoint(
        model=os.getenv("HF_MODEL_REPO_ID", ""),
        task="text-generation",
        temperature=0.1,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)


_PROVIDERS: dict[str, Callable[[], BaseChatModel]] = {
    "huggingface": _create_huggingface,
}


def create_chat_model() -> BaseChatModel:
    """Instantiate the chat model for the configured provider.

    Reads the ``LLM_PROVIDER`` environment variable to select the backend.

    Returns:
        A LangChain-compatible chat model instance.

    Raises:
        ValueError: If the configured provider is not supported.
    """
    provider = os.getenv("LLM_PROVIDER", "huggingface")
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available: {', '.join(_PROVIDERS)}"
        )
    return _PROVIDERS[provider]()
