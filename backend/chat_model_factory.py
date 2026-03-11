"""Factory for creating chat model instances based on provider configuration."""

import os
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


def _create_huggingface() -> BaseChatModel:
    model_repo_id = os.getenv("HF_MODEL_REPO_ID", "")
    llm = HuggingFaceEndpoint(
        model=model_repo_id,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)


_PROVIDERS: dict[str, Callable[[], BaseChatModel]] = {
    "huggingface": _create_huggingface,
}


def create_chat_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "huggingface")
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available: {', '.join(_PROVIDERS)}"
        )
    return _PROVIDERS[provider]()
