from collections.abc import Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def _create_huggingface(model_repo_id: str) -> BaseChatModel:
    """Instantiate a HuggingFace chat model.

    Args:
        model_repo_id: HuggingFace model repository identifier.

    Returns:
        A LangChain-compatible chat model backed by a HuggingFace endpoint.
    """
    llm = HuggingFaceEndpoint(
        model=model_repo_id,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)


def _create_anthropic(model_repo_id: str) -> BaseChatModel:
    """Instantiate an Anthropic chat model.

    Args:
        model_repo_id: Anthropic model identifier (e.g. ``"claude-sonnet-4-6"``).

    Returns:
        A LangChain-compatible chat model backed by the Anthropic API.
    """
    return ChatAnthropic(model_name=model_repo_id)  # type: ignore[call-arg]


def _create_google(model_repo_id: str) -> BaseChatModel:
    """Instantiate a Google Gemini chat model.

    Args:
        model_repo_id: Gemini model identifier (e.g. ``"gemini-2.0-flash"``).

    Returns:
        A LangChain-compatible chat model backed by the Google Generative AI API.
    """
    return ChatGoogleGenerativeAI(model=model_repo_id)


def _create_groq(model_repo_id: str) -> BaseChatModel:
    """Instantiate a Groq chat model.

    Args:
        model_repo_id: Groq model identifier (e.g. ``"llama-3.3-70b-versatile"``).

    Returns:
        A LangChain-compatible chat model backed by the Groq API.
    """
    return ChatGroq(model=model_repo_id)


_PROVIDERS: dict[str, Callable[[str], BaseChatModel]] = {
    "huggingface": _create_huggingface,
    "anthropic": _create_anthropic,
    "google": _create_google,
    "groq": _create_groq,
}


def create_chat_model(provider: str, model_repo_id: str) -> BaseChatModel:
    """Instantiate the chat model for the given provider.

    Args:
        provider: Backend provider name (e.g. ``"huggingface"``).
        model_repo_id: Model repository identifier passed to the provider.

    Returns:
        A LangChain-compatible chat model instance.

    Raises:
        ValueError: If the configured provider is not supported.
    """
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Available: {', '.join(_PROVIDERS)}")
    return _PROVIDERS[provider](model_repo_id)
