from collections.abc import Callable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


def _create_huggingface(model: str) -> Embeddings:

    return HuggingFaceEmbeddings(model_name=model)


_PROVIDERS: dict[str, Callable[[str], Embeddings]] = {
    "huggingface": _create_huggingface,
}


def create_embeddings(provider: str, model: str) -> Embeddings:
    """Instantiate an embeddings model for the given provider.

    Args:
        provider: Backend provider name (e.g. ``"huggingface"``).
        model: Model identifier passed to the provider factory.

    Returns:
        A LangChain-compatible embeddings instance.

    Raises:
        ValueError: If the provider is not registered.
    """
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. Available: {', '.join(_PROVIDERS)}"
        )
    return _PROVIDERS[provider](model)
