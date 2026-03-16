"""Ingestion script for building a ChromaDB vectorstore from registry assets.

Reads papers (PDF), text files, and code files defined in the project registry,
splits them into chunks, embeds them, and persists the result to a local
ChromaDB instance.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _create_huggingface_embeddings(model: str) -> Embeddings:
    """Create a HuggingFace sentence-transformers embedding instance.

    Args:
        model: HuggingFace model identifier
            (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``).

    Returns:
        A LangChain-compatible embeddings instance.
    """
    return HuggingFaceEmbeddings(model_name=model)


_EMBEDDING_PROVIDERS: dict[str, Callable[[str], Embeddings]] = {
    "huggingface": _create_huggingface_embeddings,
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
    if provider not in _EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            f"Available: {', '.join(_EMBEDDING_PROVIDERS)}"
        )
    return _EMBEDDING_PROVIDERS[provider](model)


def build_vectorstore(
    registry_path: str,
    data_dir: str,
    persist_directory: str,
    embedding_provider: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Chroma:
    """Build a ChromaDB vectorstore from all projects in the registry.

    Iterates over every project defined in the JSON registry, loads papers,
    text files, and code files, splits them into chunks, enriches each chunk
    with project and document metadata, and persists the embeddings to disk.

    Args:
        registry_path: Path to the JSON registry file.
        data_dir: Root directory containing the data assets referenced
            by the registry.
        persist_directory: Directory where the ChromaDB vectorstore will
            be persisted.
        embedding_provider: Embedding backend name (e.g. ``"huggingface"``).
        embedding_model: Model identifier for the embedding provider.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive
            chunks.

    Returns:
        The populated ChromaDB vectorstore instance.
    """
    persist_directory_ = Path(persist_directory)
    data_dir_ = Path(data_dir)

    if not persist_directory_.exists():
        persist_directory_.mkdir(parents=True)

    with open(registry_path) as f:
        registry = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[Document] = []

    for project_name, project in registry.items():
        base_metadata = {"project": project_name}
        logger.info("Processing project '%s'...", project_name)

        # Papers (each entry is a dict with path, title, authors)
        for paper in project.get("papers", []):
            if not paper.get("path"):
                continue
            full_path = data_dir_ / paper["path"]
            logger.info("  Loading paper %s...", full_path.name)
            pages = PyPDFLoader(str(full_path)).load()
            chunks = splitter.split_documents(pages)
            paper_metadata = {
                **base_metadata,
                "filename": full_path.name,
                "type": "paper",
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
            }
            for chunk in chunks:
                chunk.metadata.update(paper_metadata)
            all_chunks.extend(chunks)

        # Text and code files
        for file_type in ("text", "code"):
            for path in project.get(file_type, []):
                full_path = data_dir_ / path
                logger.info("  Loading %s %s...", file_type, full_path.name)
                content = full_path.read_text(encoding="utf-8")
                doc = Document(
                    page_content=content,
                    metadata={"source": str(full_path)},
                )
                chunks = splitter.split_documents([doc])
                file_metadata = {
                    **base_metadata,
                    "filename": full_path.name,
                    "type": file_type,
                }
                for chunk in chunks:
                    chunk.metadata.update(file_metadata)
                all_chunks.extend(chunks)

        logger.info("  %d total chunks so far", len(all_chunks))

    logger.info("Total chunks across all projects: %d", len(all_chunks))

    embeddings = create_embeddings(embedding_provider, embedding_model)
    logger.info("Creating vectorstore with %s embeddings...", embedding_provider)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    logger.info("Vectorstore persisted to %s", persist_directory)
    return vectorstore


if __name__ == "__main__":
    from backend.config import load_config

    config = load_config()
    build_vectorstore(
        registry_path=str(config.registry),
        data_dir=str(config.data_dir),
        persist_directory=str(config.data_dir / "vectorstore"),
        embedding_provider=config.embeddings.provider,
        embedding_model=config.embeddings.model,
        chunk_size=config.embeddings.chunk_size,
        chunk_overlap=config.embeddings.chunk_overlap,
    )
