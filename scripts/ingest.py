"""Ingestion script for building a ChromaDB vectorstore from registry assets.

Reads papers (PDF), text files, and code files defined in the project registry,
splits them into chunks, embeds them, and persists the result to a local
ChromaDB instance.
"""

import json
import logging
import logging.config
import tempfile
from pathlib import Path

import yaml
from docling.document_converter import DocumentConverter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import load_config
from backend.embeddings import create_embeddings
from scripts.config import load_ingest_config
from pypdf import PdfReader, PdfWriter
from pyprojroot import here

_logging_cfg = here("logging.yaml")
Path(_logging_cfg).parent.joinpath("logs").mkdir(exist_ok=True)
with open(_logging_cfg) as _f:
    logging.config.dictConfig(yaml.safe_load(_f))

_rapidocr_logger = logging.getLogger("RapidOCR")
_rapidocr_logger.handlers.clear()
_rapidocr_logger.addHandler(logging.NullHandler())
_rapidocr_logger.propagate = False

logger = logging.getLogger(__name__)


def _convert_pdf_in_chunks(
    pdf_path: Path,
    page_chunk_size: int,
) -> str:
    """Convert a PDF by splitting it into page chunks for Docling.

    Each chunk is written to a temporary file, converted independently via
    Docling, and the resulting markdown sections are merged.

    Args:
        pdf_path: Path to the source PDF.
        page_chunk_size: Maximum number of pages per chunk.

    Returns:
        The merged markdown text from all chunks.
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    if total_pages <= page_chunk_size:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        md = result.document.export_to_markdown()
        if not md.strip():
            logger.warning(
                "    Empty text extracted from %s (all %d pages)", pdf_path.name, total_pages
            )
        return md

    markdown_parts: list[str] = []

    for start in range(0, total_pages, page_chunk_size):
        end = min(start + page_chunk_size, total_pages)
        logger.info("    Converting pages %d–%d / %d...", start + 1, end, total_pages)

        writer = PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            writer.write(tmp)
            tmp.flush()

            converter = DocumentConverter()
            result = converter.convert(tmp.name)

        md = result.document.export_to_markdown()
        if not md.strip():
            logger.warning(
                "    Empty text extracted from %s pages %d–%d", pdf_path.name, start + 1, end
            )
        markdown_parts.append(md)

    return "\n\n".join(markdown_parts)


def build_vectorstore(
    registry_path: str,
    data_dir: str,
    persist_directory: str,
    embedding_provider: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    pdf_page_chunk_size: int = 20,
) -> Chroma:
    """Build a ChromaDB vectorstore from all projects in the registry.

    Iterates over every project defined in the JSON registry, loads papers,
    text files, and code files, splits them into chunks, enriches each chunk
    with project and document metadata, and persists the embeddings to disk.
    Tables found in papers are extracted as CSVs and added to the registry.

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

        # PDFs (each entry is a dict with path, title, authors, type)
        for pdf_entry in project.get("pdfs", []):
            if not pdf_entry.get("path"):
                continue
            full_path = data_dir_ / pdf_entry["path"]
            pdf_type = pdf_entry.get("type", "paper")
            logger.info("  Loading %s %s...", pdf_type, full_path.name)

            markdown = _convert_pdf_in_chunks(full_path, pdf_page_chunk_size)

            doc = Document(
                page_content=markdown,
                metadata={"source": str(full_path)},
            )
            chunks = splitter.split_documents([doc])
            pdf_metadata = {
                **base_metadata,
                "filename": full_path.name,
                "type": pdf_type,
                "title": pdf_entry.get("title", ""),
                "authors": pdf_entry.get("authors", ""),
            }
            for chunk in chunks:
                chunk.metadata.update(pdf_metadata)
            all_chunks.extend(chunks)

        # Tables (each entry is a dict with path, origin, etc.)
        for table_entry in project.get("tables", []):
            table_path = table_entry["path"]
            full_path = data_dir_ / table_path
            if not full_path.exists():
                logger.warning("  Table file not found: %s", full_path)
                continue
            logger.info("  Loading table %s...", full_path.name)
            content = full_path.read_text(encoding="utf-8")
            doc = Document(
                page_content=content,
                metadata={"source": str(full_path)},
            )
            chunks = splitter.split_documents([doc])
            table_metadata = {
                **base_metadata,
                "filename": full_path.name,
                "type": "table",
                "origin": table_entry.get("origin", "unknown"),
            }
            for chunk in chunks:
                chunk.metadata.update(table_metadata)
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

    all_chunks = filter_complex_metadata(all_chunks)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    logger.info("Vectorstore persisted to %s", persist_directory)
    return vectorstore


if __name__ == "__main__":
    config = load_config()
    ingest_config = load_ingest_config()
    build_vectorstore(
        registry_path=str(config.registry),
        data_dir=str(config.data_dir),
        persist_directory=str(config.data_dir / "vectorstore"),
        embedding_provider=config.embeddings.provider,
        embedding_model=config.embeddings.model,
        chunk_size=ingest_config.chunk_size,
        chunk_overlap=ingest_config.chunk_overlap,
        pdf_page_chunk_size=ingest_config.pdf_page_chunk_size,
    )
