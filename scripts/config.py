from dataclasses import dataclass
from pathlib import Path

import yaml
from pyprojroot import here


@dataclass
class IngestionConfig:
    """Document ingestion configuration.

    Attributes:
        pdf_page_chunk_size: Number of pages per chunk when splitting large
            PDFs for Docling conversion.  Recommended range: 10–30.
        chunk_size: Maximum number of characters per text chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
    """

    pdf_page_chunk_size: int
    chunk_size: int
    chunk_overlap: int


def load_ingest_config() -> IngestionConfig:
    """Load ingestion configuration from ``config.yaml`` at the project root.

    Returns:
        A populated ``IngestionConfig`` instance.
    """
    path: Path = here("config.yaml")
    with open(path) as f:
        raw: dict = yaml.safe_load(f)
    return IngestionConfig(**raw["ingestion"])
