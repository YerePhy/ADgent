"""Abstract base class for all analysis pipelines.

Every pipeline must subclass ``BasePipeline`` and implement ``run()``.
This enforces a uniform interface so the worker can dispatch to any pipeline
without knowing its internals (Strategy Pattern).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AnalysisResult:
    """Standardised output returned by every pipeline.

    Attributes:
        pipeline_type: Identifier matching the registry key (e.g. ``"pet_tau"``).
        metrics: Dict of extracted measurements. Keys are metric names, values
            can be floats or nested dicts for region-level data.
            Example: ``{"left_temporal_suvr": 1.42, "right_parietal_suvr": 1.38}``
        output_s3_keys: S3 keys of any derivative files the pipeline produced
            (processed images, reports, etc.).
        summary: Short human-readable summary suitable for an LLM prompt.
        completed_at: Timestamp when the analysis finished.
    """

    pipeline_type: str
    metrics: dict[str, float | dict] = field(default_factory=dict)
    output_s3_keys: list[str] = field(default_factory=list)
    summary: str = ""
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BasePipeline(abc.ABC):
    """Interface every analysis pipeline must implement."""

    @property
    @abc.abstractmethod
    def pipeline_type(self) -> str:
        """Unique identifier for this pipeline (used as registry key)."""
        ...

    @abc.abstractmethod
    async def run(self, s3_key: str, analysis_id: str) -> AnalysisResult:
        """Execute the analysis.

        Args:
            s3_key: Location of the input image in S3.
            analysis_id: Database ID of the analysis record.

        Returns:
            An ``AnalysisResult`` with extracted metrics and metadata.
        """
        ...

    @abc.abstractmethod
    async def validate_input(self, s3_key: str) -> bool:
        """Check whether the input file is valid for this pipeline.

        Returns ``True`` if the file can be processed, ``False`` otherwise.
        """
        ...
