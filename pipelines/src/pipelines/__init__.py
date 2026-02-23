"""Pipelines package — registry of available analysis pipelines.

Usage::

    from pipelines import get_pipeline, list_pipelines

    pipeline = get_pipeline("pet_tau")
    result = await pipeline.run(s3_key, analysis_id)
"""

from __future__ import annotations

from pipelines.base import AnalysisResult, BasePipeline

# ---------------------------------------------------------------------------
# Registry — maps pipeline_type strings to pipeline instances
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, BasePipeline] = {}


def register(pipeline: BasePipeline) -> None:
    """Register a pipeline instance. Called at import time by each pipeline module."""
    if pipeline.pipeline_type in _REGISTRY:
        raise ValueError(f"Duplicate pipeline_type: {pipeline.pipeline_type!r}")
    _REGISTRY[pipeline.pipeline_type] = pipeline


def get_pipeline(pipeline_type: str) -> BasePipeline:
    """Look up a pipeline by its type identifier.

    Raises ``KeyError`` if the pipeline is not registered.
    """
    try:
        return _REGISTRY[pipeline_type]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown pipeline_type {pipeline_type!r}. Available: {available}"
        ) from None


def list_pipelines() -> list[str]:
    """Return the identifiers of all registered pipelines."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Auto-register all built-in pipelines on import
# ---------------------------------------------------------------------------
from pipelines.pet_tau import PetTauPipeline  # noqa: E402

register(PetTauPipeline())

__all__ = [
    "AnalysisResult",
    "BasePipeline",
    "get_pipeline",
    "list_pipelines",
    "register",
]
