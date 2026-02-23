"""PET Tau analysis pipeline — stub implementation.

Replace the ``run()`` body with real NiBabel / FreeSurfer / custom model logic.
"""

from __future__ import annotations

import logging

from pipelines.base import AnalysisResult, BasePipeline

logger = logging.getLogger(__name__)


class PetTauPipeline(BasePipeline):
    """Stub pipeline for PET Tau SUVR analysis."""

    @property
    def pipeline_type(self) -> str:
        return "pet_tau"

    async def validate_input(self, s3_key: str) -> bool:
        """Check the file looks like a NIfTI PET image."""
        lower = s3_key.lower()
        # Stub: just check extension. Real version would download & inspect header.
        return lower.endswith((".nii", ".nii.gz"))

    async def run(self, s3_key: str, analysis_id: str) -> AnalysisResult:
        """Execute PET Tau SUVR analysis.

        STUB: Returns dummy metrics. Replace with real processing:
        1. Download image from S3
        2. Load with NiBabel
        3. Segment regions (FreeSurfer / deep learning model)
        4. Compute SUVr per region against reference region
        5. Upload derivative files to S3
        """
        logger.info("STUB: Running PET Tau analysis for %s (%s)", analysis_id, s3_key)

        # Fake metrics — replace with real computation
        metrics: dict[str, float | dict] = {
            "left_temporal_suvr": 1.42,
            "right_temporal_suvr": 1.35,
            "left_parietal_suvr": 1.38,
            "right_parietal_suvr": 1.29,
            "left_frontal_suvr": 1.15,
            "right_frontal_suvr": 1.12,
            "cerebellar_reference_suvr": 1.00,
            "global_cortical_suvr": 1.28,
        }

        summary = (
            f"PET Tau analysis completed for {s3_key}. "
            f"Elevated uptake detected in left temporal (SUVr 1.42) and "
            f"left parietal (SUVr 1.38) regions, above normative threshold of 1.20."
        )

        return AnalysisResult(
            pipeline_type=self.pipeline_type,
            metrics=metrics,
            output_s3_keys=[],  # No derivative files in stub
            summary=summary,
        )
