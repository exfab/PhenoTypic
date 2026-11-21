"""A ``MeasureDinoEmbedding`` that fails one image on disk (#282 P4).

``prepare_regions`` raises :class:`ValueError` when the file
``$PHENOTYPIC_TEST_FAULT_DIR/<image name>`` exists. Stage 4's loader calls
``prepare_regions`` inside spawned DataLoader workers, which import this class
afresh and inherit the environment but never see a ``monkeypatch``, so a
per-image fault is planted as a file (``tests._tiny_run.plant_stage4_fault``).
The inline path prepares through ``_prepare_inline`` and never faults.
"""

from __future__ import annotations

import os
from pathlib import Path

from phenotypic.measure import MeasureDinoEmbedding

#: The directory whose files name the images to fail.
FAULT_DIR_ENV = "PHENOTYPIC_TEST_FAULT_DIR"


class FaultyDinoEmbedding(MeasureDinoEmbedding):
    """:class:`MeasureDinoEmbedding` whose ``prepare_regions`` fails a planted image."""

    def prepare_regions(self, image, **kwargs):  # type: ignore[override]
        fault_dir = os.environ.get(FAULT_DIR_ENV)
        if fault_dir and (Path(fault_dir) / str(image.name)).exists():
            raise ValueError(f"planted Stage-4 fault for {image.name}")
        return super().prepare_regions(image, **kwargs)
