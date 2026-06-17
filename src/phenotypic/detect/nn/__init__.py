"""Neural network (PyTorch) operations for PhenoTypic.

Provides GPU-accelerated detectors backed by foundation models.
Requires: ``pip install phenotypic[torch]``
"""

from __future__ import annotations

import importlib.util


def _check_sam2_deps() -> bool:
    return all(
        importlib.util.find_spec(p) is not None for p in ["torch", "sam2"]
    )


def _check_microsam_deps() -> bool:
    return all(
        importlib.util.find_spec(p) is not None
        for p in ["torch", "micro_sam"]
    )


def _check_foundation_deps() -> bool:
    # SAM3 + DINO collapse to transformers (the foundation extra). torch is a
    # transitive dep of foundation, but we gate on transformers — it is the
    # symbol the import-smoke + functional tests need.
    return importlib.util.find_spec("transformers") is not None


SAM2_AVAILABLE = _check_sam2_deps()
MICROSAM_AVAILABLE = _check_microsam_deps()
FOUNDATION_AVAILABLE = _check_foundation_deps()


def __getattr__(name: str):  # type: ignore[misc]
    if name == "Sam2Detector":
        from ._sam2_detector import Sam2Detector

        return Sam2Detector
    if name == "MicroSamDetector":
        from ._microsam_detector import MicroSamDetector

        return MicroSamDetector
    if name == "Sam3Detector":
        from ._sam3_detector import Sam3Detector

        return Sam3Detector
    if name == "DinoSam2Detector":
        from ._dinosam2_detector import DinoSam2Detector

        return DinoSam2Detector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Sam2Detector",
    "MicroSamDetector",
    "Sam3Detector",
    "DinoSam2Detector",
    "SAM2_AVAILABLE",
    "MICROSAM_AVAILABLE",
    "FOUNDATION_AVAILABLE",
]
