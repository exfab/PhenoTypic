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


SAM2_AVAILABLE = _check_sam2_deps()
MICROSAM_AVAILABLE = _check_microsam_deps()


def __getattr__(name: str):  # type: ignore[misc]
    if name == "Sam2Detector":
        from ._sam2_detector import Sam2Detector

        return Sam2Detector
    if name == "MicroSamDetector":
        from ._microsam_detector import MicroSamDetector

        return MicroSamDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Sam2Detector",
    "MicroSamDetector",
    "SAM2_AVAILABLE",
    "MICROSAM_AVAILABLE",
]
