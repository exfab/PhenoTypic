# src/phenotypic/enhance/_focus_edge_two_k_phase.py
"""FocusEdgeTwoKPhase — two-scale-k phase-congruency hysteresis enhancer."""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from phenotypic.abc_._enhance_markers._focus_edge import FocusEdge
from phenotypic.sdk_.mixin import NormalizedOutputMixin
from phenotypic.sdk_.typing_ import TuneSpec

from ._two_k_phase_kernel import two_k_phase

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class FocusEdgeTwoKPhase(NormalizedOutputMixin, FocusEdge):
    """Branch-response enhancer via two-k phase-congruency hysteresis.

    Runs phase congruency at a strict k (clean seeds) and a loose k (full
    candidates), keeps loose candidates that touch a strict seed, and writes the
    loose-k magnitude gated by that mask into ``detect_mat``. Continuous output;
    the inoculum center hole is preserved (this is a pure branch enhancer — center
    filling belongs to the detector, not here).

    Assumes ``detect_mat`` is already illumination-flattened + contrast-stretched
    upstream (same contract as FocusEdgePhase).
    """

    n_orient: Annotated[int, TuneSpec(4, 8)] = Field(8, ge=1)
    min_wavelength: Annotated[float, TuneSpec(2.0, 10.0)] = Field(5.0, ge=2.0)
    k_strict: Annotated[float, TuneSpec(4.0, 8.0)] = Field(6.0, ge=0.0)
    k_loose: Annotated[float, TuneSpec(3.5, 6.0)] = Field(4.5, ge=0.0)
    seed_thresh: Literal["otsu", "triangle"] = "otsu"
    cand_thresh: Literal["otsu", "triangle"] = "triangle"

    def _operate(self, image: "Image") -> "Image":
        gated, _loose = two_k_phase(
            image.detect_mat[:],
            k_strict=self.k_strict,
            k_loose=self.k_loose,
            seed_thresh=self.seed_thresh,
            cand_thresh=self.cand_thresh,
            n_orient=self.n_orient,
            min_wavelength=self.min_wavelength,
        )
        image.detect_mat[:] = self._apply_norm(gated)
        return image
