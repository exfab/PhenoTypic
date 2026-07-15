# src/phenotypic/enhance/_two_k_phase_kernel.py
"""Two-scale-k phase-congruency hysteresis kernel.

Runs phase congruency at a strict and a loose noise-threshold k, keeps the loose
candidates that touch a strict seed (morphological reconstruction), and returns the
loose-k magnitude gated by that hysteresis mask. Shared by FocusEdgeTwoKPhase (the
enhancer) and TwoKFilamentousDetector (which also reuses the returned loose result
for its Dijkstra cost surface, so phase congruency runs only twice total).
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.morphology import reconstruction

from ._focus_edge_phase import FocusEdgePhase, _PhaseCong3Result

_THRESHOLDS = {"otsu": threshold_otsu, "triangle": threshold_triangle}


def two_k_phase(
    detect_mat: np.ndarray,
    *,
    k_strict: float,
    k_loose: float,
    seed_thresh: Literal["otsu", "triangle"],
    cand_thresh: Literal["otsu", "triangle"],
    n_orient: int,
    min_wavelength: float,
) -> Tuple[np.ndarray, _PhaseCong3Result]:
    """Two-k phase-congruency hysteresis.

    Args:
        detect_mat: Prepared (flattened + contrast-stretched) 2D detection matrix.
        k_strict: Strict noise-threshold k -> clean, fragmented seeds.
        k_loose: Loose noise-threshold k -> full branches + agar candidates.
        seed_thresh: Threshold rule for seeds (strict map). "otsu" verified best.
        cand_thresh: Threshold rule for candidates (loose map). "triangle" verified best.
        n_orient: Phase-congruency angular resolution.
        min_wavelength: Smallest log-Gabor wavelength (skips agar micro-texture).

    Returns:
        (gated_response, loose_result):
          - gated_response: loose.pc_sum where the hysteresis mask confirms a real
            branch, 0 elsewhere. Continuous magnitude; inoculum center hole preserved.
          - loose_result: the loose-k _PhaseCong3Result (M/m/orientation/pc_sum).
    """
    strict = FocusEdgePhase(
        n_orient=n_orient, k=k_strict, min_wavelength=min_wavelength,
    )._phasecong3(detect_mat)
    loose = FocusEdgePhase(
        n_orient=n_orient, k=k_loose, min_wavelength=min_wavelength,
    )._phasecong3(detect_mat)

    seed = strict.pc_sum > _THRESHOLDS[seed_thresh](strict.pc_sum)
    cand = loose.pc_sum > _THRESHOLDS[cand_thresh](loose.pc_sum)
    mask = reconstruction(
        (seed & cand).astype(np.uint8), cand.astype(np.uint8), method="dilation",
    ).astype(bool)

    gated = (loose.pc_sum * mask).astype(np.float32)
    return gated, loose
