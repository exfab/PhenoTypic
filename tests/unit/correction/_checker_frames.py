"""Rig-shaped synthetic colour-checker frames shared by the correction tests.

Two vertical bands, each a transposed 6x2 half-card of real ColorChecker24
colours, with the lattice supplied as a prior -- the production path.
"""

from __future__ import annotations

import warnings

import numpy as np

from phenotypic import Image
from phenotypic.correction import CalibrateColorRpcc
from phenotypic.correction._color_correction._checker_roi import (
    CheckerLattice,
    ColumnLattice,
)

CHECKER = "ColorChecker24 - After November 2014"
PITCH, TILE, TOP = 60, 48, 80
BAND_H = 2 * TOP + 6 * PITCH
GAP = 200
#: Keeps every patch off the uint8 floor, so the clipped-pixel gate stays quiet
#: on a clean frame (black and cyan otherwise have a channel at 0).
FLOOR, GAIN = 0.04, 0.85


def _patch_srgb() -> tuple[list[str], dict[str, np.ndarray]]:
    import colour

    from phenotypic.correction._color_correction._color_checker_profile import (
        _load_reference_data,
    )

    ref_lab, ref_linear, _ = _load_reference_data(CHECKER, "D65")
    names = list(ref_lab)
    return names, {
        name: colour.cctf_encoding(np.clip(ref_linear[name], 0, 1), function="sRGB")
        for name in names
    }


NAMES, SRGB = _patch_srgb()


def _band_patch(band: int, row: int, col: int) -> str:
    """Band *band* holds chart rows ``2*band`` and ``2*band+1``, transposed."""
    return NAMES[(2 * band + col) * 6 + row]


def render_frame(
        *,
        band_w: int = 140,
        col_x: tuple[int, int] = (25, 85),
        dx: int = 0,
        dy: int = 0,
        gain: float = GAIN,
        overrides: dict[tuple[int, int, int], np.ndarray] | None = None,
        seed: int = 0,
) -> np.ndarray:
    """A uint8 sRGB frame with two card bands at the left and right edges.

    ``overrides`` maps ``(band, row, col)`` to a replacement sRGB colour.
    """
    width = 2 * band_w + GAP
    rng = np.random.default_rng(seed)
    frame = np.full((BAND_H, width, 3), 0.55)
    for band, left in ((0, 0), (1, width - band_w)):
        region = np.full((BAND_H, band_w, 3), 0.12)
        for row in range(6):
            for col in range(2):
                colour = (overrides or {}).get(
                        (band, row, col), SRGB[_band_patch(band, row, col)]
                )
                y0 = TOP + dy + row * PITCH
                x0 = col_x[col] + dx
                ys, xs = max(0, y0), max(0, x0)
                region[ys:y0 + TILE, xs:x0 + TILE] = FLOOR + colour * gain
        frame[:, left:left + band_w] = region
    frame = np.clip(frame + rng.normal(0, 0.004, frame.shape), 0, 1)
    return (frame * 255).round().astype(np.uint8)


def band_rois(band_w: int = 140) -> list[list[int]]:
    width = 2 * band_w + GAP
    return [[0, 0, BAND_H, band_w], [0, width - band_w, BAND_H, width]]


def band_prior(
        col_x: tuple[int, int] = (25, 85), nrows: int = 6
) -> CheckerLattice:
    return CheckerLattice(
            columns=[
                ColumnLattice(
                        x0=x, x1=x + TILE, start=float(TOP), pitch=float(PITCH),
                        duty=TILE / PITCH,
                )
                for x in col_x
            ],
            nrows=nrows,
    )


def frozen_op(**kwargs) -> CalibrateColorRpcc:
    prior = band_prior()
    kwargs.setdefault("rois", band_rois())
    kwargs.setdefault("lattice_prior", [prior, prior])
    kwargs.setdefault("refine_method", "frozen")
    return CalibrateColorRpcc(**kwargs)


def quietly(operation: CalibrateColorRpcc, image: Image) -> Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return operation.apply(image)
