"""Harness for the SnP plate rig: frame loading and half-card band extraction.

The rig holds the ColorChecker as two 2x6 half-cards clamped at the left and
right edges of every plate frame.  Positions are near-identical between
frames, so detection is a constrained refinement of a stored prior rather
than a global search.  This module supplies the frames and the generous
search bands; detectors live in ``chipdetect.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import tifffile
from skimage import color

DATA = ("/Users/alex/Library/CloudStorage/ProtonDrive-alxxander.nguyen"
        "@protonmail.com-folder/Research/Wheeldon/ProjectData/RhodotorulaScreening")

FRAME_PATHS = {
    "d000220_300_021": f"{DATA}/d000220_300_021_2025-09-27_15-32-18.tiff",
    "d000220_300_022": f"{DATA}/d000220_300_022_2025-09-30_15-32-50.tiff",
    "d000220_300_038": f"{DATA}/d000220_300_038_2025-09-30_15-46-23.tiff",
    "d000320_300_003": "renders/d000320_300_003_2026-02-02_08-31-48.tif",
}
SESSION = {"d000220_300_021": "2025-09-27", "d000220_300_022": "2025-09-30",
           "d000220_300_038": "2025-09-30", "d000320_300_003": "2026-02-02"}

# Generous search bands: wide enough that the cards cannot leave them under
# any plausible rig displacement, tight enough to exclude the plate itself.
BAND_Y = (950, 3100)
BAND_W = 340


@dataclass(frozen=True)
class Band:
    """A half-card search band cut out of a full frame."""
    frame: str
    side: str            # "left" | "right"
    rgb: np.ndarray      # (h, w, 3) float in [0, 1]
    y0: int              # band origin in full-frame coordinates
    x0: int

    @property
    def lab(self) -> np.ndarray:
        return color.rgb2lab(self.rgb)

    def to_frame(self, yx: np.ndarray) -> np.ndarray:
        """Map band (row, col) coordinates to full-frame coordinates."""
        return np.asarray(yx, float) + np.array([self.y0, self.x0], float)


def load_frame(name: str) -> np.ndarray:
    """Load a plate frame as float RGB in [0, 1] (linear 16-bit source)."""
    path = FRAME_PATHS[name]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return tifffile.imread(path).astype(np.float32) / 65535.0


def bands(img: np.ndarray, frame: str = "", band_y=BAND_Y, band_w=BAND_W):
    """Cut the left and right half-card search bands from a frame."""
    y0, y1 = band_y
    h, w = img.shape[:2]
    return {
        "left": Band(frame, "left", img[y0:y1, 0:band_w], y0, 0),
        "right": Band(frame, "right", img[y0:y1, w - band_w:w], y0, w - band_w),
    }


def srgb8(x: np.ndarray) -> np.ndarray:
    """Display encoding for linear float RGB."""
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
