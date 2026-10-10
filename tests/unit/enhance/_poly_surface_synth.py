"""Synthetic plates for the SubtractPolySurface tests.

Mirrors docs/superpowers/logic_validation_scripts/2026-10-08-subtract-poly-surface/
robust_and_subsample.py, whose measurements set every threshold the tests assert.
"""

from __future__ import annotations

import numpy as np

NOISE = 0.01


def grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return float64 (i, j, u, v) grids; u, v are normalized to [-1, 1] (design.md §4.1)."""
    i, j = np.mgrid[0:height, 0:width].astype(float)
    return i, j, 2 * j / (width - 1) - 1, 2 * i / (height - 1) - 1


def colony_domes(height: int, width: int, cover: float, rng: np.random.Generator,
                 radius: float = 16.0) -> np.ndarray:
    """Soft-edged colony domes (amplitude 0.15-0.45) until ``cover`` of the pixels are colony."""
    i, j, _, _ = grid(height, width)
    colonies = np.zeros((height, width))
    while (colonies > 0).mean() < cover:
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        rr = np.hypot(i - cy, j - cx) / radius
        colonies = np.maximum(colonies, rng.uniform(0.15, 0.45) * np.sqrt(np.clip(1 - rr**2, 0, None)))
    return colonies


def surface_plate(height: int = 300, width: int = 450, cover: float = 0.25,
                  seed: int = 2025) -> tuple[np.ndarray, np.ndarray]:
    """A plate whose background is exactly a tensor-order-3 polynomial. Returns (z, background)."""
    rng = np.random.default_rng(seed)
    _, _, u, v = grid(height, width)
    background = 0.30 + 0.08 * u - 0.05 * v + 0.06 * u * u + 0.04 * v * v - 0.03 * u * v + 0.02 * u**3 * v
    return background + colony_domes(height, width, cover, rng) + rng.normal(0, NOISE, (height, width)), background


def line_plate(height: int = 300, width: int = 450, cover: float = 0.25,
               seed: int = 2026) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A plate whose background is exactly degree 1 along every row (per-row offset and slope).

    Returns (z, background, colony_mask).
    """
    rng = np.random.default_rng(seed)
    _, _, u, v = grid(height, width)
    background = (0.30 + 0.08 * u - 0.05 * v + 0.04 * v * v
                  + rng.normal(0, 0.03, (height, 1)) + rng.normal(0, 0.02, (height, 1)) * u)
    colonies = colony_domes(height, width, cover, rng)
    return background + colonies + rng.normal(0, NOISE, (height, width)), background, colonies > 0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
