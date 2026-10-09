"""Deterministic inputs and case list for the SubtractPolySurface Gwyddion oracle.

Writes inputs.npz (float64 arrays A, B, C) and cases.json into the directory given as argv[1].
Imports numpy only; never phenotypic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SHAPES = {"A": (23, 31), "B": (32, 32), "C": (17, 40)}


def make_input(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i, j = np.mgrid[0:height, 0:width].astype(float)
    u, v = 2 * j / (width - 1) - 1, 2 * i / (height - 1) - 1
    smooth = 0.45 + 0.12 * u - 0.08 * v + 0.05 * u * v + 0.04 * u**3 - 0.03 * v**2 * u
    rows = rng.normal(0.0, 0.02, (height, 1)) + rng.normal(0.0, 0.01, (height, 1)) * u
    return smooth + rows + rng.normal(0.0, 0.01, (height, width))


def make_cases() -> list[dict]:
    cases = []
    for x in SHAPES:
        cases.append(dict(name=f"offset__{x}", input=x, method="offset",
                          order=None, independent=None, line_order=None, line_axis=None))
        cases.append(dict(name=f"plane__{x}", input=x, method="plane",
                          order=None, independent=None, line_order=None, line_axis=None))
    for x in ("A", "C"):
        for order in (2, 3, 5):
            for independent in (True, False):
                cases.append(dict(name=f"poly{order}_{'ind' if independent else 'tot'}__{x}", input=x,
                                  method="polynomial", order=order, independent=independent,
                                  line_order=None, line_axis=None))
        for degree in (0, 1, 3):
            for axis in ("row", "column"):
                cases.append(dict(name=f"line{degree}_{axis}__{x}", input=x, method="line",
                                  order=None, independent=None, line_order=degree, line_axis=axis))
    return cases


def write_oracle_inputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: make_input(h, w, seed=101 + k) for k, (name, (h, w)) in enumerate(SHAPES.items())}
    np.savez(out_dir / "inputs.npz", **arrays)
    (out_dir / "cases.json").write_text(json.dumps(make_cases(), indent=1))


if __name__ == "__main__":
    write_oracle_inputs(Path(sys.argv[1]))
