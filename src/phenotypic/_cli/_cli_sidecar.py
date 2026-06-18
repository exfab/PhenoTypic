"""Per-image objmap sidecar for the staged GPU engine (Spec 1 §5, D13).

Stage 2 writes the GPU result here (HDF opened read-only); Stage 3 merges it
into the final HDF and deletes it. Writes are atomic (temp + os.replace).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from phenotypic.sdk_ import results_dir

_OBJMAP_LAYER = "objmap"


def sidecar_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """``<output>/results/<dataset>/objmap/<stem>.npy``."""
    return results_dir(output_dir) / dataset / _OBJMAP_LAYER / f"{image_stem}.npy"


def write_sidecar(
    output_dir: Path, dataset: str, image_stem: str, array: np.ndarray
) -> Path:
    """Atomically write *array* to the sidecar (temp + os.replace)."""
    final = sidecar_path(output_dir, dataset, image_stem)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".{image_stem}.npy.tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, final)  # atomic on POSIX same-filesystem
    return final


def load_sidecar(output_dir: Path, dataset: str, image_stem: str) -> np.ndarray:
    return np.load(sidecar_path(output_dir, dataset, image_stem))


def sidecar_exists(output_dir: Path, dataset: str, image_stem: str) -> bool:
    return sidecar_path(output_dir, dataset, image_stem).is_file()


def delete_sidecar(output_dir: Path, dataset: str, image_stem: str) -> None:
    sidecar_path(output_dir, dataset, image_stem).unlink(missing_ok=True)
