"""Frozen input artifacts for the pydantic-migration golden harness.

This module owns the small set of immutable input objects every
operation/analyzer scenario runs against. The artifacts are stored as
**arrays** (not whole-object serialization) under ``_inputs/`` so they:

* stay inspectable and diff-able under version control,
* survive the soon-to-break ``conftest.py`` session fixtures, and
* reconstruct deterministically through the public ``Image`` API.

Five frozen inputs are defined (see :data:`FROZEN_INPUT_NAMES`):

* ``raw_plate`` -- a plain :class:`~phenotypic.Image` of a synthetic
  yeast plate with no detected objects (input for enhancers, detectors,
  correctors).
* ``detected_plate`` -- the same plate with ``OtsuDetector`` already
  applied, carrying an object map (input for refiners and non-grid
  measurers).
* ``raw_grid`` -- the plate wrapped as a :class:`~phenotypic.GridImage`
  with no objects (input for grid finders).
* ``detected_grid`` -- a detected :class:`~phenotypic.GridImage` (input
  for grid measurers).
* ``reference_measurements`` -- a measurement :class:`~pandas.DataFrame`
  spanning two synthetic plates and three timepoints, built from real
  grid measurements (input for analyzers).

``capture_frozen_inputs()`` writes the artifacts; ``load_frozen_input()``
reconstructs them. The capture step is invoked by
``scripts/capture_migration_goldens.py`` and is read-only with respect to
``src/phenotypic/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phenotypic import GridImage, Image

# Directory holding the frozen input artifacts (next to this module).
INPUTS_DIR = Path(__file__).parent / "_inputs"

# Canonical ordered list of frozen input names.
FROZEN_INPUT_NAMES: tuple[str, ...] = (
    "raw_plate",
    "detected_plate",
    "raw_grid",
    "detected_grid",
    "reference_measurements",
)

# Plate geometry used for every GridImage reconstruction.
_GRID_NROWS = 8
_GRID_NCOLS = 12

# Synthetic plate / timepoint axes used to widen the reference frame so
# the time-dependent analyzers (LogGrowthModel, EdgeCorrector,
# ReplicateAgreement) have something non-degenerate to fit.
_REF_PLATES: tuple[str, ...] = ("PlateA", "PlateB")
_REF_TIMES: tuple[int, ...] = (12, 24, 48)


def _source_plate() -> "Image":
    """Load the canonical synthetic yeast plate from package data.

    Returns:
        The :class:`~phenotypic.Image` produced by
        :func:`phenotypic.data.load_synth_yeast_plate`, which already
        carries a detected object map loaded from a stored PNG.
    """
    from phenotypic.data import load_synth_yeast_plate

    return load_synth_yeast_plate()


def _new_image(rgb: np.ndarray, name: str) -> "Image":
    """Construct a fresh :class:`~phenotypic.Image` from an RGB array.

    Args:
        rgb: ``(H, W, 3)`` uint8 RGB array.
        name: Human-readable image name.

    Returns:
        A new ``Image`` with 8-bit depth, D65 illuminant and sRGB gamma.
    """
    from phenotypic import Image
    from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS

    return Image(
        arr=rgb,
        name=name,
        bit_depth=8,
        illuminant="D65",
        gamma=GAMMA_ENCODINGS.SRGB,
    )


def _new_grid_image(rgb: np.ndarray, name: str) -> "GridImage":
    """Construct a fresh :class:`~phenotypic.GridImage` from an RGB array.

    Args:
        rgb: ``(H, W, 3)`` uint8 RGB array.
        name: Human-readable image name.

    Returns:
        A new ``GridImage`` with the canonical 8x12 plate geometry.
    """
    from phenotypic import GridImage
    from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS

    return GridImage(
        arr=rgb,
        name=name,
        nrows=_GRID_NROWS,
        ncols=_GRID_NCOLS,
        bit_depth=8,
        illuminant="D65",
        gamma=GAMMA_ENCODINGS.SRGB,
    )


def _build_reference_measurements() -> pd.DataFrame:
    """Build the analyzer reference measurement frame.

    The frame is assembled from *real* grid measurements
    (:class:`~phenotypic.measure.MeasureSize` with metadata) so every
    grid/metadata column an analyzer might require is genuinely present
    and correctly typed. Plate name and timepoint are varied
    synthetically across :data:`_REF_PLATES` and :data:`_REF_TIMES`, and
    ``Size_Area`` is scaled by a deterministic log-of-time factor so the
    growth-curve fit has signal.

    Returns:
        A measurement :class:`~pandas.DataFrame` with one row per
        (plate, timepoint, colony).
    """
    from phenotypic.measure import MeasureSize

    src = _source_plate()
    rgb = np.ascontiguousarray(src.rgb[:])
    objmap = np.ascontiguousarray(src.objmap[:])

    frames: list[pd.DataFrame] = []
    for plate in _REF_PLATES:
        for time in _REF_TIMES:
            grid = _new_grid_image(rgb, plate)
            grid.objmap[:] = objmap
            frame = MeasureSize().measure(grid, include_meta=True)
            frame = frame.copy()
            frame["Metadata_ImageName"] = plate
            frame["Metadata_Time"] = time
            # Deterministic, monotone-in-time growth signal.
            frame["Size_Area"] = frame["Size_Area"] * (
                0.5 + 0.2 * float(np.log1p(time))
            )
            # A delimited tag column so post/ExpandMetadata has a
            # genuine multi-field string to split.
            frame["Metadata_Tag"] = f"{plate}-{time}-rep1"
            frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def capture_frozen_inputs() -> dict[str, Path]:
    """Capture every frozen input artifact to :data:`INPUTS_DIR`.

    Dense arrays are written as ``.npz``; the sparse object map is
    written via :func:`scipy.sparse.save_npz`; the reference frame is
    written as parquet. Called once by
    ``scripts/capture_migration_goldens.py`` against the unmigrated
    library.

    Returns:
        A mapping from frozen-input name to the primary artifact path.
    """
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    src = _source_plate()
    rgb = np.ascontiguousarray(src.rgb[:])

    # raw_plate / raw_grid share the same RGB array; no object map.
    raw_path = INPUTS_DIR / "raw_plate.npz"
    np.savez_compressed(raw_path, rgb=rgb)
    written["raw_plate"] = raw_path

    raw_grid_path = INPUTS_DIR / "raw_grid.npz"
    np.savez_compressed(raw_grid_path, rgb=rgb)
    written["raw_grid"] = raw_grid_path

    # detected_plate / detected_grid: capture the OtsuDetector object map
    # produced on the CURRENT code. The dense map is stored in the .npz
    # for direct reconstruction; the sparse form is stored alongside as a
    # scipy artifact so the sparse round-trip is itself frozen.
    from phenotypic.detect import OtsuDetector

    detected = _new_image(rgb, "detected_plate")
    OtsuDetector().apply(detected, inplace=True)
    detected_objmap = np.ascontiguousarray(detected.objmap[:])

    detected_path = INPUTS_DIR / "detected_plate.npz"
    np.savez_compressed(detected_path, rgb=rgb, objmap=detected_objmap)
    written["detected_plate"] = detected_path

    sp.save_npz(
        INPUTS_DIR / "detected_plate_objmap_sparse.npz",
        sp.csc_matrix(detected_objmap),
    )

    detected_grid_path = INPUTS_DIR / "detected_grid.npz"
    np.savez_compressed(
        detected_grid_path, rgb=rgb, objmap=detected_objmap
    )
    written["detected_grid"] = detected_grid_path

    # reference measurements frame for analyzers.
    ref = _build_reference_measurements()
    ref_path = INPUTS_DIR / "reference_measurements.parquet"
    ref.to_parquet(ref_path, index=False)
    written["reference_measurements"] = ref_path

    return written


def _load_objmap(npz_path: Path) -> np.ndarray:
    """Return the object map from a frozen ``.npz`` artifact.

    Prefers the dense ``objmap`` array; falls back to the sibling sparse
    artifact when only that is present.

    Args:
        npz_path: Path to the dense ``.npz`` artifact.

    Returns:
        The ``(H, W)`` integer object map.
    """
    with np.load(npz_path) as data:
        if "objmap" in data:
            return np.ascontiguousarray(data["objmap"])
    sparse_path = npz_path.with_name(
        npz_path.stem + "_objmap_sparse.npz"
    )
    return np.ascontiguousarray(sp.load_npz(sparse_path).toarray())


def load_frozen_input(name: str):
    """Reconstruct a frozen input by name.

    Each call returns a *fresh* object so scenarios never mutate a shared
    instance.

    Args:
        name: One of :data:`FROZEN_INPUT_NAMES`.

    Returns:
        An :class:`~phenotypic.Image`, :class:`~phenotypic.GridImage` or
        :class:`~pandas.DataFrame` depending on ``name``.

    Raises:
        ValueError: If ``name`` is not a recognized frozen input.
        FileNotFoundError: If the artifact has not been captured yet.
    """
    if name not in FROZEN_INPUT_NAMES:
        raise ValueError(
            f"Unknown frozen input {name!r}; "
            f"expected one of {FROZEN_INPUT_NAMES}."
        )

    if name == "reference_measurements":
        path = INPUTS_DIR / "reference_measurements.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen input {name!r} not captured: {path} missing. "
                "Run scripts/capture_migration_goldens.py first."
            )
        return pd.read_parquet(path)

    npz_path = INPUTS_DIR / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Frozen input {name!r} not captured: {npz_path} missing. "
            "Run scripts/capture_migration_goldens.py first."
        )

    with np.load(npz_path) as data:
        rgb = np.ascontiguousarray(data["rgb"])

    if name == "raw_plate":
        return _new_image(rgb, "raw_plate")
    if name == "raw_grid":
        return _new_grid_image(rgb, "raw_grid")
    if name == "detected_plate":
        image = _new_image(rgb, "detected_plate")
        image.objmap[:] = _load_objmap(npz_path)
        return image
    if name == "detected_grid":
        grid = _new_grid_image(rgb, "detected_grid")
        grid.objmap[:] = _load_objmap(npz_path)
        return grid

    raise ValueError(f"Unhandled frozen input {name!r}.")  # noqa: E501
