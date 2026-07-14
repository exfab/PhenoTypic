"""Generate the A10 FilFinder 1.8 external-oracle fixture.

Run with::

    PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH=docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/filfinder/upstream \
      uv run --with fil-finder==1.8 python \
      docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/filfinder/generate_fixture.py
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import pathlib
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import astropy.units as u
import numpy as np
from fil_finder import FilFinder2D
from scipy import ndimage


REFERENCE_DIR = pathlib.Path(__file__).resolve().parent
REPOSITORY_ROOT = REFERENCE_DIR.parents[5]
FIXTURE_PATH = REPOSITORY_ROOT / "tests/fixtures/reconnect/filfinder/oracle.json"
THRESHOLD = 0.5
BEAMWIDTH_PX = 1.0
BRANCH_THRESHOLD_PX: float | None = None
RELATIVE_INTENSITY_THRESHOLD = 0.2
MAX_PRUNE_ITERATIONS = 10
RNG_SEED = 0
EXPECTED_SOURCE = REFERENCE_DIR / "upstream/fil_finder/filfinder2D.py"


def paint_square(image: np.ndarray, row: int, col: int, value: float) -> None:
    """Paint a clipped 3-by-3 square around one path coordinate."""
    row_start = max(row - 1, 0)
    row_stop = min(row + 2, image.shape[0])
    col_start = max(col - 1, 0)
    col_stop = min(col + 2, image.shape[1])
    image[row_start:row_stop, col_start:col_stop] = value


def synthetic_cases() -> dict[str, np.ndarray]:
    """Return deterministic masks covering the wrapper-visible topology cases."""
    straight = np.zeros((17, 17), dtype=np.float64)
    straight[3:14, 7:10] = 0.8
    straight[8, 8] = 1.0

    y_spur = np.zeros((21, 21), dtype=np.float64)
    for row in range(4, 17):
        paint_square(y_spur, row, 10, 0.75)
    for offset in range(0, 7):
        paint_square(y_spur, 10 - offset, 10 - offset, 0.9)
        paint_square(y_spur, 10 - offset, 10 + offset, 0.65)

    disconnected = np.zeros((21, 21), dtype=np.float64)
    disconnected[3:16, 3:6] = 0.7
    disconnected[6:19, 15:18] = 0.85

    loop_branch = np.zeros((25, 25), dtype=np.float64)
    loop_branch[5:8, 5:19] = 0.8
    loop_branch[17:20, 5:19] = 0.8
    loop_branch[5:20, 5:8] = 0.8
    loop_branch[5:20, 16:19] = 0.8
    loop_branch[11:14, 18:23] = 0.6

    noise = np.fromfunction(
        lambda row, col: ((row * 17 + col * 11) % 47) / 100.0,
        (19, 23),
        dtype=int,
    ).astype(np.float64)
    noise[2:5, 2:5] = 0.55
    noise[8:11, 11:16] = 0.65
    noise[14:17, 19:22] = 0.75

    symmetric_tie = np.zeros((21, 21), dtype=np.float64)
    for row in range(10, 18):
        paint_square(symmetric_tie, row, 10, 0.8)
    for offset in range(0, 7):
        paint_square(symmetric_tie, 10 - offset, 10 - offset, 0.8)
        paint_square(symmetric_tie, 10 - offset, 10 + offset, 0.8)

    threshold_boundary = np.zeros((7, 9), dtype=np.float64)
    threshold_boundary[2:5, 2] = np.nextafter(THRESHOLD, 0.0)
    threshold_boundary[2:5, 4] = THRESHOLD
    threshold_boundary[2:5, 6] = np.nextafter(THRESHOLD, 1.0)

    empty = np.full((7, 9), np.nextafter(THRESHOLD, 0.0), dtype=np.float64)

    return {
        "straight": straight,
        "y_spur": y_spur,
        "disconnected": disconnected,
        "loop_branch": loop_branch,
        "noise": noise,
        "symmetric_tie": symmetric_tie,
        "threshold_boundary": threshold_boundary,
        "empty": empty,
    }


def encode_array(array: np.ndarray) -> list[object]:
    """Convert an oracle array into JSON-native nested lists."""
    return np.asarray(array).tolist()


def label_eight_connected(mask: np.ndarray) -> np.ndarray:
    """Label a selected raster in deterministic row-major component order."""
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return labels.astype(np.int64, copy=False)


def warning_counts(caught: list[warnings.WarningMessage]) -> dict[str, int]:
    """Count repeated upstream warnings without bloating the fixture."""
    counts: dict[str, int] = {}
    for item in caught:
        message = str(item.message)
        counts[message] = counts.get(message, 0) + 1
    return dict(sorted(counts.items()))


def analyze_case(name: str, image: np.ndarray) -> dict[str, object]:
    """Capture every wrapper-visible stage from one fresh FilFinder object."""
    threshold_mask = image >= THRESHOLD
    record: dict[str, object] = {
        "name": name,
        "image": encode_array(image),
        "threshold_mask": encode_array(threshold_mask.astype(np.uint8)),
    }

    with ProcessPoolExecutor(max_workers=1) as pool:
        filfinder = FilFinder2D(
            image.copy(),
            beamwidth=BEAMWIDTH_PX * u.pix,
            mask=threshold_mask.copy(),
            pool=pool,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filfinder.create_mask(use_existing_mask=True)
        record["create_mask_warning_counts"] = warning_counts(caught)
        record["filfinder_mask"] = encode_array(
            np.asarray(filfinder.mask, dtype=np.uint8)
        )
        record["mask_labels_8_connected"] = encode_array(
            label_eight_connected(filfinder.mask)
        )

        if not threshold_mask.any():
            record.update(
                {
                    "empty_short_circuit": True,
                    "medial_axis_distance_px": None,
                    "skeleton_pre_prune": None,
                    "skeleton_post_prune": None,
                    "skeleton_longest_path": None,
                    "skeleton_labels_8_connected": None,
                    "longest_path_labels_8_connected": None,
                    "filament_lengths_px": [],
                    "branch_lengths_px": [],
                }
            )
            return record

        filfinder.medskel(rng=RNG_SEED)
        skeleton_pre_prune = np.asarray(filfinder.skeleton, dtype=bool).copy()
        record["empty_short_circuit"] = False
        record["medial_axis_distance_px"] = encode_array(
            np.asarray(filfinder.medial_axis_distance.value, dtype=np.float64)
        )
        record["skeleton_pre_prune"] = encode_array(
            skeleton_pre_prune.astype(np.uint8)
        )
        record["skeleton_labels_8_connected"] = encode_array(
            label_eight_connected(skeleton_pre_prune)
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            branch_threshold = (
                None
                if BRANCH_THRESHOLD_PX is None
                else BRANCH_THRESHOLD_PX * u.pix
            )
            filfinder.analyze_skeletons(
                prune_criteria="all",
                relintens_thresh=RELATIVE_INTENSITY_THRESHOLD,
                skel_thresh=1.0 * u.pix,
                branch_thresh=branch_threshold,
                max_prune_iter=MAX_PRUNE_ITERATIONS,
            )
        record["analyze_skeleton_warning_counts"] = warning_counts(caught)
        record["skeleton_post_prune"] = encode_array(
            np.asarray(filfinder.skeleton, dtype=np.uint8)
        )
        record["skeleton_longest_path"] = encode_array(
            np.asarray(filfinder.skeleton_longpath, dtype=np.uint8)
        )
        record["longest_path_labels_8_connected"] = encode_array(
            label_eight_connected(filfinder.skeleton_longpath)
        )
        record["filament_lengths_px"] = [
            float(filament.length(u.pix).value) for filament in filfinder.filaments
        ]
        record["branch_lengths_px"] = [
            [float(value.value) for value in lengths]
            for lengths in filfinder.branch_properties["length"]
        ]
        record["effective_skeleton_threshold_px"] = float(
            filfinder.skel_thresh.value
        )
        record["effective_branch_threshold_px"] = int(
            filfinder.branch_thresh.value
        )
        return record


def dependency_versions() -> dict[str, str]:
    """Return the full pinned oracle dependency vector."""
    names = (
        "fil-finder",
        "astropy",
        "numpy",
        "networkx",
        "scipy",
        "matplotlib",
        "scikit-image",
        "h5py",
        "skan",
        "numba",
        "pandas",
    )
    return {name: importlib.metadata.version(name) for name in names}


def verify_authoritative_source() -> None:
    """Fail unless the oracle imports the committed sdist source file."""
    actual = pathlib.Path(inspect.getfile(FilFinder2D)).resolve()
    if actual != EXPECTED_SOURCE.resolve():
        raise RuntimeError(
            "oracle did not import the committed FilFinder source: "
            f"{actual} != {EXPECTED_SOURCE.resolve()}"
        )


def generate_filfinder_fixture() -> None:
    """Regenerate the deterministic FilFinder 1.8 fixture JSON."""
    verify_authoritative_source()
    fixture = {
        "schema_version": 1,
        "authority": "fil-finder 1.8 sdist and v1.8 tag commit",
        "parameters": {
            "threshold": THRESHOLD,
            "threshold_comparison": ">=",
            "beamwidth_px": BEAMWIDTH_PX,
            "prune_criteria": "all",
            "relative_intensity_threshold": RELATIVE_INTENSITY_THRESHOLD,
            "skeleton_threshold_px": 1.0,
            "branch_threshold_px": BRANCH_THRESHOLD_PX,
            "max_prune_iterations": MAX_PRUNE_ITERATIONS,
            "rng_seed": RNG_SEED,
            "label_connectivity": 8,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "cases": [
            analyze_case(name, image) for name, image in synthetic_cases().items()
        ],
    }
    FIXTURE_PATH.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate_filfinder_fixture()
