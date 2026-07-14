"""Validate only the PhenoTypic-owned A10 FilFinder adapter logic.

This script never imports :mod:`phenotypic`, FilFinder, scikit-image, or SciPy.
It independently verifies inclusive thresholding, threshold monotonicity,
deterministic 8-connected row-major labeling, mask/map consistency, empty
input, and input-layer preservation. FilFinder skeletonization, pruning,
longest-path selection, lengths, and distances are intentionally validated by
the pinned external fixture and behavioral controls, not reimplemented here.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import deque

import numpy as np


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[4]
FIXTURE_PATH = REPOSITORY_ROOT / "tests/fixtures/reconnect/filfinder/oracle.json"


def inclusive_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    """Return the frozen threshold after ImageData's float32 coercion seam."""
    source = np.asarray(image, dtype=np.float32).astype(np.float64)
    return source >= threshold


def label_eight_connected(mask: np.ndarray) -> np.ndarray:
    """Label components by first row-major pixel using an independent flood fill."""
    binary = np.asarray(mask, dtype=bool)
    labels = np.zeros(binary.shape, dtype=np.int64)
    next_label = 0
    height, width = binary.shape
    for start_row in range(height):
        for start_col in range(width):
            if not binary[start_row, start_col] or labels[start_row, start_col] != 0:
                continue
            next_label += 1
            labels[start_row, start_col] = next_label
            queue = deque([(start_row, start_col)])
            while queue:
                row, col = queue.popleft()
                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        if row_offset == 0 and col_offset == 0:
                            continue
                        neighbor_row = row + row_offset
                        neighbor_col = col + col_offset
                        if not (
                            0 <= neighbor_row < height and 0 <= neighbor_col < width
                        ):
                            continue
                        if (
                            binary[neighbor_row, neighbor_col]
                            and labels[neighbor_row, neighbor_col] == 0
                        ):
                            labels[neighbor_row, neighbor_col] = next_label
                            queue.append((neighbor_row, neighbor_col))
    return labels


def validate_inclusive_threshold_and_monotonicity() -> None:
    """Pin equality and prove that increasing the threshold cannot add pixels."""
    center = np.float32(0.5)
    values = np.array(
        [
            np.nextafter(center, np.float32(0.0)),
            center,
            np.nextafter(center, np.float32(1.0)),
        ],
        dtype=np.float32,
    )
    actual = inclusive_threshold(values, float(center))
    expected = np.array([False, True, True])
    if not np.array_equal(actual, expected):
        raise AssertionError(f"inclusive threshold mismatch: {actual} != {expected}")

    image = np.linspace(0.0, 1.0, 1001, dtype=np.float32).reshape(7, 143)
    previous = inclusive_threshold(image, 0.0)
    for threshold in np.linspace(0.001, 1.0, 1000):
        current = inclusive_threshold(image, float(threshold))
        if np.any(current & ~previous):
            raise AssertionError("raising threshold added a foreground pixel")
        previous = current


def validate_label_connectivity_and_order() -> None:
    """Pin 8-connectivity, scan order, label dtype, and disconnected ties."""
    mask = np.zeros((7, 8), dtype=bool)
    mask[0, 6] = True
    mask[1, 5] = True  # diagonal: same 8-connected component as (0, 6)
    mask[3, 1:3] = True
    mask[5, 5] = True
    expected = np.zeros_like(mask, dtype=np.int64)
    expected[0, 6] = 1
    expected[1, 5] = 1
    expected[3, 1:3] = 2
    expected[5, 5] = 3
    actual = label_eight_connected(mask)
    if actual.dtype != np.int64 or not np.array_equal(actual, expected):
        raise AssertionError(f"label contract mismatch:\n{actual}\n!=\n{expected}")
    if not np.array_equal(actual > 0, mask):
        raise AssertionError("objmask differs from objmap > 0")


def validate_empty_and_layer_preservation() -> None:
    """Prove empty output and that adapter-owned work need not mutate inputs."""
    rgb = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    gray = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(4, 5)
    detect_mat = gray.copy()
    originals = (rgb.copy(), gray.copy(), detect_mat.copy())
    empty_labels = label_eight_connected(inclusive_threshold(detect_mat, 2.0))
    if empty_labels.shape != detect_mat.shape or np.any(empty_labels):
        raise AssertionError("empty threshold result was not an all-zero label map")
    for actual, expected in zip((rgb, gray, detect_mat), originals, strict=True):
        if not np.array_equal(actual, expected):
            raise AssertionError("adapter work modified an image source layer")


def validate_external_fixture_translation() -> None:
    """Re-derive threshold and label translations around pinned external rasters."""
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    threshold = float(fixture["parameters"]["threshold"])
    for case in fixture["cases"]:
        image = np.asarray(case["image"], dtype=np.float64)
        expected_threshold = np.asarray(case["threshold_mask"], dtype=bool)
        if not np.array_equal(inclusive_threshold(image, threshold), expected_threshold):
            raise AssertionError(f"fixture threshold mismatch for {case['name']}")

        mask = np.asarray(case["filfinder_mask"], dtype=bool)
        expected_mask_labels = np.asarray(
            case["mask_labels_8_connected"], dtype=np.int64
        )
        if not np.array_equal(label_eight_connected(mask), expected_mask_labels):
            raise AssertionError(f"mask label mismatch for {case['name']}")

        for raster_key, label_key in (
            ("skeleton_pre_prune", "skeleton_labels_8_connected"),
            ("skeleton_longest_path", "longest_path_labels_8_connected"),
        ):
            if case[raster_key] is None:
                if case[label_key] is not None:
                    raise AssertionError(
                        f"empty raster has labels for {case['name']}:{raster_key}"
                    )
                continue
            raster = np.asarray(case[raster_key], dtype=bool)
            expected_labels = np.asarray(case[label_key], dtype=np.int64)
            if not np.array_equal(label_eight_connected(raster), expected_labels):
                raise AssertionError(f"label mismatch for {case['name']}:{raster_key}")


def validate_filfinder_adapter_contract() -> None:
    """Run every source-independent adapter derivation."""
    validate_inclusive_threshold_and_monotonicity()
    validate_label_connectivity_and_order()
    validate_empty_and_layer_preservation()
    validate_external_fixture_translation()


if __name__ == "__main__":
    try:
        validate_filfinder_adapter_contract()
    except Exception as error:
        print(f"A10 FilFinder adapter logic validation FAILED: {error}", file=sys.stderr)
        raise
    print("A10 FilFinder adapter logic validation PASSED")
