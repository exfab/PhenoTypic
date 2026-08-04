"""Lossless, bounded-memory helpers for SAM2 uncompressed RLE masks."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def validate_uncompressed_rle(
    rle: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, ...]]:
    """Validate an upstream SAM2 uncompressed, Fortran-order RLE.

    Args:
        rle: Mapping with ``size=[height, width]`` and integer run ``counts``.
        expected_shape: Optional image shape that ``size`` must match.

    Returns:
        The validated shape and immutable run counts.

    Raises:
        ValueError: If the RLE is malformed or does not cover exactly one mask.
    """
    if not isinstance(rle, Mapping) or "size" not in rle or "counts" not in rle:
        raise ValueError("SAM2 RLE must contain 'size' and 'counts'")

    size = rle["size"]
    if (
        not isinstance(size, Sequence)
        or isinstance(size, (str, bytes))
        or len(size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in size
        )
    ):
        raise ValueError("SAM2 RLE 'size' must contain two integers")
    shape = (int(size[0]), int(size[1]))
    if shape[0] < 0 or shape[1] < 0:
        raise ValueError("SAM2 RLE dimensions must be non-negative")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(
            f"SAM2 RLE shape {shape} does not match image shape {expected_shape}"
        )

    counts_value = rle["counts"]
    if (
        not isinstance(counts_value, Sequence)
        or isinstance(counts_value, (str, bytes))
        or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in counts_value
        )
    ):
        raise ValueError("SAM2 uncompressed RLE 'counts' must be integer runs")
    counts = tuple(int(value) for value in counts_value)
    if any(value < 0 for value in counts):
        raise ValueError("SAM2 RLE runs must be non-negative")
    if sum(counts) != shape[0] * shape[1]:
        raise ValueError("SAM2 RLE runs must cover exactly height * width pixels")
    return shape, counts


def decode_uncompressed_rle(
    rle: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int] | None = None,
):
    """Decode one SAM2 uncompressed RLE using its Fortran-order convention."""
    import numpy as np

    shape, counts = validate_uncompressed_rle(rle, expected_shape=expected_shape)
    flat: np.ndarray = np.empty(shape[0] * shape[1], dtype=bool)
    offset = 0
    value = False
    for count in counts:
        flat[offset : offset + count] = value
        offset += count
        value = not value
    return flat.reshape(shape, order="F")


def encode_uncompressed_rle(mask) -> dict[str, list[int]]:
    """Encode a 2-D mask into upstream-compatible Fortran-order RLE."""
    import numpy as np

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("SAM2 masks must be two-dimensional")
    flat = array.reshape(-1, order="F")
    counts: list[int] = []
    value = False
    run = 0
    for pixel in flat:
        pixel_value = bool(pixel)
        if pixel_value == value:
            run += 1
        else:
            counts.append(run)
            run = 1
            value = pixel_value
    counts.append(run)
    return {"size": [array.shape[0], array.shape[1]], "counts": counts}


def segmentation_as_rle(
    segmentation: Any, *, expected_shape: tuple[int, int]
) -> Mapping[str, Any]:
    """Return a validated RLE, accepting Boolean masks for compatibility."""
    if isinstance(segmentation, Mapping):
        validate_uncompressed_rle(segmentation, expected_shape=expected_shape)
        return segmentation
    rle = encode_uncompressed_rle(segmentation)
    validate_uncompressed_rle(rle, expected_shape=expected_shape)
    return rle


def normalize_rle_records(
    records: Sequence[MutableMapping[str, Any]],
    *,
    expected_shape: tuple[int, int],
) -> None:
    """Normalize SAM2 record segmentations to validated RLE in place."""
    for record in records:
        record["segmentation"] = segmentation_as_rle(
            record["segmentation"], expected_shape=expected_shape
        )
        record.setdefault("area", rle_area(record["segmentation"]))


def rle_area(rle: Mapping[str, Any]) -> int:
    """Return the exact foreground area without decoding the mask."""
    _, counts = validate_uncompressed_rle(rle)
    return sum(counts[1::2])


def rle_iou(rle_a: Mapping[str, Any], rle_b: Mapping[str, Any]) -> float:
    """Calculate exact mask IoU directly from two uncompressed RLE streams."""
    shape_a, counts_a = validate_uncompressed_rle(rle_a)
    shape_b, counts_b = validate_uncompressed_rle(rle_b)
    if shape_a != shape_b:
        raise ValueError(f"Cannot compare RLE shapes {shape_a} and {shape_b}")

    index_a = index_b = 0
    remaining_a = counts_a[0] if counts_a else 0
    remaining_b = counts_b[0] if counts_b else 0
    value_a = value_b = False
    intersection = union = 0
    total = shape_a[0] * shape_a[1]
    consumed = 0
    while consumed < total:
        while remaining_a == 0:
            index_a += 1
            value_a = not value_a
            remaining_a = counts_a[index_a]
        while remaining_b == 0:
            index_b += 1
            value_b = not value_b
            remaining_b = counts_b[index_b]
        span = min(remaining_a, remaining_b)
        if value_a and value_b:
            intersection += span
        if value_a or value_b:
            union += span
        remaining_a -= span
        remaining_b -= span
        consumed += span
    return intersection / union if union else 0.0


def stable_area_order(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return records largest-first, preserving input order for area ties."""
    return sorted(records, key=lambda record: int(record["area"]), reverse=True)


def merge_rle_records_by_iou(
    records: Sequence[Mapping[str, Any]], iou_thresh: float
) -> list[Mapping[str, Any]]:
    """Greedily deduplicate RLE records by exact IoU, largest-first."""
    ordered = stable_area_order(records)
    kept: list[Mapping[str, Any]] = []
    for candidate in ordered:
        candidate_rle = candidate["segmentation"]
        if any(
            rle_iou(candidate_rle, record["segmentation"]) > iou_thresh
            for record in kept
        ):
            continue
        kept.append(candidate)
    return kept


def paint_rle_records(
    records: Sequence[Mapping[str, Any]],
    shape: tuple[int, int],
    *,
    detector_name: str,
    truncate_before_sort: bool,
):
    """Paint records largest-first while decoding at most one mask at a time."""
    import warnings

    import numpy as np

    selected = list(records)
    max_labels = int(np.iinfo(np.uint16).max)
    if truncate_before_sort and len(selected) > max_labels:
        warnings.warn(
            f"{detector_name} generated {len(selected)} masks, exceeding uint16 "
            f"range. Only the first {max_labels} will be labeled.",
            UserWarning,
            stacklevel=2,
        )
        selected = selected[:max_labels]
    selected = stable_area_order(selected)
    if not truncate_before_sort and len(selected) > max_labels:
        warnings.warn(
            f"{detector_name} kept {len(selected)} proposals, exceeding uint16 "
            f"range. Only the first {max_labels} (largest) will be labeled.",
            UserWarning,
            stacklevel=2,
        )
        selected = selected[:max_labels]

    objmap: np.ndarray = np.zeros(shape, dtype=np.uint16)
    for label, record in enumerate(selected, start=1):
        mask = decode_uncompressed_rle(
            record["segmentation"], expected_shape=shape
        )
        objmap[mask] = label
        del mask
    return objmap
