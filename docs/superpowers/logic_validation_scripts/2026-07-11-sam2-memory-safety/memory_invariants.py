"""Re-derive the numeric invariants behind SAM2 memory-safe processing.

This script is intentionally independent of :mod:`phenotypic`.  It validates
the byte-exact legacy normalization path, lossless Fortran-order RLE, and the
load-bearing full-resolution mask-memory calculations used by the design.
"""

from __future__ import annotations

import numpy as np


PLATE_HEIGHT = 3002
PLATE_WIDTH = 5086
MASKS_PER_POINT = 3
FLOAT32_BYTES = 4


def legacy_image_max_scale(values: np.ndarray, maximum: int) -> np.ndarray:
    """Return the historical whole-array uint16-to-uint8 conversion."""
    if maximum == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    return (values / maximum * 255).astype(np.uint8)


def chunked_image_max_scale(
    values: np.ndarray, maximum: int, chunk_size: int = 257
) -> np.ndarray:
    """Return the same conversion while bounding temporary allocations."""
    output = np.empty(values.shape, dtype=np.uint8)
    for start in range(0, values.size, chunk_size):
        stop = min(start + chunk_size, values.size)
        output[start:stop] = legacy_image_max_scale(values[start:stop], maximum)
    return output


def encode_fortran_rle(mask: np.ndarray) -> dict[str, object]:
    """Encode a Boolean mask using SAM2's uncompressed Fortran-order RLE."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    counts: list[int] = []
    value = False
    run = 0
    for pixel in flat:
        pixel_value = bool(pixel)
        if pixel_value == value:
            run += 1
        else:
            counts.append(run)
            value = pixel_value
            run = 1
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def decode_fortran_rle(rle: dict[str, object]) -> np.ndarray:
    """Decode uncompressed Fortran-order RLE into its exact Boolean mask."""
    height, width = (int(value) for value in rle["size"])  # type: ignore[index]
    flat = np.empty(height * width, dtype=bool)
    offset = 0
    value = False
    for count_value in rle["counts"]:  # type: ignore[union-attr]
        count = int(count_value)
        flat[offset : offset + count] = value
        offset += count
        value = not value
    assert offset == flat.size
    return flat.reshape((height, width), order="F")


def validate_scaling_equivalence() -> None:
    """Prove chunking preserves legacy bytes for representative maxima."""
    for maximum in (0, 1, 257, 1023, 4095, 30_000, 65_522, 65_535):
        values = np.arange(maximum + 1, dtype=np.uint16)
        expected = legacy_image_max_scale(values, maximum)
        actual = chunked_image_max_scale(values, maximum)
        np.testing.assert_array_equal(actual, expected)


def validate_rle_round_trips() -> None:
    """Prove lossless RLE for edge cases and deterministic random masks."""
    rng = np.random.default_rng(42)
    masks = [
        np.zeros((3, 5), dtype=bool),
        np.ones((3, 5), dtype=bool),
        np.indices((4, 7)).sum(axis=0) % 2 == 0,
        rng.random((17, 11)) > 0.7,
    ]
    for mask in masks:
        np.testing.assert_array_equal(decode_fortran_rle(encode_fortran_rle(mask)), mask)


def validate_memory_arithmetic() -> None:
    """Validate exact bytes without embedding an opaque decimal constant."""
    pixels = PLATE_HEIGHT * PLATE_WIDTH
    assert pixels == 15_268_172
    batch_64 = 64 * MASKS_PER_POINT * pixels * FLOAT32_BYTES
    batch_8 = 8 * MASKS_PER_POINT * pixels * FLOAT32_BYTES
    assert batch_64 == 11_725_956_096
    assert batch_8 == 1_465_744_512
    assert batch_64 == 8 * batch_8


if __name__ == "__main__":
    validate_scaling_equivalence()
    validate_rle_round_trips()
    validate_memory_arithmetic()
    print("SAM2 memory invariants verified")
