"""Independently re-derive load-bearing Method B numeric invariants."""

from __future__ import annotations

import sys

import numpy as np


def segment_sse(matrix: np.ndarray, start: int, stop: int) -> float:
    segment = matrix[start:stop]
    return float(np.square(segment - segment.mean(axis=0)).sum())


def exact_fit(
    matrix: np.ndarray, support: np.ndarray, minimum: int
) -> tuple[float, int, int] | None:
    best = None
    count = matrix.shape[0]
    for first in range(minimum, count - 2 * minimum + 1):
        if support[first:].mean() < support[:first].mean():
            continue
        for second in range(first + minimum, count - minimum + 1):
            if not support[first:second].any() or not support[second:].any():
                continue
            candidate = (
                segment_sse(matrix, 0, first)
                + segment_sse(matrix, first, second)
                + segment_sse(matrix, second, count),
                first,
                second,
            )
            if best is None or candidate < best:
                best = candidate
    return best


def validate_method_b_invariants() -> None:
    distances = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.max(distances) == 3.0
    assert np.percentile(distances, 50.0, method="linear") == 1.5
    outer = 3.0
    internal = 2.0
    dense = (distances >= 1.0) & (distances < internal)
    sparse = (distances >= internal) & (
        distances < np.nextafter(outer, np.inf)
    )
    assert dense.tolist() == [False, True, False, False]
    assert sparse.tolist() == [False, False, True, True]

    support = np.array([False, True, False, True, False])
    bridged = support.copy()
    bridged[2] = True
    assert bridged.tolist() == [False, True, True, True, False]

    tied = exact_fit(np.zeros((12, 4)), np.ones(12, dtype=bool), 3)
    assert tied == (0.0, 3, 6)

    feature = np.concatenate(
        [np.zeros((3, 1)), np.ones((3, 1)), np.full((3, 1), 2.0)]
    )
    supported = np.ones(9, dtype=bool)
    separated = exact_fit(feature, supported, 3)
    assert separated is not None
    assert separated[1:] == (3, 6)
    assert separated[0] == 0.0

    ring_width = 8.0
    radii = (np.arange(12) + 0.5) * ring_width
    core_end = radii[3] - ring_width / 2.0
    dense_end = radii[6] - ring_width / 2.0
    assert core_end == 24.0
    assert dense_end == 48.0


if __name__ == "__main__":
    try:
        validate_method_b_invariants()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise
    print("PASS: Method B numeric invariants")
