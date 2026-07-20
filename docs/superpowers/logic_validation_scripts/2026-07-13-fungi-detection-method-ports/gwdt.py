"""Independently validate the numeric claims behind the exact APP2 GWDT port.

This script never imports :mod:`phenotypic`. It parses the pinned Vaa3D lookup table
and uses two-phase initialization plus whole-grid Bellman-Ford relaxation, which is
structurally independent of the production heap.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests" / "fixtures" / "reconnect" / "gwdt" / "app2_source.npz"
MACRO = (
    ROOT
    / "docs/superpowers/specs/2026-07-13-fungi-detection-method-ports"
    / "refs/gwdt/vaa3d/app2/fastmarching_macro.h"
)
SOURCE_INFINITY = float(np.float32(1e20))


def neighbors(connectivity: int) -> tuple[tuple[int, int, float], ...]:
    """Return Vaa3D's selected one-slice neighborhood in source loop order."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    result = []
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            offset = abs(row_offset) + abs(column_offset)
            if offset == 0 or (connectivity == 4 and offset > 1):
                continue
            result.append((row_offset, column_offset, math.sqrt(offset)))
    return tuple(result)


def validate_inputs(image: np.ndarray, background: np.ndarray) -> None:
    """Validate the frozen Python boundary independently."""
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("image shape")
    if image.dtype.kind not in "iuf":
        raise TypeError("image dtype")
    if not np.all(np.isfinite(image)) or np.any(image < 0):
        raise ValueError("image values")
    if background.dtype != np.bool_ or background.shape != image.shape:
        raise ValueError("background")


def source_frontier(
    image: np.ndarray,
    background: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce only Vaa3D's asymmetric initialization in float64."""
    validate_inputs(image, background)
    values = image.astype(np.float64, copy=False)
    distance = np.full(values.shape, SOURCE_INFINITY)
    distance[background] = values[background]
    trial = np.zeros(values.shape, dtype=bool)
    rows, columns = values.shape
    edges = neighbors(connectivity)

    for seed_row, seed_column in np.argwhere(background):
        for row_offset, column_offset, _ in edges:
            row = int(seed_row) + row_offset
            column = int(seed_column) + column_offset
            if not (0 <= row < rows and 0 <= column < columns):
                continue
            if background[row, column] or trial[row, column]:
                continue
            minimum_row = int(seed_row)
            minimum_column = int(seed_column)
            if distance[minimum_row, minimum_column] > 0.0:
                for alive_row_offset, alive_column_offset, _ in edges:
                    alive_row = row + alive_row_offset
                    alive_column = column + alive_column_offset
                    if (
                        0 <= alive_row < rows
                        and 0 <= alive_column < columns
                        and background[alive_row, alive_column]
                        and distance[alive_row, alive_column]
                        < distance[minimum_row, minimum_column]
                    ):
                        minimum_row = alive_row
                        minimum_column = alive_column
            distance[row, column] = (
                distance[minimum_row, minimum_column] + values[row, column]
            )
            trial[row, column] = True
    return distance, trial


def bellman_ford_source_oracle(
    image: np.ndarray,
    background: np.ndarray,
    connectivity: int,
) -> np.ndarray:
    """Relax APP2's ordinary recurrence after its exact frontier phase."""
    values = image.astype(np.float64, copy=False)
    distance, _ = source_frontier(values, background, connectivity)
    rows, columns = values.shape
    edges = neighbors(connectivity)

    for _ in range(values.size):
        previous = distance.copy()
        for row in range(rows):
            for column in range(columns):
                if background[row, column]:
                    continue
                for row_offset, column_offset, length in edges:
                    source_row = row + row_offset
                    source_column = column + column_offset
                    if 0 <= source_row < rows and 0 <= source_column < columns:
                        distance[row, column] = min(
                            distance[row, column],
                            previous[source_row, source_column]
                            + values[row, column] * length,
                        )
        if np.array_equal(previous, distance):
            return distance
    raise AssertionError("Bellman-Ford did not converge within |V| passes")


def parse_source_givals() -> np.ndarray:
    """Parse all 256 active lookup values from the pinned C++ macro."""
    source = MACRO.read_text(encoding="utf-8")
    match = re.search(r"givals\[256\] = \{(.*?)\};", source, flags=re.DOTALL)
    if match is None:
        raise AssertionError("could not locate pinned givals table")
    values = np.fromstring(match.group(1).replace("\n", " "), sep=",")
    assert values.shape == (256,)
    return values


def exact_source_cost(distance: np.ndarray) -> np.ndarray:
    """Apply the active truncating 256-bin source lookup."""
    minimum = float(np.min(distance))
    span = float(np.max(distance)) - minimum
    if span == 0.0:
        return np.ones(distance.shape)
    indices = ((distance - minimum) / span * 255.0).astype(np.intp)
    return parse_source_givals()[indices]


def float32_rounding_bound(image: np.ndarray) -> float:
    """Bound one float32 rounding per accumulated path term."""
    terms = max(1, image.size)
    epsilon = np.finfo(np.float32).eps
    gamma = terms * epsilon / (1.0 - terms * epsilon)
    path_magnitude = (
        float(np.max(image)) + terms * float(np.max(image)) * math.sqrt(2.0)
    )
    return gamma * path_magnitude


def enumerate_post_frontier_costs(
    image: np.ndarray,
    background: np.ndarray,
    target: tuple[int, int],
) -> list[float]:
    """Enumerate simple 8-neighbor routes from exact initialized frontier nodes."""
    initial, trial = source_frontier(image, background, 8)
    rows, columns = image.shape
    costs: list[float] = []

    def visit(
        position: tuple[int, int],
        seen: frozenset[tuple[int, int]],
        cost: float,
    ) -> None:
        if position == target:
            costs.append(cost)
            return
        row, column = position
        for row_offset, column_offset, length in neighbors(8):
            next_position = (row + row_offset, column + column_offset)
            next_row, next_column = next_position
            if (
                0 <= next_row < rows
                and 0 <= next_column < columns
                and not background[next_position]
                and next_position not in seen
            ):
                visit(
                    next_position,
                    seen | {next_position},
                    cost + image[next_position] * length,
                )

    for start in zip(*np.nonzero(trial), strict=True):
        visit(start, frozenset({start}), float(initial[start]))
    return sorted(set(costs))


def verify_source_fixture() -> list[tuple[str, int, float, float]]:
    """Check every source-generated map and return printed tolerance evidence."""
    cases = (
        ("standard", "image", "background"),
        ("diagonal", "diagonal_image", "diagonal_background"),
        ("threshold", "threshold_image", "threshold_background"),
        ("all_background", "all_background_image", "all_background"),
        ("no_background", "no_background_image", "no_background"),
        (
            "post_frontier_diagonal",
            "post_frontier_diagonal_image",
            "post_frontier_diagonal_background",
        ),
    )
    evidence = []
    with np.load(FIXTURE) as fixture:
        for case_name, image_key, background_key in cases:
            image = fixture[image_key]
            background = fixture[background_key]
            for connectivity in (4, 8):
                source = fixture[f"source_{case_name}_distance_{connectivity}"]
                oracle = bellman_ford_source_oracle(image, background, connectivity)
                bound = float32_rounding_bound(image)
                error = float(np.max(np.abs(source.astype(np.float64) - oracle)))
                assert error <= bound, (case_name, connectivity, error, bound)
                evidence.append((case_name, connectivity, error, bound))

                cost_key = f"source_{case_name}_cost_{connectivity}"
                if cost_key in fixture.files:
                    np.testing.assert_array_equal(
                        exact_source_cost(source),
                        fixture[cost_key],
                    )

        assert fixture["source_threshold_distance_4"][0, 0] == 1.0
        assert fixture["source_threshold_distance_4"][0, 1] == 2.0
        assert fixture["source_threshold_distance_4"][0, 2] == 7.0
        np.testing.assert_array_equal(
            fixture["source_all_background_distance_8"],
            fixture["all_background_image"].astype(np.float32),
        )
        np.testing.assert_array_equal(
            fixture["source_no_background_distance_8"],
            np.float32(SOURCE_INFINITY),
        )
        assert fixture["source_diagonal_distance_8"][1, 1] == 1.0
        assert fixture["source_post_frontier_diagonal_distance_8"][2, 2] == np.float32(
            1.0 + math.sqrt(2.0)
        )

        routes = enumerate_post_frontier_costs(
            fixture["post_frontier_diagonal_image"].astype(np.float64),
            fixture["post_frontier_diagonal_background"],
            (2, 2),
        )
        bound = float32_rounding_bound(fixture["post_frontier_diagonal_image"])
        source_target = float(
            fixture["source_post_frontier_diagonal_distance_8"][2, 2]
        )
        oracle_target = float(
            bellman_ford_source_oracle(
                fixture["post_frontier_diagonal_image"],
                fixture["post_frontier_diagonal_background"],
                8,
            )[2, 2]
        )
        assert abs(source_target - routes[0]) <= bound
        assert abs(oracle_target - routes[0]) <= bound
        assert routes[1] - routes[0] > bound
    return evidence


def verify_behavioral_controls() -> None:
    """Check analytic, geometric, monotonic, and polarity controls."""
    one_dimensional = np.array([[0.0, 2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(
        bellman_ford_source_oracle(one_dimensional, one_dimensional == 0.0, 4),
        [[0.0, 2.0, 5.0, 9.0]],
    )

    baseline_image = np.array([[0.0, 1.0, 2.0, 3.0]])
    baseline = bellman_ford_source_oracle(
        baseline_image,
        baseline_image == 0.0,
        4,
    )
    increased = baseline_image.copy()
    increased[0, 2] += 5.0
    assert np.all(
        bellman_ford_source_oracle(increased, increased == 0.0, 4) >= baseline
    )
    np.testing.assert_allclose(
        bellman_ford_source_oracle(7.0 * baseline_image, baseline_image == 0.0, 4),
        7.0 * baseline,
    )
    added_seed_image = baseline_image.copy()
    added_seed_image[0, 3] = 0.0
    added_seed_distance = bellman_ford_source_oracle(
        added_seed_image,
        added_seed_image == 0.0,
        4,
    )
    assert np.all(added_seed_distance <= baseline)

    multiple_image = np.array([[0.0, 1.0, 1.0, 1.0, 0.0]])
    np.testing.assert_array_equal(
        bellman_ford_source_oracle(multiple_image, multiple_image == 0.0, 4),
        [[0.0, 1.0, 2.0, 1.0, 0.0]],
    )

    two_route = np.array([[1.0, 1.0, 1.0], [0.0, 10.0, 1.0]])
    two_route_distance = bellman_ford_source_oracle(
        two_route,
        two_route == 0.0,
        8,
    )
    assert two_route_distance[1, 2] == 1.0 + math.sqrt(2.0)

    ordinary_diagonal = np.array(
        [[0.0, 100.0, 100.0], [100.0, 1.0, 100.0], [100.0, 100.0, 1.0]]
    )
    ordinary_distance = bellman_ford_source_oracle(
        ordinary_diagonal,
        ordinary_diagonal == 0.0,
        8,
    )
    assert ordinary_distance[1, 1] == 1.0
    assert ordinary_distance[2, 2] == 1.0 + math.sqrt(2.0)

    asymmetric = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0], [2.0, 6.0, 0.0]])
    asymmetric_distance = bellman_ford_source_oracle(
        asymmetric,
        asymmetric == 0.0,
        8,
    )
    for transform in (np.transpose, np.fliplr, np.flipud, np.rot90):
        transformed = transform(asymmetric)
        np.testing.assert_allclose(
            bellman_ford_source_oracle(transformed, transformed == 0.0, 8),
            transform(asymmetric_distance),
            atol=1e-14,
        )
    assert np.all(
        bellman_ford_source_oracle(asymmetric, asymmetric == 0.0, 4)
        >= asymmetric_distance
    )

    bar = np.zeros((5, 7))
    bar[1:4] = 1.0
    bar_distance = bellman_ford_source_oracle(bar, bar == 0.0, 4)
    bar_cost = exact_source_cost(bar_distance)
    assert np.all(bar_distance[2] > bar_distance[1])
    assert np.all(bar_cost[2] < bar_cost[1])
    assert np.all(bar_cost[1] < bar_cost[0])

    all_background_image = np.array([[1.0, 2.0]])
    np.testing.assert_array_equal(
        bellman_ford_source_oracle(
            all_background_image,
            np.ones_like(all_background_image, dtype=bool),
            8,
        ),
        all_background_image,
    )
    np.testing.assert_array_equal(
        bellman_ford_source_oracle(
            all_background_image,
            np.zeros_like(all_background_image, dtype=bool),
            8,
        ),
        SOURCE_INFINITY,
    )


def verify_failures() -> None:
    """Prove deterministic rejection of invalid Python-boundary inputs."""
    invalid_images = (
        np.empty((0, 2)),
        np.array([[0.0, -1.0]]),
        np.array([[0.0, np.nan]]),
        np.array([[0.0, np.inf]]),
    )
    for image in invalid_images:
        try:
            validate_inputs(image, np.zeros(image.shape, dtype=bool))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid image accepted: {image!r}")

    try:
        validate_inputs(np.ones((1, 2)), np.zeros((1, 2), dtype=np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError("non-boolean mask accepted")

    try:
        validate_inputs(np.ones((1, 2)), np.zeros((2, 1), dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched mask shape accepted")


if __name__ == "__main__":
    tolerance_evidence = verify_source_fixture()
    verify_behavioral_controls()
    verify_failures()
    print("Assumptions: destination intensity; exact APP2 frontier; float32 source maps")
    for case, connectivity, error, bound in tolerance_evidence:
        print(
            f"{case} connectivity={connectivity}: max_error={error:.9g}, "
            f"derived_float32_bound={bound:.9g}"
        )
    print("GWDT logic validation passed: source maps, exact lookup, controls, failures")
