"""Validate the A11 cubical-persistence numerical contract independently.

This script uses only the standard library and NumPy. It does not import GUDHI,
SciPy, scikit-image, or ``phenotypic``. Persistence is recomputed by explicitly
building the closed 2-D cubical complex and reducing its boundary matrix over
``F_2`` with Python integer bitsets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
FIXTURE_PATH = (
    ROOT
    / "docs/superpowers/specs/2026-07-13-fungi-detection-method-ports"
    / "refs/persistence/fixture.json"
)
Filtration = Literal["sublevel", "superlevel"]


@dataclass(frozen=True)
class Cell:
    """One cell in a closed rectangular cubical complex."""

    kind: Literal["vertex", "horizontal", "vertical", "square"]
    row: int
    column: int
    dimension: int
    filtration_value: float


def _incident_pixel_minimum(
    image: np.ndarray, pixel_coordinates: list[tuple[int, int]]
) -> float:
    """Return the lower-star filtration value of a face."""
    height, width = image.shape
    values = [
        image[row, column]
        for row, column in pixel_coordinates
        if 0 <= row < height and 0 <= column < width
    ]
    if not values:
        raise AssertionError("Every cell must be incident to a top-dimensional cell")
    return float(min(values))


def _build_cells(image: np.ndarray) -> list[Cell]:
    """Enumerate and filtration-sort all vertices, edges, and squares."""
    height, width = image.shape
    cells: list[Cell] = []
    for row in range(height + 1):
        for column in range(width + 1):
            value = _incident_pixel_minimum(
                image,
                [
                    (row - 1, column - 1),
                    (row - 1, column),
                    (row, column - 1),
                    (row, column),
                ],
            )
            cells.append(Cell("vertex", row, column, 0, value))
    for row in range(height + 1):
        for column in range(width):
            value = _incident_pixel_minimum(
                image, [(row - 1, column), (row, column)]
            )
            cells.append(Cell("horizontal", row, column, 1, value))
    for row in range(height):
        for column in range(width + 1):
            value = _incident_pixel_minimum(
                image, [(row, column - 1), (row, column)]
            )
            cells.append(Cell("vertical", row, column, 1, value))
    for row in range(height):
        for column in range(width):
            cells.append(Cell("square", row, column, 2, float(image[row, column])))

    kind_order = {"vertex": 0, "horizontal": 1, "vertical": 2, "square": 3}
    cells.sort(
        key=lambda cell: (
            cell.filtration_value,
            cell.dimension,
            kind_order[cell.kind],
            cell.row,
            cell.column,
        )
    )
    return cells


def _boundary_keys(cell: Cell) -> tuple[tuple[str, int, int], ...]:
    """Return the boundary-cell keys of one cell over ``F_2``."""
    row, column = cell.row, cell.column
    if cell.kind == "vertex":
        return ()
    if cell.kind == "horizontal":
        return (("vertex", row, column), ("vertex", row, column + 1))
    if cell.kind == "vertical":
        return (("vertex", row, column), ("vertex", row + 1, column))
    return (
        ("horizontal", row, column),
        ("horizontal", row + 1, column),
        ("vertical", row, column),
        ("vertical", row, column + 1),
    )


def _reduce_boundary_matrix(
    filtration_image: np.ndarray,
) -> tuple[list[list[tuple[float, float]]], list[list[float]]]:
    """Compute regular and essential intervals by bitset column reduction."""
    cells = _build_cells(filtration_image)
    positions = {
        (cell.kind, cell.row, cell.column): index
        for index, cell in enumerate(cells)
    }
    reduced_columns: dict[int, int] = {}
    zero_columns: set[int] = set()
    paired_births: set[int] = set()
    regular: list[list[tuple[float, float]]] = [[], []]

    for column_index, cell in enumerate(cells):
        column = 0
        for boundary_key in _boundary_keys(cell):
            row_index = positions[boundary_key]
            if row_index >= column_index:
                raise AssertionError("A boundary cell did not precede its coface")
            column ^= 1 << row_index
        while column:
            low = column.bit_length() - 1
            previous = reduced_columns.get(low)
            if previous is None:
                break
            column ^= previous
        if column:
            low = column.bit_length() - 1
            reduced_columns[low] = column
            paired_births.add(low)
            birth = cells[low]
            if birth.dimension < 2:
                regular[birth.dimension].append(
                    (birth.filtration_value, cell.filtration_value)
                )
        else:
            zero_columns.add(column_index)

    essential: list[list[float]] = [[], []]
    for birth_index in sorted(zero_columns - paired_births):
        birth = cells[birth_index]
        if birth.dimension < 2:
            essential[birth.dimension].append(birth.filtration_value)
    return regular, essential


def _public_intervals(
    image: np.ndarray,
    filtration: Filtration,
    min_persistence: float,
) -> list[list[tuple[float, float, float]]]:
    """Return independent intervals in original image intensity coordinates."""
    filtration_image = image if filtration == "sublevel" else -image
    regular, essential = _reduce_boundary_matrix(filtration_image)
    output: list[list[tuple[float, float, float]]] = [[], []]
    for dimension in range(2):
        for birth_f, death_f in regular[dimension]:
            lifetime = death_f - birth_f
            if lifetime > min_persistence:
                if filtration == "sublevel":
                    output[dimension].append((birth_f, death_f, lifetime))
                else:
                    output[dimension].append((-birth_f, -death_f, lifetime))
        for birth_f in essential[dimension]:
            if filtration == "sublevel":
                output[dimension].append((birth_f, np.inf, np.inf))
            else:
                output[dimension].append((-birth_f, -np.inf, np.inf))
    return output


def _decode_float(value: float | str) -> float:
    """Decode the fixture's standards-compliant infinity strings."""
    if value == "+inf":
        return np.inf
    if value == "-inf":
        return -np.inf
    return float(value)


def _fixture_intervals(case: dict[str, object]) -> list[list[tuple[float, float, float]]]:
    """Extract public interval triples from one fixture case."""
    public = case["public_contract"]
    assert isinstance(public, dict)
    births = public["birth_values"]
    deaths = public["death_values"]
    lifetimes = public["lifetimes"]
    assert isinstance(births, list)
    assert isinstance(deaths, list)
    assert isinstance(lifetimes, list)
    return [
        [
            (_decode_float(birth), _decode_float(death), _decode_float(lifetime))
            for birth, death, lifetime in zip(
                births[dimension], deaths[dimension], lifetimes[dimension], strict=True
            )
        ]
        for dimension in range(2)
    ]


def _canonical(
    intervals: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Sort an interval multiset without depending on representative cells."""
    return sorted(intervals, key=lambda item: tuple(float(value) for value in item))


def _component_count(mask: np.ndarray, connectivity: Literal[4, 8]) -> int:
    """Count binary components with an independent stack flood fill."""
    if connectivity == 4:
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        offsets = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    count = 0
    for start_row, start_column in np.argwhere(mask):
        if seen[start_row, start_column]:
            continue
        count += 1
        stack = [(int(start_row), int(start_column))]
        seen[start_row, start_column] = True
        while stack:
            row, column = stack.pop()
            for dr, dc in offsets:
                rr, cc = row + dr, column + dc
                if (
                    0 <= rr < height
                    and 0 <= cc < width
                    and mask[rr, cc]
                    and not seen[rr, cc]
                ):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
    return count


def _hole_count(mask: np.ndarray) -> int:
    """Count 4-connected background components enclosed by foreground."""
    padded_background = np.pad(~mask, 1, constant_values=True)
    return _component_count(padded_background, 4) - 1


def _betti_from_intervals(
    intervals: list[list[tuple[float, float, float]]],
    filtration: Filtration,
    level: float,
) -> tuple[int, int]:
    """Evaluate beta-0 and beta-1 at one public intensity threshold."""
    counts = []
    for dimension in range(2):
        alive = 0
        for birth, death, _ in intervals[dimension]:
            if filtration == "sublevel":
                alive += int(birth <= level < death)
            else:
                alive += int(birth >= level > death)
        counts.append(alive)
    return counts[0], counts[1]


def _assert_betti_curves(
    image: np.ndarray,
    filtration: Filtration,
    intervals: list[list[tuple[float, float, float]]],
) -> None:
    """Cross-check interval Betti curves against digital topology."""
    for level in np.unique(image):
        foreground = image <= level if filtration == "sublevel" else image >= level
        expected = (
            _component_count(foreground, 8),
            _hole_count(foreground),
        )
        actual = _betti_from_intervals(intervals, filtration, float(level))
        if actual != expected:
            raise AssertionError(
                f"Betti mismatch at {filtration=} {level=}: {actual=} {expected=}"
            )


def validate_persistence_contract() -> None:
    """Re-derive all load-bearing A11 numerical claims and fixture intervals."""
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if fixture["oracle"]["version"] != "3.13.0":
        raise AssertionError("Fixture did not come from pinned GUDHI 3.13.0")

    for name, case in fixture["cases"].items():
        image = np.asarray(case["input"], dtype=np.float64)
        filtration = case["filtration"]
        min_persistence = float(case["min_persistence"])
        independent = _public_intervals(image, filtration, min_persistence)
        expected = _fixture_intervals(case)
        for dimension in range(2):
            if _canonical(independent[dimension]) != _canonical(expected[dimension]):
                raise AssertionError(
                    f"{name} beta-{dimension} interval mismatch: "
                    f"{independent[dimension]} != {expected[dimension]}"
                )
        unthresholded = _public_intervals(image, filtration, 0.0)
        _assert_betti_curves(image, filtration, unthresholded)

    cases = fixture["cases"]
    peaks = _fixture_intervals(cases["four_peaks_superlevel"])
    assert sorted(value[2] for value in peaks[0] if np.isfinite(value[2])) == [
        1.0,
        2.0,
        3.0,
    ]
    assert peaks[1] == [(1.0, 0.0, 1.0)]

    equality = _fixture_intervals(cases["four_peaks_threshold_equality"])
    assert equality[0] == [(4.0, 1.0, 3.0), (5.0, -np.inf, np.inf)]
    assert equality[1] == []

    hole = _fixture_intervals(cases["one_hole_sublevel"])
    assert hole[1] == [(0.0, 2.0, 2.0)]

    diagonal = _fixture_intervals(cases["diagonal_touch_superlevel"])
    assert diagonal[0] == [(2.0, -np.inf, np.inf)]
    assert diagonal[1] == []

    assert _fixture_intervals(cases["single_sublevel"])[0] == [
        (2.5, np.inf, np.inf)
    ]
    assert _fixture_intervals(cases["single_superlevel"])[0] == [
        (2.5, -np.inf, np.inf)
    ]

    plateau = cases["plateau_superlevel"]
    plateau_birth = tuple(plateau["public_contract"]["essential_cells"][0][0])
    plateau_image = np.asarray(plateau["input"], dtype=np.float64)
    assert plateau_image[plateau_birth] == plateau_image.max()

    non_square = cases["non_square_sublevel"]
    essential_id = non_square["source_visible"]["essential_coface_ids"][0][0]
    expected_coordinate = tuple(
        int(value)
        for value in np.unravel_index(
            essential_id, np.asarray(non_square["input"]).shape, order="F"
        )
    )
    actual_coordinate = tuple(non_square["public_contract"]["essential_cells"][0][0])
    assert actual_coordinate == expected_coordinate

    print("A11 persistence logic validation: PASS")
    print("- independent F_2 cubical boundary reduction matches every interval multiset")
    print("- 8-connected foreground / 4-connected background Betti curves agree")
    print("- polarity, essential signs, strict equality, plateaus, and Fortran IDs agree")


if __name__ == "__main__":
    validate_persistence_contract()
