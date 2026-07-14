"""Source-faithful cellular-automaton evolution from TrickTrack 1.0.9."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrickTrackCAResult:
    """Results of cellular-automaton evolution and exact-length extraction.

    Attributes:
        states: Final unsigned 8-bit state for every input cell.
        retained_root_indices: Retained cell indices in supplied root order.
        path_offsets: Offsets delimiting paths in ``path_cell_indices``.
        path_cell_indices: Flattened exact-length paths in depth-first order.
        ordinary_rounds: Number of globally synchronous evolution rounds.
    """

    states: np.ndarray
    retained_root_indices: np.ndarray
    path_offsets: np.ndarray
    path_cell_indices: np.ndarray
    ordinary_rounds: int


def _validated_index_array(array: np.ndarray, *, name: str) -> np.ndarray:
    """Validate one one-dimensional int64 index array without copying it."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if array.dtype != np.int64:
        raise TypeError(f"{name} must have dtype int64")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _validate_tricktrack_inputs(
    outer_neighbor_offsets: np.ndarray,
    outer_neighbor_indices: np.ndarray,
    root_cell_indices: np.ndarray,
    min_hits_per_track: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate the Python CSR boundary for the source-faithful core."""
    offsets = _validated_index_array(
        outer_neighbor_offsets, name="outer_neighbor_offsets"
    )
    neighbors = _validated_index_array(
        outer_neighbor_indices, name="outer_neighbor_indices"
    )
    roots = _validated_index_array(root_cell_indices, name="root_cell_indices")

    if offsets.size == 0:
        raise ValueError("outer_neighbor_offsets must contain at least one offset")
    if offsets[0] != 0:
        raise ValueError("outer_neighbor_offsets must start at zero")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("outer_neighbor_offsets must be nondecreasing")
    if offsets[-1] != neighbors.size:
        raise ValueError(
            "outer_neighbor_offsets must end at the number of neighbor indices"
        )

    number_of_cells = offsets.size - 1
    if np.any(neighbors < 0) or np.any(neighbors >= number_of_cells):
        raise ValueError("outer_neighbor_indices contains an out-of-range cell index")
    if np.any(roots < 0) or np.any(roots >= number_of_cells):
        raise ValueError("root_cell_indices contains an out-of-range cell index")

    if (
        isinstance(min_hits_per_track, (bool, np.bool_))
        or not isinstance(min_hits_per_track, (int, np.integer))
        or not 3 <= int(min_hits_per_track) <= 257
    ):
        raise ValueError("min_hits_per_track must be an integer from 3 through 257")

    return offsets, neighbors, roots, int(min_hits_per_track)


def _find_first_equal_neighbors(
    states: np.ndarray,
    outer_neighbor_offsets: np.ndarray,
    outer_neighbor_indices: np.ndarray,
    cell_indices: np.ndarray,
) -> np.ndarray:
    """Return the first equal-state outer neighbor, or ``-1``, for each cell."""
    first_matches = np.full(cell_indices.size, -1, dtype=np.int64)
    for position, raw_cell_index in enumerate(cell_indices):
        cell_index = int(raw_cell_index)
        state = states[cell_index]
        first = int(outer_neighbor_offsets[cell_index])
        last = int(outer_neighbor_offsets[cell_index + 1])
        for neighbor_position in range(first, last):
            neighbor_index = int(outer_neighbor_indices[neighbor_position])
            if states[neighbor_index] == state:
                first_matches[position] = neighbor_index
                break
    return first_matches


def _apply_synchronous_round(
    states: np.ndarray,
    outer_neighbor_offsets: np.ndarray,
    outer_neighbor_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one source-ordered check pass followed by a separate update pass."""
    cell_indices = np.arange(states.size, dtype=np.int64)
    first_matches = _find_first_equal_neighbors(
        states, outer_neighbor_offsets, outer_neighbor_indices, cell_indices
    )
    flags = (first_matches >= 0).astype(np.uint8)
    states += flags
    return first_matches, flags


def _apply_immediate_root_pass(
    states: np.ndarray,
    outer_neighbor_offsets: np.ndarray,
    outer_neighbor_indices: np.ndarray,
    root_cell_indices: np.ndarray,
    minimum_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply and trace TrickTrack's ordered, immediate root-only final pass."""
    first_matches = np.empty(root_cell_indices.size, dtype=np.int64)
    flags = np.empty(root_cell_indices.size, dtype=np.uint8)
    states_after_root = np.empty(root_cell_indices.size, dtype=np.uint8)
    retained: list[int] = []

    for position, raw_root_index in enumerate(root_cell_indices):
        root_index = int(raw_root_index)
        match = _find_first_equal_neighbors(
            states,
            outer_neighbor_offsets,
            outer_neighbor_indices,
            root_cell_indices[position : position + 1],
        )[0]
        first_matches[position] = match
        flag = np.uint8(match >= 0)
        flags[position] = flag
        states[root_index] = np.uint8(
            (int(states[root_index]) + int(flag)) & 0xFF
        )
        states_after_root[position] = states[root_index]
        if states[root_index] >= minimum_state:
            retained.append(root_index)

    return (
        np.asarray(retained, dtype=np.int64),
        first_matches,
        flags,
        states_after_root,
    )


def _extract_exact_length_paths(
    outer_neighbor_offsets: np.ndarray,
    outer_neighbor_indices: np.ndarray,
    retained_root_indices: np.ndarray,
    path_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate exact-length paths in source depth-first insertion order."""
    paths: list[list[int]] = []

    def visit(cell_index: int, current_path: list[int]) -> None:
        if len(current_path) == path_length:
            paths.append(current_path.copy())
            return
        first = int(outer_neighbor_offsets[cell_index])
        last = int(outer_neighbor_offsets[cell_index + 1])
        for neighbor_position in range(first, last):
            neighbor_index = int(outer_neighbor_indices[neighbor_position])
            current_path.append(neighbor_index)
            visit(neighbor_index, current_path)
            current_path.pop()

    for raw_root_index in retained_root_indices:
        root_index = int(raw_root_index)
        visit(root_index, [root_index])

    offsets = np.arange(len(paths) + 1, dtype=np.int64) * path_length
    if not paths:
        return offsets, np.empty(0, dtype=np.int64)
    return offsets, np.asarray(paths, dtype=np.int64).reshape(-1)


def tricktrack_ca(
    outer_neighbor_offsets: np.ndarray,
    outer_neighbor_indices: np.ndarray,
    root_cell_indices: np.ndarray,
    *,
    min_hits_per_track: int,
) -> TrickTrackCAResult:
    """Evolve TrickTrack's CA over a caller-supplied ordered friendship graph.

    Cells and their stored outer-neighbor vectors are represented as int64 CSR.
    Their order and the supplied root order are semantic and are never sorted.
    All states start at zero. Ordinary rounds are globally synchronous, while the
    final root-only pass updates each root immediately before checking the next.
    Extraction then emits every path of exactly ``min_hits_per_track - 1`` cells
    in depth-first neighbor order. Cycles are allowed because extraction has a
    fixed depth; a cell may consequently occur more than once in one path.

    Args:
        outer_neighbor_offsets: Int64 CSR offsets with shape ``(M + 1,)``.
        outer_neighbor_indices: Int64 stored outer-neighbor indices.
        root_cell_indices: Int64 root indices in final-pass order.
        min_hits_per_track: Requested hit count from 3 through 257, inclusive.

    Returns:
        Final uint8 states, retained roots, flattened exact-length paths and
        offsets, and the fixed ordinary-round count. Returned arrays own their
        data and do not alias the inputs.

    Raises:
        TypeError: An index input has the wrong container or dtype.
        ValueError: An input dimension, CSR structure, index, or minimum is invalid.
    """
    offsets, neighbors, roots, min_hits = _validate_tricktrack_inputs(
        outer_neighbor_offsets,
        outer_neighbor_indices,
        root_cell_indices,
        min_hits_per_track,
    )
    states = np.zeros(offsets.size - 1, dtype=np.uint8)
    ordinary_rounds = min_hits - 3

    for _ in range(ordinary_rounds):
        _apply_synchronous_round(states, offsets, neighbors)

    retained_roots, _, _, _ = _apply_immediate_root_pass(
        states,
        offsets,
        neighbors,
        roots,
        min_hits - 2,
    )
    path_offsets, path_cells = _extract_exact_length_paths(
        offsets, neighbors, retained_roots, min_hits - 1
    )
    return TrickTrackCAResult(
        states=states,
        retained_root_indices=retained_roots,
        path_offsets=path_offsets,
        path_cell_indices=path_cells,
        ordinary_rounds=ordinary_rounds,
    )


__all__ = ["TrickTrackCAResult", "tricktrack_ca"]
