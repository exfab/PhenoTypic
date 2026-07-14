"""Source-fidelity and behavioral tests for TrickTrack CA evolution."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from phenotypic.sdk_.reconnect._cellular_automaton import (
    TrickTrackCAResult,
    _apply_immediate_root_pass,
    _apply_synchronous_round,
    _extract_exact_length_paths,
    _find_first_equal_neighbors,
    tricktrack_ca,
)


_FIXTURE_PATH = Path(
    "tests/fixtures/reconnect/cellular_automaton/tricktrack_source.json"
)
_MANIFEST_PATH = _FIXTURE_PATH.with_name("manifest.json")


def _csr(neighbor_lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Build int64 CSR while preserving the nested-list order."""
    offsets = [0]
    indices: list[int] = []
    for neighbors in neighbor_lists:
        indices.extend(neighbors)
        offsets.append(len(indices))
    return np.asarray(offsets, dtype=np.int64), np.asarray(indices, dtype=np.int64)


def _result_paths(result: TrickTrackCAResult) -> list[list[int]]:
    """Expand a result's ragged path representation for assertions."""
    return [
        result.path_cell_indices[result.path_offsets[i] : result.path_offsets[i + 1]]
        .astype(int)
        .tolist()
        for i in range(result.path_offsets.size - 1)
    ]


def test_source_generated_fixture_matches_every_trace_and_result_field() -> None:
    """The port matches all instrumented source states, flags, roots, and paths."""
    fixture_bytes = _FIXTURE_PATH.read_bytes()
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture_bytes).hexdigest() == manifest["fixture_sha256"]
    fixture = json.loads(fixture_bytes)
    assert fixture["source_commit"] == "b164fad1361505ff8dbf328107b645753ce331ac"

    for case in fixture["cases"]:
        offsets = np.asarray(case["outer_neighbor_offsets"], dtype=np.int64)
        neighbors = np.asarray(case["outer_neighbor_indices"], dtype=np.int64)
        roots = np.asarray(case["root_cell_indices"], dtype=np.int64)
        states = np.zeros(offsets.size - 1, dtype=np.uint8)

        observed_matches: list[list[int]] = []
        observed_flags: list[list[int]] = []
        observed_states: list[list[int]] = []
        for _ in range(case["ordinary_rounds"]):
            matches, flags = _apply_synchronous_round(states, offsets, neighbors)
            observed_matches.append(matches.tolist())
            observed_flags.append(flags.astype(int).tolist())
            observed_states.append(states.astype(int).tolist())

        assert observed_matches == case["ordinary_first_equal_neighbors"]
        assert observed_flags == case["ordinary_flags"]
        assert observed_states == case["ordinary_states"]

        retained, root_matches, root_flags, root_states = _apply_immediate_root_pass(
            states, offsets, neighbors, roots, case["min_hits_per_track"] - 2
        )
        assert root_matches.tolist() == case["root_first_equal_neighbors"]
        assert root_flags.astype(int).tolist() == case["root_flags"]
        assert root_states.astype(int).tolist() == case["states_after_each_root"]
        assert retained.tolist() == case["retained_root_indices"]

        path_offsets, path_cells = _extract_exact_length_paths(
            offsets,
            neighbors,
            retained,
            case["min_hits_per_track"] - 1,
        )
        assert path_offsets.tolist() == case["path_offsets"]
        assert path_cells.tolist() == case["path_cell_indices"]

        result = tricktrack_ca(
            offsets,
            neighbors,
            roots,
            min_hits_per_track=case["min_hits_per_track"],
        )
        assert result.ordinary_rounds == case["ordinary_rounds"]
        assert result.states.dtype == np.uint8
        assert result.states.astype(int).tolist() == case["final_states"]
        assert result.retained_root_indices.tolist() == case["retained_root_indices"]
        assert result.path_offsets.tolist() == case["path_offsets"]
        assert result.path_cell_indices.tolist() == case["path_cell_indices"]


def test_ordinary_updates_are_synchronous_not_in_place() -> None:
    offsets, neighbors = _csr([[1], [2], []])
    states = np.zeros(3, dtype=np.uint8)

    _apply_synchronous_round(states, offsets, neighbors)

    assert_array_equal(states, np.array([1, 1, 0], dtype=np.uint8))


def test_first_equal_scan_stops_at_first_stored_neighbor() -> None:
    offsets, neighbors = _csr([[2, 1], [], []])
    states = np.array([4, 4, 4], dtype=np.uint8)

    matches = _find_first_equal_neighbors(
        states, offsets, neighbors, np.array([0], dtype=np.int64)
    )

    assert_array_equal(matches, np.array([2], dtype=np.int64))


def test_evolution_scans_stored_outer_not_transposed_neighbors() -> None:
    offsets, neighbors = _csr([[1], [], []])

    result = tricktrack_ca(
        offsets,
        neighbors,
        np.array([0], dtype=np.int64),
        min_hits_per_track=4,
    )

    assert_array_equal(result.states, np.array([1, 0, 0], dtype=np.uint8))
    assert result.retained_root_indices.size == 0


def test_fixed_round_count_does_not_run_to_convergence() -> None:
    offsets, neighbors = _csr([[1], [2], [3], []])

    result = tricktrack_ca(
        offsets,
        neighbors,
        np.array([0], dtype=np.int64),
        min_hits_per_track=4,
    )

    assert result.ordinary_rounds == 1
    assert_array_equal(result.states, np.array([2, 1, 1, 0], dtype=np.uint8))


def test_root_pass_is_immediate_and_preserves_supplied_order() -> None:
    offsets, neighbors = _csr([[2], [0], [3], []])
    roots = np.array([0, 1], dtype=np.int64)

    result = tricktrack_ca(offsets, neighbors, roots, min_hits_per_track=4)
    reversed_result = tricktrack_ca(
        offsets, neighbors, roots[::-1].copy(), min_hits_per_track=4
    )

    assert_array_equal(result.states, np.array([2, 1, 1, 0], dtype=np.uint8))
    assert result.retained_root_indices.tolist() == [0]
    assert_array_equal(
        reversed_result.states, np.array([2, 2, 1, 0], dtype=np.uint8)
    )
    assert reversed_result.retained_root_indices.tolist() == [1, 0]


def test_root_threshold_is_inclusive() -> None:
    offsets, neighbors = _csr([[1], []])

    result = tricktrack_ca(
        offsets,
        neighbors,
        np.array([0], dtype=np.int64),
        min_hits_per_track=3,
    )

    assert result.states[0] == 1
    assert result.retained_root_indices.tolist() == [0]
    assert _result_paths(result) == [[0, 1]]


def test_dfs_preserves_neighbor_order_and_emits_every_exact_length_fork() -> None:
    offsets, neighbors = _csr(
        [[2, 1], [5], [4, 3], [7], [6], [8], [], [], []]
    )

    result = tricktrack_ca(
        offsets,
        neighbors,
        np.array([0], dtype=np.int64),
        min_hits_per_track=5,
    )

    assert _result_paths(result) == [
        [0, 2, 4, 6],
        [0, 2, 3, 7],
        [0, 1, 5, 8],
    ]
    assert all(len(path) == 4 for path in _result_paths(result))


def test_dfs_does_not_enforce_descending_states_or_cycle_rejection() -> None:
    offsets, neighbors = _csr([[1], [0]])

    result = tricktrack_ca(
        offsets,
        neighbors,
        np.array([0], dtype=np.int64),
        min_hits_per_track=4,
    )

    assert_array_equal(result.states, np.array([2, 1], dtype=np.uint8))
    assert _result_paths(result) == [[0, 1, 0]]


def test_upper_minimum_preserves_uint8_state_and_full_exact_path() -> None:
    number_of_cells = 256
    offsets, neighbors = _csr(
        [[cell + 1] if cell + 1 < number_of_cells else [] for cell in range(256)]
    )

    result = tricktrack_ca(
        offsets,
        neighbors,
        np.array([0], dtype=np.int64),
        min_hits_per_track=257,
    )

    assert result.ordinary_rounds == 254
    assert result.states.dtype == np.uint8
    assert result.states[0] == 255
    assert result.retained_root_indices.tolist() == [0]
    assert_array_equal(result.path_offsets, np.array([0, 256], dtype=np.int64))
    assert_array_equal(result.path_cell_indices, np.arange(256, dtype=np.int64))


def test_empty_graph_and_isolated_root_have_deterministic_empty_results() -> None:
    empty = tricktrack_ca(
        np.array([0], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        min_hits_per_track=3,
    )
    isolated = tricktrack_ca(
        np.array([0, 0], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([0], dtype=np.int64),
        min_hits_per_track=3,
    )

    for result, expected_states in ((empty, []), (isolated, [0])):
        assert result.states.astype(int).tolist() == expected_states
        assert result.retained_root_indices.size == 0
        assert_array_equal(result.path_offsets, np.array([0], dtype=np.int64))
        assert result.path_cell_indices.size == 0


def test_inputs_are_not_mutated_or_aliased_by_outputs() -> None:
    offsets, neighbors = _csr([[1], []])
    roots = np.array([0], dtype=np.int64)
    before = (offsets.copy(), neighbors.copy(), roots.copy())

    result = tricktrack_ca(offsets, neighbors, roots, min_hits_per_track=3)
    result.retained_root_indices[0] = 1

    for actual, expected in zip((offsets, neighbors, roots), before, strict=True):
        assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("offsets", "neighbors", "roots", "minimum", "error", "message"),
    [
        (
            [0],
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            3,
            TypeError,
            "outer_neighbor_offsets must be a numpy.ndarray",
        ),
        (
            np.array([0], dtype=np.int32),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            3,
            TypeError,
            "outer_neighbor_offsets must have dtype int64",
        ),
        (
            np.array([[0]], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            3,
            ValueError,
            "outer_neighbor_offsets must be one-dimensional",
        ),
        (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            3,
            ValueError,
            "must contain at least one offset",
        ),
        (
            np.array([1], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            3,
            ValueError,
            "must start at zero",
        ),
        (
            np.array([0, 2, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int64),
            3,
            ValueError,
            "must be nondecreasing",
        ),
        (
            np.array([0, 0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int64),
            3,
            ValueError,
            "must end at the number",
        ),
        (
            np.array([0, 1], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([], dtype=np.int64),
            3,
            ValueError,
            "out-of-range cell index",
        ),
        (
            np.array([0, 0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            3,
            ValueError,
            "root_cell_indices contains an out-of-range",
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            True,
            ValueError,
            "must be an integer from 3 through 257",
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            2,
            ValueError,
            "must be an integer from 3 through 257",
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            258,
            ValueError,
            "must be an integer from 3 through 257",
        ),
    ],
)
def test_invalid_inputs_raise(
    offsets: object,
    neighbors: np.ndarray,
    roots: np.ndarray,
    minimum: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        tricktrack_ca(  # type: ignore[arg-type]
            offsets,
            neighbors,
            roots,
            min_hits_per_track=minimum,
        )
