"""Independently validate the pinned TrickTrack CA numeric and ordering claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIRECTORY = REPOSITORY_ROOT / "tests/fixtures/reconnect/cellular_automaton"


def _load_verified_fixture() -> dict[str, object]:
    """Read the source fixture only after checking its committed byte hash."""
    manifest = json.loads(
        (FIXTURE_DIRECTORY / "manifest.json").read_text(encoding="utf-8")
    )
    fixture_bytes = (FIXTURE_DIRECTORY / manifest["fixture"]).read_bytes()
    observed_hash = hashlib.sha256(fixture_bytes).hexdigest()
    if observed_hash != manifest["fixture_sha256"]:
        raise AssertionError("cellular-automaton fixture checksum mismatch")
    fixture = json.loads(fixture_bytes)
    if fixture["source_commit"] != manifest["source_commit"]:
        raise AssertionError("fixture and manifest source revisions disagree")
    return fixture


def _ordered_adjacency(case: dict[str, object]) -> list[tuple[int, ...]]:
    """Expand CSR to immutable tuples without calling production code."""
    offsets = [int(value) for value in case["outer_neighbor_offsets"]]
    indices = [int(value) for value in case["outer_neighbor_indices"]]
    return [
        tuple(indices[offsets[cell] : offsets[cell + 1]])
        for cell in range(len(offsets) - 1)
    ]


def _independent_oracle(case: dict[str, object]) -> dict[str, object]:
    """Use immutable snapshots and an iterative DFS to re-derive every trace."""
    adjacency = _ordered_adjacency(case)
    minimum = int(case["min_hits_per_track"])
    states = np.zeros(len(adjacency), dtype=np.uint8)
    ordinary_matches: list[list[int]] = []
    ordinary_flags: list[list[int]] = []
    ordinary_states: list[list[int]] = []

    for _ in range(minimum - 3):
        snapshot = tuple(int(value) for value in states)
        matches = [
            next(
                (
                    neighbor
                    for neighbor in adjacency[cell]
                    if snapshot[neighbor] == snapshot[cell]
                ),
                -1,
            )
            for cell in range(len(adjacency))
        ]
        increments = np.fromiter(
            (match >= 0 for match in matches), dtype=np.uint8, count=len(matches)
        )
        states = np.add(states, increments, dtype=np.uint8)
        ordinary_matches.append(matches)
        ordinary_flags.append(increments.astype(int).tolist())
        ordinary_states.append(states.astype(int).tolist())

    root_matches: list[int] = []
    root_flags: list[int] = []
    states_after_roots: list[int] = []
    retained: list[int] = []
    for raw_root in case["root_cell_indices"]:
        root = int(raw_root)
        match = next(
            (
                neighbor
                for neighbor in adjacency[root]
                if states[neighbor] == states[root]
            ),
            -1,
        )
        flag = int(match >= 0)
        states[root] = np.uint8((int(states[root]) + flag) % 256)
        root_matches.append(match)
        root_flags.append(flag)
        states_after_roots.append(int(states[root]))
        if int(states[root]) >= minimum - 2:
            retained.append(root)

    path_length = minimum - 1
    paths: list[list[int]] = []
    for root in retained:
        stack: list[tuple[int, tuple[int, ...]]] = [(root, (root,))]
        while stack:
            cell, path = stack.pop()
            if len(path) == path_length:
                paths.append(list(path))
                continue
            for neighbor in reversed(adjacency[cell]):
                stack.append((neighbor, (*path, neighbor)))

    return {
        "final_states": states.astype(int).tolist(),
        "ordinary_first_equal_neighbors": ordinary_matches,
        "ordinary_flags": ordinary_flags,
        "ordinary_rounds": minimum - 3,
        "ordinary_states": ordinary_states,
        "path_cell_indices": [cell for path in paths for cell in path],
        "path_offsets": [index * path_length for index in range(len(paths) + 1)],
        "retained_root_indices": retained,
        "root_first_equal_neighbors": root_matches,
        "root_flags": root_flags,
        "states_after_each_root": states_after_roots,
    }


def validate_cellular_automaton_claims() -> None:
    """Check exact source traces and load-bearing scheduling counterexamples."""
    fixture = _load_verified_fixture()
    checked_fields = {
        "final_states",
        "ordinary_first_equal_neighbors",
        "ordinary_flags",
        "ordinary_rounds",
        "ordinary_states",
        "path_cell_indices",
        "path_offsets",
        "retained_root_indices",
        "root_first_equal_neighbors",
        "root_flags",
        "states_after_each_root",
    }
    cases = {case["name"]: case for case in fixture["cases"]}
    for name, case in cases.items():
        observed = _independent_oracle(case)
        for field in checked_fields:
            if observed[field] != case[field]:
                raise AssertionError(f"{name}: source mismatch in {field}")

    immediate = cases["immediate-root-order"]
    if immediate["final_states"] != [2, 1, 1, 0]:
        raise AssertionError("immediate root-pass counterexample was weakened")
    if immediate["ordinary_states"] != [[1, 1, 1, 0]]:
        raise AssertionError("synchronous ordinary-round counterexample was weakened")
    upper = cases["upper-bound-chain"]
    if upper["final_states"][0] != 255 or upper["ordinary_rounds"] != 254:
        raise AssertionError("uint8 upper-bound claim was not re-derived")


if __name__ == "__main__":
    validate_cellular_automaton_claims()
    print("A06 cellular-automaton logic validation passed")
