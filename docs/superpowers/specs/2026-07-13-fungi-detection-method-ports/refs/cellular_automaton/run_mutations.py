"""Apply required A06 mutants to temporary copies and prove focused tests kill them."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
PRODUCTION_MODULE = (
    REPOSITORY_ROOT / "src/phenotypic/sdk_/reconnect/_cellular_automaton.py"
)
TEST_MODULE = "tests/unit/sdk_/reconnect/test_cellular_automaton.py"


@dataclass(frozen=True)
class Mutation:
    """One exact source replacement and the focused test expected to kill it."""

    name: str
    old: str
    new: str
    test: str


MUTATIONS = (
    Mutation(
        "ordinary-updates-in-place",
        """    first_matches = _find_first_equal_neighbors(
        states, outer_neighbor_offsets, outer_neighbor_indices, cell_indices
    )
    flags = (first_matches >= 0).astype(np.uint8)
    states += flags
""",
        """    first_matches = np.full(states.size, -1, dtype=np.int64)
    flags = np.zeros(states.size, dtype=np.uint8)
    for cell_index in cell_indices:
        match = _find_first_equal_neighbors(
            states,
            outer_neighbor_offsets,
            outer_neighbor_indices,
            cell_indices[cell_index : cell_index + 1],
        )[0]
        first_matches[cell_index] = match
        flags[cell_index] = match >= 0
        states[cell_index] += flags[cell_index]
""",
        "test_source_generated_fixture_matches_every_trace_and_result_field",
    ),
    Mutation(
        "scan-transposed-neighbors",
        """        first = int(outer_neighbor_offsets[cell_index])
        last = int(outer_neighbor_offsets[cell_index + 1])
        for neighbor_position in range(first, last):
            neighbor_index = int(outer_neighbor_indices[neighbor_position])
            if states[neighbor_index] == state:
                first_matches[position] = neighbor_index
                break
""",
        """        for possible_inner in range(states.size):
            first = int(outer_neighbor_offsets[possible_inner])
            last = int(outer_neighbor_offsets[possible_inner + 1])
            if cell_index in outer_neighbor_indices[first:last]:
                neighbor_index = possible_inner
                if states[neighbor_index] == state:
                    first_matches[position] = neighbor_index
                    break
""",
        "test_evolution_scans_stored_outer_not_transposed_neighbors",
    ),
    Mutation(
        "retain-last-equal-neighbor",
        """                first_matches[position] = neighbor_index
                break
""",
        """                first_matches[position] = neighbor_index
                continue
""",
        "test_first_equal_scan_stops_at_first_stored_neighbor",
    ),
    Mutation(
        "one-extra-ordinary-round",
        "ordinary_rounds = min_hits - 3",
        "ordinary_rounds = min_hits - 2",
        "test_fixed_round_count_does_not_run_to_convergence",
    ),
    Mutation(
        "root-flags-from-global-snapshot",
        """    for position, raw_root_index in enumerate(root_cell_indices):
        root_index = int(raw_root_index)
        match = _find_first_equal_neighbors(
            states,
""",
        """    root_snapshot = states.copy()
    for position, raw_root_index in enumerate(root_cell_indices):
        root_index = int(raw_root_index)
        match = _find_first_equal_neighbors(
            root_snapshot,
""",
        "test_root_pass_is_immediate_and_preserves_supplied_order",
    ),
    Mutation(
        "exclusive-root-threshold",
        "if states[root_index] >= minimum_state:",
        "if states[root_index] > minimum_state:",
        "test_root_threshold_is_inclusive",
    ),
    Mutation(
        "sort-root-pass",
        "for position, raw_root_index in enumerate(root_cell_indices):",
        "for position, raw_root_index in enumerate(np.sort(root_cell_indices)):",
        "test_root_pass_is_immediate_and_preserves_supplied_order",
    ),
    Mutation(
        "sort-dfs-neighbors",
        """        for neighbor_position in range(first, last):
            neighbor_index = int(outer_neighbor_indices[neighbor_position])
            current_path.append(neighbor_index)
""",
        """        positions = sorted(
            range(first, last),
            key=lambda position: int(outer_neighbor_indices[position]),
        )
        for neighbor_position in positions:
            neighbor_index = int(outer_neighbor_indices[neighbor_position])
            current_path.append(neighbor_index)
""",
        "test_dfs_preserves_neighbor_order_and_emits_every_exact_length_fork",
    ),
    Mutation(
        "keep-first-fork-only",
        "for neighbor_position in range(first, last):\n            neighbor_index = int(outer_neighbor_indices[neighbor_position])\n            current_path.append",
        "for neighbor_position in range(first, min(first + 1, last)):\n            neighbor_index = int(outer_neighbor_indices[neighbor_position])\n            current_path.append",
        "test_dfs_preserves_neighbor_order_and_emits_every_exact_length_fork",
    ),
    Mutation(
        "continue-beyond-exact-length",
        "if len(current_path) == path_length:",
        "if len(current_path) > path_length:",
        "test_dfs_preserves_neighbor_order_and_emits_every_exact_length_fork",
    ),
    Mutation(
        "promote-states-to-int64",
        "states = np.zeros(offsets.size - 1, dtype=np.uint8)",
        "states = np.zeros(offsets.size - 1, dtype=np.int64)",
        "test_upper_minimum_preserves_uint8_state_and_full_exact_path",
    ),
    Mutation(
        "extract-one-extra-cell",
        "offsets, neighbors, retained_roots, min_hits - 1",
        "offsets, neighbors, retained_roots, min_hits",
        "test_dfs_preserves_neighbor_order_and_emits_every_exact_length_fork",
    ),
)


def _mutated_source(source: str, mutation: Mutation) -> str:
    """Apply exactly one mutation and reject ambiguous source matches."""
    count = source.count(mutation.old)
    if count != 1:
        raise RuntimeError(f"{mutation.name}: expected one source match, found {count}")
    return source.replace(mutation.old, mutation.new, 1)


def prove_required_mutants_are_killed() -> None:
    """Run one named focused test per mutant without modifying the worktree."""
    source = PRODUCTION_MODULE.read_text(encoding="utf-8")
    failures: list[str] = []
    for mutation in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix=f"a06-{mutation.name}-") as temporary:
            temporary_path = Path(temporary)
            mutant = temporary_path / "mutant.py"
            mutant.write_text(_mutated_source(source, mutation), encoding="utf-8")
            bootstrap = temporary_path / "sitecustomize.py"
            bootstrap.write_text(
                "import importlib.util\n"
                "import sys\n"
                f"path = {str(mutant)!r}\n"
                "name = 'phenotypic.sdk_.reconnect._cellular_automaton'\n"
                "spec = importlib.util.spec_from_file_location(name, path)\n"
                "module = importlib.util.module_from_spec(spec)\n"
                "sys.modules[name] = module\n"
                "spec.loader.exec_module(module)\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            existing_path = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = str(temporary_path) + (
                os.pathsep + existing_path if existing_path else ""
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    f"{TEST_MODULE}::{mutation.test}",
                    "-q",
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        if completed.returncode == 0:
            failures.append(mutation.name)
            print(f"SURVIVED {mutation.name}: {mutation.test}")
        else:
            print(f"KILLED   {mutation.name}: {mutation.test}")
    if failures:
        raise SystemExit("surviving A06 mutants: " + ", ".join(failures))


if __name__ == "__main__":
    prove_required_mutants_are_killed()
