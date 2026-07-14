"""Inject each APP2 detector-seam mutant and run its named killing test."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
SOURCE_RELATIVE = Path("src/phenotypic/detect/_filamentous_fungi_detector.py")
TEST = "tests/unit/detect/test_filamentous_fungi_gwdt_seam.py"


Mutation = tuple[str, str, str, str]


MUTATIONS: tuple[Mutation, ...] = (
    (
        "S01 raw cumulative GWDT",
        "return app2_gwdt_cost(distance)",
        "return distance.astype(np.float64)",
        "test_full_image_adapter_applies_gi_lookup_not_raw_distance",
    ),
    (
        "S02 destination-only GI",
        """edge_cost = (
                (float(costs[row, column]) + float(costs[neighbor_row, neighbor_column]))
                * factor
                / 2.0
            )""",
        """edge_cost = (
                float(costs[neighbor_row, neighbor_column]) * factor
            )""",
        "test_app2_axis_edges_use_endpoint_average_not_destination_cost",
    ),
    (
        "S03 exact square-root diagonal",
        """* factor
                / 2.0""",
        """* (np.sqrt(2.0) if factor != 1.0 else 1.0)
                / 2.0""",
        "test_app2_diagonal_edges_use_pinned_source_factor",
    ),
    (
        "S04 skip full-image transform",
        """app2_gi_cost = self._compute_full_image_app2_gi_cost(
                    enhanced_arr,
                    background=~overall_objmask.astype(np.bool_, copy=False),
            )""",
        "app2_gi_cost = np.ones(enhanced_arr.shape, dtype=np.float64)",
        "test_app2_cost_is_computed_once_on_full_image_before_tiling",
    ),
    (
        "S05 feed GI to legacy kernel",
        "dijkstra = _run_app2_gwdt_dijkstra(tile_app2_gi, tile_colony)",
        "dijkstra = run_multisource_dijkstra(tile_app2_gi, tile_colony, self.delta)",
        "test_tile_dispatch_keeps_app2_separate_from_legacy_dijkstra",
    ),
    (
        "S06 change disabled strategy",
        "if tile_app2_gi is None:",
        "if False:",
        "test_tile_dispatch_keeps_app2_separate_from_legacy_dijkstra",
    ),
)


def replace_once(source: str, old: str, new: str) -> str:
    """Apply one conceptual textual mutation at exactly one site."""
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"mutation site count is {count}, expected 1: {old!r}"
        )
    return source.replace(old, new, 1)


def verify_mutations_are_killed() -> None:
    """Run each mutant in an isolated source-tree copy."""
    original = (ROOT / SOURCE_RELATIVE).read_text(encoding="utf-8")
    for index, (name, old, new, test_name) in enumerate(MUTATIONS):
        with tempfile.TemporaryDirectory(
            prefix="phenotypic-gwdt-seam-"
        ) as temp:
            temporary_root = Path(temp)
            temporary_src = temporary_root / "src"
            shutil.copytree(ROOT / "src", temporary_src)
            mutant_path = temporary_root / SOURCE_RELATIVE
            mutant_path.write_text(
                replace_once(original, old, new),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(temporary_src)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    f"{TEST}::{test_name}",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            output = completed.stdout + completed.stderr
            if (
                completed.returncode == 0
                or " failed" not in output
                or "ERROR" in output
            ):
                raise AssertionError(
                    f"{name} survived or did not reach its assertion:\n{output[-4000:]}"
                )
            print(f"{index + 1}/{len(MUTATIONS)} {name}: KILLED")


if __name__ == "__main__":
    verify_mutations_are_killed()
