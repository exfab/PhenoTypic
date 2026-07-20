"""Inject each GWDT core mutant separately and execute its named killing probe."""

from __future__ import annotations

import importlib.util
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[6]
SOURCE = ROOT / "src/phenotypic/sdk_/reconnect/_gwdt.py"
FIXTURE = ROOT / "tests/fixtures/reconnect/gwdt/app2_source.npz"


def load_module(path: Path, name: str) -> ModuleType:
    """Load one isolated helper module."""
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def replace_once(source: str, old: str, new: str) -> str:
    """Apply exactly one textual mutation site."""
    if source.count(old) != 1:
        raise RuntimeError(f"mutation site count is {source.count(old)}, expected 1: {old!r}")
    return source.replace(old, new, 1)


def assert_distance(
    module: ModuleType,
    case_name: str,
    image_key: str,
    background_key: str,
    connectivity: int,
) -> None:
    """Assert one complete distance map against the source fixture."""
    with np.load(FIXTURE) as fixture:
        actual = module.grey_weighted_distance(
            fixture[image_key],
            fixture[background_key],
            connectivity=connectivity,
        )
        expected = fixture[f"source_{case_name}_distance_{connectivity}"]
    np.testing.assert_array_equal(actual, expected)


def assert_standard_4(module: ModuleType) -> None:
    assert_distance(module, "standard", "image", "background", 4)


def assert_standard_8(module: ModuleType) -> None:
    assert_distance(module, "standard", "image", "background", 8)


def assert_threshold_4(module: ModuleType) -> None:
    assert_distance(module, "threshold", "threshold_image", "threshold_background", 4)


def assert_diagonal_4(module: ModuleType) -> None:
    assert_distance(module, "diagonal", "diagonal_image", "diagonal_background", 4)


def assert_diagonal_8(module: ModuleType) -> None:
    assert_distance(module, "diagonal", "diagonal_image", "diagonal_background", 8)


def assert_post_diagonal_8(module: ModuleType) -> None:
    assert_distance(
        module,
        "post_frontier_diagonal",
        "post_frontier_diagonal_image",
        "post_frontier_diagonal_background",
        8,
    )


def assert_all_background(module: ModuleType) -> None:
    assert_distance(
        module,
        "all_background",
        "all_background_image",
        "all_background",
        8,
    )


def assert_no_background(module: ModuleType) -> None:
    assert_distance(
        module,
        "no_background",
        "no_background_image",
        "no_background",
        8,
    )


def assert_cost(module: ModuleType) -> None:
    with np.load(FIXTURE) as fixture:
        actual = module.app2_gwdt_cost(fixture["source_standard_distance_8"])
        expected = fixture["source_standard_cost_8"]
    np.testing.assert_array_equal(actual, expected)


def assert_invalid_values(module: ModuleType) -> None:
    for image in (np.array([[0.0, -1.0]]), np.array([[0.0, np.nan]])):
        try:
            module.grey_weighted_distance(image, np.array([[True, False]]))
        except ValueError:
            continue
        raise AssertionError("invalid input was accepted")


def assert_increasing_cost(module: ModuleType) -> None:
    expected = np.array([[22026.5, 85.1526], [3.03773, 1.0]])
    np.testing.assert_array_equal(
        module.app2_gwdt_cost(np.arange(1.0, 5.0).reshape(2, 2)),
        expected,
    )


Mutation = tuple[str, str, str, Callable[[ModuleType], None]]


MUTATIONS: tuple[Mutation, ...] = (
    (
        "M01",
        "seeds = _validated_background(background, values.shape)",
        "seeds = ~_validated_background(background, values.shape)",
        assert_standard_8,
    ),
    (
        "M02",
        "source_values: NDArray[np.float32] = values.astype(np.float32)",
        "source_values: NDArray[np.float32] = "
        "np.reciprocal(values + 1e-6).astype(np.float32)",
        assert_standard_4,
    ),
    (
        "M03",
        "float(source_values[neighbor_row, neighbor_column]) * step_length",
        "float(source_values[row, column]) * step_length",
        assert_standard_8,
    ),
    (
        "M04",
        "+ float(source_values[row, column])\n            )",
        "+ 0.5 * (float(source_values[row, column]) + "
        "float(source_values[minimum_row, minimum_column]))\n            )",
        assert_threshold_4,
    ),
    (
        "M05",
        "+ float(source_values[row, column])\n            )",
        "+ float(source_values[row, column]) * step_length\n            )",
        assert_diagonal_8,
    ),
    (
        "M06",
        "float(source_values[neighbor_row, neighbor_column]) * step_length",
        "float(source_values[neighbor_row, neighbor_column]) * 1.0",
        assert_post_diagonal_8,
    ),
    (
        "M07",
        "neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8",
        "neighbors = _NEIGHBORS_4",
        assert_standard_8,
    ),
    (
        "M08",
        "distance\n                + float(source_values[neighbor_row, neighbor_column])",
        "distance\n                * float(source_values[neighbor_row, neighbor_column])",
        assert_standard_4,
    ),
    (
        "M10",
        "source_values: NDArray[np.float32] = values.astype(np.float32)",
        "source_values: NDArray[np.float32] = "
        "(values / np.max(values)).astype(np.float32)",
        assert_standard_4,
    ),
    (
        "M11",
        "return _GIVALS[lookup_indices]",
        "return _GIVALS[255 - lookup_indices]",
        assert_cost,
    ),
    (
        "M12",
        "if not np.all(np.isfinite(values)):\n"
        "        raise ValueError(f\"{name} must contain only finite values\")\n"
        "    if np.any(values < 0.0):\n"
        "        raise ValueError(f\"{name} must contain only nonnegative values\")",
        "if False:\n"
        "        raise ValueError(f\"{name} must contain only finite values\")\n"
        "    if False:\n"
        "        raise ValueError(f\"{name} must contain only nonnegative values\")",
        assert_invalid_values,
    ),
    (
        "M13",
        "neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8",
        "neighbors = _NEIGHBORS_8",
        assert_diagonal_4,
    ),
    (
        "M14",
        "distances[seeds] = source_values[seeds]",
        "distances[seeds] = 0.0",
        assert_all_background,
    ),
    (
        "M15",
        "_SOURCE_INFINITY = np.float32(1e20)",
        "_SOURCE_INFINITY = np.float32(np.inf)",
        assert_no_background,
    ),
    (
        "M16",
        "minimum = float(np.min(values))\n    span = float(np.max(values)) - minimum",
        "maximum = 0.0\n"
        "    minimum = 1e20\n"
        "    for value in values.flat:\n"
        "        if value > maximum:\n"
        "            maximum = float(value)\n"
        "        elif value < minimum:\n"
        "            minimum = float(value)\n"
        "    span = maximum - minimum",
        assert_increasing_cost,
    ),
)


def assert_threshold_policy_mutant(module: ModuleType) -> None:
    """Inject the `<` policy at the mask seam and prove equality is load-bearing."""
    with np.load(FIXTURE) as fixture:
        image = fixture["threshold_image"]
        mutated_mask = image < 2
        expected = fixture["source_threshold_distance_4"]
    actual = module.grey_weighted_distance(image, mutated_mask, connectivity=4)
    np.testing.assert_array_equal(actual, expected)


def execute_mutations() -> None:
    """Run baselines, then prove every isolated mutant is killed."""
    source = SOURCE.read_text(encoding="utf-8")
    original = load_module(SOURCE, "gwdt_mutation_baseline")
    for _, _, _, probe in MUTATIONS:
        probe(original)
    assert_threshold_4(original)

    results = []
    with tempfile.TemporaryDirectory(prefix="phenotypic-gwdt-mutants-") as temporary:
        directory = Path(temporary)
        for mutant_id, old, new, probe in MUTATIONS:
            mutant_path = directory / f"{mutant_id.lower()}.py"
            mutant_path.write_text(replace_once(source, old, new), encoding="utf-8")
            mutant = load_module(mutant_path, f"gwdt_{mutant_id.lower()}")
            try:
                probe(mutant)
            except Exception as error:
                results.append((mutant_id, "KILLED", type(error).__name__))
            else:
                raise AssertionError(f"{mutant_id} survived")

        try:
            assert_threshold_policy_mutant(original)
        except AssertionError as error:
            results.append(("M09", "KILLED", type(error).__name__))
        else:
            raise AssertionError("M09 survived")

    for mutant_id, status, reason in sorted(results):
        print(f"{mutant_id}: {status} ({reason})")


if __name__ == "__main__":
    execute_mutations()
