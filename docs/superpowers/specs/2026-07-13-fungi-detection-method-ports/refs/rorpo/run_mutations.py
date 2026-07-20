"""Inject isolated RORPO mutants and prove each is killed by a named probe."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[6]
SOURCE = ROOT / "src/phenotypic/sdk_/reconnect/_rorpo.py"
FIXTURE = Path(__file__).with_name("source_fixture.json")


def _load_module(path: Path, name: str) -> ModuleType:
    """Load one isolated implementation module."""
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str) -> str:
    """Apply exactly one textual mutation site."""
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"mutation site count is {count}, expected 1: {old!r}")
    return source.replace(old, new, 1)


def _fixture() -> dict[str, object]:
    """Read the sanitized, source-free fixture."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _case(name: str) -> dict[str, object]:
    """Return one named fixture case."""
    cases = _fixture()["cases"]
    return next(case for case in cases if case["name"] == name)  # type: ignore[union-attr]


def _decode(encoded: dict[str, object]) -> np.ndarray:
    """Decode one fixture array."""
    shape = tuple(int(value) for value in encoded["shape"])  # type: ignore[arg-type]
    return np.asarray(encoded["data"]).reshape(shape)


def _in_bounds(shape: tuple[int, int], row: int, column: int) -> bool:
    return 0 <= row < shape[0] and 0 <= column < shape[1]


def _enumerate_paths(
    mask: np.ndarray,
    steps: tuple[tuple[int, int], ...],
    length: int,
) -> Iterable[tuple[tuple[int, int], ...]]:
    """Enumerate paper-complete paths without using production dynamic programming."""

    def extend(path: tuple[tuple[int, int], ...]):
        if len(path) == length:
            yield path
            return
        row, column = path[-1]
        for row_step, column_step in steps:
            neighbor = row + row_step, column + column_step
            if _in_bounds(mask.shape, *neighbor) and mask[neighbor]:
                yield from extend(path + (neighbor,))

    for row, column in np.argwhere(mask):
        yield from extend(((int(row), int(column)),))


def _paper_opening(
    image: np.ndarray,
    length: int,
    steps: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Re-derive the atomic upper-level-set opening by exhaustive enumeration."""
    output = np.zeros_like(image)
    for level in np.unique(image):
        if level == 0:
            continue
        mask = image >= level
        survivors = np.zeros(mask.shape, dtype=bool)
        for path in _enumerate_paths(mask, steps, length):
            for point in path:
                survivors[point] = True
        output[survivors] = level
    return output


def _assert_fixture_case(module: ModuleType, name: str) -> None:
    """Match every source-visible stable single-scale product."""
    case = _case(name)
    arrays = case["arrays"]
    image = _decode(arrays["input"]).astype(np.uint8)  # type: ignore[index]
    result = module._single_scale_rorpo(
        image,
        int(case["path_length"]),
        int(case["robustness"]),
    )
    for orientation in range(4):
        np.testing.assert_array_equal(
            result.path_openings[orientation],
            _decode(arrays[f"po_raw_o{orientation + 1}"]),  # type: ignore[index]
        )
        np.testing.assert_array_equal(
            result.robust_openings[orientation],
            _decode(arrays[f"rpo_o{orientation + 1}"]),  # type: ignore[index]
        )
    np.testing.assert_array_equal(
        result.response,
        _decode(arrays["intensity_reconstructed"]),  # type: ignore[index]
    )


def assert_all_orientations(module: ModuleType) -> None:
    """Kill axis-bank and rank mutations across all four direction families."""
    for name in ("horizontal_l", "vertical_l", "diagonal_o3_l", "diagonal_o4_l"):
        _assert_fixture_case(module, name)


def assert_vertex_length(module: ModuleType) -> None:
    """Kill edge-count and complete-path boundary mutations."""
    image = np.zeros((7, 7), dtype=np.uint8)
    image[3, 2:5] = 200
    np.testing.assert_equal(np.count_nonzero(module._path_opening(image, 3, 1)), 3)
    np.testing.assert_equal(np.count_nonzero(module._path_opening(image, 4, 1)), 0)


def assert_curved_paths(module: ModuleType) -> None:
    """Kill a straight-line substitution for admissible curved paths."""
    case = _case("curved_admissible")
    arrays = case["arrays"]
    image = _decode(arrays["input"]).astype(np.uint8)  # type: ignore[index]
    np.testing.assert_array_equal(
        module._path_opening(image, 4, 2),
        _decode(arrays["po_raw_o3"]),  # type: ignore[index]
    )


def assert_atomic_levels(module: ModuleType) -> None:
    """Kill gray-level ordering that overwrites stronger surviving levels."""
    case = _case("multilevel")
    image = _decode(case["arrays"]["input"]).astype(np.uint8)  # type: ignore[index]
    expected = _paper_opening(image, 4, module.ORIENTATION_STEPS[1])
    np.testing.assert_array_equal(module._path_opening(image, 4, 1), expected)


def assert_robustness(module: ModuleType) -> None:
    """Kill dilation-radius, filter-polarity, and anti-extensivity mutations."""
    case = _case("gap_robustness_2")
    image = _decode(case["arrays"]["input"]).astype(np.uint8)  # type: ignore[index]
    result = module._single_scale_rorpo(image, 5, 2)
    radius = 1
    dilated = np.zeros_like(image)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            dilated[row, column] = np.max(
                image[
                    max(0, row - radius) : row + radius + 1,
                    max(0, column - radius) : column + radius + 1,
                ]
            )
    for orientation in range(4):
        paper = _paper_opening(dilated, 5, module.ORIENTATION_STEPS[orientation])
        np.testing.assert_array_equal(
            result.robust_openings[orientation],
            np.minimum(image, paper),
        )
    assert np.all(result.response <= image)


def assert_rank_and_response(module: ModuleType) -> None:
    """Kill descending-rank and wrong-order-statistic response mutations."""
    _assert_fixture_case(module, "rank_tie")
    _assert_fixture_case(module, "multilevel")


def _one_pixel_direction(
    module: ModuleType,
    values: tuple[int, int, int, int],
    orientations: tuple[int, int, int, int],
    response: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = module._direction_from_ranked(
        np.asarray(values, dtype=np.uint8).reshape(1, 1, 4),
        np.asarray(orientations, dtype=np.int64).reshape(1, 1, 4),
        np.asarray([[response]], dtype=np.uint8),
    )
    return outputs[0], outputs[8], outputs[9]


def assert_direction_threshold(module: ModuleType) -> None:
    """Kill a non-strict response threshold."""
    threshold, direction, valid = _one_pixel_direction(
        module, (0, 0, 0, 1), (0, 1, 2, 3), 1
    )
    assert not threshold[0, 0] and not valid[0, 0]
    np.testing.assert_array_equal(direction, 0.0)


def assert_unique_cost(module: ModuleType) -> None:
    """Kill removal of the unique truncated split-cost predicate."""
    _, direction, valid = _one_pixel_direction(
        module, (0, 0, 0, 2), (0, 1, 2, 3), 2
    )
    assert not valid[0, 0]
    np.testing.assert_array_equal(direction, 0.0)


def assert_strict_boundary(module: ModuleType) -> None:
    """Kill removal of the strict rank-boundary predicate."""
    original = module._truncated_split_cost
    module._truncated_split_cost = (
        lambda _values, high_count: {1: 2, 2: 0, 3: 1}[high_count]
    )
    try:
        _, direction, valid = _one_pixel_direction(
            module, (0, 5, 5, 10), (1, 3, 2, 0), 10
        )
    finally:
        module._truncated_split_cost = original
    assert not valid[0, 0]
    np.testing.assert_array_equal(direction, 0.0)


def assert_unique_signs(module: ModuleType) -> None:
    """Kill removal of the unique correction-sign predicate."""
    _, direction, valid = _one_pixel_direction(
        module, (0, 0, 10, 10), (2, 3, 1, 0), 10
    )
    assert not valid[0, 0]
    np.testing.assert_array_equal(direction, 0.0)


def assert_nonzero_sum(module: ModuleType) -> None:
    """Kill removal of the nonzero corrected-vector predicate."""
    original = module._pairwise_angle_objective
    module._pairwise_angle_objective = lambda vectors: (
        0 if np.linalg.norm(np.sum(vectors, axis=0)) == 0.0 else 1
    )
    try:
        _, direction, valid = _one_pixel_direction(
            module, (0, 10, 10, 10), (3, 0, 1, 2), 10
        )
    finally:
        module._pairwise_angle_objective = original
    assert not valid[0, 0]
    np.testing.assert_array_equal(direction, 0.0)


def assert_public_direction(module: ModuleType) -> None:
    """Kill coordinate-swap and axial-canonicalization mutations."""
    expected = {
        "horizontal_l": (0.0, 1.0),
        "vertical_l": (1.0, 0.0),
        "diagonal_o4_l": (2**-0.5, -(2**-0.5)),
    }
    for name, vector in expected.items():
        case = _case(name)
        image = _decode(case["arrays"]["input"]).astype(np.uint8)  # type: ignore[index]
        result = module.rorpo(image, (int(case["path_length"]),), 0)
        assert np.any(result.direction_valid)
        np.testing.assert_allclose(
            result.direction_vector[result.direction_valid],
            np.broadcast_to(vector, result.direction_vector[result.direction_valid].shape),
            rtol=0.0,
            atol=8.0 * np.finfo(np.float64).eps,
        )


def assert_multiscale(module: ModuleType) -> None:
    """Kill strict-tie, inversion, and public-response dtype mutations."""
    fixture = _fixture()["multiscale"]
    arrays = fixture["arrays"]
    image = _decode(arrays["input"]).astype(np.uint8)  # type: ignore[index]
    result = module.rorpo(image, (2, 3, 5), 0)
    np.testing.assert_array_equal(result.response, _decode(arrays["intensity_source"]))  # type: ignore[index]
    np.testing.assert_array_equal(result.winning_scale, _decode(arrays["winning_scale"]))  # type: ignore[index]
    assert result.response.dtype == np.uint8

    tied = module.rorpo(image, (3, 2, 3), 0)
    three = module._single_scale_rorpo(image, 3, 0).response
    two = module._single_scale_rorpo(image, 2, 0).response
    positive_ties = (three == two) & (three > 0)
    assert np.any(positive_ties)
    np.testing.assert_array_equal(tied.winning_scale[positive_ties], 3)


Mutation = tuple[str, str, str, Callable[[ModuleType], None]]


MUTATIONS: tuple[Mutation, ...] = (
    (
        "M01_vertex_edge_count",
        "forward + backward - 1 >= length",
        "forward + backward - 1 >= length - 1",
        assert_vertex_length,
    ),
    (
        "M02_orientation_bank",
        "((1, -1), (1, 0), (1, 1)),",
        "((1, 0), (1, 0), (1, 1)),",
        assert_all_orientations,
    ),
    (
        "M03_straight_only",
        "steps = ORIENTATION_STEPS[orientation]",
        "steps = (ORIENTATION_STEPS[orientation][1],) * 3",
        assert_curved_paths,
    ),
    (
        "M04_descending_levels",
        "for level in np.unique(image):",
        "for level in np.unique(image)[::-1]:",
        assert_atomic_levels,
    ),
    (
        "M05_robust_radius",
        "radius = robustness // 2",
        "radius = robustness",
        assert_robustness,
    ),
    (
        "M06_minimum_filter",
        "ndimage.maximum_filter(image, size=size, mode=\"constant\", cval=0)",
        "ndimage.minimum_filter(image, size=size, mode=\"constant\", cval=0)",
        assert_robustness,
    ),
    (
        "M07_remove_anti_extensive_min",
        "tuple(np.minimum(image, opened) for opened in path_openings)",
        "tuple(opened for opened in path_openings)",
        assert_robustness,
    ),
    (
        "M08_descending_ranks",
        "orientation_stack, axis=2, kind=\"stable\"",
        "-orientation_stack, axis=2, kind=\"stable\"",
        assert_rank_and_response,
    ),
    (
        "M09_wrong_response_order_statistic",
        "rank_values[..., 3] - rank_values[..., 0]",
        "rank_values[..., 3] - rank_values[..., 1]",
        assert_rank_and_response,
    ),
    (
        "M10_non_strict_direction_threshold",
        "threshold = response > 1",
        "threshold = response > 0",
        assert_direction_threshold,
    ),
    (
        "M11_remove_unique_cost",
        "cost_is_unique = np.count_nonzero(costs == minimum_cost) == 1",
        "cost_is_unique = True",
        assert_unique_cost,
    ),
    (
        "M12_remove_strict_boundary",
        "boundary_is_strict = bool(values[boundary - 1] < values[boundary])",
        "boundary_is_strict = True",
        assert_strict_boundary,
    ),
    (
        "M13_remove_unique_sign_assignment",
        "assignment_is_unique = objectives.count(minimum_objective) == 1",
        "assignment_is_unique = True",
        assert_unique_signs,
    ),
    (
        "M14_remove_nonzero_sum",
        "and norm > 0.0",
        "and True",
        assert_nonzero_sum,
    ),
    (
        "M15_omit_coordinate_swap",
        "(private_direction[1], private_direction[0]), dtype=np.float64",
        "(private_direction[0], private_direction[1]), dtype=np.float64",
        assert_public_direction,
    ),
    (
        "M16_omit_axial_canonicalization",
        "if public_direction[0] < 0.0 or (\n"
        "            public_direction[0] == 0.0 and public_direction[1] < 0.0\n"
        "        ):\n"
        "            public_direction *= -1.0",
        "if False:\n            public_direction *= -1.0",
        assert_public_direction,
    ),
    (
        "M17_non_strict_scale_tie",
        "update = scale.response > response",
        "update = scale.response >= response",
        assert_multiscale,
    ),
    (
        "M18_public_float_response",
        "response = np.zeros(values.shape, dtype=np.uint8)",
        "response = np.zeros(values.shape, dtype=np.float64)",
        assert_multiscale,
    ),
    (
        "M19_dark_ridge_inversion",
        "values = _validate_image(image)",
        "values = np.uint8(255) - _validate_image(image)",
        assert_multiscale,
    ),
)


def execute_mutations() -> None:
    """Run baseline probes, then prove every isolated mutant is killed."""
    source = SOURCE.read_text(encoding="utf-8")
    baseline = _load_module(SOURCE, "rorpo_mutation_baseline")
    probes = tuple(dict.fromkeys(mutation[3] for mutation in MUTATIONS))
    for probe in probes:
        probe(baseline)

    results: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="phenotypic-rorpo-mutants-") as temporary:
        directory = Path(temporary)
        for mutant_id, old, new, probe in MUTATIONS:
            mutant_path = directory / f"{mutant_id}.py"
            mutant_path.write_text(_replace_once(source, old, new), encoding="utf-8")
            mutant = _load_module(mutant_path, f"rorpo_{mutant_id}")
            try:
                probe(mutant)
            except Exception as error:
                results.append((mutant_id, type(error).__name__))
            else:
                raise AssertionError(f"{mutant_id} survived {probe.__name__}")

    for mutant_id, reason in results:
        print(f"{mutant_id}: KILLED by {reason}")


if __name__ == "__main__":
    execute_mutations()
