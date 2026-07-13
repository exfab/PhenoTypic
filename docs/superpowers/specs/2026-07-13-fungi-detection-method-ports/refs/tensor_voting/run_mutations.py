"""Inject every tensor-voting mutant separately and run its killing probe."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[6]
SOURCE = ROOT / "src/phenotypic/sdk_/reconnect/_tensor_voting.py"
FIXTURE = ROOT / "tests/fixtures/reconnect/tensor_voting/linton_calc_vote_stick_r2023b.mat"


def load_module(path: Path, name: str) -> ModuleType:
    """Load one isolated helper module from a temporary source copy."""
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def apply_exact_mutation(source: str, old: str, new: str) -> str:
    """Replace one unique site and prove no prefix or suffix text changed."""
    if source.count(old) != 1:
        raise RuntimeError(
            f"mutation site count is {source.count(old)}, expected 1: {old!r}"
        )
    start = source.index(old)
    mutated = source[:start] + new + source[start + len(old) :]
    if mutated[:start] != source[:start]:
        raise AssertionError("mutation changed text before its declared site")
    if mutated[start + len(new) :] != source[start + len(old) :]:
        raise AssertionError("mutation changed text after its declared site")
    return mutated


def assert_fixture(module: ModuleType) -> None:
    """Match every source-visible numeric output with a roundoff bound."""
    fixture = loadmat(FIXTURE)
    response = fixture["response"]
    theta = fixture["theta"]
    sigma = float(fixture["sigma"][0, 0])
    component_a, component_b, component_d = module._tensor_vote_components(
        response, theta, sigma
    )
    stick, ball = module.tensor_vote(response, theta, sigma)
    for actual, key in (
        (component_a, "accumulated_a"),
        (component_b, "accumulated_b"),
        (component_d, "accumulated_d"),
        (stick, "stick"),
        (ball, "ball"),
    ):
        np.testing.assert_allclose(actual, fixture[key], rtol=0.0, atol=3e-14)


def assert_rotation_and_transpose(module: ModuleType) -> None:
    """Run the row/column and Cartesian-angle covariance probe."""
    response = np.zeros((11, 13), dtype=np.float64)
    theta = np.zeros_like(response)
    response[5, 2] = 1.0
    response[7, 8] = 0.7
    theta[7, 8] = 0.3
    stick, ball = module.tensor_vote(response, theta, 2.25)
    rotated = module.tensor_vote(
        np.rot90(response), np.rot90(theta) + np.pi / 2.0, 2.25
    )
    transposed = module.tensor_vote(response.T, np.pi / 2.0 - theta.T, 2.25)
    bound = 32.0 * np.finfo(np.float64).eps
    np.testing.assert_allclose(rotated[0], np.rot90(stick), rtol=0.0, atol=bound)
    np.testing.assert_allclose(rotated[1], np.rot90(ball), rtol=0.0, atol=bound)
    np.testing.assert_allclose(transposed[0], stick.T, rtol=0.0, atol=bound)
    np.testing.assert_allclose(transposed[1], ball.T, rtol=0.0, atol=bound)


def assert_linearity(module: ModuleType) -> None:
    """Run the raw-amplitude positive-linearity probe."""
    response = np.zeros((9, 11), dtype=np.float64)
    theta = np.zeros_like(response)
    response[4, 2] = 0.4
    response[4, 8] = 1.2
    theta[4, 8] = np.pi / 6.0
    stick, ball = module.tensor_vote(response, theta, 2.25)
    scaled_stick, scaled_ball = module.tensor_vote(3.5 * response, theta, 2.25)
    np.testing.assert_allclose(scaled_stick, 3.5 * stick, rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(scaled_ball, 3.5 * ball, rtol=3e-15, atol=3e-15)


def assert_isolated_self_vote(module: ModuleType) -> None:
    """Run the retained-input plus cast-center self-vote probe."""
    response = np.zeros((7, 7), dtype=np.float64)
    theta = np.zeros_like(response)
    response[3, 3] = 1.75
    component_a, component_b, component_d = module._tensor_vote_components(
        response, theta, 2.25
    )
    stick, ball = module.tensor_vote(response, theta, 2.25)
    np.testing.assert_allclose(
        [component_a[3, 3], component_b[3, 3], component_d[3, 3]],
        [0.0, 0.0, 3.5],
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        [stick[3, 3], ball[3, 3]], [3.5, 0.0], rtol=0.0, atol=1e-14
    )


def assert_boundary_crop(module: ModuleType) -> None:
    """Run the clipped-boundary, no-wrap probe."""
    response = np.zeros((15, 15), dtype=np.float64)
    theta = np.zeros_like(response)
    response[0, 0] = 1.0
    stick, ball = module.tensor_vote(response, theta, 2.25)
    np.testing.assert_array_equal(
        [stick[0, -1], ball[0, -1], stick[-1, -1], ball[-1, -1]],
        np.zeros(4),
    )


def assert_closed_form_saliency(module: ModuleType) -> None:
    """Run the eigenvalue-to-stick/ball output-order probe."""
    response = np.zeros((9, 9), dtype=np.float64)
    theta = np.zeros_like(response)
    response[4, 2] = 1.0
    response[4, 6] = 1.0
    theta[4, 6] = np.pi / 2.0
    component_a, component_b, component_d = module._tensor_vote_components(
        response, theta, 2.25
    )
    eigenvalues = np.linalg.eigvalsh(
        np.stack(
            (
                np.stack((component_a, component_b), axis=-1),
                np.stack((component_b, component_d), axis=-1),
            ),
            axis=-2,
        )
    )
    stick, ball = module.tensor_vote(response, theta, 2.25)
    np.testing.assert_allclose(stick, eigenvalues[..., 1] - eigenvalues[..., 0])
    np.testing.assert_allclose(ball, eigenvalues[..., 0], rtol=0.0, atol=4e-15)


def assert_source_radius(module: ModuleType) -> None:
    """Run the nested support-rounding boundary probe."""
    sigma = 0.52
    doubled = np.sqrt(-np.log(0.01) * sigma**2) * 2.0
    expected = int(np.floor(np.ceil(doubled) / 2.0))
    if module._source_window_radius(sigma) != expected:
        raise AssertionError("source support-window rounding changed")


Mutation = tuple[str, str, str, str, Callable[[ModuleType], None]]


MUTATIONS: tuple[Mutation, ...] = (
    ("TV-M01", "test_source_generated_fixture_matches_all_outputs", "arc_length * arc_length\n                            + curvature_weight", "0.0\n                            + curvature_weight", assert_fixture),
    ("TV-M02", "test_source_generated_fixture_matches_all_outputs", "+ curvature_weight * curvature * curvature", "+ 0.0", assert_fixture),
    ("TV-M03", "test_rotation_and_transpose_controls", "offset_x = float(target_column - voter_column)", "offset_x = float(target_row - voter_row)", assert_rotation_and_transpose),
    ("TV-M04", "test_source_generated_fixture_matches_all_outputs", "tangent_x = -principal_y\n            tangent_y = principal_x", "tangent_x = -principal_x\n            tangent_y = -principal_y", assert_fixture),
    ("TV-M05", "test_source_generated_fixture_matches_all_outputs", "attenuation_angle *= 4.0", "attenuation_angle *= 1.0", assert_fixture),
    ("TV-M06", "test_positive_linearity_for_fixed_active_mask", "weighted_vote = vote_amplitude * decay", "weighted_vote = decay", assert_linearity),
    ("TV-M07", "test_source_generated_fixture_matches_all_outputs", "component_d[target_row, target_column] += (", "component_d[target_row, target_column] = (", assert_fixture),
    ("TV-M08", "test_isolated_token_has_retained_and_cast_self_vote", "component_d[row, column] = amplitude * normal_y * normal_y", "component_d[row, column] = 0.0", assert_isolated_self_vote),
    ("TV-M09", "test_isolated_token_has_retained_and_cast_self_vote", "offset_x = float(target_column - voter_column)\n\n                    local_x", "offset_x = float(target_column - voter_column)\n                    if offset_x == 0.0 and offset_y == 0.0:\n                        continue\n\n                    local_x", assert_isolated_self_vote),
    (
        "TV-M10",
        "test_boundary_vote_is_cropped_not_wrapped",
        "first_row = max(0, voter_row - radius)\n            last_row = min(height - 1, voter_row + radius)\n            first_column = max(0, voter_column - radius)\n            last_column = min(width - 1, voter_column + radius)\n\n            for target_row in range(first_row, last_row + 1):\n                offset_y = float(voter_row - target_row)\n                for target_column in range(first_column, last_column + 1):\n                    offset_x = float(target_column - voter_column)",
        "for row_offset in range(-radius, radius + 1):\n                target_row = (voter_row + row_offset) % height\n                offset_y = float(-row_offset)\n                for column_offset in range(-radius, radius + 1):\n                    target_column = (voter_column + column_offset) % width\n                    offset_x = float(column_offset)",
        assert_boundary_crop,
    ),
    ("TV-M11", "test_closed_form_saliency_decomposition", "stick_saliency = delta", "stick_saliency = (component_a + component_d - delta) / 2.0", assert_closed_form_saliency),
    ("TV-M12", "test_source_generated_fixture_matches_all_outputs", "component_d = np.empty((height, width), dtype=np.float64)", "component_d = np.empty((height, width), dtype=np.float32)", assert_fixture),
    ("TV-M13", "test_positive_linearity_for_fixed_active_mask", "stick_saliency = delta", "stick_saliency = delta / np.max(delta)", assert_linearity),
    ("TV-M14", "test_source_generated_fixture_matches_all_outputs", "vote_angle = 2.0 * local_angle + base_angle", "vote_angle = -2.0 * local_angle + base_angle", assert_fixture),
    (
        "TV-M15",
        "test_support_radius_matches_source_rounding_not_plain_ceiling",
        "window_size = (\n        math.floor(\n            math.ceil(math.sqrt(-math.log(0.01) * sigma**2) * 2.0) / 2.0\n        )\n        * 2\n        + 1\n    )\n    return (window_size - 1) // 2",
        "return math.ceil(math.sqrt(-math.log(0.01) * sigma**2))",
        assert_source_radius,
    ),
)


def execute_mutations() -> None:
    """Prove the baseline, isolate every mutant, and reject survivors."""
    source = SOURCE.read_text(encoding="utf-8")
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    results: list[tuple[str, str, str, str]] = []
    with tempfile.TemporaryDirectory(prefix="phenotypic-tensor-voting-mutants-") as temp:
        directory = Path(temp)
        baseline_path = directory / "baseline.py"
        baseline_path.write_text(source, encoding="utf-8")
        baseline = load_module(baseline_path, "tensor_voting_mutation_baseline")
        for _, _, _, _, probe in MUTATIONS:
            probe(baseline)

        for mutant_id, probe_name, old, new, probe in MUTATIONS:
            mutant_text = apply_exact_mutation(source, old, new)
            mutant_path = directory / f"{mutant_id.lower()}.py"
            mutant_path.write_text(mutant_text, encoding="utf-8")
            if mutant_path.read_text(encoding="utf-8") != mutant_text:
                raise AssertionError(f"{mutant_id} temporary copy changed on write")
            mutant = load_module(mutant_path, f"tensor_voting_{mutant_id.lower()}")
            try:
                probe(mutant)
            except Exception as error:
                results.append((mutant_id, "KILLED", probe_name, type(error).__name__))
            else:
                raise AssertionError(f"{mutant_id} survived {probe_name}")

    restored = SOURCE.read_text(encoding="utf-8")
    restored_digest = hashlib.sha256(restored.encode("utf-8")).hexdigest()
    if restored_digest != source_digest or restored != source:
        raise AssertionError("reviewed production source changed during mutation run")
    if len(results) != len(MUTATIONS):
        raise AssertionError("not every declared mutant produced a result")
    for mutant_id, status, probe_name, reason in results:
        print(f"{mutant_id}: {status} by {probe_name} ({reason})")
    print(f"Baseline restored: sha256={restored_digest}")


if __name__ == "__main__":
    execute_mutations()
