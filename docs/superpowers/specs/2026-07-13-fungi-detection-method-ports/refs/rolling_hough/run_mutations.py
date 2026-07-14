"""Inject each A09 Rolling Hough mutant and execute its killing probe."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[6]
SOURCE = ROOT / "src/phenotypic/sdk_/reconnect/_rolling_hough.py"
FIXTURE = ROOT / "tests/fixtures/reconnect/rolling_hough/clark_rht_source.npz"


def _load_module(path: Path, name: str) -> ModuleType:
    """Load one isolated temporary production module."""
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str) -> str:
    """Replace one unique mutation site and preserve all surrounding text."""
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


def _assert_all_outputs(module: ModuleType) -> None:
    """Match every public result field for all five source cases."""
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        for case_index in range(1, 6):
            prefix = f"c{case_index:02d}_"
            result = module.clark_rolling_hough(
                fixture[prefix + "image"],
                int(fixture[prefix + "window_diameter"]),
                int(fixture[prefix + "smoothing_radius"]),
                float(fixture[prefix + "threshold_fraction"]),
            )
            for actual, key in (
                (result.theta, "theta"),
                (result.support_counts, "support_counts"),
                (result.raw_counts, "raw_counts"),
                (result.threshold_residual, "threshold_residual"),
                (result.response, "raw_response"),
                (result.orientation, "derived_orientation"),
                (result.eligible, "window_mask"),
                (result.valid, "valid"),
            ):
                np.testing.assert_array_equal(actual, fixture[prefix + key])


def _assert_preprocessing(module: ModuleType) -> None:
    """Match every captured source preprocessing intermediate."""
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        for case_index in range(1, 6):
            prefix = f"c{case_index:02d}_"
            actual = module._clark_preprocessing(
                fixture[prefix + "image"],
                int(fixture[prefix + "window_diameter"]),
                int(fixture[prefix + "smoothing_radius"]),
            )
            for observed, key in zip(
                actual,
                (
                    "smoothing_kernel",
                    "smoothing_mask",
                    "window_mask",
                    "correlated",
                    "smoothed",
                    "unsharp",
                    "bitmask",
                ),
                strict=True,
            ):
                np.testing.assert_array_equal(observed, fixture[prefix + key])


def _assert_geometry(module: ModuleType) -> None:
    """Match every captured theta grid, circle, and rho-zero center line."""
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        for case_index in range(1, 4):
            prefix = f"c{case_index:02d}_"
            theta = fixture[prefix + "theta"]
            circle, center_lines = module._center_line_geometry(
                int(fixture[prefix + "window_diameter"]), theta
            )
            np.testing.assert_array_equal(
                circle, fixture[prefix + "circular_window"]
            )
            np.testing.assert_array_equal(
                center_lines, fixture[prefix + "center_lines"]
            )


def _assert_signed_threshold_zero(module: ModuleType) -> None:
    """Preserve equality plus the source rejected-bin negative-zero bits."""
    residual = module._threshold_counts(
        np.array([[2, 1, 0]], dtype=np.int64),
        np.array([2, 2, 2], dtype=np.int64),
        1.0,
    )
    np.testing.assert_array_equal(residual, [[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(np.signbit(residual), [[False, True, True]])


def _assert_invalid_dtype(module: ModuleType) -> None:
    """Reject an implicit float32 extension of source arithmetic."""
    try:
        module.clark_rolling_hough(
            np.ones((9, 9), dtype=np.float32), 5, 1, 0.5
        )
    except TypeError:
        return
    raise AssertionError("float32 input was accepted")


def _assert_output_dtypes(module: ModuleType) -> None:
    """Require the frozen int64 count and Boolean validity representations."""
    result = module.clark_rolling_hough(
        np.ones((9, 9), dtype=np.float64), 5, 1, 0.5
    )
    if (
        result.support_counts.dtype != np.int64
        or result.raw_counts.dtype != np.int64
    ):
        raise AssertionError("count dtype drifted")
    if result.eligible.dtype != np.bool_ or result.valid.dtype != np.bool_:
        raise AssertionError("mask dtype drifted")


Mutation = tuple[str, str, str, str, Callable[[ModuleType], None]]


MUTATIONS: tuple[Mutation, ...] = (
    (
        "RH-M01",
        "test_nonfinite_pixels_invalidate_both_source_halos",
        "np.logical_not(np.isfinite(image))",
        "np.isnan(image)",
        _assert_preprocessing,
    ),
    (
        "RH-M02",
        "test_source_fixture_matches_every_core_output_and_intermediate",
        "_circular_kernel(2 * smoothing_radius + 1)",
        "_circular_kernel(2 * smoothing_radius - 1)",
        _assert_preprocessing,
    ),
    (
        "RH-M03",
        "test_source_fixture_matches_every_core_output_and_intermediate",
        "correlated: Float64Array = ndimage.correlate(image, smoothing_kernel)",
        'correlated: Float64Array = ndimage.correlate(\n            image, smoothing_kernel, mode="constant"\n        )',
        _assert_preprocessing,
    ),
    (
        "RH-M04",
        "test_constant_image_returns_defined_empty_result",
        "np.logical_and(smoothing_mask, np.greater(unsharp, 0.0))",
        "np.logical_and(smoothing_mask, np.greater_equal(unsharp, 0.0))",
        _assert_all_outputs,
    ),
    (
        "RH-M05",
        "test_nonfinite_pixels_invalidate_both_source_halos",
        "np.logical_not(smoothing_mask),",
        "np.zeros_like(smoothing_mask),",
        _assert_preprocessing,
    ),
    (
        "RH-M06",
        "test_outputs_have_frozen_shapes_and_dtypes",
        "np.pi * (diameter - 1)",
        "np.pi * diameter",
        _assert_all_outputs,
    ),
    (
        "RH-M07",
        "test_source_fixture_matches_every_core_output_and_intermediate",
        "endpoint=False, dtype=np.float64",
        "endpoint=True, dtype=np.float64",
        _assert_all_outputs,
    ),
    (
        "RH-M08",
        "test_diameter_eleven_geometry_counts_and_angles_match_source",
        "column_coordinates[:, :, None] * np.cos(\n        theta\n    ) + row_coordinates[:, :, None] * np.sin(theta)",
        "row_coordinates[:, :, None] * np.cos(\n        theta\n    ) + column_coordinates[:, :, None] * np.sin(theta)",
        _assert_geometry,
    ),
    (
        "RH-M09",
        "test_geometry_contains_round_to_nearest_even_half_ties",
        "np.round(distances) == 0.0",
        "np.floor(distances) == 0.0",
        _assert_geometry,
    ),
    (
        "RH-M10",
        "test_source_fixture_matches_every_core_output_and_intermediate",
        'support_counts = np.einsum(\n        "ijt,ij->t", center_lines, circular_window, dtype=np.int64\n    )',
        "support_counts = np.full(\n        theta_count, np.sum(circular_window), dtype=np.int64\n    )",
        _assert_all_outputs,
    ),
    (
        "RH-M11",
        "test_source_fixture_matches_every_core_output_and_intermediate",
        "residual = np.true_divide(counts, support_counts) - threshold_fraction",
        "residual = counts.astype(np.float64) - threshold_fraction",
        _assert_all_outputs,
    ),
    (
        "RH-M12",
        "test_threshold_equality_is_zero_and_rejected_values_keep_negative_zero",
        "residual *= np.greater_equal(residual, 0.0)",
        "residual = np.maximum(residual, 0.0)",
        _assert_signed_threshold_zero,
    ),
    (
        "RH-M13",
        "test_constant_image_returns_defined_empty_result",
        "valid = np.any(threshold_residual > 0.0, axis=2)",
        "valid = np.any(raw_counts > 0, axis=2)",
        _assert_all_outputs,
    ),
    (
        "RH-M14",
        "test_raw_response_is_not_globally_normalized",
        "response = cast(\n        Float64Array,\n        np.sum(threshold_residual, axis=2, dtype=np.float64),\n    )",
        "response = cast(\n        Float64Array,\n        np.sum(threshold_residual, axis=2, dtype=np.float64),\n    )\n    response /= np.max(response)",
        _assert_all_outputs,
    ),
    (
        "RH-M15",
        "test_orientation_is_axial_hough_normal_with_source_mapping",
        "return float(np.pi - math.fmod(float(rough_angle + np.pi), float(np.pi)))",
        "return float(\n        (np.pi - math.fmod(float(rough_angle + np.pi), float(np.pi))\n        + np.pi / 2.0) % np.pi\n    )",
        _assert_all_outputs,
    ),
    (
        "RH-M16",
        "test_constant_image_returns_defined_empty_result",
        "orientation: Float64Array = np.full(values.shape, np.nan, dtype=np.float64)",
        "orientation: Float64Array = np.full(values.shape, np.pi, dtype=np.float64)",
        _assert_all_outputs,
    ),
    (
        "RH-M17",
        "test_outputs_have_frozen_shapes_and_dtypes",
        "raw_counts = np.zeros((height, width, theta_count), dtype=np.int64)",
        "raw_counts = np.zeros((height, width, theta_count), dtype=np.int32)",
        _assert_output_dtypes,
    ),
    (
        "RH-M18",
        "test_invalid_images_raise",
        'if image.dtype != np.dtype(np.float64):\n        raise TypeError("image must have dtype exactly float64")\n    return image',
        "return image.astype(np.float64, copy=False)",
        _assert_invalid_dtype,
    ),
)


def execute_mutations() -> None:
    """Prove the baseline, kill every isolated mutant, and verify restoration."""
    source = SOURCE.read_text(encoding="utf-8")
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    results: list[tuple[str, str, str, str]] = []
    with tempfile.TemporaryDirectory(
        prefix="phenotypic-rolling-hough-mutants-"
    ) as temp:
        directory = Path(temp)
        baseline_path = directory / "baseline.py"
        baseline_path.write_text(source, encoding="utf-8")
        baseline = _load_module(
            baseline_path, "rolling_hough_mutation_baseline"
        )
        _assert_all_outputs(baseline)
        _assert_preprocessing(baseline)
        _assert_geometry(baseline)
        _assert_signed_threshold_zero(baseline)
        _assert_invalid_dtype(baseline)
        _assert_output_dtypes(baseline)

        for mutant_id, probe_name, old, new, probe in MUTATIONS:
            mutant_text = _replace_once(source, old, new)
            mutant_path = directory / f"{mutant_id.lower()}.py"
            mutant_path.write_text(mutant_text, encoding="utf-8")
            mutant = _load_module(
                mutant_path, f"rolling_hough_{mutant_id.lower()}"
            )
            try:
                probe(mutant)
            except Exception as error:
                results.append(
                    (mutant_id, "KILLED", probe_name, type(error).__name__)
                )
            else:
                raise AssertionError(f"{mutant_id} survived {probe_name}")

    restored = SOURCE.read_text(encoding="utf-8")
    restored_digest = hashlib.sha256(restored.encode("utf-8")).hexdigest()
    if restored != source or restored_digest != source_digest:
        raise AssertionError(
            "reviewed production source changed during mutation run"
        )
    if len(results) != len(MUTATIONS):
        raise AssertionError("not every declared mutant produced a result")
    for mutant_id, status, probe_name, reason in results:
        print(f"{mutant_id}: {status} by {probe_name} ({reason})")
    print(f"Baseline restored: sha256={restored_digest}")


if __name__ == "__main__":
    execute_mutations()
