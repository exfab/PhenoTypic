"""Tests for the canonical Method B branch-orientation zone resolver."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from phenotypic import Image, ImagePipeline
from phenotypic.abc_ import BaseOperation
from phenotypic.measure import MeasureOrientationZones, MeasureSymZones
from phenotypic.measure._orientation_zone_segmentation import (
    OrientationChangePointParams,
    bridge_short_gaps,
    collapsed_one_change_point,
    exact_two_change_points,
    fit_orientation_zones,
    selected_outer_radius,
)
from phenotypic.sdk_.typing_ import OperationField


def test_outer_percentile_uses_exact_full_extent_and_linear_percentile():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3:7] = True
    rows, cols = np.indices(mask.shape, dtype=float)
    distances = np.hypot(rows - 3.0, cols - 3.0)

    p100, full, retained = selected_outer_radius(mask, distances, 100.0)
    p50, full_again, retained_50 = selected_outer_radius(mask, distances, 50.0)

    assert p100 == 3.0
    assert full == full_again == 3.0
    assert retained == 1.0
    assert p50 == pytest.approx(1.5)
    assert retained_50 == 0.5


def test_short_gap_bridging_fills_only_bounded_interior_runs():
    support = np.array([False, True, False, True, False], dtype=bool)
    bridged = bridge_short_gaps(support, maximum_gap=1)
    assert bridged.tolist() == [False, True, True, True, False]


def test_exact_solver_breaks_cost_ties_at_earliest_boundaries():
    matrix = np.zeros((12, 3), dtype=float)
    support = np.ones(12, dtype=bool)
    result = exact_two_change_points(matrix, support, 3, 0.0)
    assert result == (0.0, 3, 6)


def test_collapsed_solver_finds_supported_outer_transition():
    support = np.array([False] * 4 + [True] * 8, dtype=bool)
    objective, boundary = collapsed_one_change_point(support, 3) or (None, None)
    assert objective == pytest.approx(0.0)
    assert boundary == 4


def test_unsupported_object_returns_missing_without_legacy_boundaries():
    rows, cols = np.indices((81, 81), dtype=float)
    mask = np.hypot(rows - 40.0, cols - 40.0) <= 30.0
    signal = np.ones(mask.shape, dtype=float)
    fitted = fit_orientation_zones(
        mask,
        signal,
        (40.0, 40.0),
        OrientationChangePointParams(ring_width=4.0, minimum_segment=2),
    )
    assert fitted.result.method_used == "missing"
    assert fitted.result.method_code == 4
    assert np.isnan(fitted.result.core_zone_radius)
    assert np.isnan(fitted.result.dense_radius)
    assert np.isfinite(fitted.result.outer_radius)


def _radial_spoke_image() -> Image:
    size = 181
    center = (90, 90)
    rows, cols = np.indices((size, size), dtype=float)
    radius = np.hypot(rows - center[0], cols - center[1])
    angle = np.arctan2(rows - center[0], cols - center[1])
    mask = radius <= 12.0
    for spoke in np.linspace(-np.pi, np.pi, 24, endpoint=False):
        axial_delta = 0.5 * np.arctan2(
            np.sin(2.0 * (angle - spoke)),
            np.cos(2.0 * (angle - spoke)),
        )
        mask |= (radius <= 75.0) & (np.abs(axial_delta) <= 0.018)
    signal = np.zeros((size, size), dtype=np.float32)
    signal[mask] = 1.0
    rgb = np.repeat(signal[..., None], 3, axis=2)
    image = Image(rgb)
    image.detect_mat[:] = signal
    image.objmap[:] = mask.astype(np.int32)
    return image


def test_both_measurers_use_identical_canonical_zone_geometry():
    image = _radial_spoke_image()
    params = dict(
        radial_ring_width=4.0,
        zone_minimum_segment=2,
        zone_min_crossings=1,
        zone_min_resultant=0.15,
        zone_min_ring_coherence=0.15,
    )
    symmetric = MeasureSymZones(**params).measure(image)
    orientation_op = MeasureOrientationZones(
        include_diagnostics=True, **params
    )
    orientation = orientation_op.measure(image)

    symmetric_radii = symmetric[
        [
            "SymZones_CoreEndRadius",
            "SymZones_DenseEndRadius",
            "SymZones_SparseEndRadius",
        ]
    ].iloc[0].to_numpy(dtype=float)
    orientation_radii = orientation[
        [
            "OrientZones_CoreZoneEndRadius",
            "OrientZones_DenseRadius",
            "OrientZones_OuterRadius",
        ]
    ].iloc[0].to_numpy(dtype=float)
    np.testing.assert_allclose(symmetric_radii, orientation_radii)
    assert orientation["OrientZones_ZoneSegmentationMethodCode"].iloc[0] == 1.0
    assert np.all(np.diff(orientation_radii) > 0.0)
    assert orientation_op._cache[1]["zone_resolution"]["method_used"] == "exact"


def test_collapsed_canonical_fit_has_zero_dense_area_and_missing_dense_metrics():
    image = _radial_spoke_image()
    params = dict(
        radial_ring_width=8.0,
        zone_minimum_segment=4,
        zone_min_crossings=1,
        zone_min_resultant=0.15,
        zone_min_ring_coherence=0.15,
    )
    symmetric = MeasureSymZones(**params).measure(image)
    orientation = MeasureOrientationZones(
        include_diagnostics=True, **params
    ).measure(image)

    assert symmetric["SymZones_CoreEndRadius"].iloc[0] == symmetric[
        "SymZones_DenseEndRadius"
    ].iloc[0]
    assert symmetric["SymZones_DenseArea"].iloc[0] == 0.0
    assert orientation["OrientZones_ZoneSegmentationMethodCode"].iloc[0] == 2.0
    dense_metrics = orientation.filter(regex=r"-Dense$").iloc[0]
    assert not dense_metrics.empty
    assert dense_metrics.isna().all()


def test_canonical_measurement_failure_never_exposes_legacy_zone_radii():
    rows, cols = np.indices((81, 81), dtype=float)
    mask = np.hypot(rows - 40.0, cols - 40.0) <= 30.0
    signal = mask.astype(np.float32)
    image = Image(np.repeat(signal[..., None], 3, axis=2))
    image.detect_mat[:] = signal
    image.objmap[:] = mask.astype(np.int32)

    result = MeasureSymZones(zone_minimum_segment=20).measure(image).iloc[0]
    assert result[
        [
            "SymZones_CoreEndRadius",
            "SymZones_DenseEndRadius",
            "SymZones_SparseEndRadius",
        ]
    ].isna().all()


@pytest.mark.parametrize("operation", [MeasureSymZones, MeasureOrientationZones])
def test_new_instances_are_canonical_but_old_serialized_payloads_migrate(operation):
    assert operation().legacy_mode is False
    old_payload = {"class": operation.__name__, "params": {}}
    restored = BaseOperation.from_json(json.dumps(old_payload))
    assert restored.legacy_mode is True
    new_payload = json.loads(operation().to_json())
    assert new_payload["params"]["legacy_mode"] is False


def test_old_pipeline_measurement_payload_migrates_to_legacy_mode():
    payload = json.loads(ImagePipeline(meas=[MeasureSymZones()]).to_json())
    del payload["meas"]["MeasureSymZones"]["params"]["legacy_mode"]
    restored = ImagePipeline.from_json(json.dumps(payload))
    assert restored._meas["MeasureSymZones"].legacy_mode is True


def test_old_nested_operation_payload_migrates_to_legacy_mode():
    class _OperationHost(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        operation: OperationField  # type: ignore[valid-type]

    restored = _OperationHost.model_validate(
        {"operation": {"class": "MeasureSymZones", "params": {}}}
    )
    assert restored.operation.legacy_mode is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outer_zone_percentile": True},
        {"outer_zone_percentile": 0.0},
        {"zone_minimum_segment": 0},
        {"zone_min_crossings": False},
        {"zone_min_resultant": 0.1},
        {"zone_support_weight": -1.0},
        {"zone_outer_support_margin": 1.1},
        {"zone_maximum_gap": -1},
        {"method": "intensity"},
    ],
)
def test_canonical_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        MeasureSymZones(**kwargs)
