"""Tests for the canonical Method B branch-orientation zone resolver."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from phenotypic import Image, ImagePipeline
from phenotypic.abc_ import BaseOperation
from phenotypic.detect import HysteresisDetector
from phenotypic.enhance import BlurGauss
from phenotypic.measure import MeasureOrientationZones, MeasureSymZones
from phenotypic.measure._orientation_zone_segmentation import (
    OrientationChangePointParams,
    bridge_short_gaps,
    collapsed_one_change_point,
    exact_two_change_points,
    fit_orientation_zones,
    selected_outer_radius,
)
from phenotypic.measure._measure_orientation_zones import (
    radial_ring_orientation_profile,
    zone_selector,
)
from phenotypic.measure._zone_segmentation import detected_center_coordinates
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


def test_canonical_tensor_and_boundaries_are_positive_affine_scale_invariant():
    image = _radial_spoke_image()
    mask = image.objmap[:] == 1
    source = image.detect_mat[:].astype(np.float64)
    params = OrientationChangePointParams(
        ring_width=4.0,
        minimum_segment=2,
        min_crossings=1,
    )
    reference = fit_orientation_zones(mask, source, (90.0, 90.0), params)
    assert reference.context is not None
    assert reference.result.method_used == "exact"

    transformed_signals = (
        source * 1e-9 + 7e-10,
        source * 1e9 - 3e8,
        source + 1e6,
    )
    for transformed in transformed_signals:
        fitted = fit_orientation_zones(mask, transformed, (90.0, 90.0), params)
        assert fitted.context is not None
        assert fitted.result.method_used == reference.result.method_used
        np.testing.assert_allclose(
            [
                fitted.result.core_zone_radius,
                fitted.result.dense_radius,
                fitted.result.outer_radius,
                fitted.result.objective,
            ],
            [
                reference.result.core_zone_radius,
                reference.result.dense_radius,
                reference.result.outer_radius,
                reference.result.objective,
            ],
            rtol=1e-7,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            fitted.context.coherence,
            reference.context.coherence,
            rtol=1e-6,
            atol=1e-8,
        )


def test_canonical_outer_boundary_is_included_but_internal_boundary_is_half_open():
    distances = np.array([[1.0, 2.0, 3.0]])
    mask = np.ones_like(distances, dtype=bool)

    dense = zone_selector(distances, 1.0, 2.0, mask, "Mask")
    sparse = zone_selector(
        distances,
        2.0,
        3.0,
        mask,
        "Mask",
        include_upper=True,
    )

    assert dense.tolist() == [[True, False, False]]
    assert sparse.tolist() == [[False, True, True]]


def test_long_range_profile_includes_exact_global_outer_boundary():
    shape = (1, 4)
    tilt = np.zeros(shape, dtype=float)
    polar = np.zeros(shape, dtype=float)
    coherence = np.ones(shape, dtype=float)
    distance = np.array([[0.5, 1.5, 2.0, 2.0]])
    selector = np.ones(shape, dtype=bool)

    _radii, excluded, _resultant = radial_ring_orientation_profile(
        tilt,
        polar,
        coherence,
        distance,
        selector,
        0.0,
        2.0,
        2.0,
        1,
    )
    _radii, included, _resultant = radial_ring_orientation_profile(
        tilt,
        polar,
        coherence,
        distance,
        selector,
        0.0,
        2.0,
        2.0,
        1,
        include_outer=True,
    )

    assert np.isnan(excluded[0, 0])
    assert included[0, 0] == 0.0


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
    signal[mask] = 1.0 + radius[mask] / 75.0
    rgb = np.repeat((signal / signal.max())[..., None], 3, axis=2)
    image = Image(rgb)
    image.detect_mat[:] = signal
    image.objmap[:] = mask.astype(np.int32)
    return image


def _off_center_core_image(*, center_overlaps_object: bool = True) -> Image:
    """Build one colony with a threshold-isolatable compact center."""
    rows, cols = np.indices((121, 121), dtype=float)
    object_mask = np.hypot(rows - 60.0, cols - 60.0) <= 45.0
    core_center = (45.0, 35.0) if center_overlaps_object else (8.0, 8.0)
    core_mask = np.hypot(
        rows - core_center[0], cols - core_center[1]
    ) <= 5.0
    signal = np.zeros(object_mask.shape, dtype=np.float32)
    signal[object_mask] = 0.2
    signal[core_mask] = 1.0
    image = Image(np.repeat(signal[..., None], 3, axis=2))
    image.detect_mat[:] = signal
    image.objmap[:] = object_mask.astype(np.int32)
    return image


def _compact_center_detector() -> HysteresisDetector:
    return HysteresisDetector(
        low=0.8,
        high=0.8,
        ignore_borders=False,
    )


def test_detected_centers_are_associated_by_overlap_with_deterministic_ties():
    objects = np.zeros((8, 12), dtype=np.int32)
    objects[:, :6] = 1
    objects[:, 6:] = 2
    centers = np.zeros_like(objects)
    centers[1:3, 1:3] = 3
    centers[4:6, 1:3] = 2
    centers[2:4, 5:7] = 4
    centers[5:7, 8:10] = 5

    coordinates = detected_center_coordinates(objects, centers)

    # Labels 2 and 3 have equal overlap with object 1, so label 2 wins.
    assert coordinates[1] == (4.5, 1.5)
    assert coordinates[2] == (5.5, 8.5)


def test_center_detector_supplies_the_shared_center_to_both_measurers():
    image = _off_center_core_image()
    symmetric = MeasureSymZones(center_detector=_compact_center_detector())
    orientation = MeasureOrientationZones(
        center_detector=_compact_center_detector(),
        include_diagnostics=True,
    )

    symmetric.measure(image)
    orientation.measure(image)

    symmetric_center = symmetric._MeasureSymZones__cache_intermediates[
        1
    ].centroid_global
    orientation_center = orientation._cache[1]["centroid_global"]
    np.testing.assert_allclose(symmetric_center, (45.0, 35.0), atol=1e-12)
    np.testing.assert_allclose(orientation_center, symmetric_center, atol=1e-12)


def test_requested_center_detector_without_overlap_is_canonical_failure():
    image = _off_center_core_image(center_overlaps_object=False)
    symmetric = MeasureSymZones(
        center_detector=_compact_center_detector()
    ).measure(image).iloc[0]
    orientation = MeasureOrientationZones(
        center_detector=_compact_center_detector(),
        include_diagnostics=True,
    ).measure(image).iloc[0]

    assert symmetric.drop(labels="Object_Label").isna().all()
    assert orientation["OrientZones_ZoneSegmentationMethodCode"] == 4.0
    assert orientation.filter(like="OrientZones_").drop(
        labels="OrientZones_ZoneSegmentationMethodCode"
    ).isna().all()


def test_center_detector_is_serialized_and_rejects_non_detectors():
    operation = MeasureSymZones(center_detector=_compact_center_detector())
    restored = MeasureSymZones.from_json(operation.to_json())

    assert isinstance(restored.center_detector, HysteresisDetector)
    assert restored.center_detector.low == 0.8
    assert "center_detector" in MeasureSymZones.model_json_schema()["properties"]
    assert "center_detector" in MeasureOrientationZones.model_json_schema()[
        "properties"
    ]
    with pytest.raises(ValueError, match="center_detector must be"):
        MeasureSymZones(center_detector=BlurGauss())


def test_center_detector_accepts_and_roundtrips_an_image_pipeline():
    center_pipeline = ImagePipeline(ops=[_compact_center_detector()])
    operation = MeasureOrientationZones(center_detector=center_pipeline)
    restored = MeasureOrientationZones.from_json(operation.to_json())

    assert isinstance(restored.center_detector, ImagePipeline)
    centers = restored._detected_centers(_off_center_core_image())
    assert centers is not None
    np.testing.assert_allclose(centers[1], (45.0, 35.0), atol=1e-12)


def test_legacy_mode_ignores_center_detector(monkeypatch):
    image = _off_center_core_image()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("legacy mode must not run center_detector")

    monkeypatch.setattr(HysteresisDetector, "apply", fail_if_called)
    result = MeasureSymZones(
        legacy_mode=True,
        center_detector=_compact_center_detector(),
    ).measure(image)

    assert len(result) == 1


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

    np.testing.assert_allclose(
        symmetric[
            [
                "SymZones_CoreEndRadius",
                "SymZones_DenseEndRadius",
                "SymZones_SparseEndRadius",
            ]
        ].iloc[0].to_numpy(dtype=float),
        orientation[
            [
                "OrientZones_CoreZoneEndRadius",
                "OrientZones_DenseRadius",
                "OrientZones_OuterRadius",
            ]
        ].iloc[0].to_numpy(dtype=float),
    )
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
    orientation_op = MeasureOrientationZones(
        zone_minimum_segment=20,
        include_diagnostics=True,
    )
    orientation = orientation_op.measure(image).iloc[0]
    assert orientation["OrientZones_ZoneSegmentationMethodCode"] == 4.0
    cached = orientation_op._cache[1]["radii"]
    assert np.isnan(
        [cached["core_end"], cached["dense_end"], cached["sparse_end"]]
    ).all()


def test_tiny_canonical_object_is_missing_for_both_measurers_and_code_four():
    signal = np.zeros((9, 9), dtype=np.float32)
    signal[4, 3:6] = 1.0
    image = Image(np.repeat(signal[..., None], 3, axis=2))
    image.detect_mat[:] = signal
    image.objmap[:] = (signal > 0).astype(np.int32)

    symmetric = MeasureSymZones().measure(image).iloc[0]
    orientation_op = MeasureOrientationZones(include_diagnostics=True)
    orientation = orientation_op.measure(image).iloc[0]

    assert symmetric.drop(labels="Object_Label").isna().all()
    assert orientation.filter(like="OrientZones_").drop(
        labels="OrientZones_ZoneSegmentationMethodCode"
    ).isna().all()
    assert orientation["OrientZones_ZoneSegmentationMethodCode"] == 4.0
    cached = orientation_op._cache[1]["radii"]
    assert np.isnan(
        [cached["core_end"], cached["dense_end"], cached["sparse_end"]]
    ).all()

    legacy = MeasureSymZones(legacy_mode=True).measure(image).iloc[0]
    legacy_zone_columns = [
        "SymZones_CoreEndRadius",
        "SymZones_DenseEndRadius",
        "SymZones_SparseEndRadius",
        "SymZones_CoreArea",
        "SymZones_DenseArea",
        "SymZones_SparseArea",
    ]
    assert (legacy[legacy_zone_columns] == 0.0).all()


def test_canonical_mode_does_not_execute_legacy_colony_ness(
    monkeypatch: pytest.MonkeyPatch,
):
    from phenotypic.measure import _zone_segmentation

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("legacy colony-ness executed in canonical mode")

    monkeypatch.setattr(
        _zone_segmentation,
        "compute_colony_ness_profile",
        fail_if_called,
    )
    result = MeasureSymZones(
        radial_ring_width=4.0,
        zone_minimum_segment=2,
        zone_min_crossings=1,
    ).measure(_radial_spoke_image())
    assert np.isfinite(result["SymZones_SparseEndRadius"].iloc[0])


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
