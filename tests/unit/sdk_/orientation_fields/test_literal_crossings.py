"""Tests for literal skeleton-ring crossing transforms."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.morphology import skeletonize

from phenotypic.sdk_.orientation_fields import (
    LiteralSkeletonRingCrossing,
    LiteralSkeletonRingCrossingTransform,
    literal_crossing_ring_profile,
    literal_skeleton_ring_crossings,
)


def _radial_cross_transform() -> LiteralSkeletonRingCrossingTransform:
    """Return crossings of a four-arm radial synthetic object."""
    size = 41
    center = (20.0, 20.0)
    rows, cols = np.indices((size, size))
    distance = np.hypot(rows - center[0], cols - center[1])
    polar = np.arctan2(rows - center[0], cols - center[1])
    mask = np.zeros((size, size), dtype=bool)
    mask[20, 4:37] = True
    mask[4:37, 20] = True
    selector = mask & (distance >= 4.0)
    return literal_skeleton_ring_crossings(
        mask,
        polar,
        np.ones_like(distance),
        distance,
        center,
        np.asarray([6.0, 12.0]),
        selector=selector,
        crossing_half_width=0.6,
    )


def _profile_transform(
    ring_tilts_degrees: list[list[float]],
) -> LiteralSkeletonRingCrossingTransform:
    """Build a transform with controlled crossing tilts."""
    crossings: list[LiteralSkeletonRingCrossing] = []
    for ring_index, tilts in enumerate(ring_tilts_degrees):
        for tilt in tilts:
            radians = float(np.radians(tilt))
            crossings.append(
                LiteralSkeletonRingCrossing(
                    point_id=len(crossings),
                    ring_index=ring_index,
                    radius=float(ring_index + 1),
                    row=5.0,
                    col=float(ring_index + 1),
                    anchor_row=5,
                    anchor_col=ring_index + 1,
                    fiber_axis=radians,
                    radial_tilt=radians,
                    coherence=1.0,
                    resultant=1.0,
                    pixel_count=1,
                )
            )
    return LiteralSkeletonRingCrossingTransform(
        crossings=tuple(crossings),
        reliable_skeleton=np.zeros((11, 11), dtype=bool),
        radii=np.arange(1, len(ring_tilts_degrees) + 1, dtype=float),
        center=(5.0, 5.0),
        crossing_half_width=1.5,
    )


def test_literal_transform_collects_one_sample_per_arm_and_ring() -> None:
    """Each connected ring intersection should contribute one crossing."""
    transform = _radial_cross_transform()

    assert len(transform.crossings) == 8
    assert [point.ring_index for point in transform.crossings].count(0) == 4
    assert [point.ring_index for point in transform.crossings].count(1) == 4
    assert np.allclose(
        [point.radial_tilt for point in transform.crossings], 0.0
    )
    assert not transform.reliable_skeleton[20, 20]


def test_literal_transform_uses_coherence_weighted_axial_mean() -> None:
    """A connected crossing should average its pixels in doubled-angle space."""
    size = 21
    center = (10.0, 10.0)
    rows, cols = np.indices((size, size))
    distance = np.hypot(rows - center[0], cols - center[1])
    mask = np.zeros((size, size), dtype=bool)
    mask[10, 2:19] = True
    axes = np.zeros((size, size), dtype=float)
    coherence = np.ones((size, size), dtype=float)
    axes[10, 17] = 0.25 * np.pi
    coherence[10, 17] = 0.5

    transform = literal_skeleton_ring_crossings(
        mask,
        axes,
        coherence,
        distance,
        center,
        np.asarray([6.0]),
        crossing_half_width=1.1,
    )
    positive_crossing = max(transform.crossings, key=lambda point: point.col)
    expected_axis = 0.5 * np.arctan2(0.5 / 2.5, 2.0 / 2.5)

    assert positive_crossing.pixel_count == 3
    assert positive_crossing.fiber_axis == pytest.approx(expected_axis)
    assert positive_crossing.radial_tilt == pytest.approx(expected_axis)


def test_literal_transform_rejects_incoherent_crossing_component() -> None:
    """Balanced orthogonal pixels should fail the within-crossing guard."""
    size = 21
    center = (10.0, 10.0)
    rows, cols = np.indices((size, size))
    distance = np.hypot(rows - center[0], cols - center[1])
    mask = np.zeros((size, size), dtype=bool)
    mask[10, 2:19] = True
    axes = np.zeros((size, size), dtype=float)
    axes[10, (3, 17)] = 0.5 * np.pi

    transform = literal_skeleton_ring_crossings(
        mask,
        axes,
        np.ones((size, size), dtype=float),
        distance,
        center,
        np.asarray([6.5]),
        crossing_half_width=0.6,
    )

    assert transform.crossings == ()


def test_inoculum_selector_is_applied_after_full_object_skeletonization() -> (
    None
):
    """Inoculum exclusion must gate the full-object skeleton, not reshape it."""
    size = 31
    center = (15.0, 15.0)
    rows, cols = np.indices((size, size))
    distance = np.hypot(rows - center[0], cols - center[1])
    mask = distance <= 10.0
    selector = mask & (distance >= 4.0)

    transform = literal_skeleton_ring_crossings(
        mask,
        np.zeros((size, size), dtype=float),
        np.ones((size, size), dtype=float),
        distance,
        center,
        np.asarray([6.0]),
        selector=selector,
    )

    assert np.array_equal(
        transform.reliable_skeleton, skeletonize(mask) & selector
    )
    assert (
        skeletonize(mask & selector).sum() > transform.reliable_skeleton.sum()
    )


def test_eligible_consensus_is_invariant_to_uniform_crossing_replication() -> (
    None
):
    """Uniform replication must not change an already-eligible ring angle."""
    sparse = _profile_transform([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0]])
    dense = _profile_transform(
        [
            [10.0, 20.0, 30.0] * 5,
            [20.0, 30.0, 40.0] * 5,
        ]
    )

    sparse_profile = literal_crossing_ring_profile(sparse)
    dense_profile = literal_crossing_ring_profile(dense)

    assert np.allclose(
        sparse_profile.consensus_tilt, dense_profile.consensus_tilt
    )
    assert np.allclose(sparse_profile.resultant, dense_profile.resultant)
    assert np.allclose(
        sparse_profile.contiguous_change,
        dense_profile.contiguous_change,
    )
    assert np.array_equal(
        dense_profile.crossing_count, 5 * sparse_profile.crossing_count
    )


def test_minimum_point_support_is_intentionally_count_sensitive() -> None:
    """The support guard should distinguish one crossing from three copies."""
    one_crossing = _profile_transform([[20.0]])
    three_crossings = _profile_transform([[20.0] * 3])

    guarded_sparse = literal_crossing_ring_profile(one_crossing)
    guarded_dense = literal_crossing_ring_profile(three_crossings)
    relaxed_sparse = literal_crossing_ring_profile(
        one_crossing, minimum_points=1
    )

    assert np.isnan(guarded_sparse.consensus_tilt[0])
    assert np.degrees(guarded_dense.consensus_tilt[0]) == pytest.approx(20.0)
    assert np.degrees(relaxed_sparse.consensus_tilt[0]) == pytest.approx(20.0)
    assert guarded_sparse.crossing_count[0] == 1
    assert guarded_dense.crossing_count[0] == 3


@pytest.mark.parametrize("minimum_points", [True, 1.5, np.nan])
def test_ring_profile_rejects_non_integer_minimum_points(
    minimum_points: object,
) -> None:
    """The minimum-count guard must not be silently disabled."""
    transform = _profile_transform([[20.0] * 3])

    with pytest.raises(ValueError, match="integer"):
        literal_crossing_ring_profile(
            transform,
            minimum_points=minimum_points,  # type: ignore[arg-type]
        )


def test_ring_profile_rejects_incoherent_ring_population() -> None:
    """Orthogonal crossing populations should fail the resultant guard."""
    transform = _profile_transform([[0.0, 0.0, 90.0, 90.0]])

    profile = literal_crossing_ring_profile(transform)

    assert profile.resultant[0] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(profile.consensus_tilt[0])


def test_ring_profile_unwraps_across_axial_seam() -> None:
    """Equivalent axial angles on opposite seam sides should change smoothly."""
    transform = _profile_transform(
        [
            [80.0, 80.0, 80.0],
            [-85.0, -85.0, -85.0],
            [-70.0, -70.0, -70.0],
        ]
    )

    profile = literal_crossing_ring_profile(transform)

    assert np.allclose(
        np.degrees(profile.contiguous_change), [0.0, 15.0, 30.0]
    )
    assert np.array_equal(profile.run_id, [0, 0, 0])


def test_ring_profile_restarts_after_a_missing_ring() -> None:
    """A missing population must break, not bridge, the accumulated run."""
    transform = _profile_transform(
        [
            [10.0] * 3,
            [20.0] * 3,
            [],
            [70.0] * 3,
            [80.0] * 3,
        ]
    )

    profile = literal_crossing_ring_profile(transform)

    assert np.allclose(
        np.degrees(profile.contiguous_change[[0, 1, 3, 4]]),
        [0.0, 10.0, 0.0, 10.0],
    )
    assert np.isnan(profile.contiguous_change[2])
    assert np.array_equal(profile.run_id, [0, 0, -1, 1, 1])


def test_ring_profile_marks_exact_ninety_degree_step_ambiguous() -> None:
    """An axial 90-degree step has no defensible signed continuation."""
    transform = _profile_transform(
        [
            [0.0] * 3,
            [90.0] * 3,
            [80.0] * 3,
            [70.0] * 3,
        ]
    )

    profile = literal_crossing_ring_profile(transform)

    assert np.allclose(
        np.degrees(profile.contiguous_change[[0, 2, 3]]),
        [0.0, 0.0, -10.0],
    )
    assert np.isnan(profile.contiguous_change[1])
    assert np.array_equal(profile.run_id, [0, -1, 1, 1])


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"radii": np.asarray([6.0, 6.0])}, "strictly increasing"),
        ({"coherence": np.full((41, 41), 1.1)}, "coherence values"),
        ({"crossing_half_width": 0.0}, "crossing_half_width"),
    ],
)
def test_literal_transform_rejects_invalid_inputs(
    replacement: dict[str, object],
    message: str,
) -> None:
    """Invalid public parameters should fail before calculating crossings."""
    size = 41
    center = (20.0, 20.0)
    rows, cols = np.indices((size, size))
    distance = np.hypot(rows - center[0], cols - center[1])
    arguments: dict[str, object] = {
        "object_mask": np.ones((size, size), dtype=bool),
        "fiber_axis": np.zeros((size, size)),
        "coherence": np.ones((size, size)),
        "distance_map": distance,
        "center": center,
        "radii": np.asarray([6.0, 12.0]),
    }
    arguments.update(replacement)

    with pytest.raises(ValueError, match=message):
        literal_skeleton_ring_crossings(**arguments)  # type: ignore[arg-type]
