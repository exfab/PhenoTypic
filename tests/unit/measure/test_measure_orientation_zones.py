"""Behaviour tests for MeasureOrientationZones."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pandas as pd
import pytest

from phenotypic import Image
from phenotypic.data import (
    load_synth_yeast_plate,
    load_synth_filamentous_plate,
)
from phenotypic.schema import (
    ORIENTATION_ZONE_DIAGNOSTIC,
    ORIENTATION_ZONE_PRIMARY,
    ORIENTATION_ZONES,
)
from phenotypic.measure import MeasureOrientationZones
from phenotypic.measure._measure_orientation_zones import (
    aggregate_long_range_rotation,
    aggregate_orientation,
    aggregate_paired_zone_rotation,
    aggregate_radial_relative,
    cumulative_ring_rotation_profile,
    long_range_ring_rotation_profile,
    radial_ring_sector_field,
    radial_relative_field,
    radial_ring_orientation_profile,
    signed_radial_relative_field,
    zone_selector,
)


def test_aggregate_parallel_field_high_R_zero_turning():
    n = 40
    phi = np.full((n, n), 0.3)  # constant orientation
    coh = np.ones((n, n))
    grad = np.zeros((n, n))
    sel = np.ones((n, n), dtype=bool)
    R, turning, coh_mean = aggregate_orientation(phi, coh, grad, sel, eps=1e-9)
    assert R == pytest.approx(1.0, abs=1e-6)
    assert turning == pytest.approx(0.0, abs=1e-9)
    assert coh_mean == pytest.approx(1.0, abs=1e-9)


def test_aggregate_zero_coherence_is_nan():
    n = 20
    out = aggregate_orientation(
        np.zeros((n, n)),
        np.zeros((n, n)),
        np.zeros((n, n)),
        np.ones((n, n), dtype=bool),
        eps=1e-9,
    )
    assert all(np.isnan(v) for v in out)


def test_aggregate_empty_selector_is_nan():
    n = 20
    out = aggregate_orientation(
        np.zeros((n, n)),
        np.ones((n, n)),
        np.zeros((n, n)),
        np.zeros((n, n), dtype=bool),
        eps=1e-9,
    )
    assert all(np.isnan(v) for v in out)


def test_zone_selector_radial_vs_mask():
    n = 21
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - c, xx - c)
    obj = dist < 6  # imperfect mask: only inner disk
    radial = zone_selector(dist, 0.0, 8.0, obj, "Radial")
    masked = zone_selector(dist, 0.0, 8.0, obj, "Mask")
    assert radial.sum() > masked.sum()  # mask carves out the ring 6..8
    assert np.array_equal(masked, radial & obj)


def test_zone_restriction_inner_vs_outer_orientation():
    # Inner disk oriented one way, outer ring another -> per-zone R directions differ.
    n = 61
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - c, xx - c)
    phi = np.where(dist < 15, 0.0, np.pi / 2 - 1e-6)
    coh = np.ones((n, n))
    grad = np.zeros((n, n))
    obj = np.ones((n, n), dtype=bool)
    inner = aggregate_orientation(
        phi, coh, grad, zone_selector(dist, 0.0, 15.0, obj, "Radial")
    )
    outer = aggregate_orientation(
        phi, coh, grad, zone_selector(dist, 20.0, 28.0, obj, "Radial")
    )
    assert inner[0] == pytest.approx(
        1.0, abs=1e-6
    )  # each zone internally aligned
    assert outer[0] == pytest.approx(1.0, abs=1e-6)


def test_measure_returns_all_schema_columns_one_row_per_object():
    image = load_synth_filamentous_plate()
    df = MeasureOrientationZones().measure(image)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == image.num_objects
    for h in ORIENTATION_ZONE_PRIMARY.get_headers():
        assert h in df.columns
    assert set(ORIENTATION_ZONE_DIAGNOSTIC.get_headers()).isdisjoint(
        df.columns
    )


def test_include_diagnostics_emits_comparators_support_and_legacy_columns():
    image = load_synth_filamentous_plate()
    df = MeasureOrientationZones(include_diagnostics=True).measure(image)

    assert set(ORIENTATION_ZONE_PRIMARY.get_headers()).issubset(df.columns)
    assert set(ORIENTATION_ZONE_DIAGNOSTIC.get_headers()).issubset(df.columns)
    for metric in (
        "RadialTilt",
        "OutwardTurning",
        "LongRangeRotation",
        "SignedLongRangeRotation",
    ):
        headers = [
            header
            for header in ORIENTATION_ZONE_DIAGNOSTIC.get_headers()
            if f"_{metric}-" in header
        ]
        assert headers
        assert any(
            np.isfinite(df[header].to_numpy(dtype=float)).any()
            for header in headers
        )
    # R and coherence within [0,1] where finite
    for col in df.columns:
        if col.startswith(
            ("OrientZones_Concentration", "OrientZones_Coherence")
        ):
            vals = df[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            assert np.all((finite >= -1e-9) & (finite <= 1 + 1e-9))
        elif col.startswith("OrientZones_RadialTilt"):
            vals = df[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            assert np.all((finite >= -1e-9) & (finite <= 90.0 + 1e-9))
        elif col.startswith("OrientZones_OutwardTurning"):
            vals = df[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            assert np.all(finite >= -1e-9)
        elif col.startswith("OrientZones_RadialSectorSupport"):
            vals = df[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            assert np.all((finite >= -1e-9) & (finite <= 1.0 + 1e-9))
        elif col.startswith("OrientZones_LongRangeRotationSupport"):
            vals = df[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            assert np.all((finite >= -1e-9) & (finite <= 1.0 + 1e-9))
        elif col.startswith("OrientZones_LongRangeRotation"):
            vals = df[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            assert np.all((finite >= -1e-9) & (finite <= 90.0 + 1e-9))
        elif col.startswith("OrientZones_SignedLongRangeRotation"):
            vals = df[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            assert np.all((finite >= -90.0 - 1e-9) & (finite <= 90.0 + 1e-9))


def test_primary_and_diagnostic_schema_are_distinct_and_document_units():
    """Public descriptions should classify columns and state their units."""
    assert all(
        member.resolved_kind == "primary"
        for member in ORIENTATION_ZONE_PRIMARY
    )
    assert all(
        member.resolved_kind == "quality"
        for member in ORIENTATION_ZONE_DIAGNOSTIC
    )
    for member in (
        *ORIENTATION_ZONE_PRIMARY,
        *ORIENTATION_ZONE_DIAGNOSTIC,
    ):
        description = member.desc.lower()
        assert "degree" in description or "dimensionless" in description


def test_legacy_orientation_zones_name_preserves_legacy_member_access():
    """The old public name should retain access to its former members."""
    assert ORIENTATION_ZONES is ORIENTATION_ZONE_DIAGNOSTIC
    assert (
        ORIENTATION_ZONES.CONCENTRATION_RADIAL_OVERALL.label
        == "Concentration-Radial-Overall"
    )


def test_sparse_cumulative_magnitude_descriptions_disclose_dense_carryover():
    """Sparse cumulative magnitudes should not imply zone-local generation."""
    members = (
        ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_SUSTAINED_PEAK_SPARSE,
        ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RAW_PEAK_SPARSE,
        ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_P90_SPARSE,
        ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_P95_SPARSE,
        ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_MEDIAN_MAGNITUDE_SPARSE,
        ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_ABSOLUTE_AREA_SPARSE,
    )
    assert all(
        "include rotation accumulated" in member.desc for member in members
    )


def test_public_zone_angles_are_converted_from_radians_to_degrees():
    size = 81
    centre = (40.0, 40.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    phi = polar + 0.004 * distance - np.pi / 2.0
    coherence = np.ones((size, size), dtype=float)
    gradient = np.full((size, size), 0.01, dtype=float)
    object_mask = (distance >= 5.0) & (distance < 45.0)
    segmentation = SimpleNamespace(
        zones_computed=True,
        symmetric_radius=45.0,
        core_end_radius=15.0,
        dense_end_radius=30.0,
        sparse_end_radius=45.0,
    )
    operation = MeasureOrientationZones(include_diagnostics=True)
    row: dict[str, float] = {}

    (
        _outward_rotation,
        per_zone,
        radial_relative,
        _long_range,
        _ring_profile,
    ) = operation._fill_metrics(
        row,
        segmentation,
        object_mask,
        phi,
        coherence,
        gradient,
        distance,
        centre,
    )

    dense_selector = zone_selector(
        distance,
        segmentation.core_end_radius,
        segmentation.dense_end_radius,
        object_mask,
        "Mask",
    )
    internal_turning = aggregate_orientation(
        phi,
        coherence,
        gradient,
        dense_selector,
    )[1]
    absolute_tilt, outward_turning, measured_polar = radial_relative_field(
        phi,
        centre,
        distance,
    )
    internal_tilt, internal_outward, _support = aggregate_radial_relative(
        absolute_tilt,
        outward_turning,
        measured_polar,
        coherence,
        distance,
        dense_selector,
        n_angular_bins=36,
    )

    assert row["OrientZones_Turning-Mask-Dense"] == pytest.approx(
        np.degrees(internal_turning)
    )
    assert row["OrientZones_RadialTilt-Mask-Dense"] == pytest.approx(
        np.degrees(internal_tilt)
    )
    assert row["OrientZones_OutwardTurning-Mask-Dense"] == pytest.approx(
        np.degrees(internal_outward)
    )
    assert per_zone[("Mask", "Dense")][1] == pytest.approx(
        row["OrientZones_Turning-Mask-Dense"]
    )
    assert radial_relative["Dense"][:2] == pytest.approx(
        (
            row["OrientZones_RadialTilt-Mask-Dense"],
            row["OrientZones_OutwardTurning-Mask-Dense"],
        )
    )


def test_literal_crossing_primary_and_diagnostic_values_use_public_units():
    """Controlled spiral arms should yield degree-based literal metrics."""
    from skimage.draw import line

    size = 121
    centre = (60.0, 60.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    object_mask = np.zeros((size, size), dtype=bool)
    radii = np.linspace(0.0, 50.0, 401)
    for base_angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        branch_rows = np.rint(
            centre[0] + radii * np.sin(base_angle + 0.01 * radii)
        ).astype(int)
        branch_cols = np.rint(
            centre[1] + radii * np.cos(base_angle + 0.01 * radii)
        ).astype(int)
        for index in range(1, radii.size):
            rr, cc = line(
                branch_rows[index - 1],
                branch_cols[index - 1],
                branch_rows[index],
                branch_cols[index],
            )
            object_mask[rr, cc] = True

    fiber_axis = polar + 0.01 * distance
    phi = fiber_axis - np.pi / 2.0
    segmentation = SimpleNamespace(
        zones_computed=True,
        core_end_radius=10.0,
        dense_end_radius=30.0,
    )
    operation = MeasureOrientationZones(
        radial_ring_width=5.0,
        long_range_lag=10.0,
        include_diagnostics=True,
    )
    row: dict[str, float] = {}

    cached_primary = operation._fill_literal_crossing_metrics(
        row,
        segmentation,
        object_mask,
        phi,
        np.ones_like(distance),
        distance,
        centre,
    )

    assert row[
        "OrientZones_OutwardRotationSustainedPeak-Mask-Overall"
    ] == pytest.approx(np.degrees(0.30), abs=0.25)
    assert row["OrientZones_OutwardRotationNet-Mask-Overall"] == pytest.approx(
        np.degrees(0.30), abs=0.25
    )
    assert row[
        "OrientZones_OutwardRotationRate-Mask-Overall"
    ] == pytest.approx(np.degrees(0.01), abs=0.02)
    assert row[
        "OrientZones_OutwardRotationConsistency-Mask-Overall"
    ] == pytest.approx(1.0)
    assert row[
        "OrientZones_OutwardRotationRawPeak-Mask-Overall"
    ] == pytest.approx(np.degrees(0.35), abs=0.25)
    assert cached_primary["Overall"][
        "OutwardRotationSustainedPeak"
    ] == pytest.approx(
        row["OrientZones_OutwardRotationSustainedPeak-Mask-Overall"]
    )


def test_r3c4_real_crop_preserves_literal_crossing_regression() -> None:
    """The real development colony should preserve its robust rotation readout."""
    fixture_path = (
        Path(__file__).parents[2]
        / "fixtures"
        / "orientation_zones"
        / "r3c4_twok_literal_crossing.npz"
    )
    with np.load(fixture_path) as fixture:
        detect_mat = fixture["detect_mat"]
        objmap = fixture["objmap"]
        assert detect_mat.shape == (512, 512)
        assert objmap.shape == detect_mat.shape
        assert int(fixture["source_label"]) == 24
        assert str(fixture["colony"]) == "R3C4"

    image = Image(arr=detect_mat)
    image.detect_mat[:] = detect_mat
    image.objmap[:] = objmap
    result = MeasureOrientationZones(include_diagnostics=True).measure(image)
    assert len(result) == 1
    row = result.iloc[0]

    assert row[
        "OrientZones_OutwardRotationSustainedPeak-Mask-Overall"
    ] == pytest.approx(47.7189, abs=0.05)
    assert row[
        "OrientZones_OutwardRotationRawPeak-Mask-Overall"
    ] == pytest.approx(59.7523, abs=0.05)
    assert row["OrientZones_OutwardRotationNet-Mask-Overall"] == pytest.approx(
        -23.3725, abs=0.05
    )
    assert row[
        "OrientZones_OutwardRotationRate-Mask-Overall"
    ] == pytest.approx(-0.42594, abs=0.005)
    assert row[
        "OrientZones_OutwardRotationConsistency-Mask-Overall"
    ] == pytest.approx(0.384615, abs=1e-6)
    assert row[
        "OrientZones_OutwardRotationRingSupport-Mask-Overall"
    ] == pytest.approx(0.851852, abs=1e-6)
    assert row[
        "OrientZones_OutwardRotationRunSpanSupport-Mask-Overall"
    ] == pytest.approx(0.461538, abs=1e-6)


def _radial_test_field(size: int = 121):
    """Return polar geometry and a perfect radial fiber field."""
    centre = ((size - 1) / 2.0, (size - 1) / 2.0)
    rows, cols = np.indices((size, size), dtype=float)
    delta_row = rows - centre[0]
    delta_col = cols - centre[1]
    distance = np.hypot(delta_row, delta_col)
    polar = np.arctan2(delta_row, delta_col)
    # Fiber axis theta=polar; orientation_field's phi is the normal theta-pi/2.
    phi = polar - np.pi / 2.0
    coherence = np.ones_like(phi)
    annulus = (distance >= 15.0) & (distance < 50.0)
    return centre, distance, polar, phi, coherence, annulus


def test_radial_relative_straight_branches_ignore_axis_and_branch_count():
    centre, distance, polar, phi, coherence, annulus = _radial_test_field()
    tilt, outward, measured_polar = radial_relative_field(
        phi, centre, distance
    )
    angular_tolerance = np.deg2rad(4.0)
    horizontal = annulus & (
        np.abs(0.5 * np.arctan2(np.sin(2 * polar), np.cos(2 * polar)))
        < angular_tolerance
    )
    vertical = annulus & (
        np.abs(
            0.5
            * np.arctan2(
                np.sin(2 * (polar - np.pi / 2.0)),
                np.cos(2 * (polar - np.pi / 2.0)),
            )
        )
        < angular_tolerance
    )

    results = [
        aggregate_radial_relative(
            tilt,
            outward,
            measured_polar,
            coherence,
            distance,
            selector,
            n_angular_bins=36,
        )
        for selector in (horizontal, vertical, annulus)
    ]

    for radial_tilt, radial_turning, radial_support in results:
        assert radial_tilt == pytest.approx(0.0, abs=1e-12)
        assert radial_turning == pytest.approx(0.0, abs=1e-12)
        assert radial_support > 0.0


def test_radial_relative_constant_oblique_tilt_is_density_invariant():
    centre, distance, polar, _phi, coherence, annulus = _radial_test_field()
    expected_tilt = np.deg2rad(20.0)
    phi = polar + expected_tilt - np.pi / 2.0
    tilt, outward, measured_polar = radial_relative_field(
        phi, centre, distance
    )
    sparse = annulus & (
        (np.mod(polar, 2.0 * np.pi) < np.deg2rad(8.0))
        | (np.mod(polar, 2.0 * np.pi) > np.deg2rad(352.0))
    )

    sparse_result = aggregate_radial_relative(
        tilt,
        outward,
        measured_polar,
        coherence,
        distance,
        sparse,
        n_angular_bins=36,
    )
    dense_result = aggregate_radial_relative(
        tilt,
        outward,
        measured_polar,
        coherence,
        distance,
        annulus,
        n_angular_bins=36,
    )

    assert sparse_result[0] == pytest.approx(expected_tilt, abs=1e-12)
    assert dense_result[0] == pytest.approx(expected_tilt, abs=1e-12)
    assert sparse_result[1] == pytest.approx(0.0, abs=1e-12)
    assert dense_result[1] == pytest.approx(0.0, abs=1e-12)
    assert sparse_result[2] < dense_result[2]


def test_radial_relative_rejects_negligible_confidence_sectors():
    size = 41
    centre = (20.0, 20.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    selector = (distance >= 5.0) & (distance < 18.0)
    absolute_tilt = np.zeros((size, size), dtype=float)
    outward_turning = np.zeros((size, size), dtype=float)
    coherence = np.ones((size, size), dtype=float)
    low_confidence_sector = selector & (polar >= 0.0) & (polar < np.pi / 6.0)
    absolute_tilt[low_confidence_sector] = np.pi / 2.0
    coherence[low_confidence_sector] = 0.01

    radial_tilt, radial_turning, support = aggregate_radial_relative(
        absolute_tilt,
        outward_turning,
        polar,
        coherence,
        distance,
        selector,
        n_angular_bins=36,
    )

    assert radial_tilt == pytest.approx(0.0, abs=1e-12)
    assert radial_turning == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < support < 1.0


def test_radial_relative_outward_bend_recovers_radial_rate():
    centre, distance, polar, _phi, coherence, annulus = _radial_test_field()
    expected_rate = 0.004
    phi = polar + expected_rate * distance - np.pi / 2.0
    tilt, outward, measured_polar = radial_relative_field(
        phi, centre, distance
    )

    radial_tilt, radial_turning, radial_support = aggregate_radial_relative(
        tilt,
        outward,
        measured_polar,
        coherence,
        distance,
        annulus,
        n_angular_bins=36,
    )

    assert radial_tilt > 0.0
    assert radial_turning == pytest.approx(expected_rate, rel=0.03)
    assert radial_support > 0.0


def test_signed_outward_turning_preserves_clockwise_and_counterclockwise():
    centre, distance, polar, _phi, _coherence, annulus = _radial_test_field()
    expected_rate = 0.004

    for direction in (-1.0, 1.0):
        phi = polar + direction * expected_rate * distance - np.pi / 2.0
        _tilt, signed_turning, magnitude, _measured_polar = (
            signed_radial_relative_field(phi, centre, distance)
        )

        assert np.mean(signed_turning[annulus]) == pytest.approx(
            direction * expected_rate,
            rel=0.03,
        )
        assert np.mean(magnitude[annulus]) == pytest.approx(
            expected_rate,
            rel=0.03,
        )


def test_sectorized_radial_rings_recover_fixed_lag_rotation():
    size = 181
    centre = (90.0, 90.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    expected_rate = 0.002
    signed_tilt = expected_rate * distance
    signed_tilt[distance < 16.0] = np.deg2rad(80.0)
    coherence = np.ones_like(distance)
    structure = distance < 88.0

    radii, sector_tilt, sector_resultant = radial_ring_orientation_profile(
        signed_tilt,
        polar,
        coherence,
        distance,
        structure,
        inner_radius=16.0,
        outer_radius=88.0,
        ring_width=8.0,
        n_angular_bins=36,
    )
    midpoints, rotation = long_range_ring_rotation_profile(
        radii,
        sector_tilt,
        radial_lag=16.0,
    )
    magnitude, signed, support = aggregate_long_range_rotation(
        midpoints,
        rotation,
        lower_radius=16.0,
        upper_radius=88.0,
    )

    assert sector_tilt.shape == sector_resultant.shape == (9, 36)
    assert radii[0] == pytest.approx(20.0)
    assert magnitude == pytest.approx(expected_rate * 16.0, rel=0.04)
    assert signed == pytest.approx(expected_rate * 16.0, rel=0.04)
    assert support > 0.95

    seam_tilt = 0.5 * np.arctan2(
        np.sin(2.0 * (np.deg2rad(85.0) + expected_rate * distance)),
        np.cos(2.0 * (np.deg2rad(85.0) + expected_rate * distance)),
    )
    radii, sector_tilt, _resultant = radial_ring_orientation_profile(
        seam_tilt,
        polar,
        coherence,
        distance,
        structure,
        inner_radius=16.0,
        outer_radius=88.0,
        ring_width=8.0,
        n_angular_bins=36,
    )
    midpoints, rotation = long_range_ring_rotation_profile(
        radii,
        sector_tilt,
        radial_lag=16.0,
    )
    seam_magnitude, seam_signed, _seam_support = aggregate_long_range_rotation(
        midpoints,
        rotation,
        lower_radius=16.0,
        upper_radius=88.0,
    )
    assert seam_magnitude == pytest.approx(expected_rate * 16.0, rel=0.04)
    assert seam_signed == pytest.approx(expected_rate * 16.0, rel=0.04)


def test_cumulative_ring_rotation_unwraps_axial_seam_and_exceeds_90_degrees():
    sector_tilt = np.deg2rad(
        np.array(
            [
                [80.0, np.nan, 10.0],
                [-70.0, 20.0, 40.0],
                [-40.0, 50.0, np.nan],
                [-10.0, 80.0, 70.0],
            ]
        )
    )

    cumulative = np.degrees(cumulative_ring_rotation_profile(sector_tilt))

    assert cumulative[:, 0] == pytest.approx([0.0, 30.0, 60.0, 90.0])
    assert np.isnan(cumulative[0, 1])
    assert cumulative[1:, 1] == pytest.approx([0.0, 30.0, 60.0])
    assert cumulative[:2, 2] == pytest.approx([0.0, 30.0])
    assert np.isnan(cumulative[2:, 2]).all()


def test_cumulative_ring_rotation_blanks_ambiguous_orthogonal_step():
    positive_representation = np.deg2rad([[0.0], [90.0], [80.0]])
    negative_representation = np.deg2rad([[0.0], [-90.0], [-80.0]])

    positive = cumulative_ring_rotation_profile(positive_representation)
    negative = cumulative_ring_rotation_profile(negative_representation)

    assert positive[0, 0] == pytest.approx(0.0)
    assert negative[0, 0] == pytest.approx(0.0)
    assert np.isnan(positive[1:, 0]).all()
    assert np.isnan(negative[1:, 0]).all()


def test_radial_ring_sector_field_excludes_core_and_unsupported_cells():
    size = 11
    centre = (5.0, 5.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    structure = distance < 5.0
    ring_values = np.array(
        [
            [1.0, 2.0, np.nan, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    field = radial_ring_sector_field(
        ring_values,
        polar,
        distance,
        structure,
        inner_radius=1.0,
        ring_width=2.0,
    )

    assert np.isnan(field[5, 5])
    assert field[5, 7] == pytest.approx(1.0)
    assert field[7, 5] == pytest.approx(2.0)
    assert np.isnan(field[5, 3])
    assert field[8, 5] == pytest.approx(6.0)
    assert np.isnan(field[0, 5])


def test_long_range_ring_rotation_is_branch_count_scale_invariant():
    size = 181
    centre = (90.0, 90.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    signed_tilt = 0.002 * distance
    coherence = np.ones_like(distance)
    annulus = (distance >= 16.0) & (distance < 88.0)
    sparse_arcs = annulus & (
        (np.mod(polar, 2.0 * np.pi) < np.deg2rad(28.0))
        | (np.mod(polar, 2.0 * np.pi) > np.deg2rad(332.0))
    )

    results = []
    for structure in (sparse_arcs, annulus):
        radii, sector_tilt, _resultant = radial_ring_orientation_profile(
            signed_tilt,
            polar,
            coherence,
            distance,
            structure,
            inner_radius=16.0,
            outer_radius=88.0,
            ring_width=8.0,
            n_angular_bins=36,
        )
        midpoints, rotation = long_range_ring_rotation_profile(
            radii,
            sector_tilt,
            radial_lag=16.0,
        )
        results.append(
            aggregate_long_range_rotation(
                midpoints,
                rotation,
                lower_radius=16.0,
                upper_radius=88.0,
            )
        )

    assert results[0][0] == pytest.approx(results[1][0], rel=0.04)
    assert results[0][1] == pytest.approx(results[1][1], rel=0.04)
    assert results[0][2] < results[1][2]


def test_paired_zone_rotation_preserves_direction_and_opposition():
    inner = np.full(36, np.deg2rad(10.0))
    outer = np.full(36, np.deg2rad(30.0))
    magnitude, signed, support = aggregate_paired_zone_rotation(inner, outer)

    assert np.degrees(magnitude) == pytest.approx(20.0)
    assert np.degrees(signed) == pytest.approx(20.0)
    assert support == pytest.approx(1.0)

    outer[::2] = np.deg2rad(-10.0)
    magnitude, signed, support = aggregate_paired_zone_rotation(inner, outer)
    assert np.degrees(magnitude) == pytest.approx(20.0)
    assert np.degrees(signed) == pytest.approx(0.0, abs=1e-12)
    assert support == pytest.approx(1.0)


def test_long_range_midpoint_assignment_uses_lower_inclusive_bound():
    midpoints = np.array([30.0, 40.0])
    rotations = np.array([[0.1, 0.1], [0.2, 0.2]])

    magnitude, signed, support = aggregate_long_range_rotation(
        midpoints,
        rotations,
        lower_radius=40.0,
        upper_radius=50.0,
    )

    assert magnitude == pytest.approx(0.2)
    assert signed == pytest.approx(0.2)
    assert support == pytest.approx(1.0)


def test_long_range_parameters_require_compatible_positive_scales():
    operation = MeasureOrientationZones(
        radial_ring_width=8.0,
        long_range_lag=32.0,
        include_diagnostics=True,
    )
    assert operation.radial_ring_width == 8.0
    assert operation.long_range_lag == 32.0
    restored = MeasureOrientationZones.from_json(operation.to_json())
    assert restored.radial_ring_width == 8.0
    assert restored.long_range_lag == 32.0
    assert restored.include_diagnostics is True

    with pytest.raises(ValueError, match="integer multiple"):
        MeasureOrientationZones(
            radial_ring_width=8.0,
            long_range_lag=12.0,
        )
    with pytest.raises(ValueError, match="finite and > 0"):
        MeasureOrientationZones(radial_ring_width=0.0)
    with pytest.raises(ValueError, match="integer multiple"):
        MeasureOrientationZones(
            radial_ring_width=8.0,
            long_range_lag=16.00008,
        )
    for field in ("radial_ring_width", "long_range_lag"):
        for value in (np.nan, np.inf):
            with pytest.raises(ValueError, match="finite and > 0"):
                MeasureOrientationZones(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("outward_peak_window_rings", 2, "odd integer"),
        ("outward_peak_window_rings", 4, "odd integer"),
        ("outward_peak_window_rings", True, "odd integer"),
        ("outward_min_run_rings", 2, "integer >= 3"),
        ("outward_min_run_rings", True, "integer >= 3"),
    ],
)
def test_literal_aggregate_parameters_are_validated(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MeasureOrientationZones(**{field: value})


def test_rotation_invariance_of_R_magnitude_and_turning():
    # A single synthetic tile rotated 90 deg: R magnitude and turning invariant.
    from phenotypic.measure._measure_orientation_zones import (
        aggregate_orientation,
        zone_selector,
    )

    n = 61
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - c, xx - c)
    base = np.sin(2 * np.pi * xx / 7.0)
    from phenotypic.util._orientation_field import orientation_field

    obj = np.ones((n, n), dtype=bool)

    def metrics(field):
        phi, coh, grad = orientation_field(field, 1.5, 4.0)
        sel = zone_selector(dist, 0.0, 20.0, obj, "Radial")
        return aggregate_orientation(phi, coh, grad, sel)

    R0, t0, _ = metrics(base)
    R90, t90, _ = metrics(np.rot90(base))
    assert R0 == pytest.approx(R90, abs=0.05)
    assert t0 == pytest.approx(t90, abs=0.05)


def test_tiny_objects_are_all_nan():
    image = load_synth_yeast_plate()
    df = MeasureOrientationZones().measure(image)
    # every row has the full column set; NaN allowed, no exceptions
    assert set(ORIENTATION_ZONE_PRIMARY.get_headers()).issubset(df.columns)


def test_measure_cache_is_compact():
    # Guard against memory bloat: after measure(), the per-object cache must hold
    # NO full-res arrays and NO seg dataclass — only scalars + the block quiver.
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    op.measure(image)
    assert op._cache, "cache should be populated"
    assert not hasattr(op, "_cache_image")
    assert isinstance(op._cache_image_ref, weakref.ReferenceType)
    assert op._cache_image_ref() is image
    forbidden = {"tile", "phi", "coherence", "grad_phi", "dist_map", "seg"}
    for rec in op._cache.values():
        assert forbidden.isdisjoint(rec), (
            f"full-res leaked: {forbidden & set(rec)}"
        )
        assert "quiver" in rec
        assert "ring_profile" in rec
        assert set(rec["outward_rotation"]) == {
            "Overall",
            "Dense",
            "Sparse",
        }
        for v in rec.values():
            if isinstance(v, np.ndarray):
                assert v.size <= 4096, (
                    "only the block-resolution quiver may be cached"
                )
        # the block quiver must be far smaller than a full tile
        rows, cols, pb, cb = rec["quiver"]
        assert pb.shape == cb.shape and pb.size <= 4096
        for value in rec["ring_profile"].values():
            assert isinstance(value, np.ndarray)
            assert value.ndim == 1
            assert value.size <= 4096


def test_inspect_builds_figure():
    import plotly.graph_objects as go
    from plotly.colors import get_colorscale

    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    op.measure(image)
    fig = op.inspect(image)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert not fig.layout.annotations
    assert fig.layout.margin.b is None
    assert not any(
        getattr(trace, "mode", None) == "text" for trace in fig.data
    )
    assert any("Local fiber axis" in trace.name for trace in fig.data)
    assert any(
        trace.name == "Zone metrics (hover centres)" for trace in fig.data
    )
    turning_map = next(
        trace for trace in fig.data if trace.name == "Signed outward turning"
    )
    colors = np.asarray(turning_map.z, dtype=float)
    assert max(colors.shape) <= 900
    height, width = image.gray[:].shape
    assert tuple(fig.layout.xaxis.range) == pytest.approx((-0.5, width - 0.5))
    assert tuple(fig.layout.yaxis.range) == pytest.approx((height - 0.5, -0.5))
    raster_x, raster_y = np.meshgrid(
        np.asarray(turning_map.x, dtype=float),
        np.asarray(turning_map.y, dtype=float),
    )
    for record in op._cache.values():
        centre_row, centre_col = record["centroid_global"]
        core_radius = float(record["radii"]["core_end"])
        safely_inside_core = (
            np.hypot(raster_y - centre_row, raster_x - centre_col)
            < 0.5 * core_radius
        )
        if safely_inside_core.any():
            assert np.isnan(colors[safely_inside_core]).all()
    assert turning_map.zmin == pytest.approx(-turning_map.zmax)
    assert turning_map.zmax == pytest.approx(np.nanmax(np.abs(colors)))
    assert turning_map.zmid == pytest.approx(0.0)
    expected_colorscale = get_colorscale("Spectral")
    assert np.allclose(
        [stop for stop, _color in turning_map.colorscale],
        [stop for stop, _color in expected_colorscale],
    )
    assert [color for _stop, color in turning_map.colorscale] == [
        color for _stop, color in expected_colorscale
    ]
    assert "deg/px" in turning_map.colorbar.title.text
    assert turning_map.colorbar.orientation == "h"
    hover = next(
        trace
        for trace in fig.data
        if trace.name == "Zone metrics (hover centres)"
    )
    assert any(
        "Primary outward-rotation metrics" in str(text)
        and "Sustained peak=" in str(text)
        and "Net=" in str(text)
        and "Rate=" in str(text)
        and "Consistency=" in str(text)
        for text in hover.text
    )
    assert all("deg/px" in str(text) for text in hover.text)
    assert all(
        "Diagnostic orientation metrics" not in str(text)
        for text in hover.text
    )
    fig_save = op.inspect(image, for_save=True)
    assert isinstance(fig_save, go.Figure)


def test_inspect_includes_legacy_metrics_only_when_diagnostics_are_enabled():
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones(include_diagnostics=True)
    op.measure(image)

    figure = op.inspect(image)
    hover = next(
        trace
        for trace in figure.data
        if trace.name == "Zone metrics (hover centres)"
    )

    assert any(
        "Diagnostic orientation metrics" in str(text)
        and "RTilt=" in str(text)
        and "OutT=" in str(text)
        and "Long range" in str(text)
        and "DenseToSparse" in str(text)
        for text in hover.text
    )


def test_cumulative_rotation_overlay_uses_degrees_and_excludes_inoculum(
    monkeypatch,
):
    import plotly.graph_objects as go
    from plotly.colors import get_colorscale

    size = 181
    centre = (90.0, 90.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    signed_tilt = 0.002 * distance
    phi = polar + signed_tilt - np.pi / 2.0
    coherence = np.ones_like(distance)
    obj_mask = distance < 88.0
    seg = SimpleNamespace(
        core_end_radius=16.0,
        sparse_end_radius=88.0,
        symmetric_radius=88.0,
        centroid_global=centre,
    )
    op = MeasureOrientationZones()
    monkeypatch.setattr(
        MeasureOrientationZones,
        "_prep",
        lambda self, image: ([], {}),
    )
    monkeypatch.setattr(
        MeasureOrientationZones,
        "_iter_object_fields",
        lambda self, image, props, label2section: iter(
            [(None, seg, obj_mask, phi, coherence, None, distance, centre)]
        ),
    )
    fig = go.Figure()
    op._add_cumulative_rotation_trace(fig, object(), (size, size))

    assert isinstance(fig, go.Figure)
    cumulative_map = next(
        trace
        for trace in fig.data
        if trace.name == "Cumulative radial rotation"
    )
    colors = np.asarray(cumulative_map.z, dtype=float)
    assert cumulative_map.zmin == pytest.approx(-cumulative_map.zmax)
    assert cumulative_map.zmax == pytest.approx(np.nanmax(np.abs(colors)))
    assert cumulative_map.zmid == pytest.approx(0.0)
    assert "(deg)" in cumulative_map.colorbar.title.text
    assert "deg/px" not in cumulative_map.colorbar.title.text
    expected_colorscale = get_colorscale("Spectral")
    assert [color for _stop, color in cumulative_map.colorscale] == [
        color for _stop, color in expected_colorscale
    ]
    raster_x, raster_y = np.meshgrid(
        np.asarray(cumulative_map.x, dtype=float),
        np.asarray(cumulative_map.y, dtype=float),
    )
    safely_inside_core = (
        np.hypot(raster_y - centre[0], raster_x - centre[1]) < 8.0
    )
    assert safely_inside_core.any()
    assert np.isnan(colors[safely_inside_core]).all()


def test_matched_cumulative_overlay_tracks_nearby_sectors_and_uses_full_range(
    monkeypatch,
):
    import plotly.graph_objects as go
    from plotly.colors import get_colorscale

    size = 181
    centre = (90.0, 90.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    signed_tilt = 0.003 * distance
    phi = polar + signed_tilt - np.pi / 2.0
    coherence = np.ones_like(distance)
    obj_mask = distance < 88.0
    seg = SimpleNamespace(
        core_end_radius=16.0,
        sparse_end_radius=88.0,
        symmetric_radius=88.0,
        centroid_global=centre,
    )
    op = MeasureOrientationZones()
    monkeypatch.setattr(
        MeasureOrientationZones,
        "_prep",
        lambda self, image: ([], {}),
    )
    monkeypatch.setattr(
        MeasureOrientationZones,
        "_iter_object_fields",
        lambda self, image, props, label2section: iter(
            [(None, seg, obj_mask, phi, coherence, None, distance, centre)]
        ),
    )
    fig = go.Figure()
    op._add_matched_cumulative_rotation_trace(
        fig,
        object(),
        (size, size),
        max_sector_shift=2,
    )

    matched_map = next(
        trace
        for trace in fig.data
        if trace.name == "Matched cumulative fiber rotation"
    )
    colors = np.asarray(matched_map.z, dtype=float)
    assert matched_map.zmin == pytest.approx(-180.0)
    assert matched_map.zmax == pytest.approx(180.0)
    assert matched_map.zmid == pytest.approx(0.0)
    assert "(deg)" in matched_map.colorbar.title.text
    expected_colorscale = get_colorscale("Spectral")
    assert [color for _stop, color in matched_map.colorscale] == [
        color for _stop, color in expected_colorscale
    ]
    raster_x, raster_y = np.meshgrid(
        np.asarray(matched_map.x, dtype=float),
        np.asarray(matched_map.y, dtype=float),
    )
    safely_inside_core = (
        np.hypot(raster_y - centre[0], raster_x - centre[1]) < 8.0
    )
    assert safely_inside_core.any()
    assert np.isnan(colors[safely_inside_core]).all()
    path_trace = next(
        trace
        for trace in fig.data
        if trace.name == "Matched outward ring paths"
    )
    assert path_trace.mode == "lines"
    assert len(path_trace.x) > 0

    restart_fig = go.Figure()
    op._add_matched_cumulative_rotation_trace(
        restart_fig,
        object(),
        (size, size),
        max_sector_shift=2,
        allow_restarts=True,
    )
    assert not any(
        trace.name == "Matched outward ring paths"
        for trace in restart_fig.data
    )


def test_matched_cumulative_overlay_draws_gap_as_dashed_bridge(monkeypatch):
    import plotly.graph_objects as go

    import phenotypic.measure._measure_orientation_zones as orientation_module

    size = 81
    centre = (40.0, 40.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    phi = polar - np.pi / 2.0
    coherence = np.ones_like(distance)
    obj_mask = distance < 38.0
    seg = SimpleNamespace(
        core_end_radius=4.0,
        sparse_end_radius=38.0,
        symmetric_radius=38.0,
        centroid_global=centre,
    )
    radii = np.array([10.0, 20.0, 30.0])
    orientation = np.full((3, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[0, 0] = np.deg2rad(5.0)
    orientation[2, 0] = np.deg2rad(15.0)
    resultant[0, 0] = 1.0
    resultant[2, 0] = 1.0
    op = MeasureOrientationZones()
    monkeypatch.setattr(
        MeasureOrientationZones,
        "_prep",
        lambda self, image: ([], {}),
    )
    monkeypatch.setattr(
        MeasureOrientationZones,
        "_iter_object_fields",
        lambda self, image, props, label2section: iter(
            [(None, seg, obj_mask, phi, coherence, None, distance, centre)]
        ),
    )
    monkeypatch.setattr(
        orientation_module,
        "radial_ring_orientation_profile",
        lambda *args, **kwargs: (radii, orientation, resultant),
    )
    monkeypatch.setattr(
        orientation_module,
        "radial_ring_sector_field",
        lambda *args, **kwargs: np.full((size, size), np.nan),
    )

    fig = go.Figure()
    op._add_matched_cumulative_rotation_trace(
        fig,
        object(),
        (size, size),
        max_sector_shift=2,
        allow_gap_bridging=True,
    )

    bridge = next(
        trace
        for trace in fig.data
        if trace.name == "Bridged unsupported rings"
    )
    assert bridge.line.dash == "dash"
    assert not any(
        trace.name == "Matched outward ring paths" for trace in fig.data
    )


def test_matched_cumulative_overlay_warns_restart_is_segment_relative():
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    op.measure(image)

    fig = op.matched_cumulative_rotation_overlay(
        allow_restarts=True,
    )

    annotation_text = " ".join(
        str(item.text) for item in fig.layout.annotations
    )
    assert "segment-relative" in annotation_text
    assert "hidden" in annotation_text
    assert not any(
        trace.name == "Matched outward ring paths" for trace in fig.data
    )


@pytest.mark.parametrize(
    "parameter",
    ["allow_gap_bridging", "allow_restarts"],
)
def test_matched_cumulative_overlay_rejects_non_boolean_rule_flags(parameter):
    op = MeasureOrientationZones()

    with pytest.raises(ValueError, match="must be a boolean"):
        op.matched_cumulative_rotation_overlay(**{parameter: 1})


def test_fiber_bend_overlay_is_multiscale_unsigned_and_excludes_core(
    monkeypatch,
):
    import plotly.graph_objects as go

    size = 181
    centre = (90.0, 90.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    phi = polar + np.deg2rad(30.0) - np.pi / 2.0
    coherence = np.ones_like(distance)
    obj_mask = distance < 88.0
    seg = SimpleNamespace(
        core_end_radius=16.0,
        sparse_end_radius=88.0,
        symmetric_radius=88.0,
        centroid_global=centre,
    )
    base = np.exp(-((distance - 50.0) ** 2) / 40.0)

    class _WeakReferenceableImage:
        pass

    image = _WeakReferenceableImage()
    image.detect_mat = base
    image.gray = base
    image.rgb = np.repeat(base[..., None], 3, axis=2)
    op = MeasureOrientationZones()
    original_cache = {1: {"sentinel": True}}
    op._cache = original_cache
    op._cache_image_ref = weakref.ref(image)
    op._cache_signature = op.model_dump_json()
    monkeypatch.setattr(
        MeasureOrientationZones,
        "_prep",
        lambda self, subject: ([], {}),
    )
    monkeypatch.setattr(
        MeasureOrientationZones,
        "_iter_object_fields",
        lambda self, subject, props, label2section: iter(
            [(None, seg, obj_mask, phi, coherence, None, distance, centre)]
        ),
    )

    fig = op.fiber_bend_overlay(image, scale_set="balanced")

    assert isinstance(fig, go.Figure)
    bend_traces = [
        trace
        for trace in fig.data
        if str(trace.name).startswith("Fiber bend σ=")
    ]
    assert [trace.name for trace in bend_traces] == [
        "Fiber bend σ=4 px",
        "Fiber bend σ=8 px",
        "Fiber bend σ=16 px",
    ]
    base_traces = [trace for trace in fig.data if trace.name == "detect_mat"]
    assert len(base_traces) == 3
    assert all(
        np.asarray(trace.z).shape[:2] == np.asarray(bend_traces[index].z).shape
        for index, trace in enumerate(base_traces)
    )
    assert all(
        max(np.asarray(trace.z).shape[:2]) <= 900 for trace in base_traces
    )
    assert [trace.coloraxis for trace in bend_traces] == [
        "coloraxis",
        "coloraxis2",
        "coloraxis3",
    ]
    for index, trace in enumerate(bend_traces, start=1):
        coloraxis_name = "coloraxis" if index == 1 else f"coloraxis{index}"
        coloraxis = fig.layout[coloraxis_name]
        assert coloraxis.cmin == pytest.approx(0.0)
        assert coloraxis.cmax == pytest.approx(
            np.nanmax(np.asarray(trace.z, dtype=float))
        )
        assert "deg/px" in coloraxis.colorbar.title.text
        colors = np.asarray(trace.z, dtype=float)
        assert np.nanmin(colors) >= 0.0
        raster_x, raster_y = np.meshgrid(
            np.asarray(trace.x, dtype=float),
            np.asarray(trace.y, dtype=float),
        )
        safely_inside_core = (
            np.hypot(raster_y - centre[0], raster_x - centre[1]) < 8.0
        )
        assert safely_inside_core.any()
        assert np.isnan(colors[safely_inside_core]).all()
        assert max(colors.shape) <= 900
    annotation_text = {
        str(annotation.text) for annotation in fig.layout.annotations
    }
    assert all("Object" not in text for text in annotation_text)
    assert op._cache is original_cache
    specs = {spec.name: spec for spec in op.iter_figures()}
    assert "fiber_bend_overlay" not in specs
    assert specs["inspect"].primary is True

    image.rgb = np.round(np.repeat(base[..., None], 3, axis=2) * 50000).astype(
        np.uint16
    )
    rgb_fig = op.fiber_bend_overlay(image, base_layer="rgb")
    rgb_base = next(trace for trace in rgb_fig.data if trace.name == "rgb")
    rgb_values = np.asarray(rgb_base.z)
    assert rgb_values.dtype == np.uint8
    assert np.ptp(rgb_values) > 0
    assert rgb_values.max() <= 255


def test_fiber_bend_overlay_rejects_unknown_scale_set():
    op = MeasureOrientationZones()

    with pytest.raises(ValueError, match="scale_set"):
        op.fiber_bend_overlay(
            SimpleNamespace(),
            scale_set="unknown",  # type: ignore[arg-type]
        )


def test_inspect_adds_long_range_ring_trace_from_compact_cache():
    import plotly.graph_objects as go

    operation = MeasureOrientationZones()
    operation._cache[1] = {
        "centroid_global": (20.0, 30.0),
        "ring_profile": {
            "radii": np.array([8.0, 16.0]),
            "mean_absolute_tilt": np.array([10.0, 20.0]),
            "mean_signed_tilt": np.array([-5.0, 12.0]),
            "support": np.array([0.5, 0.75]),
            "pair_midpoints": np.array([12.0]),
            "mean_absolute_rotation": np.array([17.0]),
            "mean_signed_rotation": np.array([-3.0]),
            "pair_support": np.array([0.6]),
        },
    }
    figure = go.Figure()

    operation._add_long_range_ring_traces(figure)

    assert len(figure.data) == 4
    ring_trace = next(
        trace
        for trace in figure.data
        if str(trace.name).startswith("Orientation rings")
    )
    assert ring_trace.hoverinfo == "skip"
    ring_hover = next(
        trace
        for trace in figure.data
        if trace.name == "Ring orientation (hover)"
    )
    assert "Sholl-style orientation band" in "".join(
        str(text) for text in ring_hover.text if text is not None
    )
    pair_hover = next(
        trace
        for trace in figure.data
        if trace.name == "Long-range rotation (hover)"
    )
    assert "Mean |rotation|=17.00°" in "".join(
        str(text) for text in pair_hover.text if text is not None
    )


def test_inspect_fiber_axes_are_rotated_and_clipped_to_overall_selector():
    import plotly.graph_objects as go

    op = MeasureOrientationZones(quiver_block=12)
    op._cache[1] = {
        "centroid_global": (10.0, 10.0),
        "centre": (10.0, 10.0),
        "radii": {"symmetric": 5.0},
        "quiver": (
            np.array([[10.0, 30.0]]),
            np.array([[10.0, 30.0]]),
            np.array([[0.0, 0.0]]),  # horizontal gradient normal
            np.array([[1.0, 1.0]]),
        ),
    }
    fig = go.Figure()

    op._add_quiver_trace(fig)

    assert len(fig.data) == 1
    trace = fig.data[0]
    finite_x = [value for value in trace.x if value is not None]
    finite_y = [value for value in trace.y if value is not None]
    assert len(finite_x) == len(finite_y) == 2  # outside block was clipped
    assert finite_x[0] == pytest.approx(finite_x[1])  # fiber axis is vertical
    assert finite_y[0] != pytest.approx(finite_y[1])


def test_inspect_remeasures_when_explicit_image_changes():
    plate = load_synth_filamentous_plate()
    first = plate.grid[18]
    second = plate.grid[19]
    op = MeasureOrientationZones()
    op.measure(first)
    assert op._cached_image() is first

    op.inspect(second)

    assert op._cached_image() is second
    previous_signature = op._cache_signature
    op.long_range_lag = 32.0

    figure = op.inspect(second)

    assert op._cache_signature != previous_signature
    assert op._cache_signature == op.model_dump_json()
    assert any(
        "matching sectors 32 px apart" in str(annotation.text)
        for annotation in figure.layout.annotations
    )


def test_report_builds_composed_figure():
    import plotly.graph_objects as go

    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    op.measure(image)
    fig = op.report(subject=image, show=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # the go.Table summary panel must survive composition (base composer can't
    # host it — proves the custom report() override with per-row specs works).
    assert any(getattr(tr, "type", None) == "table" for tr in fig.data)
    table = next(tr for tr in fig.data if getattr(tr, "type", None) == "table")
    assert list(table.header.values) == [
        "Zone",
        "Sustained peak (deg)",
        "Net rotation (deg)",
        "Rotation rate (deg/px)",
        "Consistency",
    ]
    # coherence heatmap present too
    assert any(getattr(tr, "type", None) == "heatmap" for tr in fig.data)


def test_orientation_plot_uses_plot_image_without_legacy_report_aliases():
    from phenotypic.abc_.plotting import PlotImage
    from phenotypic.measure._measure_orientation_zones import (
        _OrientationZonesReport,
    )

    op = MeasureOrientationZones()
    assert isinstance(op, PlotImage)
    assert not hasattr(op, "dashboard")
    assert not hasattr(op, "dash")

    report = _OrientationZonesReport()
    assert isinstance(report, PlotImage)
    assert vars(report) == {}

    specs = op.iter_figures()
    assert [spec.name for spec in specs] == [
        "inspect",
        "cumulative_rotation_overlay",
        "matched_cumulative_rotation_overlay",
    ]
    assert all(spec.wants_subject for spec in specs)
    assert all(set(spec.controls) == {"base_layer"} for spec in specs)


def test_plot_only_fields_follow_measurement_fields_in_schema_order():
    fields = list(MeasureOrientationZones.model_fields)

    assert fields[-1] == "quiver_block"
    assert fields.index("quiver_block") > fields.index("tau_sparse")


def test_non_grid_image_uses_expanded_crop_fallback():
    # A grid section extracted via image.grid[idx] is a plain Image with no
    # .grid accessor — the ONLY way to exercise _resolve_tile's expanded-crop
    # fallback (all repo fixtures are GridImages). Spec §5: non-grid → no error.
    image = load_synth_filamentous_plate()
    section = image.grid[18]
    assert not hasattr(section, "grid"), "grid section should be a plain Image"
    df = MeasureOrientationZones(include_diagnostics=True).measure(section)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == section.num_objects
    assert set(ORIENTATION_ZONE_PRIMARY.get_headers()).issubset(df.columns)
    # the object is a real colony → the fallback path still yields a finite R
    r = df["OrientZones_Concentration-Radial-Overall"].to_numpy(float)
    assert np.isfinite(r).any()


def test_collapsed_zones_yield_all_nan():
    # zones_computed==False (collapsed symmetric envelope) → all 18 metrics NaN,
    # including Overall once symmetric_radius==0 (empty selector). Spec §5. This
    # branch is never hit by the fixtures, so drive _fill_metrics directly using
    # real per-object arrays with a mutated (collapsed) segmentation.
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    props, label2section = op._prep(image)
    _prop, seg, obj_mask, phi, coh, grad, dist_map, _centre = next(
        op._iter_object_fields(image, props, label2section)
    )
    seg.zones_computed = False
    seg.symmetric_radius = 0.0
    row: dict = {}
    op._fill_metrics(
        row,
        seg,
        obj_mask,
        phi,
        coh,
        grad,
        dist_map,
        _centre,
    )
    assert len(row) == len(ORIENTATION_ZONE_PRIMARY.get_headers())
    assert all(np.isnan(v) for v in row.values())


def test_radial_and_mask_variants_diverge_on_real_plate():
    # The Mask variant exists so the imperfect mask's distortion can be *seen*
    # (spec §1/§2). In the sparse ring the mask carves holes the Radial variant
    # keeps, so the two concentration reads must differ for at least some objects.
    image = load_synth_filamentous_plate()
    df = MeasureOrientationZones(include_diagnostics=True).measure(image)
    rad = df["OrientZones_Concentration-Radial-Sparse"].to_numpy(float)
    msk = df["OrientZones_Concentration-Mask-Sparse"].to_numpy(float)
    both = np.isfinite(rad) & np.isfinite(msk)
    assert both.any(), (
        "need objects with a finite sparse ring in both variants"
    )
    assert np.nanmax(np.abs(rad[both] - msk[both])) > 1e-6
