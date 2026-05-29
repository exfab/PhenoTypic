"""Tests for ManualRefine: init, centers coercion, _operate, and napari.

Since the pydantic migration ``centers`` is stored as a JSON-native
``list[tuple[int, int]]`` (not an ``np.ndarray``). A
``field_validator(mode="before")`` coerces ``np.ndarray`` / list / tuple
input to that list form — replacing the ``PointPickerMixin.__setattr__``
coercion that was removed when the operation became a pydantic model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from phenotypic.refine import ManualRefine


# ---- Helpers ----


def _label_centroid(objmap: np.ndarray, label: int) -> tuple[int, int]:
    """Return the (y, x) centroid of *label* in *objmap* — a pixel well inside
    the object body, so a small footprint stamp won't graze neighbours."""
    from scipy.ndimage import center_of_mass

    cy, cx = center_of_mass(objmap == label)
    cy_int, cx_int = int(round(cy)), int(round(cx))
    # Centroid can fall on a nearby background pixel for non-convex labels;
    # nudge to the nearest pixel actually carrying the label.
    if objmap[cy_int, cx_int] != label:
        ys, xs = np.where(objmap == label)
        d2 = (ys - cy_int) ** 2 + (xs - cx_int) ** 2
        k = int(np.argmin(d2))
        cy_int, cx_int = int(ys[k]), int(xs[k])
    return cy_int, cx_int


# ---- Init Tests ----


class TestManualSelectorInit:
    """Field defaults and ``centers`` coercion to a list of pairs."""

    def test_default_init(self):
        sel = ManualRefine()
        assert sel.centers is None
        assert sel.shape == "disk"
        assert sel.width == 15

    def test_init_with_centers_list(self):
        sel = ManualRefine(centers=[[50, 60], [100, 120]])
        # Stored as a JSON-native list of (y, x) pairs.
        assert sel.centers == [(50, 60), (100, 120)]

    def test_init_with_tuples(self):
        sel = ManualRefine(centers=[(10, 20), (30, 40)])
        assert sel.centers == [(10, 20), (30, 40)]

    def test_init_coerces_ndarray_to_list(self):
        arr = np.array([[5, 6]])
        sel = ManualRefine(centers=arr)
        # The before-validator normalizes an ndarray to a list of pairs.
        assert sel.centers == [(5, 6)]

    def test_setattr_coercion(self):
        sel = ManualRefine()
        sel.centers = [[10, 20]]
        assert sel.centers == [(10, 20)]

    def test_setattr_none_passthrough(self):
        sel = ManualRefine(centers=[[10, 20]])
        sel.centers = None
        assert sel.centers is None

    def test_inherits_point_picker_mixin(self):
        """ManualRefine mixes in PointPickerMixin and exposes its marker."""
        from phenotypic.tools_.mixin import PointPickerMixin

        sel = ManualRefine()
        assert isinstance(sel, PointPickerMixin)
        assert sel._point_picker_param_name == "centers"


# ---- _operate Tests ----


class TestManualSelectorOperate:
    """_operate behaviour via .apply()."""

    def test_none_centers_is_noop(self, synth_plate_detected):
        before = synth_plate_detected.objmap[:].copy()
        result = ManualRefine(centers=None).apply(synth_plate_detected, inplace=False)
        np.testing.assert_array_equal(result.objmap[:], before)

    def test_empty_centers_is_noop(self, synth_plate_detected):
        before = synth_plate_detected.objmap[:].copy()
        result = ManualRefine(centers=np.empty((0, 2), dtype=int)).apply(
                synth_plate_detected, inplace=False
        )
        np.testing.assert_array_equal(result.objmap[:], before)

    def test_empty_objmap_is_noop(self, synth_plate_detected):
        detected = synth_plate_detected.copy()
        detected.objmap[:] = np.zeros_like(detected.objmap[:])
        result = ManualRefine(centers=[(10, 10), (50, 50)], width=15).apply(
                detected, inplace=False
        )
        assert not result.objmap[:].any()
        assert not result.objmask[:].any()

    def test_single_center_keeps_only_target_label(self, synth_plate_detected):
        original_objmap = synth_plate_detected.objmap[:]
        target_label = int(np.unique(original_objmap)[1])  # first non-zero
        cy, cx = _label_centroid(original_objmap, target_label)

        result = ManualRefine(centers=[(cy, cx)], width=3).apply(
                synth_plate_detected, inplace=False
        )

        surviving = set(np.unique(result.objmap[:])) - {0}
        assert surviving == {target_label}
        # Original label is preserved (not relabelled to 1)
        assert result.objmap[cy, cx] == target_label
        # objmask view stays consistent with objmap
        np.testing.assert_array_equal(result.objmask[:], result.objmap[:] > 0)

    def test_multiple_centers_keep_multiple_labels(self, synth_plate_detected):
        original_objmap = synth_plate_detected.objmap[:]
        labels = np.unique(original_objmap)
        labels = labels[labels > 0]
        # Pick two well-separated labels
        target_a = int(labels[0])
        target_b = int(labels[len(labels) // 2])
        cy_a, cx_a = _label_centroid(original_objmap, target_a)
        cy_b, cx_b = _label_centroid(original_objmap, target_b)

        result = ManualRefine(
                centers=[(cy_a, cx_a), (cy_b, cx_b)], width=3
        ).apply(synth_plate_detected, inplace=False)

        surviving = set(np.unique(result.objmap[:])) - {0}
        assert surviving == {target_a, target_b}
        assert result.objmap[cy_a, cx_a] == target_a
        assert result.objmap[cy_b, cx_b] == target_b

    def test_center_on_background_drops_everything(self, synth_plate_detected):
        original_objmap = synth_plate_detected.objmap[:]
        # Find a background pixel far from any object
        ys, xs = np.where(original_objmap == 0)
        # Pick one arbitrarily; width=3 keeps the stamp tightly in background
        idx = len(ys) // 2
        cy, cx = int(ys[idx]), int(xs[idx])
        # Guard: confirm a 3x3 neighbourhood around (cy, cx) is entirely background
        # so the disk footprint of width=3 doesn't clip a nearby object.
        while original_objmap[
            max(0, cy - 2): cy + 3, max(0, cx - 2): cx + 3
        ].any():
            idx += 1
            cy, cx = int(ys[idx]), int(xs[idx])

        result = ManualRefine(centers=[(cy, cx)], width=3).apply(
                synth_plate_detected, inplace=False
        )
        assert not result.objmap[:].any()
        assert not result.objmask[:].any()

    def test_any_overlap_keeps_object(self):
        """A footprint spanning two separate labels keeps both."""
        from phenotypic import Image

        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        img = Image(rgb)
        objmap = np.zeros((50, 50), dtype=np.uint16)
        objmap[20:25, 10:20] = 5  # label 5
        objmap[20:25, 22:32] = 9  # label 9 (gap of 2 pixels between them)
        img.objmap[:] = objmap

        # Center at (22, 21) with width=15 (disk radius 7) spans both labels
        result = ManualRefine(centers=[(22, 21)], width=15).apply(
                img, inplace=False
        )
        surviving = set(np.unique(result.objmap[:])) - {0}
        assert surviving == {5, 9}

    def test_off_image_coordinates_do_not_raise(self, synth_plate_detected):
        # Off-image centers should be clipped by _stamp_footprint without errors.
        detected = synth_plate_detected.copy()
        h, w = detected.objmap[:].shape
        result = ManualRefine(
                centers=[(-10, -10), (h + 50, w + 50)], width=15
        ).apply(detected, inplace=False)
        # No stamp lands inside the image, so nothing is selected.
        assert not result.objmap[:].any()

    def test_protected_image_data(self, synth_plate_detected):
        """rgb, gray, detect_mat must not be modified."""
        detected = synth_plate_detected.copy()
        rgb = detected.rgb[:].copy()
        gray = detected.gray[:].copy()
        detect_mat = detected.detect_mat[:].copy()

        ys, xs = np.where(detected.objmap[:] > 0)
        cy, cx = int(ys[0]), int(xs[0])
        result = ManualRefine(centers=[(cy, cx)], width=3).apply(
                detected, inplace=False
        )

        np.testing.assert_array_equal(result.rgb[:], rgb)
        np.testing.assert_array_equal(result.gray[:], gray)
        np.testing.assert_array_equal(result.detect_mat[:], detect_mat)

    def test_inplace_vs_copy(self, synth_plate_detected):
        detected = synth_plate_detected.copy()
        original_objmap = detected.objmap[:].copy()

        ys, xs = np.where(original_objmap > 0)
        cy, cx = int(ys[0]), int(xs[0])
        sel = ManualRefine(centers=[(cy, cx)], width=3)

        result_copy = sel.apply(detected, inplace=False)
        np.testing.assert_array_equal(detected.objmap[:], original_objmap)
        assert not np.array_equal(result_copy.objmap[:], original_objmap)

        detected2 = synth_plate_detected.copy()
        sel.apply(detected2, inplace=True)
        assert not np.array_equal(detected2.objmap[:], original_objmap)


# ---- napari Tests ----


class TestManualSelectorNapari:
    """napari() integration via mocked PointPickerWidget."""

    def test_napari_sets_centers(self):
        sel = ManualRefine()
        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            mock_instance = MockWidget.return_value
            mock_instance.run.return_value = np.array([[50, 60], [100, 120]])
            result = sel.napari(MagicMock())

        # napari's setattr triggers the centers before-validator under
        # validate_assignment, normalizing the ndarray to a list of pairs.
        assert sel.centers == [(50, 60), (100, 120)]
        assert result is sel

    def test_napari_empty_result_preserves_existing_centers(self):
        original = [[10, 20], [30, 40]]
        sel = ManualRefine(centers=original)
        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            mock_instance = MockWidget.return_value
            mock_instance.run.return_value = np.empty((0, 2))
            sel.napari(MagicMock())
        np.testing.assert_array_equal(sel.centers, original)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
