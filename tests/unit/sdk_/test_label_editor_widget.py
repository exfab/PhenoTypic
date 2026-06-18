"""Tests for LabelEditorWidget and _LabelEditorPanelLogic."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# LabelEditorWidget public API
# ---------------------------------------------------------------------------


class TestLabelEditorWidget:
    def test_run_raises_import_error_without_napari(self):
        from phenotypic.sdk_.napari_ import LabelEditorWidget

        w = LabelEditorWidget()
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            False,
        ):
            with pytest.raises(ImportError, match="napari is required"):
                w.run(MagicMock(), "objmap")


# ---------------------------------------------------------------------------
# _LabelEditorPanelLogic (tested without real Qt)
# ---------------------------------------------------------------------------


def _make_mock_panel(*, image, accessor_name: str, layer_data: np.ndarray) -> MagicMock:
    """Mimic a label-editor panel for logic testing without Qt.

    Build a ``MagicMock`` with the same attributes and bind the real
    ``_LabelEditorPanelLogic`` methods so ``_save``/``_discard`` run against a
    real Image without constructing a QWidget.
    """
    from phenotypic.sdk_.napari_._label_editor_widget import _LabelEditorPanelLogic

    panel = MagicMock()

    labels_layer = MagicMock()
    labels_layer.data = layer_data.copy()
    panel._labels_layer = labels_layer

    panel._image = image
    panel._accessor_name = accessor_name
    panel._viewer = MagicMock()
    panel.saved_labels = None

    panel._save = lambda: _LabelEditorPanelLogic._save(panel)
    panel._discard = lambda: _LabelEditorPanelLogic._discard(panel)

    return panel


@pytest.fixture
def detected_image():
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector

    image = load_synth_yeast_plate()
    image = OtsuDetector().apply(image)
    return image


class TestSaveObjmap:
    """objmap save preserves the edited integer labels verbatim."""

    def test_save_writes_back_preserving_ids(self, detected_image):
        edited = detected_image.objmap[:].copy()
        # Stamp a small non-contiguous high label ID to prove IDs are preserved.
        edited[0:3, 0:3] = 777
        panel = _make_mock_panel(
            image=detected_image, accessor_name="objmap", layer_data=edited
        )

        panel._save()

        np.testing.assert_array_equal(detected_image.objmap[:], edited)
        assert 777 in np.unique(detected_image.objmap[:])
        assert panel.saved_labels is not None
        panel._viewer.close.assert_called_once()


class TestSaveObjmask:
    """objmask save binarizes then relabels (sequential IDs)."""

    def test_save_binarizes_and_relabels(self, detected_image):
        # Build a 2-blob binary layer with a non-binary stray value.
        mask = np.zeros(detected_image.objmask.shape, dtype=np.uint8)
        mask[5:10, 5:10] = 1
        mask[20:25, 20:25] = 5  # stray non-1 value -> must be binarized
        panel = _make_mock_panel(
            image=detected_image, accessor_name="objmask", layer_data=mask
        )

        panel._save()

        result_mask = detected_image.objmask[:]
        np.testing.assert_array_equal(result_mask, mask > 0)
        # The stray value 5 must have been binarized, not stored verbatim:
        # relabel produces sequential IDs [1, 2] and no label equals 5.
        objmap = detected_image.objmap[:]
        assert 5 not in np.unique(objmap)
        assert objmap.max() == 2
        labels = np.unique(objmap)
        labels = labels[labels > 0]
        np.testing.assert_array_equal(labels, np.array([1, 2], dtype=labels.dtype))
        panel._viewer.close.assert_called_once()


class TestDiscard:
    """Discard closes the viewer and leaves the image untouched."""

    def test_discard_no_mutation(self, detected_image):
        before = detected_image.objmap[:].copy()
        edited = detected_image.objmap[:].copy()
        edited[0:3, 0:3] = 777
        panel = _make_mock_panel(
            image=detected_image, accessor_name="objmap", layer_data=edited
        )

        panel._discard()

        np.testing.assert_array_equal(detected_image.objmap[:], before)
        assert panel.saved_labels is None
        panel._viewer.close.assert_called_once()


class TestDrawMethod:
    """Tests for NapariLabelsMixin.draw()."""

    def test_draw_raises_import_error_without_napari(self, detected_image):
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            False,
        ):
            with pytest.raises(ImportError, match="napari is required"):
                detected_image.objmap.draw()

    def test_draw_delegates_and_returns_root_image(self, detected_image):
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            True,
        ), patch(
            "phenotypic.sdk_.napari_.LabelEditorWidget.run"
        ) as mock_run:
            mock_run.return_value = None

            result = detected_image.objmap.draw()

            mock_run.assert_called_once()
            # First positional arg is the root image, second is the accessor name.
            args, _ = mock_run.call_args
            assert args[0] is detected_image
            assert args[1] == "objmap"
            assert result is detected_image

    def test_draw_passes_objmask_accessor_name(self, detected_image):
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            True,
        ), patch(
            "phenotypic.sdk_.napari_.LabelEditorWidget.run"
        ) as mock_run:
            mock_run.return_value = None

            detected_image.objmask.draw()

            args, _ = mock_run.call_args
            assert args[1] == "objmask"


class TestAddImageLayer:
    """add_image_layer sets non-stretched, dtype-aware contrast limits."""

    def _fresh_viewer(self):
        viewer = MagicMock()
        viewer.layers.__getitem__.side_effect = KeyError  # force add_image branch
        return viewer

    def test_integer_layer_uses_full_dtype_range(self):
        from phenotypic.sdk_.napari_ import add_image_layer

        viewer = self._fresh_viewer()
        arr = np.full((10, 10, 3), 200, dtype=np.uint8)  # RGB uint8, range 200..200

        add_image_layer(viewer, arr, name="rgb")

        _, kw = viewer.add_image.call_args
        assert kw["contrast_limits"] == (0, 255)  # NOT (200, 200)
        assert kw["rgb"] is True

    def test_float_layer_uses_unit_range_not_minmax(self):
        from phenotypic.sdk_.napari_ import add_image_layer

        viewer = self._fresh_viewer()
        arr = np.full((10, 10), 0.7, dtype=np.float32)  # data range 0.7..0.7

        add_image_layer(viewer, arr, name="gray")

        _, kw = viewer.add_image.call_args
        assert kw["contrast_limits"] == (0.0, 1.0)  # NOT (0.7, 0.7) auto-stretch
        assert kw["rgb"] is False

    def test_existing_layer_is_updated_in_place(self):
        from phenotypic.sdk_.napari_ import add_image_layer

        viewer = MagicMock()
        existing = MagicMock()
        viewer.layers.__getitem__.return_value = existing  # layer already present
        arr = np.full((10, 10), 0.4, dtype=np.float32)

        add_image_layer(viewer, arr, name="gray")

        viewer.add_image.assert_not_called()
        np.testing.assert_array_equal(existing.data, arr)
        assert existing.contrast_limits == (0.0, 1.0)


class TestRealPanelConstruction:
    """Build the actual Qt dock widget (regression for the __bases__ bug).

    The MagicMock-based tests above never instantiate a real QWidget, so they
    could not catch ``TypeError: __bases__ assignment: 'QWidget' deallocator
    differs from 'object'`` raised by the old ``__new__`` trick under PyQt6.
    These tests require a live Qt binding (qt-test group, offscreen platform).
    """

    def test_factory_builds_qwidget_and_save_writes_back(self, qtbot, detected_image):
        from qtpy.QtWidgets import QWidget

        from phenotypic.sdk_.napari_._label_editor_widget import (
            _make_label_editor_panel,
        )

        edited = detected_image.objmap[:].copy()
        edited[0:3, 0:3] = 777
        labels_layer = MagicMock()
        labels_layer.data = edited
        viewer = MagicMock()

        panel = _make_label_editor_panel(viewer, labels_layer, detected_image, "objmap")
        qtbot.addWidget(panel)

        assert isinstance(panel, QWidget)

        panel._save()

        np.testing.assert_array_equal(detected_image.objmap[:], edited)
        viewer.close.assert_called_once()

    def test_factory_discard_does_not_mutate(self, qtbot, detected_image):
        from phenotypic.sdk_.napari_._label_editor_widget import (
            _make_label_editor_panel,
        )

        before = detected_image.objmap[:].copy()
        labels_layer = MagicMock()
        labels_layer.data = np.full_like(before, 9)
        viewer = MagicMock()

        panel = _make_label_editor_panel(viewer, labels_layer, detected_image, "objmap")
        qtbot.addWidget(panel)

        panel._discard()

        np.testing.assert_array_equal(detected_image.objmap[:], before)
        assert panel.saved_labels is None
        viewer.close.assert_called_once()
