"""Tests for LabelEditorWidget and _LabelEditorPanel logic."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# LabelEditorWidget public API
# ---------------------------------------------------------------------------


class TestLabelEditorWidget:
    def test_run_raises_import_error_without_napari(self):
        from phenotypic.tools_.napari_ import LabelEditorWidget

        w = LabelEditorWidget()
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            False,
        ):
            with pytest.raises(ImportError, match="napari is required"):
                w.run(MagicMock(), "objmap")


# ---------------------------------------------------------------------------
# _LabelEditorPanel logic (tested without real Qt)
# ---------------------------------------------------------------------------


def _make_mock_panel(*, image, accessor_name: str, layer_data: np.ndarray) -> MagicMock:
    """Mimic a ``_LabelEditorPanel`` for logic testing without Qt.

    Mirrors ``tests/unit/tools_/test_point_picker_widget.py::_make_mock_panel``:
    build a ``MagicMock`` with the same attributes and bind the real class
    methods so ``_save``/``_discard`` run against a real Image.
    """
    from phenotypic.tools_.napari_._label_editor_widget import _LabelEditorPanel

    panel = MagicMock()

    labels_layer = MagicMock()
    labels_layer.data = layer_data.copy()
    panel._labels_layer = labels_layer

    panel._image = image
    panel._accessor_name = accessor_name
    panel._viewer = MagicMock()
    panel.saved_labels = None

    panel._save = lambda: _LabelEditorPanel._save(panel)
    panel._discard = lambda: _LabelEditorPanel._discard(panel)

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
        # Relabel produced sequential integer IDs on the objmap.
        labels = np.unique(detected_image.objmap[:])
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
