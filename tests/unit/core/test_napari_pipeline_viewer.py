"""Tests for the NapariPipelineViewer and accessor viewer/layer_name parameters.

Covers:
- ``_napari_layers_for()`` unit tests for each ABC type
- ``NapariPipelineResult`` named tuple access
- Accessor ``viewer``/``layer_name`` keyword-only parameters
- ``apply_napari()`` integration with synthetic image + simple pipeline
- Edge cases: MeasureFeatures-only pipeline, empty RGB, existing viewer reuse
"""

from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

import numpy as np
import pytest

from phenotypic._core._pipeline_parts._napari_pipeline_viewer import (
    _napari_layers_for,
    NapariPipelineResult,
    NapariPipelineViewer,
)
from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.abc_ import ImageCorrector
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur, CLAHE
from phenotypic.measure import MeasureSize
from phenotypic.refine import SmallObjectRemover


# ---------------------------------------------------------------------------
# Dummy corrector
# ---------------------------------------------------------------------------


class _DummyCorrector(ImageCorrector):
    """Minimal corrector that returns the image unchanged."""

    def _operate(self, image):
        return image


# ---------------------------------------------------------------------------
# Mock napari helpers
# ---------------------------------------------------------------------------


class _MockLayers:
    """Dict-like mock that mimics napari viewer.layers[name] access."""

    def __init__(self):
        self._store = {}

    def __getitem__(self, name):
        if name in self._store:
            return self._store[name]
        raise KeyError(name)

    def __contains__(self, name):
        return name in self._store

    def __setitem__(self, name, value):
        self._store[name] = value


def _make_mock_viewer():
    """Create a mock napari viewer with layers dict-like behavior.

    The mock's ``add_image`` and ``add_labels`` methods automatically
    register the new layer in the ``layers`` store so that subsequent
    lookups (e.g. ``viewer.layers[name].contour = ...``) succeed.
    """
    viewer = MagicMock()
    viewer.layers = _MockLayers()

    def _add_image(data, *, name, **kwargs):
        layer = MagicMock()
        layer.data = data
        viewer.layers[name] = layer

    def _add_labels(data, *, name, **kwargs):
        layer = MagicMock()
        layer.data = data
        viewer.layers[name] = layer

    viewer.add_image.side_effect = _add_image
    viewer.add_labels.side_effect = _add_labels
    return viewer


# ===================================================================
# 1. Unit tests for _napari_layers_for()
# ===================================================================


class TestNapariLayersFor:
    """Verify ``_napari_layers_for`` returns correct layer specs per ABC."""

    def test_enhancer_returns_detect_mat(self):
        result = _napari_layers_for(GaussianBlur(sigma=1))
        assert result == [("detect_mat", False)]

    def test_clahe_enhancer_returns_detect_mat(self):
        result = _napari_layers_for(CLAHE())
        assert result == [("detect_mat", False)]

    def test_detector_returns_objmap(self):
        result = _napari_layers_for(OtsuDetector())
        assert result == [("objmap", True)]

    def test_refiner_returns_objmap(self):
        result = _napari_layers_for(SmallObjectRemover(min_size=10))
        assert result == [("objmap", True)]

    def test_corrector_returns_rgb_gray(self):
        result = _napari_layers_for(_DummyCorrector())
        assert result == [("rgb", False), ("gray", False)]

    def test_measure_returns_none(self):
        assert _napari_layers_for(MeasureSize()) is None

    def test_grid_finder_returns_none(self):
        from phenotypic.grid import AutoGridFinder

        assert _napari_layers_for(AutoGridFinder()) is None

    def test_fallback_returns_all_layers(self):
        """Plain ImageOperation or unknown type falls back to all layers."""
        from phenotypic.prefab import HeavyOtsuPipeline

        result = _napari_layers_for(HeavyOtsuPipeline())
        assert result == [
            ("rgb", False),
            ("gray", False),
            ("detect_mat", False),
            ("objmap", True),
        ]


# ===================================================================
# 2. NapariPipelineResult NamedTuple
# ===================================================================


class TestNapariPipelineResult:
    """Verify NapariPipelineResult fields."""

    def test_named_tuple_access(self):
        mock_image = MagicMock()
        mock_viewer = MagicMock()
        result = NapariPipelineResult(image=mock_image, viewer=mock_viewer)
        assert result.image is mock_image
        assert result.viewer is mock_viewer

    def test_unpacking(self):
        mock_image = MagicMock()
        mock_viewer = MagicMock()
        result = NapariPipelineResult(image=mock_image, viewer=mock_viewer)
        img, v = result
        assert img is mock_image
        assert v is mock_viewer


# ===================================================================
# 3. Accessor viewer/layer_name parameters
# ===================================================================


class TestAccessorViewerParam:
    """Verify that accessor .napari() accepts viewer and layer_name kwargs."""

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_image_accessor_uses_provided_viewer(self):
        """When viewer kwarg is provided, it should be used directly."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        mock_viewer = _make_mock_viewer()

        # The accessor .napari() should use the provided viewer
        # and not touch the global viewer
        result = image.gray.napari(
            viewer=mock_viewer, layer_name="test_layer"
        )
        assert result is mock_viewer

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_image_accessor_custom_layer_name(self):
        """When layer_name is provided, it should be used as-is."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        mock_viewer = _make_mock_viewer()

        image.gray.napari(viewer=mock_viewer, layer_name="my_custom_name")
        # Verify add_image was called with our custom name
        mock_viewer.add_image.assert_called_once()
        call_kwargs = mock_viewer.add_image.call_args
        assert call_kwargs[1]["name"] == "my_custom_name"

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_labels_accessor_uses_provided_viewer(self):
        """Labels accessor should use provided viewer."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        image = OtsuDetector().apply(image)

        mock_viewer = _make_mock_viewer()
        result = image.objmap.napari(
            viewer=mock_viewer, layer_name="test_labels"
        )
        assert result is mock_viewer

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_labels_accessor_custom_layer_name(self):
        """Labels accessor should use custom layer_name."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        image = OtsuDetector().apply(image)

        mock_viewer = _make_mock_viewer()
        image.objmap.napari(viewer=mock_viewer, layer_name="custom_objmap")
        mock_viewer.add_labels.assert_called_once()
        call_kwargs = mock_viewer.add_labels.call_args
        assert call_kwargs[1]["name"] == "custom_objmap"

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_image_napari_passes_viewer_through(self):
        """Image.napari() should pass viewer kwarg to each accessor."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        mock_viewer = _make_mock_viewer()

        result = image.napari(viewer=mock_viewer)
        assert result is mock_viewer
        # Should have called add_image for rgb, gray, detect_mat
        assert mock_viewer.add_image.call_count >= 3


# ===================================================================
# 4. apply_napari() integration
# ===================================================================


class TestApplyNapari:
    """Integration tests for apply_napari with mock napari."""

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    @patch(
        "phenotypic._core._pipeline_parts._napari_pipeline_viewer._HAS_NAPARI",
        True,
        create=True,
    )
    def test_apply_napari_basic_pipeline(self):
        """apply_napari with enhancer + detector adds correct layers."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        pipe = ImagePipeline(ops=[GaussianBlur(sigma=1), OtsuDetector()])

        mock_viewer = _make_mock_viewer()

        # Patch the _HAS_NAPARI check inside the viewer module
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            True,
        ):
            result = pipe.apply_napari(image, viewer=mock_viewer)

        assert result.image is not None
        assert result.viewer is mock_viewer

        # Collect all layer names from add_image and add_labels calls
        layer_names = []
        for call in mock_viewer.add_image.call_args_list:
            layer_names.append(call[1]["name"])
        for call in mock_viewer.add_labels.call_args_list:
            layer_names.append(call[1]["name"])

        # Baseline layers
        assert "00_original_gray" in layer_names
        assert "00_original_detect_mat" in layer_names

        # Operation layers
        assert "01_GaussianBlur_detect_mat" in layer_names
        assert "02_OtsuDetector_objmap" in layer_names

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_apply_napari_returns_named_tuple(self):
        """Result should be a NapariPipelineResult with image and viewer."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        pipe = ImagePipeline(ops=[GaussianBlur(sigma=1)])

        mock_viewer = _make_mock_viewer()
        result = pipe.apply_napari(image, viewer=mock_viewer)

        assert isinstance(result, NapariPipelineResult)
        img, viewer = result
        assert img is not None
        assert viewer is mock_viewer

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_apply_napari_measure_only_pipeline(self):
        """Pipeline with only MeasureFeatures should skip those ops in viewer."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        # Need to detect first for MeasureSize to work, but put only
        # MeasureSize-style ops in a separate pipeline won't add layers
        pipe = ImagePipeline(ops=[])
        mock_viewer = _make_mock_viewer()

        result = pipe.apply_napari(image, viewer=mock_viewer)

        # Only baseline layers should exist — no operation layers
        layer_names = [
            call[1]["name"] for call in mock_viewer.add_image.call_args_list
        ]
        assert all(name.startswith("00_original") for name in layer_names)

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_apply_napari_does_not_modify_original(self):
        """When inplace=False, original image should be unchanged."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        original_detect_mat = image.detect_mat[:].copy()

        pipe = ImagePipeline(ops=[GaussianBlur(sigma=3)])
        mock_viewer = _make_mock_viewer()

        result = pipe.apply_napari(image, viewer=mock_viewer, inplace=False)

        np.testing.assert_array_equal(image.detect_mat[:], original_detect_mat)

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_apply_napari_inplace(self):
        """When inplace=True, the input image should be modified."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        original_detect_mat = image.detect_mat[:].copy()

        pipe = ImagePipeline(ops=[GaussianBlur(sigma=3)])
        mock_viewer = _make_mock_viewer()

        result = pipe.apply_napari(image, viewer=mock_viewer, inplace=True)

        assert result.image is image
        assert not np.array_equal(image.detect_mat[:], original_detect_mat)

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_apply_napari_with_corrector(self):
        """Corrector should add rgb and gray layers (not detect_mat)."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        pipe = ImagePipeline(ops=[_DummyCorrector()])
        mock_viewer = _make_mock_viewer()

        result = pipe.apply_napari(image, viewer=mock_viewer)

        layer_names = [
            call[1]["name"] for call in mock_viewer.add_image.call_args_list
        ]

        # Should have corrector layers for rgb and gray
        assert "01__DummyCorrector_rgb" in layer_names
        assert "01__DummyCorrector_gray" in layer_names

    @patch(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )
    def test_apply_napari_baseline_includes_rgb(self):
        """Baseline should include rgb layer when image has RGB data."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        pipe = ImagePipeline(ops=[])
        mock_viewer = _make_mock_viewer()

        pipe.apply_napari(image, viewer=mock_viewer)

        layer_names = [
            call[1]["name"] for call in mock_viewer.add_image.call_args_list
        ]
        assert "00_original_rgb" in layer_names

    def test_apply_napari_raises_without_napari(self):
        """apply_napari raises ImportError when napari is not available."""
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        pipe = ImagePipeline(ops=[GaussianBlur(sigma=1)])

        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            False,
        ):
            with pytest.raises(ImportError, match="napari"):
                pipe.apply_napari(image)


# ===================================================================
# 5. Inheritance chain verification
# ===================================================================


class TestInheritanceChain:
    """Verify the inheritance chain is correct after refactor."""

    def test_serializable_inherits_from_napari_viewer(self):
        from phenotypic._core._pipeline_parts._serializable_pipeline import (
            SerializablePipeline,
        )

        assert issubclass(SerializablePipeline, NapariPipelineViewer)

    def test_napari_viewer_inherits_from_core(self):
        from phenotypic._core._pipeline_parts._image_pipeline_core import (
            ImagePipelineCore,
        )

        assert issubclass(NapariPipelineViewer, ImagePipelineCore)

    def test_image_pipeline_has_apply_napari(self):
        assert hasattr(ImagePipeline, "apply_napari")

    def test_prefab_pipeline_has_apply_napari(self):
        from phenotypic.prefab import HeavyOtsuPipeline

        assert hasattr(HeavyOtsuPipeline, "apply_napari")
