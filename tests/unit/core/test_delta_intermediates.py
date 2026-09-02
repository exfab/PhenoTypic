"""Tests for delta-based intermediate storage in ImagePipeline.

Covers the ``_layers_modified_by()`` helper, delta OME-Zarr output from
``apply_with_intermediates()``, and corrector base-store emission.

A delta store carries the layers the operation modified **plus ``gray``**:
``save_intermediate_zarr`` always co-writes the primary series, without which
the store has no anchor for its label group or its OME projection (user ruling,
2026-08-19).

The delta-intermediate *resolver/scanner* tests
(``build_layer_resolution_index`` / ``SweepOutputScanner``) were removed with
the ``phenotypic.sweep`` hard cutover (master §9): those helpers lived in the
now-deleted ``phenotypic.gui.sweep._sweep_data_model`` napari viewer, which had
no surviving production consumer. The core *write* path exercised here
(``apply_with_intermediates``) is still used by the builder GUI.
"""

import numpy as np

from phenotypic import Image
from phenotypic._core._pipeline_parts._image_pipeline_core import _layers_modified_by
from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.abc_ import ImageCorrector
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.measure import MeasureSize
from phenotypic.refine import SmallObjectRemover

_ALL_LAYERS = {"rgb", "gray", "detect_mat", "objmap"}


def _store_layers(store) -> set[str]:
    """Every layer a store actually carries: its series plus its label."""
    from phenotypic.sdk_ import ngff_

    block = ngff_.read_phenotypic_attributes(store)
    return set(block.get(ngff_.PhenotypicAttr.SERIES, {})) | set(
        block.get(ngff_.PhenotypicAttr.LABELS, {})
    )


# ---------------------------------------------------------------------------
# Dummy corrector for tests requiring an ImageCorrector instance
# ---------------------------------------------------------------------------


class _DummyCorrector(ImageCorrector):
    """Minimal corrector that returns the image unchanged."""

    def _operate(self, image):
        return image


# ===================================================================
# 1. Unit tests for _layers_modified_by()
# ===================================================================


class TestLayersModifiedBy:
    """Verify ``_layers_modified_by`` returns the correct layer tuple for each ABC."""

    def test_enhancer_returns_detect_mat(self):
        assert _layers_modified_by(BlurGauss(sigma=1)) == ("detect_mat",)

    def test_detector_returns_objmap(self):
        assert _layers_modified_by(OtsuDetector()) == ("objmap",)

    def test_refiner_returns_objmap(self):
        assert _layers_modified_by(SmallObjectRemover(min_size=10)) == ("objmap",)

    def test_corrector_returns_all_layers(self):
        result = _layers_modified_by(_DummyCorrector())
        assert result == ("rgb", "gray", "detect_mat", "objmap")

    def test_measure_returns_none(self):
        assert _layers_modified_by(MeasureSize()) is None

    def test_fallback_returns_all_layers(self):
        """Plain ImageOperation or unknown type falls back to all layers."""
        from phenotypic.prefab import HeavyOtsuPipeline

        result = _layers_modified_by(HeavyOtsuPipeline())
        assert result == ("rgb", "gray", "detect_mat", "objmap")


# ===================================================================
# 2. Integration test for delta HDF5 output
# ===================================================================


class TestDeltaIntermediatesOutput:
    """Pipeline.apply_with_intermediates writes delta OME-Zarr stores."""

    def test_delta_stores_contain_the_modified_layers_plus_gray(self, tmp_path):
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        pipeline = ImagePipeline(ops=[BlurGauss(sigma=1), OtsuDetector()])

        out_dir = tmp_path / "intermediates"
        result = pipeline.apply_with_intermediates(image, output_dir=out_dir)

        # -- base store: all 4 layers ---------------------------------
        base = out_dir / "base_00.ome.zarr"
        assert base.is_dir(), "base_00.ome.zarr should be created"
        assert _store_layers(base) == _ALL_LAYERS

        # -- enhancer delta: detect_mat, plus the co-written gray -----
        enhancer_store = out_dir / "00_BlurGauss.ome.zarr"
        assert enhancer_store.is_dir(), "00_BlurGauss.ome.zarr should be created"
        assert _store_layers(enhancer_store) == {"gray", "detect_mat"}
        # The delta really is the POST-op state, not a copy of the base.
        assert not np.array_equal(
            Image.load_layer_zarr(enhancer_store, "detect_mat"),
            Image.load_layer_zarr(base, "detect_mat"),
        )
        # ... and the co-written gray is the real gray, not a zeros filler.
        np.testing.assert_array_equal(
            Image.load_layer_zarr(enhancer_store, "gray"),
            Image.load_layer_zarr(base, "gray"),
        )

        # -- detector delta: objmap, plus the co-written gray ---------
        detector_store = out_dir / "01_OtsuDetector.ome.zarr"
        assert detector_store.is_dir(), "01_OtsuDetector.ome.zarr should be created"
        assert _store_layers(detector_store) == {"gray", "objmap"}
        assert int(Image.load_layer_zarr(detector_store, "objmap").max()) > 0

        # -- result image is valid ------------------------------------
        assert result.image is not None


# ===================================================================
# 3. Corrector emits a new base file
# ===================================================================


class TestCorrectorEmitsBase:
    """Pipeline containing a corrector produces a base_NN.ome.zarr store."""

    def test_corrector_emits_base(self, tmp_path):
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        pipeline = ImagePipeline(
            ops=[BlurGauss(sigma=1), _DummyCorrector(), OtsuDetector()],
        )

        out_dir = tmp_path / "intermediates"
        pipeline.apply_with_intermediates(image, output_dir=out_dir)

        # base_00.ome.zarr -> initial base
        assert (out_dir / "base_00.ome.zarr").is_dir()
        assert _store_layers(out_dir / "base_00.ome.zarr") == _ALL_LAYERS

        # 00_BlurGauss.ome.zarr -> detect_mat delta (+ gray)
        assert (out_dir / "00_BlurGauss.ome.zarr").is_dir()
        assert _store_layers(out_dir / "00_BlurGauss.ome.zarr") == {
            "gray", "detect_mat",
        }

        # base_01.ome.zarr -> corrector base (all layers)
        assert (out_dir / "base_01.ome.zarr").is_dir()
        assert _store_layers(out_dir / "base_01.ome.zarr") == _ALL_LAYERS

        # 02_OtsuDetector.ome.zarr -> objmap delta (+ gray)
        assert (out_dir / "02_OtsuDetector.ome.zarr").is_dir()
        assert _store_layers(out_dir / "02_OtsuDetector.ome.zarr") == {
            "gray", "objmap",
        }
