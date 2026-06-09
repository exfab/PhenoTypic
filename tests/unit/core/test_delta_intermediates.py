"""Tests for delta-based intermediate storage in ImagePipeline.

Covers the ``_layers_modified_by()`` helper, delta HDF5 output from
``apply_with_intermediates()``, and corrector base-file emission.

The delta-intermediate *resolver/scanner* tests
(``build_layer_resolution_index`` / ``SweepOutputScanner``) were removed with
the ``phenotypic.sweep`` hard cutover (master §9): those helpers lived in the
now-deleted ``phenotypic.gui.sweep._sweep_data_model`` napari viewer, which had
no surviving production consumer. The core *write* path exercised here
(``apply_with_intermediates``) is still used by the builder GUI.
"""

import h5py

from phenotypic._core._pipeline_parts._image_pipeline_core import _layers_modified_by
from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.abc_ import ImageCorrector
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.measure import MeasureSize
from phenotypic.refine import SmallObjectRemover

_ALL_LAYERS = {"rgb", "gray", "detect_mat", "objmap"}


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
        assert _layers_modified_by(GaussianBlur(sigma=1)) == ("detect_mat",)

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
    """Pipeline.apply_with_intermediates writes delta HDF5 files."""

    def test_delta_files_contain_only_modified_layers(self, tmp_path):
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        pipeline = ImagePipeline(ops=[GaussianBlur(sigma=1), OtsuDetector()])

        out_dir = tmp_path / "intermediates"
        result = pipeline.apply_with_intermediates(image, output_dir=out_dir)

        # -- base file: all 4 layers ---------------------------------
        base = out_dir / "base_00.h5"
        assert base.exists(), "base_00.h5 should be created"
        with h5py.File(base, "r") as f:
            assert "rgb" in f
            assert "gray" in f
            assert "detect_mat" in f
            assert "objmap" in f

        # -- enhancer delta: detect_mat only --------------------------
        enhancer_file = out_dir / "00_GaussianBlur.h5"
        assert enhancer_file.exists(), "00_GaussianBlur.h5 should be created"
        with h5py.File(enhancer_file, "r") as f:
            assert "detect_mat" in f
            assert "rgb" not in f
            assert "gray" not in f
            assert "objmap" not in f

        # -- detector delta: objmap only ------------------------------
        detector_file = out_dir / "01_OtsuDetector.h5"
        assert detector_file.exists(), "01_OtsuDetector.h5 should be created"
        with h5py.File(detector_file, "r") as f:
            assert "objmap" in f
            assert "rgb" not in f
            assert "gray" not in f
            assert "detect_mat" not in f

        # -- result image is valid ------------------------------------
        assert result.image is not None


# ===================================================================
# 3. Corrector emits a new base file
# ===================================================================


class TestCorrectorEmitsBase:
    """Pipeline containing a corrector produces a base_NN.h5 file."""

    def test_corrector_emits_base(self, tmp_path):
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        pipeline = ImagePipeline(
            ops=[GaussianBlur(sigma=1), _DummyCorrector(), OtsuDetector()],
        )

        out_dir = tmp_path / "intermediates"
        pipeline.apply_with_intermediates(image, output_dir=out_dir)

        # base_00.h5 -> initial base
        assert (out_dir / "base_00.h5").exists()
        with h5py.File(out_dir / "base_00.h5", "r") as f:
            assert set(f.keys()) & _ALL_LAYERS == _ALL_LAYERS

        # 00_GaussianBlur.h5 -> detect_mat delta
        assert (out_dir / "00_GaussianBlur.h5").exists()
        with h5py.File(out_dir / "00_GaussianBlur.h5", "r") as f:
            assert "detect_mat" in f
            assert "rgb" not in f

        # base_01.h5 -> corrector base (all layers)
        assert (out_dir / "base_01.h5").exists()
        with h5py.File(out_dir / "base_01.h5", "r") as f:
            assert set(f.keys()) & _ALL_LAYERS == _ALL_LAYERS

        # 02_OtsuDetector.h5 -> objmap delta
        assert (out_dir / "02_OtsuDetector.h5").exists()
        with h5py.File(out_dir / "02_OtsuDetector.h5", "r") as f:
            assert "objmap" in f
            assert "rgb" not in f
