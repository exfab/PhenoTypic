"""Tests for delta-based intermediate storage in ImagePipeline.

Covers the ``_layers_modified_by()`` helper, delta HDF5 output from
``apply_with_intermediates()``, corrector base-file emission, the
``build_layer_resolution_index()`` resolver, and backward-compatible scanning
of old full-snapshot intermediates.
"""

import h5py
import numpy as np
import pytest
from pathlib import Path

from phenotypic._core._pipeline_parts._image_pipeline_core import _layers_modified_by
from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.abc_ import ImageCorrector
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.measure import MeasureSize
from phenotypic.refine import SmallObjectRemover
from phenotypic.gui.sweep._sweep_data_model import (
    IntermediateStep,
    ResolvedLayerSources,
    SweepOutputScanner,
    build_layer_resolution_index,
)

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


# ===================================================================
# 4. Tests for build_layer_resolution_index()
# ===================================================================


class TestBuildLayerResolutionIndex:
    """Verify ``build_layer_resolution_index`` resolves layer sources."""

    def test_empty_steps_returns_empty(self):
        assert build_layer_resolution_index([]) == {}

    def test_single_base_step(self, tmp_path):
        """A single base step resolves all layers to the base path."""
        base_path = tmp_path / "base_00.h5"
        step = IntermediateStep(
            index=0,
            operation_name="base",
            h5_path=base_path,
            layers=("rgb", "gray", "detect_mat", "objmap"),
            is_base=True,
        )
        index = build_layer_resolution_index([step])

        assert 0 in index
        resolved = index[0]
        assert resolved.rgb == base_path
        assert resolved.gray == base_path
        assert resolved.detect_mat == base_path
        assert resolved.objmap == base_path

    def test_multi_step_delta_resolution(self, tmp_path):
        """Deltas override only their specific layers."""
        base_path = tmp_path / "base_00.h5"
        blur_path = tmp_path / "01_GaussianBlur.h5"
        otsu_path = tmp_path / "02_OtsuDetector.h5"

        steps = [
            IntermediateStep(
                index=0,
                operation_name="base",
                h5_path=base_path,
                layers=("rgb", "gray", "detect_mat", "objmap"),
                is_base=True,
            ),
            IntermediateStep(
                index=1,
                operation_name="GaussianBlur",
                h5_path=blur_path,
                layers=("detect_mat",),
                is_base=False,
            ),
            IntermediateStep(
                index=2,
                operation_name="OtsuDetector",
                h5_path=otsu_path,
                layers=("objmap",),
                is_base=False,
            ),
        ]

        index = build_layer_resolution_index(steps)

        # After base (index 0): all layers from base
        resolved_base = index[0]
        assert resolved_base.rgb == base_path
        assert resolved_base.gray == base_path
        assert resolved_base.detect_mat == base_path
        assert resolved_base.objmap == base_path

        # After GaussianBlur (index 1): detect_mat from blur, rest from base
        resolved_blur = index[1]
        assert resolved_blur.rgb == base_path
        assert resolved_blur.gray == base_path
        assert resolved_blur.detect_mat == blur_path
        assert resolved_blur.objmap == base_path

        # After OtsuDetector (index 2): objmap from otsu, detect_mat still blur
        resolved_otsu = index[2]
        assert resolved_otsu.rgb == base_path
        assert resolved_otsu.gray == base_path
        assert resolved_otsu.detect_mat == blur_path
        assert resolved_otsu.objmap == otsu_path

    def test_corrector_base_resets_sources(self, tmp_path):
        """A corrector base resets all layer sources."""
        base0_path = tmp_path / "base_00.h5"
        blur_path = tmp_path / "00_GaussianBlur.h5"
        base1_path = tmp_path / "base_01.h5"
        otsu_path = tmp_path / "02_OtsuDetector.h5"

        steps = [
            IntermediateStep(
                index=0,
                operation_name="base",
                h5_path=base0_path,
                layers=("rgb", "gray", "detect_mat", "objmap"),
                is_base=True,
            ),
            IntermediateStep(
                index=1,
                operation_name="GaussianBlur",
                h5_path=blur_path,
                layers=("detect_mat",),
                is_base=False,
            ),
            IntermediateStep(
                index=2,
                operation_name="base",
                h5_path=base1_path,
                layers=("rgb", "gray", "detect_mat", "objmap"),
                is_base=True,
            ),
            IntermediateStep(
                index=3,
                operation_name="OtsuDetector",
                h5_path=otsu_path,
                layers=("objmap",),
                is_base=False,
            ),
        ]

        index = build_layer_resolution_index(steps)

        # After corrector base: all layers point to base1, not base0 or blur
        resolved_corr = index[2]
        assert resolved_corr.rgb == base1_path
        assert resolved_corr.gray == base1_path
        assert resolved_corr.detect_mat == base1_path
        assert resolved_corr.objmap == base1_path

        # After OtsuDetector: objmap from otsu, rest from base1
        resolved_otsu = index[3]
        assert resolved_otsu.rgb == base1_path
        assert resolved_otsu.gray == base1_path
        assert resolved_otsu.detect_mat == base1_path
        assert resolved_otsu.objmap == otsu_path


# ===================================================================
# 5. Backward compatibility: old full-snapshot intermediates
# ===================================================================


class TestScannerBackwardCompat:
    """Old-format intermediates (every file has all 4 layers) are parsed."""

    def test_scanner_backward_compat(self, tmp_path):
        inter_dir = tmp_path / "stem1" / "pipe1" / "intermediates"
        inter_dir.mkdir(parents=True)

        for i, name in enumerate(["GaussianBlur", "OtsuDetector"]):
            path = inter_dir / f"{i:02d}_{name}.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("rgb", data=np.zeros((2, 2, 3), dtype=np.uint8))
                f.create_dataset("gray", data=np.zeros((2, 2), dtype=np.float64))
                f.create_dataset(
                    "detect_mat", data=np.zeros((2, 2), dtype=np.float64),
                )
                f.create_dataset("objmap", data=np.zeros((2, 2), dtype=np.int32))

        result = SweepOutputScanner._scan_intermediates(tmp_path)

        steps = result["stem1"]["pipe1"]
        assert len(steps) == 2
        for step in steps:
            assert set(step.layers) == _ALL_LAYERS
            assert step.is_base is False


# ===================================================================
# 6. Scanner handles delta + base files
# ===================================================================


class TestScannerDeltaAndBase:
    """Scanner parses both ``base_NN.h5`` and ``NN_OpName.h5`` files."""

    def test_scanner_delta_and_base(self, tmp_path):
        inter_dir = tmp_path / "stem1" / "pipe1" / "intermediates"
        inter_dir.mkdir(parents=True)

        # base_00.h5 with all layers
        with h5py.File(inter_dir / "base_00.h5", "w") as f:
            f.create_dataset("rgb", data=np.zeros((2, 2, 3), dtype=np.uint8))
            f.create_dataset("gray", data=np.zeros((2, 2), dtype=np.float64))
            f.create_dataset(
                "detect_mat", data=np.zeros((2, 2), dtype=np.float64),
            )
            f.create_dataset("objmap", data=np.zeros((2, 2), dtype=np.int32))

        # 00_GaussianBlur.h5 with only detect_mat
        with h5py.File(inter_dir / "00_GaussianBlur.h5", "w") as f:
            f.create_dataset(
                "detect_mat", data=np.zeros((2, 2), dtype=np.float64),
            )

        # 01_OtsuDetector.h5 with only objmap
        with h5py.File(inter_dir / "01_OtsuDetector.h5", "w") as f:
            f.create_dataset("objmap", data=np.zeros((2, 2), dtype=np.int32))

        result = SweepOutputScanner._scan_intermediates(tmp_path)

        steps = result["stem1"]["pipe1"]
        assert len(steps) == 3

        base_step = [s for s in steps if s.is_base][0]
        assert base_step.index == 0
        assert set(base_step.layers) == _ALL_LAYERS

        blur_step = [s for s in steps if s.operation_name == "GaussianBlur"][0]
        assert set(blur_step.layers) == {"detect_mat"}
        assert blur_step.is_base is False

        otsu_step = [s for s in steps if s.operation_name == "OtsuDetector"][0]
        assert set(otsu_step.layers) == {"objmap"}
        assert otsu_step.is_base is False

    def test_scanner_step_ordering(self, tmp_path):
        """Steps are sorted by index after scanning."""
        inter_dir = tmp_path / "img" / "pipe" / "intermediates"
        inter_dir.mkdir(parents=True)

        # Create files in reverse order to verify sorting
        for idx, name in [(2, "OtsuDetector"), (0, "base"), (1, "GaussianBlur")]:
            if name == "base":
                fname = f"base_{idx:02d}.h5"
                layers = _ALL_LAYERS
                is_base_file = True
            else:
                fname = f"{idx:02d}_{name}.h5"
                layers = {"detect_mat"} if name == "GaussianBlur" else {"objmap"}
                is_base_file = False

            with h5py.File(inter_dir / fname, "w") as f:
                for layer in layers:
                    if layer == "rgb":
                        f.create_dataset(
                            layer, data=np.zeros((2, 2, 3), dtype=np.uint8),
                        )
                    elif layer in ("gray", "detect_mat"):
                        f.create_dataset(
                            layer, data=np.zeros((2, 2), dtype=np.float64),
                        )
                    else:
                        f.create_dataset(
                            layer, data=np.zeros((2, 2), dtype=np.int32),
                        )

        result = SweepOutputScanner._scan_intermediates(tmp_path)
        steps = result["img"]["pipe"]

        indices = [s.index for s in steps]
        assert indices == sorted(indices), "Steps should be sorted by index"
