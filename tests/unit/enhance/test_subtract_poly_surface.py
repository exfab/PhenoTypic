"""Operation tests for SubtractPolySurface (design.md §3, §4.6, §4.7, §4.9, §7.2)."""

from __future__ import annotations

import json
from typing import get_args

import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import SubtractPolySurface
from phenotypic.enhance._poly_surface_kernels import flatten_surface
from phenotypic.sdk_.typing_ import LineAxis, SurfaceFit, SurfaceMethod

from ._poly_surface_synth import NOISE, grid, rmse, surface_plate

FLOAT32_TOL = 1e-6  # ~8 * eps32 on [0, 1] data (design.md §5)


def _image(z: np.ndarray) -> Image:
    image = Image(arr=np.clip(z, 0.0, 1.0))
    image.detect_mat[:] = z
    return image


class TestFields:
    def test_field_order_ends_with_norm(self):
        assert list(SubtractPolySurface.model_fields) == [
            "method", "order", "independent", "line_order", "line_axis",
            "fit", "clip_sigma", "max_iter", "norm"]

    def test_defaults_match_the_spec(self):
        op = SubtractPolySurface()
        assert (op.method, op.order, op.independent, op.line_order, op.line_axis,
                op.fit, op.clip_sigma, op.max_iter, op.norm) == (
            "plane", 3, True, 1, "row", "lstsq", 3.0, 10, "clip")

    def test_literal_aliases_are_the_closed_sets(self):
        assert set(get_args(SurfaceMethod)) == {"offset", "plane", "polynomial", "line"}
        assert set(get_args(SurfaceFit)) == {"lstsq", "robust"}
        assert set(get_args(LineAxis)) == {"row", "column"}

    @pytest.mark.parametrize("bad", [
        dict(order=1), dict(order=12), dict(line_order=-1), dict(line_order=6),
        dict(clip_sigma=0.0), dict(max_iter=0), dict(method="median"),
        dict(fit="huber"), dict(line_axis="diagonal"), dict(clip=True)])
    def test_invalid_values_raise_validation_error(self, bad):
        with pytest.raises((ValidationError, ValueError)):
            SubtractPolySurface(**bad)

    def test_unused_fields_are_ignored_not_rejected(self):
        z, _ = surface_plate(height=40, width=60, cover=0.1, seed=2050)
        a = SubtractPolySurface(method="plane", norm=None).apply(_image(z)).detect_mat[:]
        b = SubtractPolySurface(method="plane", order=7, line_order=4, norm=None).apply(_image(z)).detect_mat[:]
        np.testing.assert_array_equal(a, b)


class TestOperationContract:
    def test_plane_through_the_operation(self):
        height, width, a, bx, by = 21, 34, 0.35, 0.002, 0.001
        i, j, _, _ = grid(height, width)
        out = SubtractPolySurface(method="plane", norm=None).apply(_image(a + bx * j + by * i)).detect_mat[:]
        np.testing.assert_allclose(out, a + bx * width / 2 + by * height / 2, atol=FLOAT32_TOL)

    def test_dtype_rgb_and_gray_are_preserved(self):
        z, _ = surface_plate(height=40, width=60, cover=0.1, seed=2051)
        image = Image(arr=np.dstack([np.clip(z, 0.0, 1.0)] * 3))  # 2-D arrays carry no rgb
        image.detect_mat[:] = z
        rgb, gray = image.rgb[:].copy(), image.gray[:].copy()
        out = SubtractPolySurface(method="polynomial").apply(image)
        assert out.detect_mat[:].dtype == np.float32
        np.testing.assert_array_equal(out.rgb[:], rgb)
        np.testing.assert_array_equal(out.gray[:], gray)

    def test_norm_policies(self):
        z, _ = surface_plate(height=60, width=80, cover=0.2, seed=2052)
        clip = SubtractPolySurface(method="polynomial", norm="clip").apply(_image(z)).detect_mat[:]
        none = SubtractPolySurface(method="polynomial", norm=None).apply(_image(z)).detect_mat[:]
        resc = SubtractPolySurface(method="polynomial", norm="rescale").apply(_image(z)).detect_mat[:]
        assert clip.min() >= 0.0 and clip.max() <= 1.0
        assert none.min() < 0.0                       # level 0 => negative half of the noise survives
        assert resc.min() == pytest.approx(0.0, abs=FLOAT32_TOL)
        assert resc.max() == pytest.approx(1.0, abs=FLOAT32_TOL)

    def test_robust_beats_lstsq_on_a_plate(self):
        z, background = surface_plate(cover=0.25, seed=2025)
        robust = SubtractPolySurface(method="polynomial", fit="robust", norm=None).apply(_image(z)).detect_mat[:]
        plain = SubtractPolySurface(method="polynomial", norm=None).apply(_image(z)).detect_mat[:]
        assert rmse(z - robust, background) < 0.5 * NOISE
        assert rmse(z - plain, background) > 10 * rmse(z - robust, background)

    def test_grid_image_only_detect_mat_changes(self):
        """Review Focus 4: rgb and gray of a detected GridImage are untouched; objmap is cleared."""
        image = load_synth_yeast_plate()
        rgb, gray, objmap = image.rgb[:].copy(), image.gray[:].copy(), image.objmap[:].copy()
        before = image.detect_mat[:].copy()
        out = SubtractPolySurface(method="polynomial", fit="robust").apply(image)
        assert not np.array_equal(out.detect_mat[:], before)
        assert 0.0 <= out.detect_mat[:].min() and out.detect_mat[:].max() <= 1.0
        np.testing.assert_array_equal(out.rgb[:], rgb)
        np.testing.assert_array_equal(out.gray[:], gray)
        # ImageOperation clears objmap after any enhancer (stale once detect_mat
        # changes); the operation must behave exactly like its siblings here.
        assert objmap.max() > 0
        assert out.objmap[:].max() == 0

    def test_apply_time_errors_keep_their_cause(self):
        """Spec §4.7: ImageOperation wraps twice; the root cause is our ValueError."""
        image = _image(np.random.default_rng(11).random((3, 40)))
        with pytest.raises(Exception, match="order") as excinfo:
            SubtractPolySurface(method="polynomial", order=3).apply(image)
        root = excinfo.value
        while root.__cause__ is not None:
            root = root.__cause__
        assert isinstance(root, ValueError)


#: Every kernel argument off its default, so a hardcoded default cannot match.
OFF_DEFAULT = dict(order=5, independent=False, line_order=3, line_axis="column",
                   fit="robust", clip_sigma=2.25, max_iter=2)


class TestFieldForwarding:
    """Every field reaches the kernel (final review I1).

    The spy pins the call itself; the equality test pins the cast and norm around it.
    """

    def test_every_field_reaches_the_kernel(self, monkeypatch):
        calls = []

        def spy(z, **kwargs):
            calls.append(kwargs)
            return np.asarray(z, dtype=np.float64)

        monkeypatch.setattr("phenotypic.enhance._subtract_poly_surface.flatten_surface", spy)
        expected = dict(method="line", order=7, independent=False, line_order=4,
                        line_axis="column", fit="robust", clip_sigma=2.25, max_iter=4)
        z, _ = surface_plate(height=40, width=60, cover=0.1, seed=2053)
        SubtractPolySurface(**expected).apply(_image(z))
        assert calls == [expected]

    @pytest.mark.parametrize("method", get_args(SurfaceMethod))
    def test_operation_equals_the_kernel(self, method):
        """Spec §7.2 #1 through the operation: float64 kernel, then float32, then norm."""
        z, _ = surface_plate(height=60, width=90, cover=0.2, seed=2054)
        image = _image(z)
        detect = image.detect_mat[:].copy()
        op = SubtractPolySurface(method=method, norm=None, **OFF_DEFAULT)
        expected = op._apply_norm(flatten_surface(
            detect.astype(np.float64), method=method, **OFF_DEFAULT).astype(np.float32))
        np.testing.assert_allclose(op.apply(image).detect_mat[:], expected, rtol=0, atol=FLOAT32_TOL)


class TestSerialization:
    def test_pipeline_json_round_trip(self):
        op = SubtractPolySurface(method="line", line_order=2, line_axis="column",
                                 fit="robust", clip_sigma=2.5, max_iter=7, norm="rescale")
        loaded = ImagePipeline.from_json(ImagePipeline(ops=[op]).to_json())
        (restored,) = list(loaded._ops.values())
        assert isinstance(restored, SubtractPolySurface)
        assert restored.model_dump() == op.model_dump()

    def test_schema_is_json_serializable(self):
        json.dumps(SubtractPolySurface.model_json_schema())
