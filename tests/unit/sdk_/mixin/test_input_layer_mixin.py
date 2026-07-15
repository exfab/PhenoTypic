"""Contract for InputLayerMixin: field append, layer read, projection, range guard."""
from typing import Annotated

import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic.abc_ import ContrastAdjustment
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import InputLayerMixin, NormalizedOutputMixin
from phenotypic.sdk_.typing_ import TuneSpec


class _Probe(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment):
    """Probe enhancer.

    Args:
        gamma: Exponent.
        norm: Output normalization policy.
        input_layer: Source layer.
    """

    gamma: Annotated[float, TuneSpec(0.1, 5.0, log=True)] = 1.0

    def _operate(self, image):
        return image


def test_two_mixins_append_in_deterministic_order():
    """norm then input_layer, both after the op's own params."""
    assert list(_Probe.model_fields) == ["gamma", "norm", "input_layer"]


def test_order_holds_in_json_schema_and_serialization():
    assert list(_Probe.model_json_schema()["properties"]) == ["gamma", "norm", "input_layer"]
    import json
    params = json.loads(_Probe().to_json())["params"]
    assert list(params) == ["gamma", "norm", "input_layer"]


def test_tunespec_survives_both_forced_rebuilds():
    """Each mixin calls model_rebuild(force=True); Annotated metadata must persist."""
    meta = _Probe.model_fields["gamma"].metadata
    assert any(isinstance(m, TuneSpec) for m in meta)


def test_docstring_descriptions_reach_appended_fields():
    assert _Probe.model_fields["input_layer"].description == "Source layer."


def test_invalid_input_layer_rejected():
    with pytest.raises(ValidationError):
        _Probe(input_layer="gray")


def test_read_detect_mat_returns_2d():
    image = load_synth_yeast_plate()
    arr = _Probe(input_layer="detect_mat")._read_input_layer(image)
    assert arr.ndim == 2
    assert arr.dtype == np.float32


def test_read_rgb_returns_3d_float32_unit_range():
    image = load_synth_yeast_plate()
    arr = _Probe(input_layer="rgb")._read_input_layer(image)
    assert arr.ndim == 3 and arr.shape[2] == 3
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_read_rgb_accepts_normalized_float_rgb():
    """Float RGB is a valid image representation and must not call ``np.iinfo``."""
    from phenotypic import Image

    rgb = np.array([[[0.0, 0.25, 1.0]]], dtype=np.float32)
    arr = _Probe(input_layer="rgb")._read_input_layer(Image(rgb))

    assert arr.dtype == np.float32
    assert not arr.flags.writeable
    # Image stores as uint16 (bit_depth=16), so a float RGB round-trips through one
    # uint16 quantization step (1/65535 ~= 1.5e-5); assert equality within that step
    # rather than bit-exact (0.25 comes back as 0.24998856).
    np.testing.assert_allclose(arr, rgb, atol=1.0 / 65535)


def test_project_collapses_3d_via_detect_mode():
    image = load_synth_yeast_plate()
    image.set_detect_mode("MinRGB")
    op = _Probe(input_layer="rgb")
    rgb = op._read_input_layer(image)
    out = op._project_to_detect_mat(image, rgb)
    assert out.shape == image.detect_mat[:].shape
    np.testing.assert_allclose(out, np.min(rgb, axis=2), atol=1e-6)


def test_project_passes_2d_through_unchanged():
    image = load_synth_yeast_plate()
    op = _Probe()
    arr = image.detect_mat[:]
    assert op._project_to_detect_mat(image, arr) is arr


def test_guard_rescales_negative_input():
    op = _Probe(norm="clip")
    arr = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    out = op._guard_input_range(arr)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)


def test_guard_is_identity_for_in_range_input():
    op = _Probe(norm="clip")
    arr = np.array([0.25, 0.75], dtype=np.float32)
    assert op._guard_input_range(arr) is arr


def test_guard_skipped_when_norm_is_none():
    """A GAT-stabilized signal (~[1.9, 2.3]) must not be normalized on the way in."""
    op = _Probe(norm=None)
    arr = np.array([1.9185, 2.3065], dtype=np.float32)
    assert op._guard_input_range(arr) is arr


# --- Regressions from the cluster A review gate -----------------------------------


def test_rgb_on_grayscale_image_raises_rather_than_returning_degenerate_array():
    """`rgb.normed()` silently returns a (0, 3) array on a 2-D image.

    That array has ``ndim == 2``, so `_project_to_detect_mat` would take its
    identity path, the op would compute on an empty array, and a shape-mismatched
    result would reach `detect_mat` -- with no exception anywhere on the path.
    """
    from phenotypic import Image
    from phenotypic.sdk_.exceptions_ import NoArrayError

    gray_only = Image(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(NoArrayError):
        _Probe(input_layer="rgb")._read_input_layer(gray_only)


def test_guard_raises_on_nan_rather_than_silently_passing_it_through():
    """`nan < 0` and `nan > 1` are both False, so NaN slips past a naive guard.

    The negatives alongside it would then reach skimage and raise its opaque
    'non-negative values' error -- the exact failure the guard exists to prevent,
    surfacing only when a NaN happens to be present.
    """
    op = _Probe(norm="clip")
    arr = np.array([np.nan, -5.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        op._guard_input_range(arr)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_guard_raises_on_infinity(bad):
    op = _Probe(norm="clip")
    with pytest.raises(ValueError, match="non-finite"):
        op._guard_input_range(np.array([0.5, bad], dtype=np.float32))


def test_guard_still_skips_non_finite_when_norm_is_none():
    """norm=None means 'do not touch the array' -- including the finiteness check."""
    op = _Probe(norm=None)
    arr = np.array([np.nan, -5.0], dtype=np.float32)
    assert op._guard_input_range(arr) is arr


def test_both_input_layers_return_read_only_arrays():
    """Asymmetric writeability would make in-place ops work under one layer only."""
    image = load_synth_yeast_plate()
    dm = _Probe(input_layer="detect_mat")._read_input_layer(image)
    rgb = _Probe(input_layer="rgb")._read_input_layer(image)
    assert not dm.flags.writeable
    assert not rgb.flags.writeable
    for arr in (dm, rgb):
        with pytest.raises(ValueError, match="read-only"):
            arr *= 2.0


def test_read_rgb_does_not_allocate_float64_intermediate():
    """`normed()` chains uint8 -> float64 -> float32, peaking at 3x its own result."""
    import tracemalloc

    from phenotypic import Image

    image = Image(np.zeros((512, 512, 3), dtype=np.uint8))
    result_bytes = 512 * 512 * 3 * 4  # float32

    tracemalloc.start()
    tracemalloc.reset_peak()
    arr = _Probe(input_layer="rgb")._read_input_layer(image)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert arr.nbytes == result_bytes
    # A float64 intermediate would double the peak; the uint8 copy adds a quarter more.
    assert peak < result_bytes * 1.6, f"peak {peak} bytes vs result {result_bytes}"


def test_project_rejects_a_non_rgb_3d_array():
    image = load_synth_yeast_plate()
    with pytest.raises(ValueError, match="rows, cols, 3"):
        _Probe()._project_to_detect_mat(image, np.zeros((4, 4, 5), dtype=np.float32))
