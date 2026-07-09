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


@pytest.mark.xfail(reason="compute_from_rgb lands in Phase 3", strict=True)
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
