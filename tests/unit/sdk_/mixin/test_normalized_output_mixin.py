"""Contract for NormalizedOutputMixin: field append, norm semantics, clip rejection."""
import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic.abc_ import ImageEnhancer
from phenotypic.sdk_ import NormalizedOutputMixin


class _Probe(NormalizedOutputMixin, ImageEnhancer):
    """Probe enhancer.

    Args:
        sigma: Width.
        norm: Output normalization policy.
    """

    sigma: float = 1.0

    def _operate(self, image):
        return image


def test_norm_is_appended_last():
    assert list(_Probe.model_fields) == ["sigma", "norm"]


def test_norm_appended_last_in_json_schema():
    assert list(_Probe.model_json_schema()["properties"]) == ["sigma", "norm"]


def test_default_is_clip():
    assert _Probe().norm == "clip"


@pytest.mark.parametrize(
    ("norm", "expected"),
    [
        ("clip", [0.0, 0.5, 1.0]),
        ("rescale", [0.0, 0.5, 1.0]),
        (None, [-0.5, 0.5, 1.5]),
    ],
)
def test_apply_norm(norm, expected):
    arr = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
    np.testing.assert_allclose(_Probe(norm=norm)._apply_norm(arr), expected, atol=1e-6)


def test_rescale_differs_from_clip_when_input_is_inside_unit_range():
    """`clip` is the identity in-range; `rescale` stretches. Distinguishes the two."""
    arr = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    np.testing.assert_allclose(_Probe(norm="clip")._apply_norm(arr), arr, atol=1e-6)
    np.testing.assert_allclose(
        _Probe(norm="rescale")._apply_norm(arr), [0.0, 0.5, 1.0], atol=1e-6
    )


def test_legacy_clip_key_raises_migration_message():
    with pytest.raises(ValidationError, match=r"`clip` was replaced by `norm` in 0\.18\.0"):
        _Probe(clip=True)


def test_invalid_norm_rejected():
    with pytest.raises(ValidationError):
        _Probe(norm="passthrough")


def test_setattr_to_none_under_validate_assignment():
    """The GAT defer path uses setattr; validate_assignment must accept None."""
    op = _Probe(norm="clip")
    op.norm = None
    assert op.norm is None
