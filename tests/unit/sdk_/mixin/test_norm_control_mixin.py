"""NormControlMixin replaces ClipControlMixin: duck-types on `.norm`, sets None."""

import pytest

from phenotypic import ImagePipeline
from phenotypic.abc_ import ImageEnhancer
from phenotypic.enhance import GaussianBlur
from phenotypic.sdk_ import NormalizedOutputMixin, NormControlMixin


class _Probe(NormalizedOutputMixin, ImageEnhancer):
    """Probe enhancer carrying a ``norm`` field.

    Stands in for the enhancers migrated to ``norm`` in a later task; this module
    must stay green before that migration lands.

    Args:
        sigma: Width.
        norm: Output normalization policy.
    """

    sigma: float = 1.0

    def _operate(self, image):
        return image


def test_disable_normalization_sets_norm_none():
    enh = _Probe(sigma=5.0, norm="clip")
    copied = NormControlMixin._disable_normalization(enh)
    assert enh.norm == "clip", "original must be untouched"
    assert copied.norm is None


def test_disable_normalization_clears_rescale_too():
    """`rescale` is as destructive as `clip` inside a GAT domain."""
    copied = NormControlMixin._disable_normalization(_Probe(norm="rescale"))
    assert copied.norm is None


def test_op_with_norm_is_cleared_and_op_without_norm_is_untouched():
    """Pins the distinction the old `.clip` gate silently lost after migration.

    An op exposing `norm` must come back with normalization disabled; an op that
    genuinely has no `norm` must pass through unchanged and must not raise.
    """
    with_norm = _Probe(norm="clip")
    without_norm = GaussianBlur(sigma=1.0)
    assert not hasattr(without_norm, "norm")

    assert NormControlMixin._disable_normalization(with_norm).norm is None
    assert NormControlMixin._disable_normalization(without_norm) is without_norm


def test_disable_normalization_recurses_into_pipeline():
    pipe = ImagePipeline(pipe_cfgs=[GaussianBlur(sigma=1.0), _Probe(norm="clip")])
    copied = NormControlMixin._disable_normalization(pipe)
    assert list(copied._ops.values())[1].norm is None


def test_pipeline_recursion_leaves_the_original_pipeline_untouched():
    probe = _Probe(norm="clip")
    pipe = ImagePipeline(pipe_cfgs=[GaussianBlur(sigma=1.0), probe])
    NormControlMixin._disable_normalization(pipe)
    assert probe.norm == "clip"
    assert list(pipe._ops.values())[1].norm == "clip"


def test_old_symbol_is_gone():
    with pytest.raises(ImportError):
        from phenotypic.sdk_ import ClipControlMixin  # noqa: F401
