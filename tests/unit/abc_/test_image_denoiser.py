"""Tests for the :class:`ImageDenoiser` marker ABC."""

import pytest

from phenotypic.abc_ import ImageCorrector, ImageDenoiser, ImageEnhancer
from phenotypic.enhance import (
    BayesShrinkEnhancer,
    BilateralDenoise,
    BM3DDenoiser,
    NonLocalMeansDenoiser,
    VisuShrinkEnhancer,
)


class TestImageDenoiserABC:
    """Marker ABC contract: cannot instantiate; all denoisers inherit it."""

    def test_cannot_instantiate_bare(self):
        """``ImageDenoiser`` is abstract -- direct construction must fail."""
        with pytest.raises(TypeError):
            ImageDenoiser()  # type: ignore[abstract]

    def test_subclass_of_image_enhancer(self):
        """``ImageDenoiser`` participates in the enhancer hierarchy."""
        assert issubclass(ImageDenoiser, ImageEnhancer)

    def test_not_subclass_of_image_corrector(self):
        """``ImageDenoiser`` is enhancer-only; correctors should not subclass it."""
        assert not issubclass(ImageDenoiser, ImageCorrector)

    @pytest.mark.parametrize(
        "denoiser_cls",
        [
            BM3DDenoiser,
            BayesShrinkEnhancer,
            VisuShrinkEnhancer,
            NonLocalMeansDenoiser,
            BilateralDenoise,
        ],
    )
    def test_concrete_denoisers_inherit(self, denoiser_cls):
        """All five concrete denoisers are :class:`ImageDenoiser` subclasses."""
        assert issubclass(denoiser_cls, ImageDenoiser)
        assert issubclass(denoiser_cls, ImageEnhancer)

    @pytest.mark.parametrize(
        "denoiser_cls",
        [
            BM3DDenoiser,
            BayesShrinkEnhancer,
            VisuShrinkEnhancer,
            NonLocalMeansDenoiser,
            BilateralDenoise,
        ],
    )
    def test_instances_pass_isinstance(self, denoiser_cls):
        """Instances of every denoiser register as :class:`ImageDenoiser`."""
        op = denoiser_cls()
        assert isinstance(op, ImageDenoiser)
