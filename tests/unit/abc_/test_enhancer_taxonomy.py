"""Tests for the enhancer purpose-group marker ABCs.

Every concrete :class:`ImageEnhancer` is reparented under one of seven
purpose-group marker ABCs (``ImageDenoiser`` plus six added alongside it).
These tests pin the taxonomy: each marker is abstract and an
``ImageEnhancer``, each concrete enhancer subclasses the intended marker,
and the markers stay out of the ``phenotypic.enhance`` namespace so they
never leak into the GUI builder's enhancer dropdown.
"""

import inspect

import pytest

import phenotypic.enhance as enhance
from phenotypic.abc_ import (
    BackgroundSubtraction,
    FocusBlob,
    ContrastAdjustment,
    FocusEdge,
    ImageCorrector,
    ImageDenoiser,
    ImageEnhancer,
    MorphologicalFiltering,
    Smoothing,
)

# Marker ABC -> the concrete enhancer class names that must subclass it.
TAXONOMY: dict[type, tuple[str, ...]] = {
    ImageDenoiser         : (
        "BayesShrinkEnhancer",
        "EnhanceBlockMatch",
        "NonLocalMeansDenoiser",
        "VisuShrinkEnhancer",
        "LocalEdgeDenoise",
    ),
    FocusEdge             : (
        "FocusEdgePhase",
        "FocusEdgeHessian",
        "FocusEdgeMeijering",
        "FocusEdgeFrangi",
        "FocusEdgeSato",
        "FocusEdgeLaplace",
        "FocusEdgeSobel",
    ),
    FocusBlob             : ("FocusBlobLoG",),
    Smoothing             : (
        "GaussianBlur",
        "MedianFilter",
        "RankMedianEnhancer",
        "StructureSmoothing",
    ),
    BackgroundSubtraction : (
        "SubtractGaussian",
        "SubtractRollingBall",
        "SubtractOpening",
        "FlattenIllumination",
    ),
    MorphologicalFiltering: (
        "GrayOpening",
        "WhiteTophatEnhance",
        "SubtractWhiteTophat",
    ),
    ContrastAdjustment    : (
        "EnhanceLocalContrast",
        "ContrastStretching",
        "ContrastGamma",
        "ContrastLog",
        "ContrastSigmoid",
        "ImageInverter",
        "SharpenEdgeGauss",
    ),
}

NEW_MARKERS = (
    FocusEdge,
    FocusBlob,
    Smoothing,
    BackgroundSubtraction,
    MorphologicalFiltering,
    ContrastAdjustment,
)

# (concrete_class, marker) pairs for the six newly added groups (excludes the
# pre-existing ImageDenoiser, which has its own dedicated test module).
_NEW_PAIRS = [
    (getattr(enhance, name), marker)
    for marker in NEW_MARKERS
    for name in TAXONOMY[marker]
]


class TestMarkerABCContract:
    """Each purpose-group is an abstract ``ImageEnhancer`` subclass."""

    @pytest.mark.parametrize("marker", NEW_MARKERS, ids=lambda m: m.__name__)
    def test_marker_is_abstract(self, marker):
        """Markers add no ``_operate`` -- direct construction must fail."""
        assert inspect.isabstract(marker)
        with pytest.raises(TypeError):
            marker()  # type: ignore[abstract]

    @pytest.mark.parametrize("marker", NEW_MARKERS, ids=lambda m: m.__name__)
    def test_marker_subclasses_image_enhancer(self, marker):
        """Markers participate in the enhancer hierarchy (not corrector)."""
        assert issubclass(marker, ImageEnhancer)
        assert not issubclass(marker, ImageCorrector)


class TestConcreteEnhancerReparenting:
    """Every concrete enhancer subclasses its intended purpose-group."""

    @pytest.mark.parametrize(
            ("cls", "marker"),
            _NEW_PAIRS,
            ids=[f"{c.__name__}->{m.__name__}" for c, m in _NEW_PAIRS],
    )
    def test_concrete_inherits_marker(self, cls, marker):
        assert issubclass(cls, marker)
        assert issubclass(cls, ImageEnhancer)

    @pytest.mark.parametrize(
            ("cls", "marker"),
            _NEW_PAIRS,
            ids=[f"{c.__name__}->{m.__name__}" for c, m in _NEW_PAIRS],
    )
    def test_instance_passes_isinstance(self, cls, marker):
        """Enhancers construct with no args and register as their marker."""
        op = cls()
        assert isinstance(op, marker)
        assert isinstance(op, ImageEnhancer)

    def test_each_concrete_belongs_to_exactly_one_new_group(self):
        """No concrete enhancer is shared across two new purpose-groups."""
        seen: dict[str, type] = {}
        for cls, marker in _NEW_PAIRS:
            assert cls.__name__ not in seen, (
                f"{cls.__name__} appears under both "
                f"{seen.get(cls.__name__)} and {marker.__name__}"
            )
            seen[cls.__name__] = marker


class TestMarkersStayOutOfEnhanceNamespace:
    """Markers must not leak into ``phenotypic.enhance`` (GUI isolation)."""

    @pytest.mark.parametrize("marker", NEW_MARKERS, ids=lambda m: m.__name__)
    def test_marker_not_in_enhance_namespace(self, marker):
        # The GUI builder registry walks ``phenotypic.enhance`` for
        # ImageEnhancer subclasses; a leaked marker would surface as a
        # bogus (abstract) dropdown entry.
        assert not hasattr(enhance, marker.__name__)
