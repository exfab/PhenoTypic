"""
Tests for the GitterGridObjectDetector class.

Covers initialization, threshold parameter options, and end-to-end detection
on real plate images to ensure the ported gitter workflow produces labeled
colonies on a gridded plate image.
"""
import pytest
import numpy as np
import phenotypic
from phenotypic import GridImage
from phenotypic.detect import GitterGridObjectDetector
from phenotypic.data import load_plate_12hr

from .resources.TestHelper import timeit


class TestGitterGridDetectorInitialization:
    """Initialization and parameter handling."""

    @timeit
    def test_default_initialization(self):
        detector = GitterGridObjectDetector()
        assert detector.thresh_method == "gitter"
        assert detector.subtract_background is True
        assert detector.remove_noise is True
        assert detector.background_fast_height == 1000
        assert detector.gaussian_sigma is None
        assert detector.smoothing_sigma == 2.0
        assert detector.min_peak_distance is None
        assert detector.peak_prominence is None
        assert detector.edge_refinement is True
        assert detector.closing_radius == 1
        assert detector.min_fill_ratio == pytest.approx(0.01)
        assert detector.max_fill_ratio == pytest.approx(0.9)
        assert detector.rescue_method == "local"

    @timeit
    @pytest.mark.parametrize(
        "thresh_method",
        ["gitter", "otsu", "mean", "local", "triangle", "minimum", "isodata"],
    )
    def test_initialization_threshold_methods(self, thresh_method):
        detector = GitterGridObjectDetector(thresh_method=thresh_method)
        assert detector.thresh_method == thresh_method


class TestGitterGridDetectorOnGridImage:
    """Detection behavior on a gridded plate image."""

    @timeit
    def test_detection_on_grid_image(self):
        image = GridImage(load_plate_12hr(), nrows=8, ncols=12)
        detector = GitterGridObjectDetector(
            subtract_background=True,
            remove_noise=True,
            closing_radius=1,
            smoothing_sigma=2.0,
            background_fast_height=200,
            processing_height=200,
            edge_refinement=False,
            step_timeout=60.0,
        )

        result = detector.apply(image, inplace=False)

        assert isinstance(result, phenotypic.GridImage)
        assert result.objmap[:].max() > 0
        assert result.num_objects > 0
        # Grid is 8x12 = 96 wells; allow small over-segmentation tolerance
        assert result.num_objects <= 120

    @timeit
    def test_inplace_detection_changes_image(self):
        image = GridImage(load_plate_12hr(), nrows=8, ncols=12)
        before = image.objmap[:].copy()

        detector = GitterGridObjectDetector(
            subtract_background=False,
            remove_noise=False,
            edge_refinement=False,
        )
        result = detector.apply(image, inplace=True)

        assert result is image
        assert not np.array_equal(before, image.objmap[:])
        assert image.num_objects > 0
