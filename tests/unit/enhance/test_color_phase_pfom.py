"""Spec §7.1 -- Pratt's Figure of Merit, as a **ranking** regression.

**Do not attempt to reproduce CMPCM's Table 1.** Its PFOM values
(`Canny 0.8888 · Log 0.9008 · VPMM 0.9934 · PC 0.9099 · MPC 0.9321 · CMPCM 0.9989`) are
measured on Fig. 4a's "geometry image" (176x298), **whose pixels are published nowhere**; the
paper's §2 pixel spec (173x299) describes Fig. 1a1, a different image. The numbers are
transcribed in ``references.md`` for exactly one purpose: the CMPCM/VPMM gap is ``0.0055``,
so a ranking regression that separated them would need better than ~0.5% PFOM resolution.

Ours separates ``colour PC > PC > Canny``. The ``PC``-to-``Canny`` gap is enormous
(``0.25``). The ``colour PC``-to-``PC`` gap is **``0.0027``** -- real, reproducible, and
*smaller than CMPCM's own margin over VPMM*. The tests below assert the ranking and then
state that margin explicitly, so nobody reads "colour PC wins" as "colour PC wins by much".
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

import phenotypic
from phenotypic.enhance import (
    FocusEdgeColorPhase,
    FocusEdgeMonogenicPhase,
    FocusEdgePhase,
)

ALPHA = 1.0 / 9.0  # Pratt's constant


def pratt_figure_of_merit(detected: np.ndarray, ideal: np.ndarray) -> float:
    """``FOM = 1/max(N_d, N_i) * sum_i 1/(1 + alpha * d_i**2)``.

    ``d_i`` is the distance from each *detected* edge pixel to the nearest *ideal* one. The
    ``max`` in the denominator penalises both over- and under-detection.
    """
    n_detected, n_ideal = int(detected.sum()), int(ideal.sum())
    if n_detected == 0:
        return 0.0
    distance = ndimage.distance_transform_edt(~ideal)[detected]
    return float((1.0 / (1.0 + ALPHA * distance ** 2)).sum() / max(n_detected, n_ideal))


def _geometric_colour_image(size: int = 192) -> tuple[phenotypic.Image, np.ndarray]:
    """Shapes of distinct hue on a coloured ground. Ideal edges are the shape boundaries.

    Deterministic: no RNG. A rectangle, a disc, a triangle and a thin bar, so the ranking is
    not decided by a single shape family.
    """
    rgb = np.empty((size, size, 3), dtype=np.float64)
    rgb[:] = (0.45, 0.50, 0.42)
    rows, cols = np.indices((size, size))
    shapes = np.zeros((size, size), dtype=bool)

    rectangle = (rows > 24) & (rows < 76) & (cols > 24) & (cols < 92)
    rgb[rectangle] = (0.85, 0.35, 0.30)
    shapes |= rectangle

    disc = (rows - 130) ** 2 + (cols - 52) ** 2 < 32 ** 2
    rgb[disc] = (0.25, 0.70, 0.40)
    shapes |= disc

    triangle = (rows > 30) & (rows < 100) & (np.abs(cols - 140) < (rows - 30) // 2)
    rgb[triangle] = (0.30, 0.35, 0.85)
    shapes |= triangle

    bar = (rows > 120) & (rows < 132) & (cols > 100) & (cols < 175)
    rgb[bar] = (0.85, 0.80, 0.25)
    shapes |= bar

    ideal = shapes ^ ndimage.binary_erosion(shapes)
    return phenotypic.Image((rgb * 255).round().astype(np.uint8)), ideal


def _binarize(response: np.ndarray) -> np.ndarray:
    """Otsu, then thin to one pixel. The same treatment for every method."""
    finite = response[np.isfinite(response)]
    return skeletonize(response > threshold_otsu(finite))


class TestThePrattMetricItself:
    """Do not mark your own homework. The metric has closed-form answers; check them.

    Lesson ``S5``: every check up to ``verify_claims.py::check_14`` used a test signal this
    spec wrote against a ground truth this spec derived. A PFOM implementation that silently
    normalised differently would make every ranking below meaningless.
    """

    def test_a_perfect_detector_scores_exactly_one(self):
        _, ideal = _geometric_colour_image()
        assert pratt_figure_of_merit(ideal, ideal) == pytest.approx(1.0)

    def test_an_edge_displaced_by_one_pixel_scores_exactly_nine_tenths(self):
        """Every detected pixel sits at ``d = 1``, so ``FOM = 1/(1 + 1/9) = 0.9`` exactly."""
        ideal = np.zeros((32, 32), dtype=bool)
        ideal[:, 16] = True
        displaced = np.zeros_like(ideal)
        displaced[:, 17] = True
        assert pratt_figure_of_merit(displaced, ideal) == pytest.approx(0.9)

    def test_an_empty_detection_scores_zero(self):
        _, ideal = _geometric_colour_image()
        assert pratt_figure_of_merit(np.zeros_like(ideal), ideal) == 0.0

    def test_over_detection_is_penalised(self):
        """``max(N_d, N_i)`` in the denominator: firing everywhere must not score well.

        Measured: ``0.1825`` on this image (``N_d = 36864`` against ``N_i = 777``). Well
        below every real method's ``0.949``, and below the ``0.9`` a one-pixel-displaced
        detector scores -- but **not** below ``0.1``, which a first draft of this test
        guessed. The penalty is ``N_i/N_d`` times the mean proximity weight, not a wipeout.
        """
        _, ideal = _geometric_colour_image()
        everywhere = np.ones_like(ideal)
        score = pratt_figure_of_merit(everywhere, ideal)
        assert score == pytest.approx(0.1825, abs=1e-3)
        assert score < 0.9, "over-detection must score below a one-pixel-displaced edge"


class TestTheRankingRegression:
    """Spec §7.1, at ``color_space="hsv", fusion="l2"`` -- the paper's configuration.

    Computed on the **un-clipped** ``_color_phase_congruency`` output, because ``l2``'s range
    is ``[0, ||w||]`` and clipping would truncate the paper's actual quantity (drift ``C3``).
    """

    @staticmethod
    def _scores() -> dict[str, float]:
        image, ideal = _geometric_colour_image()
        rgb = image.rgb[:]
        scores = {
            "canny": pratt_figure_of_merit(canny(rgb2gray(rgb), sigma=1.0), ideal),
            "phase_congruency": pratt_figure_of_merit(
                _binarize(
                    FocusEdgePhase().apply(phenotypic.Image(rgb)).detect_mat[:].astype(float)
                ),
                ideal,
            ),
            "monogenic": pratt_figure_of_merit(
                _binarize(
                    FocusEdgeMonogenicPhase()
                    .apply(phenotypic.Image(rgb))
                    .detect_mat[:]
                    .astype(float)
                ),
                ideal,
            ),
            "colour_phase_congruency": pratt_figure_of_merit(
                _binarize(
                    FocusEdgeColorPhase(color_space="hsv", fusion="l2")
                    ._color_phase_congruency(image)
                    .pc
                ),
                ideal,
            ),
        }
        return scores

    def test_colour_pc_beats_pc_beats_canny(self):
        scores = self._scores()
        assert scores["colour_phase_congruency"] > scores["phase_congruency"] > scores["canny"], (
            f"ranking broken: {scores}"
        )

    def test_the_colour_margin_over_pc_is_small_and_that_is_the_honest_result(self):
        """Measured ``0.0027``. Assert the *order of magnitude*, not just the sign.

        If this margin ever grows past ``0.02`` on this image, the operation has changed
        materially and somebody should find out why. If it collapses below ``0.0005`` the
        ranking is no longer resolvable and the regression is worthless as evidence.

        For scale: CMPCM's own reported margin over VPMM is ``0.0055``. Our colour-over-PC
        margin is **half of that**. "Colour PC wins" is true; "colour PC wins by much" is not.
        """
        scores = self._scores()
        margin = scores["colour_phase_congruency"] - scores["phase_congruency"]
        assert 0.0005 < margin < 0.02, f"colour-over-PC margin is {margin:.5f}"

    def test_canny_is_far_behind_and_that_margin_is_robust(self):
        """Keep a material margin on both supported CI and local platforms.

        Canny's non-maximum suppression is platform-sensitive on this synthetic
        geometry: Linux scores about 0.862 while macOS scores about 0.698. The
        phase-congruency score is stable at about 0.949, so 0.05 is a robust
        cross-platform separation without weakening the ordering contract.
        """
        scores = self._scores()
        margin = scores["phase_congruency"] - scores["canny"]
        assert margin > 0.05, (
            f"phase-congruency/Canny PFOM margin is {margin:.5f}: {scores}"
        )

    def test_every_phase_method_clears_nine_tenths(self):
        """A displaced-by-one-pixel detector scores exactly ``0.9`` (see above). All four
        phase-congruency variants must beat that; Canny must not."""
        scores = self._scores()
        for method in ("phase_congruency", "monogenic", "colour_phase_congruency"):
            assert scores[method] > 0.9, f"{method} = {scores[method]:.4f}"
        assert scores["canny"] < 0.9

    def test_we_do_not_claim_to_reproduce_table_1(self):
        """A documentation test, and a deliberate one.

        CMPCM's Table 1 is measured on an image whose pixels are published nowhere. If a
        future reader is tempted to compare our absolute numbers against ``0.9989``, this
        assertion is where they will find out why they must not.
        """
        scores = self._scores()
        assert scores["colour_phase_congruency"] != pytest.approx(0.9989, abs=1e-3), (
            "coincidental agreement with CMPCM's Table 1 would be meaningless: it is "
            "measured on Fig. 4a's geometry image, which is not published"
        )
