"""Tests for AsymmetricSpurTrimmer."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import disk, rectangle

from phenotypic import Image, ImagePipeline
from phenotypic.refine import AsymmetricSpurTrimmer


# =====================================================================
# Fixtures
# =====================================================================


def _make_image(objmap: np.ndarray) -> Image:
    """Wrap an objmap in an Image with a dummy rgb backing store."""
    h, w = objmap.shape
    image = Image(arr=np.zeros((h, w, 3), dtype=np.uint8))
    image.objmap[:] = objmap.astype(image.objmap[:].dtype)
    return image


def _stamp_disk(objmap: np.ndarray, centre, radius: int, label: int) -> None:
    rr, cc = disk(centre, radius, shape=objmap.shape)
    objmap[rr, cc] = label


def _stamp_rect(objmap: np.ndarray, start, end, label: int) -> None:
    rr, cc = rectangle(start=start, end=end, shape=objmap.shape)
    objmap[rr.astype(int), cc.astype(int)] = label


def _stamp_web(
        objmap: np.ndarray,
        top_left,
        shape,
        cell_size: int,
        label: int,
) -> None:
    """Stamp a reticulated web (many enclosed small holes) into objmap.

    Creates a filled rectangle then knocks out an evenly-spaced grid of
    square holes so the topological Euler number is very negative.
    """
    h, w = shape
    r0, c0 = top_left
    objmap[r0:r0 + h, c0:c0 + w] = label

    # Punch out small square holes on a grid, leaving 1-pixel walls.
    for rr in range(r0 + 1, r0 + h - 1, cell_size):
        for cc in range(c0 + 1, c0 + w - 1, cell_size):
            r_end = min(rr + cell_size - 1, r0 + h - 1)
            c_end = min(cc + cell_size - 1, c0 + w - 1)
            if r_end - rr >= 1 and c_end - cc >= 1:
                objmap[rr:r_end, cc:c_end] = 0


@pytest.fixture
def round_colony() -> Image:
    """Single compact disk colony — should never trigger trimming."""
    objmap = np.zeros((200, 200), dtype=np.int32)
    _stamp_disk(objmap, centre=(100, 100), radius=40, label=1)
    return _make_image(objmap)


@pytest.fixture
def spurred_colony() -> Image:
    """Disk + narrow rectangular spur extending 3× body radius past envelope."""
    objmap = np.zeros((250, 400), dtype=np.int32)
    _stamp_disk(objmap, centre=(125, 100), radius=40, label=1)
    # A 6-pixel-tall strip extending far to the right.
    _stamp_rect(objmap, start=(122, 150), end=(128, 350), label=1)
    return _make_image(objmap)


@pytest.fixture
def linear_branch_colony() -> Image:
    """Disk + single thin linear branch (topologically a line — 0 holes)."""
    objmap = np.zeros((250, 400), dtype=np.int32)
    _stamp_disk(objmap, centre=(125, 100), radius=40, label=1)
    _stamp_rect(objmap, start=(123, 150), end=(127, 320), label=1)
    return _make_image(objmap)


@pytest.fixture
def web_noise_colony() -> Image:
    """Disk + reticulated web region past the envelope (many enclosed holes)."""
    objmap = np.zeros((250, 400), dtype=np.int32)
    _stamp_disk(objmap, centre=(125, 100), radius=40, label=1)
    _stamp_web(
            objmap,
            top_left=(90, 160),
            shape=(70, 180),
            cell_size=4,
            label=1,
    )
    return _make_image(objmap)


@pytest.fixture
def two_colonies_one_spurred() -> Image:
    """Two disks; only the right-hand one has a spur."""
    objmap = np.zeros((250, 500), dtype=np.int32)
    _stamp_disk(objmap, centre=(125, 60), radius=35, label=1)
    _stamp_disk(objmap, centre=(125, 300), radius=40, label=2)
    _stamp_rect(objmap, start=(123, 350), end=(127, 480), label=2)
    return _make_image(objmap)


# =====================================================================
# Tests
# =====================================================================


class TestCircularColonyNoTrim:
    """A compact disk colony should never be touched."""

    def test_pure_rsym_mode(self, round_colony: Image) -> None:
        before = round_colony.objmap[:].copy()
        refined = AsymmetricSpurTrimmer().apply(round_colony)
        np.testing.assert_array_equal(refined.objmap[:], before)

    def test_beehive_mode(self, round_colony: Image) -> None:
        before = round_colony.objmap[:].copy()
        refined = AsymmetricSpurTrimmer(beehive_threshold=0.002).apply(
                round_colony
        )
        np.testing.assert_array_equal(refined.objmap[:], before)


class TestLopsidedSpurPureRsym:
    """Pure R_sym mode should remove the spur but preserve the disk body."""

    def test_spur_removed_body_preserved(self, spurred_colony: Image) -> None:
        before_area = int((spurred_colony.objmap[:] == 1).sum())
        refined = AsymmetricSpurTrimmer().apply(spurred_colony)
        after_area = int((refined.objmap[:] == 1).sum())

        # Spur should be clearly shorter than before — a meaningful fraction trimmed.
        assert after_area < before_area, (
            "Expected spur pixels to be trimmed"
        )
        # ... but the disk body (a rough circle of radius 40) should survive.
        assert after_area >= int(np.pi * 35 ** 2), (
            "Disk body looks like it was eaten too"
        )

        # No pixels should survive at the far right where the spur ended.
        assert not (refined.objmap[:, 340:] > 0).any(), (
            "Spur tip should be gone"
        )


class TestLinearBranchPreservedBeehiveMode:
    """Beehive mode keeps long linear branches (holes = 0)."""

    def test_linear_branch_kept(self, linear_branch_colony: Image) -> None:
        before = linear_branch_colony.objmap[:].copy()
        refined = AsymmetricSpurTrimmer(beehive_threshold=0.002).apply(
                linear_branch_colony
        )
        # The linear strip past R_sym is topologically linear → holes=0 → kept.
        # Pixel count should be unchanged.
        np.testing.assert_array_equal(refined.objmap[:], before)


class TestWebNoiseTrimmedBeehiveMode:
    """Beehive mode removes reticulated web regions past R_sym."""

    def test_web_removed(self, web_noise_colony: Image) -> None:
        refined = AsymmetricSpurTrimmer(
                beehive_threshold=0.002,
        ).apply(web_noise_colony)

        # The web lives at cols 160-340 ish. After trim, little should survive there.
        web_cols_after = (refined.objmap[:, 180:330] > 0).sum()
        assert web_cols_after < 200, (
            "Web noise should be largely removed"
        )


class TestSmallObjectSkipped:
    """Objects below min_object_area are ignored entirely."""

    def test_tiny_blob_untouched(self) -> None:
        objmap = np.zeros((200, 200), dtype=np.int32)
        _stamp_disk(objmap, centre=(50, 50), radius=3, label=1)
        image = _make_image(objmap)

        before = image.objmap[:].copy()
        refined = AsymmetricSpurTrimmer(min_object_area=100).apply(image)
        np.testing.assert_array_equal(refined.objmap[:], before)


class TestSmallCcSkippedBeehiveMode:
    """Candidate CCs below min_cc_area are preserved in beehive mode."""

    def test_tiny_cc_kept(self) -> None:
        objmap = np.zeros((250, 400), dtype=np.int32)
        _stamp_disk(objmap, centre=(125, 100), radius=40, label=1)
        # A tiny 6-pixel blob well past the disk envelope.
        objmap[124:127, 260:262] = 1
        image = _make_image(objmap)

        before = image.objmap[:].copy()
        refined = AsymmetricSpurTrimmer(
                beehive_threshold=0.002, min_cc_area=50,
        ).apply(image)
        np.testing.assert_array_equal(refined.objmap[:], before)


class TestMultiColonyIndependent:
    """Each colony is handled independently in pure R_sym mode."""

    def test_only_spurred_colony_trimmed(
            self, two_colonies_one_spurred: Image
    ) -> None:
        before = two_colonies_one_spurred.objmap[:].copy()
        refined = AsymmetricSpurTrimmer().apply(two_colonies_one_spurred)
        after = refined.objmap[:]

        # Label-1 disk (no spur) must be byte-for-byte identical.
        np.testing.assert_array_equal(after == 1, before == 1)
        # Label 2's spur region must have lost pixels.
        assert (after == 2).sum() < (before == 2).sum()


class TestBeehiveThresholdMonotonic:
    """Rising beehive_threshold monotonically reduces trimmed pixels."""

    def test_monotonic_decrease(self, web_noise_colony: Image) -> None:
        trimmed_counts: list[int] = []
        originals = int((web_noise_colony.objmap[:] > 0).sum())
        for threshold in [0.0, 0.002, 0.01, 0.05, 1.0]:
            img = _make_image(web_noise_colony.objmap[:].copy())
            refined = AsymmetricSpurTrimmer(
                    beehive_threshold=threshold,
            ).apply(img)
            trimmed_counts.append(
                    originals - int((refined.objmap[:] > 0).sum())
            )

        for earlier, later in zip(trimmed_counts, trimmed_counts[1:]):
            assert later <= earlier, (
                f"Expected monotonic decrease; saw {trimmed_counts}"
            )


class TestJsonRoundtrip:
    """Serialize and deserialize via ImagePipeline; attributes survive."""

    def test_attributes_match(self) -> None:
        trimmer = AsymmetricSpurTrimmer(
                symmetry_threshold=0.4,
                beehive_threshold=0.003,
                min_cc_area=75,
                min_object_area=120,
                method="intensity",
        )
        pipeline = ImagePipeline(ops=[trimmer])
        restored = ImagePipeline.from_json(pipeline.to_json())

        ops = restored.get_ops()
        restored_trimmer = ops["AsymmetricSpurTrimmer"]

        assert isinstance(restored_trimmer, AsymmetricSpurTrimmer)
        assert restored_trimmer.symmetry_threshold == 0.4
        assert restored_trimmer.beehive_threshold == 0.003
        assert restored_trimmer.min_cc_area == 75
        assert restored_trimmer.min_object_area == 120
        assert restored_trimmer.method == "intensity"


class TestProtectedComponents:
    """RGB/gray/detect_mat must not be modified by the refiner."""

    def test_rgb_gray_detect_mat_preserved(
            self, spurred_colony: Image,
    ) -> None:
        before_rgb = spurred_colony.rgb[:].copy()
        before_gray = spurred_colony.gray[:].copy()
        before_detect = spurred_colony.detect_mat[:].copy()

        refined = AsymmetricSpurTrimmer().apply(spurred_colony)

        np.testing.assert_array_equal(refined.rgb[:], before_rgb)
        np.testing.assert_array_equal(refined.gray[:], before_gray)
        np.testing.assert_array_equal(refined.detect_mat[:], before_detect)


class TestIntensityModeSmoke:
    """Smoke-test the intensity-centroid branch end-to-end.

    The distance branch is exercised by every other test. These tests guard
    the intensity path on (a) a real grayscale image and (b) the degenerate
    all-zero-gray case that would otherwise produce a NaN weighted centroid.
    """

    def test_intensity_mode_applies_with_real_gray(self) -> None:
        """Paint a non-uniform gray under the mask so centroid_weighted is defined."""
        objmap = np.zeros((250, 400), dtype=np.int32)
        _stamp_disk(objmap, centre=(125, 100), radius=40, label=1)
        _stamp_rect(objmap, start=(122, 150), end=(128, 350), label=1)

        rgb = np.zeros((250, 400, 3), dtype=np.uint8)
        rgb[objmap > 0] = (180, 180, 180)
        image = Image(arr=rgb)
        image.objmap[:] = objmap.astype(image.objmap[:].dtype)

        refined = AsymmetricSpurTrimmer(method="intensity").apply(image)
        # Spur should still be trimmed (intensity-weighted centroid lives on
        # the disk body since the spur has the same gray level but way fewer pixels).
        assert (refined.objmap[:] > 0).sum() < (image.objmap[:] > 0).sum()

    def test_intensity_mode_tolerates_zero_gray(
            self, spurred_colony: Image,
    ) -> None:
        """All-zero gray would break centroid_weighted; op must fall back."""
        # spurred_colony has an all-zero RGB/gray backing; intensity mode must
        # still run and produce a valid result (fallback to distance centroid).
        refined = AsymmetricSpurTrimmer(method="intensity").apply(spurred_colony)
        assert refined.objmap[:].max() >= 0


class TestInvalidConstructorArgs:
    """Constructor validation surfaces obviously bad parameter values."""

    def test_symmetry_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            AsymmetricSpurTrimmer(symmetry_threshold=1.5)

    def test_negative_beehive_threshold(self) -> None:
        with pytest.raises(ValueError):
            AsymmetricSpurTrimmer(beehive_threshold=-0.01)

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError):
            AsymmetricSpurTrimmer(method="bogus")  # type: ignore[arg-type]
