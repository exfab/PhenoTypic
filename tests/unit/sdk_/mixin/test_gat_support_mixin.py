"""Tests for the optional GAT-wrapping mixin.

Covers:
- The load-bearing equivalence claim: ``EnhanceBlockMatch(use_gat=True)`` produces
  the same ``detect_mat`` as the manual ``gat_forward -> bm3d.bm3d(sigma_psd=1)
  -> gat_inverse -> /scale -> clip`` triple.
- Same equivalence for the wavelet enhancers (asserts the
  ``rescale_sigma`` deferral is wired correctly).
- Smoke tests for the wavelet correctors with ``use_gat=True`` (these
  also exercise the gray-channel write path, which has a [0, 1] range
  assertion that the mixin must bypass via ``image._data.gray``).
- Snapshot integrity: noise/defer params restored after ``_operate``.
- The ``_GAT_DEFER_VALUES`` contract: ``norm`` is ``None`` *inside* the GAT
  region and restored to its constructed value afterwards.
- Pass-through behaviour when ``use_gat=False``.
- Pipeline JSON round-trip preserves all GAT init params.
- Constructor validation rejects negative gain / sigma / scale-factor.
"""

import bm3d
import numpy as np
import pytest
from bm3d.profiles import BM3DStages
from skimage.restoration import denoise_wavelet

from phenotypic import Image, ImagePipeline
from phenotypic.correction import BayesShrinkCorrector, VisuShrinkCorrector
from phenotypic.enhance import (
    BayesShrinkEnhancer,
    EnhanceBlockMatch,
    VisuShrinkEnhancer,
)
from phenotypic.sdk_._anscombe import gat_forward, gat_inverse


# -- Fixtures --------------------------------------------------------------


@pytest.fixture
def synth_image():
    np.random.seed(42)
    arr = np.random.rand(64, 64).astype(np.float64) * 0.5 + 0.25
    return Image(arr=arr)


# -- Equivalence (key regression) ------------------------------------------


class TestEquivalenceWithExplicitTriple:
    """``use_gat=True`` matches the explicit Forward -> denoise -> Inverse triple."""

    def test_bm3d_use_gat_matches_manual_triple(self, synth_image):
        """The mixin must reproduce the manual GAT pipeline byte-for-byte (within float tol)."""
        scale = 255.0
        gain, mu, sigma = 1.0, 0.0, 0.0

        # Manual pipeline: forward -> bm3d at sigma=1 with no clip -> inverse -> /scale -> clip
        counts = synth_image.detect_mat[:] * scale
        stabilized = gat_forward(counts, mu, sigma, gain)
        profile = bm3d.BM3DProfile()
        profile.bs_ht = 8
        profile.bs_wiener = 8
        denoised_stab = bm3d.bm3d(
                stabilized,
                profile=profile,
                sigma_psd=1.0,
                stage_arg=BM3DStages.ALL_STAGES,
        )
        recovered = gat_inverse(denoised_stab, mu, sigma, gain)
        expected = (recovered / scale).clip(0.0, 1.0)

        # Mixin pipeline
        op = EnhanceBlockMatch(
                sigma_psd=0.02,
                block_size=8,
                stage_arg="all_stages",
                norm="clip",
                use_gat=True,
                gat_gain=gain,
                gat_mu=mu,
                gat_read_sigma=sigma,
                gat_scale_factor=scale,
        )
        op.apply(synth_image, inplace=True)

        np.testing.assert_allclose(
                synth_image.detect_mat[:], expected, atol=1e-6
        )

    @pytest.mark.parametrize(
            "cls,method",
            [
                (BayesShrinkEnhancer, "BayesShrink"),
                (VisuShrinkEnhancer, "VisuShrink"),
            ],
    )
    def test_wavelet_use_gat_matches_manual_triple(
            self, synth_image, cls, method
    ):
        """Wavelet enhancers: GAT mixin == manual triple with rescale_sigma=False."""
        scale = 255.0
        gain, mu, sigma = 1.0, 0.0, 0.0

        # Manual: forward -> denoise_wavelet(sigma=1, rescale_sigma=False) -> inverse
        counts = synth_image.detect_mat[:] * scale
        stabilized = gat_forward(counts, mu, sigma, gain)
        denoised_stab = denoise_wavelet(
                image=stabilized,
                sigma=1.0,
                wavelet="db2",
                mode="soft",
                wavelet_levels=None,
                method=method,
                channel_axis=None,
                rescale_sigma=False,
        )
        recovered = gat_inverse(denoised_stab, mu, sigma, gain)
        expected = (recovered / scale).clip(0.0, 1.0)

        # Mixin: user-supplied sigma + rescale_sigma must be ignored under GAT
        op = cls(
                sigma=0.05,
                wavelet="db2",
                mode="soft",
                norm="clip",
                rescale_sigma=True,
                use_gat=True,
                gat_gain=gain,
                gat_mu=mu,
                gat_read_sigma=sigma,
                gat_scale_factor=scale,
        )
        op.apply(synth_image, inplace=True)

        np.testing.assert_allclose(
                synth_image.detect_mat[:], expected, atol=1e-6
        )


# -- Wavelet corrector smoke tests (gray accessor + norm defer) -----------


class TestWaveletCorrectorGAT:
    """Wavelet correctors must not crash on the gray-channel range assert."""

    @pytest.fixture
    def gray_image(self):
        np.random.seed(123)
        arr = np.random.rand(48, 48).astype(np.float64) * 0.5 + 0.25
        return Image(arr=arr)

    @pytest.fixture
    def rgb_image(self):
        np.random.seed(123)
        arr = np.random.rand(48, 48, 3).astype(np.float64)
        return Image(arr=arr)

    @pytest.mark.parametrize(
            "cls", [BayesShrinkCorrector, VisuShrinkCorrector]
    )
    def test_gray_image_use_gat_runs(self, gray_image, cls):
        """``apply`` does not crash on the gray accessor [0, 1] range guard."""
        op = cls(use_gat=True, gat_gain=1.0, gat_scale_factor=255.0)
        out = op.apply(gray_image, inplace=False)
        assert 0.0 <= out.gray[:].min()
        assert out.gray[:].max() <= 1.0
        assert 0.0 <= out.detect_mat[:].min()
        assert out.detect_mat[:].max() <= 1.0

    @pytest.mark.parametrize(
            "cls", [BayesShrinkCorrector, VisuShrinkCorrector]
    )
    def test_rgb_image_use_gat_runs(self, rgb_image, cls):
        """RGB pass stays out of GAT but coexists with GATd gray/detect_mat."""
        expected_rgb_dtype = rgb_image.rgb[:].dtype
        op = cls(use_gat=True, gat_gain=1.0, gat_scale_factor=255.0)
        out = op.apply(rgb_image, inplace=False)
        assert out.rgb[:].dtype == expected_rgb_dtype
        assert 0.0 <= out.gray[:].min() <= out.gray[:].max() <= 1.0
        assert 0.0 <= out.detect_mat[:].min() <= out.detect_mat[:].max() <= 1.0

    @pytest.mark.parametrize(
            "cls", [BayesShrinkCorrector, VisuShrinkCorrector]
    )
    def test_norm_attr_restored(self, gray_image, cls):
        """``norm`` (in ``_GAT_DEFER_VALUES``) must roll back after the GAT region."""
        op = cls(use_gat=True, norm="clip", gat_scale_factor=255.0)
        op.apply(gray_image, inplace=True)
        assert op.norm == "clip"
        assert op.rescale_sigma is True


# -- Snapshot integrity ----------------------------------------------------


class TestSnapshotIntegrity:
    """Inner attrs must be restored after the GAT region exits."""

    def test_sigma_psd_restored(self, synth_image):
        op = EnhanceBlockMatch(
                sigma_psd=0.02, use_gat=True, gat_scale_factor=255.0
        )
        op.apply(synth_image, inplace=True)
        assert op.sigma_psd == 0.02

    def test_norm_restored(self, synth_image):
        op = EnhanceBlockMatch(
                sigma_psd=0.02, norm="clip", use_gat=True, gat_scale_factor=255.0
        )
        op.apply(synth_image, inplace=True)
        assert op.norm == "clip"

    def test_attrs_restored_even_on_inner_failure(self, synth_image):
        """``finally`` clause guarantees attrs roll back even if inner raises."""
        op = EnhanceBlockMatch(
                sigma_psd=0.02, use_gat=True, gat_scale_factor=255.0
        )

        def boom(_image):
            raise RuntimeError("simulated failure")

        with pytest.raises(RuntimeError, match="simulated failure"):
            op._gat_apply(synth_image, "detect_mat", boom)
        assert op.sigma_psd == 0.02
        assert op.norm == "clip"


# -- Pass-through when use_gat=False ---------------------------------------


class TestPassThrough:
    """``use_gat=False`` must be identical to running without the mixin code path."""

    def test_use_gat_false_matches_default_bm3d(self, synth_image):
        # Run with use_gat=False
        op_off = EnhanceBlockMatch(sigma_psd=0.02, use_gat=False)
        img_off = Image(arr=synth_image.detect_mat[:].copy())
        op_off.apply(img_off, inplace=True)

        # Run the same body manually (no GAT wrapping at all)
        profile = bm3d.BM3DProfile()
        profile.bs_ht = 8
        profile.bs_wiener = 8
        expected = bm3d.bm3d(
                synth_image.detect_mat[:],
                profile=profile,
                sigma_psd=0.02,
                stage_arg=BM3DStages.ALL_STAGES,
        ).clip(0.0, 1.0)

        # BM3D has small non-determinism in float reductions across separate
        # invocations; tolerance loosened to 1e-5 (well below any observable
        # behavioral difference). The point is that no GAT bookkeeping has
        # corrupted the output -- the result matches direct skimage usage.
        np.testing.assert_allclose(
                img_off.detect_mat[:], expected, atol=1e-5
        )


# -- Serialization round-trip ----------------------------------------------


class TestSerialization:
    """All GAT params survive ``ImagePipeline.to_json`` -> ``from_json``."""

    def test_gat_params_round_trip(self):
        op = EnhanceBlockMatch(
                sigma_psd=0.05,
                use_gat=True,
                gat_gain=2.0,
                gat_mu=1.0,
                gat_read_sigma=0.5,
                gat_scale_factor=65535.0,
        )
        pipeline = ImagePipeline(pipe_cfgs=[op])
        json_str = pipeline.to_json()
        loaded = ImagePipeline.from_json(json_str)

        loaded_op = list(loaded._ops.values())[0]
        assert isinstance(loaded_op, EnhanceBlockMatch)
        assert loaded_op.sigma_psd == 0.05
        assert loaded_op.use_gat is True
        assert loaded_op.gat_gain == 2.0
        assert loaded_op.gat_mu == 1.0
        assert loaded_op.gat_read_sigma == 0.5
        assert loaded_op.gat_scale_factor == 65535.0

    def test_default_gat_params_round_trip(self):
        op = EnhanceBlockMatch(sigma_psd=0.02)
        pipeline = ImagePipeline(pipe_cfgs=[op])
        loaded = ImagePipeline.from_json(pipeline.to_json())
        loaded_op = list(loaded._ops.values())[0]
        assert loaded_op.use_gat is False
        assert loaded_op.gat_gain == 1.0
        assert loaded_op.gat_mu == 0.0
        assert loaded_op.gat_read_sigma == 0.0
        assert loaded_op.gat_scale_factor is None


# -- Constructor validation ------------------------------------------------


class TestValidation:
    """Mixin rejects bad GAT parameters at construction time."""

    def test_gat_gain_zero_raises(self):
        with pytest.raises(
                ValueError,
                match=r"gat_gain\s+Input should be greater than 0",
        ):
            EnhanceBlockMatch(use_gat=True, gat_gain=0.0)

    def test_gat_gain_negative_raises(self):
        with pytest.raises(
                ValueError,
                match=r"gat_gain\s+Input should be greater than 0",
        ):
            EnhanceBlockMatch(use_gat=True, gat_gain=-1.0)

    def test_gat_read_sigma_negative_raises(self):
        with pytest.raises(
                ValueError,
                match=r"gat_read_sigma\s+Input should be greater than or equal to 0",
        ):
            EnhanceBlockMatch(use_gat=True, gat_read_sigma=-0.1)

    def test_gat_scale_factor_zero_raises(self):
        with pytest.raises(
                ValueError,
                match=r"gat_scale_factor\s+Input should be greater than 0",
        ):
            EnhanceBlockMatch(use_gat=True, gat_scale_factor=0.0)

    def test_gat_scale_factor_negative_raises(self):
        with pytest.raises(
                ValueError,
                match=r"gat_scale_factor\s+Input should be greater than 0",
        ):
            EnhanceBlockMatch(use_gat=True, gat_scale_factor=-255.0)

    def test_valid_gat_params_accepted(self):
        op = EnhanceBlockMatch(
                use_gat=True,
                gat_gain=2.0,
                gat_mu=0.5,
                gat_read_sigma=1.0,
                gat_scale_factor=65535.0,
        )
        assert op.gat_gain == 2.0
        assert op.gat_mu == 0.5
        assert op.gat_read_sigma == 1.0
        assert op.gat_scale_factor == 65535.0


# -- _GAT_DEFER_VALUES contract --------------------------------------------


class TestDeferValues:
    """``_GAT_DEFER_VALUES`` maps each deferred attribute to its inert value."""

    def test_defer_values_is_a_mapping_with_correct_inert_values(self):
        assert VisuShrinkEnhancer._GAT_DEFER_VALUES == {
            "norm"         : None,
            "rescale_sigma": False,
        }

    def test_norm_is_none_inside_the_gat_region_and_restored_after(
            self, synth_image, monkeypatch
    ):
        """The crown-jewel round-trip: rescaling a GAT-stabilized signal is fatal.

        A GAT-stabilized ``detect_mat`` lives around ~1.9-2.3, deliberately
        outside [0, 1]. If ``norm`` is not deferred to ``None``, the inner
        denoiser clips or rescales it and the inverse transform collapses.
        """
        op = VisuShrinkEnhancer(
                use_gat=True, norm="clip", gat_scale_factor=255.0
        )
        seen = {}

        original = VisuShrinkEnhancer._denoise_detect_mat

        def spy(self, image):
            seen["norm_inside"] = self.norm
            seen["rescale_sigma_inside"] = self.rescale_sigma
            seen["max_inside"] = float(image.detect_mat[:].max())
            return original(self, image)

        monkeypatch.setattr(VisuShrinkEnhancer, "_denoise_detect_mat", spy)
        op.apply(synth_image, inplace=True)

        assert seen["norm_inside"] is None
        assert seen["rescale_sigma_inside"] is False
        # Confirms the spy really ran inside the stabilized domain.
        assert seen["max_inside"] > 1.0
        assert op.norm == "clip"
        assert op.rescale_sigma is True
