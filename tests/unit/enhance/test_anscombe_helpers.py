"""Tests for the private GAT math helpers in :mod:`phenotypic.enhance._anscombe`.

Covers forward/inverse mathematics, round-trip identity, NaN handling, and
``resolve_scale_factor`` bit-depth dispatch. The user-facing classes
``AnscombeForward``/``AnscombeInverse`` were removed in favor of the
``use_gat=True`` flag on individual denoisers; this file replaces the
math-correctness portion of the deleted ``test_anscombe_denoise.py``.
"""

import numpy as np

from phenotypic import Image
from phenotypic.enhance._anscombe import (
    gat_forward,
    gat_inverse,
    resolve_scale_factor,
)


# -- Forward transform mathematics -----------------------------------------


class TestForwardTransformMathematics:
    """Mathematical correctness of the forward Anscombe transform."""

    def test_sqrt_scaling_for_large_counts(self):
        """For large counts and zero noise the forward transform ~= 2*sqrt(x)."""
        x = np.array([100.0, 400.0, 900.0])
        result = gat_forward(x, mu=0, sigma=0, gain=1.0)
        expected_approx = 2 * np.sqrt(x)
        np.testing.assert_allclose(result, expected_approx, rtol=0.05)

    def test_handles_zero_values(self):
        """Forward transform handles zero counts without NaN."""
        x = np.array([0.0, 0.0, 0.0])
        result = gat_forward(x, mu=0, sigma=0, gain=1.0)
        assert not np.any(np.isnan(result))
        assert np.all(result >= 0)

    def test_with_read_noise(self):
        """Forward transform with non-zero read noise produces valid output."""
        x = np.array([100.0, 200.0, 300.0])
        result = gat_forward(x, mu=5.0, sigma=10.0, gain=2.0)
        assert not np.any(np.isnan(result))
        assert np.all(result > 0)


# -- Inverse transform mathematics -----------------------------------------


class TestInverseTransformMathematics:
    """Mathematical correctness of the inverse Anscombe transform."""

    def test_handles_small_values(self):
        """Inverse transform handles small (clamped) transformed values."""
        x = np.array([0.5, 0.8, 1.0, 1.5])
        result = gat_inverse(x, mu=0, sigma=0, gain=1.0)
        assert not np.any(np.isnan(result))
        assert np.all(result >= 0)

    def test_handles_nan(self):
        """Inverse transform replaces NaN with 0."""
        x = np.array([np.nan, 10.0, 20.0])
        result = gat_inverse(x, mu=0, sigma=0, gain=1.0)
        assert result[0] == 0.0
        assert not np.any(np.isnan(result))


# -- Forward/inverse round-trip --------------------------------------------


class TestForwardInverseRoundtrip:
    """Forward then inverse should approximately recover the original."""

    def test_roundtrip_large_counts(self):
        x = np.array([50.0, 100.0, 200.0, 500.0, 1000.0])
        forward = gat_forward(x, mu=0, sigma=0, gain=1.0)
        inverse = gat_inverse(forward, mu=0, sigma=0, gain=1.0)
        np.testing.assert_allclose(inverse, x, rtol=0.1)

    def test_roundtrip_with_read_noise(self):
        x = np.array([100.0, 200.0, 500.0, 1000.0])
        gain, mu, sigma = 2.0, 1.0, 3.0
        forward = gat_forward(x, mu=mu, sigma=sigma, gain=gain)
        inverse = gat_inverse(forward, mu=mu, sigma=sigma, gain=gain)
        np.testing.assert_allclose(inverse, x, rtol=0.15)


# -- Scale-factor resolution -----------------------------------------------


class TestResolveScaleFactor:
    """``resolve_scale_factor`` honours overrides and falls back to bit depth."""

    def _make_image(self):
        arr = np.random.rand(32, 32).astype(np.float64)
        return Image(arr=arr)

    def test_manual_override(self):
        image = self._make_image()
        assert resolve_scale_factor(image, override=65535.0) == 65535.0

    def test_default_when_no_metadata(self):
        image = self._make_image()
        assert resolve_scale_factor(image, override=None) == 255.0

    def test_explicit_8bit_metadata(self):
        image = self._make_image()
        image.metadata.bit_depth = 8
        assert resolve_scale_factor(image, override=None) == 255.0

    def test_explicit_16bit_metadata(self):
        image = self._make_image()
        image.metadata.bit_depth = 16
        assert resolve_scale_factor(image, override=None) == 65535.0


# -- Old classes removed ---------------------------------------------------


class TestOldClassesRemoved:
    """``AnscombeForward`` and ``AnscombeInverse`` are no longer exported."""

    def test_anscombe_forward_not_in_enhance(self):
        import phenotypic.enhance as enhance_mod
        assert not hasattr(enhance_mod, "AnscombeForward")

    def test_anscombe_inverse_not_in_enhance(self):
        import phenotypic.enhance as enhance_mod
        assert not hasattr(enhance_mod, "AnscombeInverse")
