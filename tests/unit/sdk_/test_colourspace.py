"""Tests for the sRGB transfer-function helpers in ``tools_.colourspace``."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.sdk_.colourspace import decode_srgb, encode_srgb


class TestSrgbTransferFunctions:
    """Round-trip and known-value checks for decode_srgb / encode_srgb."""

    def test_roundtrip_recovers_input(self):
        """encode_srgb(decode_srgb(x)) is the identity to float precision."""
        x = np.linspace(0.0, 1.0, 257)
        np.testing.assert_allclose(encode_srgb(decode_srgb(x)), x, atol=1e-12)

    def test_roundtrip_inverse_order(self):
        """decode_srgb(encode_srgb(x)) is also the identity."""
        x = np.linspace(0.0, 1.0, 257)
        np.testing.assert_allclose(decode_srgb(encode_srgb(x)), x, atol=1e-12)

    def test_endpoints_are_fixed(self):
        """0 and 1 map to themselves under both transfer functions."""
        endpoints = np.array([0.0, 1.0])
        np.testing.assert_allclose(decode_srgb(endpoints), endpoints, atol=1e-12)
        np.testing.assert_allclose(encode_srgb(endpoints), endpoints, atol=1e-12)

    def test_decode_darkens_midtones(self):
        """sRGB decoding maps mid-gray below itself (gamma > 1 expansion)."""
        # 0.5 encoded sRGB corresponds to ~0.214 linear light.
        assert decode_srgb(np.array([0.5]))[0] < 0.5

    def test_encode_brightens_midtones(self):
        """sRGB encoding maps linear mid-gray above itself."""
        assert encode_srgb(np.array([0.5]))[0] > 0.5

    def test_preserves_shape_2d(self):
        """The transfer functions apply elementwise to a 2D channel."""
        arr = np.full((8, 8), 0.3)
        assert decode_srgb(arr).shape == arr.shape
        assert encode_srgb(arr).shape == arr.shape

    def test_preserves_shape_rgb(self):
        """The transfer functions apply elementwise to an (H, W, 3) image."""
        arr = np.full((8, 8, 3), 0.3)
        assert decode_srgb(arr).shape == arr.shape
        assert encode_srgb(arr).shape == arr.shape

    @pytest.mark.parametrize("fn", [decode_srgb, encode_srgb])
    def test_output_is_float64(self, fn):
        """Both helpers return float64 regardless of input dtype."""
        assert fn(np.full((4, 4), 0.5, dtype=np.float32)).dtype == np.float64
