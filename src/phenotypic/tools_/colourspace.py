from __future__ import annotations

import colour
import numpy as np

# Define an sRGB-like space but with D50 white and the D50-adapted gray
# Flatbed scanners assume D50 illumination as a reference point
sRGB_D50 = colour.RGB_Colourspace(
    name="sRGB_D50",
    primaries=colour.RGB_COLOURSPACES["sRGB"].primaries,
    whitepoint=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
    matrix_RGB_to_XYZ=np.array(
        [
            [0.4360747, 0.3850649, 0.1430804],
            [0.2225045, 0.7168786, 0.0606169],
            [0.0139322, 0.0971045, 0.7141733],
        ]
    ),
    matrix_XYZ_to_RGB=None,
    cctf_decoding=colour.CCTF_DECODINGS["sRGB"],
    cctf_encoding=colour.CCTF_ENCODINGS["sRGB"],
)


def decode_srgb(arr: np.ndarray) -> np.ndarray:
    """Convert sRGB gamma-encoded values to linear light.

    Applies the inverse sRGB colour component transfer function (the
    standard piecewise gamma curve) elementwise. Sensor and photon noise
    is generated in linear light, so denoisers and variance-stabilizing
    transforms (e.g. the Generalized Anscombe Transform) operate most
    correctly on linearized data rather than the display-encoded values.

    Args:
        arr: Array of sRGB gamma-encoded values normalized to ``[0, 1]``.
            Any shape; the transfer function is applied elementwise, so a
            2D channel or an ``(H, W, 3)`` image are both valid.

    Returns:
        np.ndarray: Linear-light values in ``[0, 1]`` as ``float64``,
        same shape as ``arr``.
    """
    return np.asarray(colour.CCTF_DECODINGS["sRGB"](arr), dtype=np.float64)


def encode_srgb(arr: np.ndarray) -> np.ndarray:
    """Convert linear-light values to sRGB gamma-encoded values.

    Applies the forward sRGB colour component transfer function
    elementwise. This is the inverse of :func:`decode_srgb` and restores
    the display-encoded representation after processing in linear light.

    Args:
        arr: Array of linear-light values normalized to ``[0, 1]``. Any
            shape; the transfer function is applied elementwise.

    Returns:
        np.ndarray: sRGB gamma-encoded values in ``[0, 1]`` as
        ``float64``, same shape as ``arr``.
    """
    return np.asarray(colour.CCTF_ENCODINGS["sRGB"](arr), dtype=np.float64)
