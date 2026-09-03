from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import colour


@lru_cache(maxsize=1)
def _build_srgb_d50() -> "colour.RGB_Colourspace":
    """Build the D50-white sRGB colourspace, importing colour on first use.

    Flatbed scanners assume D50 illumination as a reference point, so plate
    images are converted through an sRGB-like space with a D50 white and the
    D50-adapted gray rather than the standard D65 sRGB.

    ``lru_cache`` is load-bearing, not an optimization: callers mutate the
    returned object's ``whitepoint`` in place (see
    ``_core/_image_parts/color_space_accessors/_xyz_conversion.py``), so every
    access must yield the *same* object, exactly as the former module-level
    constant did.

    Returns:
        colour.RGB_Colourspace: The shared ``sRGB_D50`` colourspace.
    """
    import colour

    return colour.RGB_Colourspace(
        name="sRGB_D50",
        primaries=colour.RGB_COLOURSPACES["sRGB"].primaries,
        whitepoint=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            "D50"
        ],
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


def __getattr__(name: str) -> Any:
    """Build ``sRGB_D50`` on first access instead of at import.

    Constructing it eagerly was the single reason ``import phenotypic`` had to
    import colour-science — the largest cost on the startup path, and one paid
    by every CLI, tune and GUI invocation regardless of whether any colour
    conversion happens. Nothing else in this module touches colour at module
    scope, so deferring this one constant takes the whole library off the
    critical path.

    Args:
        name: Attribute being looked up.

    Returns:
        The requested member.

    Raises:
        AttributeError: For any other name.
    """
    if name == "sRGB_D50":
        return _build_srgb_d50()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    import colour

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
    import colour

    return np.asarray(colour.CCTF_ENCODINGS["sRGB"](arr), dtype=np.float64)
