"""Pure RGB -> CIE XYZ conversion, independent of any Image instance.

Lifted out of :class:`XyzAccessor` so a *substitute* RGB array (e.g. one an
enhancer has just gamma-corrected) can be projected through the same colour
pipeline the accessor uses, carrying the source image's colour configuration.
"""

from __future__ import annotations

import colour
import numpy as np

from phenotypic.sdk_.colourspace import sRGB_D50
from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS


def rgb_to_xyz(
    rgb_normed: np.ndarray,
    *,
    gamma: GAMMA_ENCODINGS,
    illuminant: str,
    observer: str,
) -> np.ndarray:
    """Convert a normalized RGB array to CIE XYZ tristimulus values.

    Args:
        rgb_normed: RGB array normalized to [0, 1], shape ``(rows, cols, 3)``.
        gamma: The image's gamma encoding (``GAMMA_ENCODINGS.SRGB`` or ``LINEAR``).
        illuminant: ``"D50"`` or ``"D65"``.
        observer: CIE standard observer name, e.g.
            ``"CIE 1931 2 Degree Standard Observer"``.

    Returns:
        np.ndarray: XYZ array, shape ``(rows, cols, 3)``, dtype float64.

    Raises:
        ValueError: If the gamma/illuminant combination is unrecognized.
    """
    match (gamma, illuminant):
        case (GAMMA_ENCODINGS.SRGB, "D50"):
            # This assignment is load-bearing, not a scratch variable: it retargets
            # the shared ``sRGB_D50`` colourspace's whitepoint so that it equals the
            # ``illuminant`` passed on the next line. ``colour`` then runs its
            # chromatic adaptation from ``colourspace.whitepoint`` to ``illuminant``,
            # i.e. as the identity. Because ``sRGB_D50`` also carries an explicit
            # ``matrix_RGB_to_XYZ`` constant, the whitepoint *value* never reaches the
            # arithmetic — swapping D50 for D65 here is bit-identical (measured 0.0).
            #
            # It remains a mutation of module-level shared state. Left in place because
            # removing it safely requires copying the colourspace per call.
            sRGB_D50.whitepoint = colour.CCS_ILLUMINANTS[observer]["D50"]
            return colour.RGB_to_XYZ(
                    RGB=rgb_normed,
                    colourspace=sRGB_D50,
                    illuminant=sRGB_D50.whitepoint,
                    apply_cctf_decoding=True,
            )
        case (GAMMA_ENCODINGS.SRGB, "D65"):
            return colour.RGB_to_XYZ(
                    RGB=rgb_normed,
                    colourspace=colour.RGB_COLOURSPACES["sRGB"],
                    illuminant=colour.CCS_ILLUMINANTS[observer]["D65"],
                    apply_cctf_decoding=True,
            )
        case (GAMMA_ENCODINGS.LINEAR, "D50"):
            # Unlike the sRGB/D50 arm above, ``sRGB_D50`` is not the colourspace here
            # (the standard D65 sRGB one is), so mutating its whitepoint served only to
            # carry a value to the next line. Use a local instead. The whitepoint value
            # IS load-bearing in this arm -- it is the chromatic-adaptation target, and
            # the CAT is not the identity (measured max|Δ| = 1.5e-01 for D50 vs D65).
            whitepoint = colour.CCS_ILLUMINANTS[observer]["D50"]
            return colour.RGB_to_XYZ(
                    RGB=rgb_normed,
                    colourspace=colour.RGB_COLOURSPACES["sRGB"],
                    illuminant=whitepoint,
                    apply_cctf_decoding=False,
            )
        case (GAMMA_ENCODINGS.LINEAR, "D65"):
            return colour.RGB_to_XYZ(
                    RGB=rgb_normed,
                    colourspace=colour.RGB_COLOURSPACES["sRGB"],
                    illuminant=colour.CCS_ILLUMINANTS[observer]["D65"],
                    apply_cctf_decoding=False,
            )
        case _:
            raise ValueError(
                    f"Unknown color_profile: {gamma} or illuminant: {illuminant}"
            )
