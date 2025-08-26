from __future__ import annotations

import colour
import numpy as np

# Define an sRGB-like space but with D50 white and the D50-adapted matrix
# Flatbed scanners assume D50 illumination as a reference point
sRGB_D50 = colour.RGB_Colourspace(
    name="sRGB_D50",
    primaries=colour.RGB_COLOURSPACES["sRGB"].primaries,
    whitepoint=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
    matrix_RGB_to_XYZ=np.array([
        [0.4360747, 0.3850649, 0.1430804],
        [0.2225045, 0.7168786, 0.0606169],
        [0.0139322, 0.0971045, 0.7141733],
    ]),
    matrix_XYZ_to_RGB=None,
    cctf_decoding=colour.CCTF_DECODINGS["sRGB"],
    cctf_encoding=colour.CCTF_ENCODINGS["sRGB"],
)
