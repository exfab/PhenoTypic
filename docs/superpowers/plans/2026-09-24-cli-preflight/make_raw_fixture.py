"""Generate ``tests/fixtures/raw/synthetic_plate.dng``: a real, tiny camera-RAW file.

Plan Task 9 Step 3 (review R9) requires the rawpy decode path, never exercised
before this change, to be checked against a real RAW file. No redistributable
camera sample was at hand, so this writes one: a 12-bit RGGB Bayer mosaic of a
dark agar plate with two bright colonies, wrapped as a DNG 1.4 by ``pidng``.
LibRaw (rawpy) decodes it exactly as it decodes a camera's DNG.

Run with ``uvx --with numpy --from pidng==4.0.9 python make_raw_fixture.py``;
it depends on numpy + pidng only and never imports ``phenotypic``. The output
is deterministic (no randomness), so the committed bytes can be regenerated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import (
    CalibrationIlluminant,
    CFAPattern,
    DNGVersion,
    Orientation,
    PhotometricInterpretation,
    PreviewColorSpace,
)

WIDTH = HEIGHT = 64
BITS = 12
BLACK = 256
WHITE = (1 << BITS) - 1


def synthetic_plate() -> np.ndarray:
    """Dark agar (just above black) with two bright colonies, as a Bayer mosaic."""
    plate = np.full((HEIGHT, WIDTH), BLACK + 200, dtype=np.uint16)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    for cy, cx, r in ((20, 20, 8), (44, 40, 10)):
        plate[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = WHITE - 400
    return plate


def write_dng(path: Path) -> None:
    tags = DNGTags()
    tags.set(Tag.ImageWidth, WIDTH)
    tags.set(Tag.ImageLength, HEIGHT)
    tags.set(Tag.TileWidth, WIDTH)
    tags.set(Tag.TileLength, HEIGHT)
    tags.set(Tag.Orientation, Orientation.Horizontal)
    tags.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    tags.set(Tag.SamplesPerPixel, 1)
    tags.set(Tag.BitsPerSample, BITS)
    tags.set(Tag.CFARepeatPatternDim, [2, 2])
    tags.set(Tag.CFAPattern, CFAPattern.RGGB)
    tags.set(Tag.BlackLevel, BLACK)
    tags.set(Tag.WhiteLevel, WHITE)
    tags.set(Tag.ColorMatrix1, [[1, 1], [0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [1, 1]])
    tags.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
    tags.set(Tag.AsShotNeutral, [[1, 1], [1, 1], [1, 1]])
    tags.set(Tag.Make, "PhenoTypic")
    tags.set(Tag.Model, "Synthetic plate fixture")
    tags.set(Tag.DNGVersion, DNGVersion.V1_4)
    tags.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
    tags.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)
    converter = RAW2DNG()
    converter.options(tags, path=str(path.parent), compress=False)
    converter.convert(synthetic_plate(), filename=path.stem)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("synthetic_plate.dng")
    write_dng(target)
    print(target, target.stat().st_size, "bytes")
