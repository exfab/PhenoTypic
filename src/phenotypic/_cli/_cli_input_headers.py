"""Read input-image headers for the run preflight, never their pixels.

Spec ``docs/superpowers/specs/2026-09-24-cli-preflight/design.md`` §7; the
behavior every rule here mirrors was established by probe and is recorded in
``docs/superpowers/reports/2026-09-24-cli-preflight/header-behavior.md``.

A header says what a file *stores*, which is not always what ``Image.imread``
*returns*: a palette PNG stores one band and decodes to RGB, an RGBA file
decodes to RGB, and a TIFF decodes its first *series* -- which for a Fiji
composite is a stack of single-sample pages that decodes to RGB. The checks
need the decoded answer, so :func:`read_input_header` maps header facts to it.
Pillow parses a header lazily and decodes nothing until ``load()``;
``tifffile`` reads the first series' shape from its tags; a PhenoTypic
OME-Zarr store answers from its root
``zarr.json``. A file whose header parses but whose pixel data is truncated is
therefore NOT caught here -- only a decode finds that, and the CLI already
isolates it as a per-image failure.

Nothing heavy is imported at module level (the lazy-entry guards in
``tests/unit/ci``).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

#: Worker threads for the header pass: opening files on shared storage is
#: latency-bound, and each read is small.
HEADER_SCAN_WORKERS = 16

#: Pillow modes by the channel count ``imread`` produces from them.
_PIL_SINGLE_CHANNEL_MODES = frozenset({"1", "L", "I", "I;16", "I;16B", "I;16L", "I;16N", "F"})
_PIL_RGB_MODES = frozenset({"P", "RGB", "YCbCr", "LAB", "HSV", "RGBA", "RGBX", "RGBa", "CMYK"})
_PIL_TWO_CHANNEL_MODES = frozenset({"LA", "La"})

_TIFF_SUFFIXES = frozenset({".tif", ".tiff"})


@dataclass(frozen=True)
class InputHeader:
    """What the run will get from one input, learned from its header alone.

    Attributes:
        path: The input, as scanned.
        channels: Channels after decoding: ``1`` (grayscale) or ``3`` (RGB);
            ``None`` when the header does not say (e.g. a third-party store).
        raw_channels: The stored channel count when ``imread`` refuses it
            (2, or 5 and more), else ``None``.
        bits: 8 or 16 from the stored dtype, or ``None`` when unknown.
        carries_phenotypic_metadata: The input restores PhenoTypic metadata
            into the image when read (a PhenoTypic store, or a file carrying
            the ``phenotypic`` metadata key).
        error: Why the header could not be read, or ``None``.
    """

    path: str
    channels: Optional[int] = None
    raw_channels: Optional[int] = None
    bits: Optional[int] = None
    carries_phenotypic_metadata: bool = False
    error: Optional[str] = None


def read_input_headers(paths: Iterable[Path | str]) -> list[InputHeader]:
    """Read every input's header, in parallel, preserving input order."""
    items = [Path(path) for path in paths]
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=HEADER_SCAN_WORKERS) as pool:
        return list(pool.map(read_input_header, items))


def read_input_header(path: Path | str) -> InputHeader:
    """Read one input's header without decoding its pixels.

    Never raises: an unreadable header becomes ``InputHeader.error``.
    """
    path = Path(path)
    try:
        from phenotypic.sdk_ import is_zarr_store_name
        from phenotypic.sdk_.constants_ import IO

        if path.is_dir() and is_zarr_store_name(path):
            return _store_header(path)
        if path.stat().st_size == 0:
            return InputHeader(str(path), error="the file is empty (zero bytes)")
        suffix = path.suffix.lower()
        if suffix in {s.lower() for s in IO.RAW_FILE_EXTENSIONS}:
            # rawpy always demosaics to 16-bit RGB (Image.imread's RAW branch).
            return InputHeader(str(path), channels=3, bits=16)
        if suffix in _TIFF_SUFFIXES:
            return _tiff_header(path)
        return _pillow_header(path)
    except Exception as exc:  # noqa: BLE001 -- a header we cannot read is a finding
        return InputHeader(str(path), error=f"{type(exc).__name__}: {exc}")


def _pillow_header(path: Path) -> InputHeader:
    from PIL import Image as PILImage

    from phenotypic.sdk_.constants_ import IO

    with PILImage.open(path) as image:
        mode = image.mode
        image_format = image.format
        carries = IO.PHENOTYPIC_METADATA_KEY in (image.info or {})
    bits = _png_bit_depth(path) if image_format == "PNG" else 8 if image_format == "JPEG" else None
    if mode in _PIL_SINGLE_CHANNEL_MODES:
        return InputHeader(str(path), channels=1, bits=bits, carries_phenotypic_metadata=carries)
    if mode in _PIL_RGB_MODES:
        return InputHeader(str(path), channels=3, bits=bits, carries_phenotypic_metadata=carries)
    if mode in _PIL_TWO_CHANNEL_MODES:
        return InputHeader(str(path), raw_channels=2, carries_phenotypic_metadata=carries)
    return InputHeader(str(path), carries_phenotypic_metadata=carries)


def _png_bit_depth(path: Path) -> Optional[int]:
    """Bits per sample from the PNG IHDR chunk (byte 24).

    Pillow reports a 16-bit RGB PNG as mode ``RGB``, so the mode cannot answer
    this; the IHDR byte can. Palette and sub-byte depths decode to 8 bits.
    """
    with open(path, "rb") as handle:
        header = handle.read(25)
    if len(header) < 25 or header[12:16] != b"IHDR":
        return None
    return 16 if header[24] == 16 else 8


def _tiff_header(path: Path) -> InputHeader:
    """Predict ``Image.imread``'s result from the first series' metadata.

    ``skimage.io.imread`` returns ``series[0]`` of a TIFF, so the prediction
    starts from that series' shape, not the first page's samples: a Fiji
    composite or an OME-TIFF stores a ``(3, H, W)`` channel stack as three
    single-sample pages of ONE series and decodes to RGB (review D1). The
    shape is then passed through :func:`_skimage_axis_order` and
    :func:`_channels_for_shape`, which transcribe the two rules ``imread``
    applies. Reading ``series[0].shape`` parses tags only.
    """
    import numpy as np
    import tifffile

    from phenotypic.sdk_.constants_ import IO

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(n) for n in series.shape)
        itemsize = np.dtype(series.dtype).itemsize if series.dtype is not None else None
        description = tif.pages[0].tags.get("ImageDescription")
        carries = bool(description) and IO.PHENOTYPIC_METADATA_KEY in str(description.value)
    bits = {1: 8, 2: 16}.get(itemsize or 0)
    channels, raw_channels = _channels_for_shape(_skimage_axis_order(shape))
    return InputHeader(
        str(path),
        channels=channels,
        raw_channels=raw_channels,
        bits=bits,
        carries_phenotypic_metadata=carries,
    )


def _skimage_axis_order(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The shape ``skimage.io.imread`` returns for an array of *shape*.

    Transcribes ``skimage/io/_io.py`` (0.25): when ``ndim > 2``, the last axis
    is not 3 or 4 long and the third-from-last is, it swaps axes ``-1, -3``
    then ``-2, -3``, moving a leading channel axis to the end.
    """
    if len(shape) > 2 and shape[-1] not in (3, 4) and shape[-3] in (3, 4):
        return (*shape[:-3], shape[-2], shape[-1], shape[-3])
    return shape


def _channels_for_shape(shape: tuple[int, ...]) -> tuple[Optional[int], Optional[int]]:
    """``(channels, raw_channels)`` the ``Image`` constructor makes of *shape*.

    Transcribes ``ImageDataManager._guess_image_format``: 2-D is grayscale; a
    3-D last axis of 1 is grayscale, of 3 or 4 is RGB, and of any other length
    raises ``"Image with {c} channels"``, which ``raw_channels`` reports. More
    dimensions (a Z or time stack) also raise, with no channel count to name,
    so the prediction is left unknown rather than guessed.
    """
    if len(shape) == 2:
        return 1, None
    if len(shape) == 3:
        last = shape[-1]
        if last == 1:
            return 1, None
        if last in (3, 4):
            return 3, None
        return None, last
    return None, None


def _store_header(path: Path) -> InputHeader:
    """A PhenoTypic store records its series in the root ``zarr.json``.

    A third-party store has no ``phenotypic`` block; its layout is decided by
    ``imread``'s own NGFF projection at run time, so this reports unknown
    channels rather than re-deriving that projection.
    """
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, _zarr_v2_marker

    root_path = path / "zarr.json"
    if not root_path.is_file() and (marker := _zarr_v2_marker(path)) is not None:
        # Image.imread refuses a v2 store too; say so rather than "not found"
        # (review D9).
        return InputHeader(
            str(path),
            error=(
                f"a Zarr v2 (NGFF 0.4) store ({marker}, no zarr.json), which "
                "Image.imread does not read; convert it to Zarr v3 / NGFF 0.5"
            ),
        )
    root = json.loads(root_path.read_text(encoding="utf-8"))
    block = root.get("attributes", {}).get(PhenotypicAttr.ROOT)
    if not isinstance(block, dict):
        return InputHeader(str(path))
    series = block.get(PhenotypicAttr.SERIES) or {}
    channels = 3 if "rgb" in series else 1 if "gray" in series else None
    return InputHeader(str(path), channels=channels, carries_phenotypic_metadata=True)
