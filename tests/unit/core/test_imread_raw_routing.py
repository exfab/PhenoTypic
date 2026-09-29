"""Camera RAW reaches rawpy (spec 2026-09-24-cli-preflight F24, §10.1).

``IO.ACCEPTED_FILE_EXTENSIONS`` includes the RAW suffixes, and the general
branch of ``Image.imread`` used to be tested first, so the rawpy branch was
unreachable and RAW went to skimage/Pillow (claim-verification report §5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic._core._image_parts import _image_io_handler as io_handler
from phenotypic.sdk_.exceptions_ import UnsupportedFileTypeError

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "raw" / "synthetic_plate.dng"
RAW_SUFFIXES = [".dng", ".nef", ".NEF", ".cr2", ".cr3", ".arw"]


class _Decoded:
    """Stand-in for a rawpy RawPy handle, returning a fixed 16-bit RGB array."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def postprocess(self, **kwargs):
        return np.full((8, 8, 3), 1000, dtype=np.uint16)


@pytest.fixture
def recorded(monkeypatch):
    calls: dict[str, list] = {"rawpy": [], "skimage": []}
    fake_rawpy = type("FakeRawpy", (), {})()
    fake_rawpy.imread = lambda path: calls["rawpy"].append(path) or _Decoded()
    fake_rawpy.DemosaicAlgorithm = type(
        "D", (), {"AMAZE": type("A", (), {"isSupported": False})(), "AHD": "AHD"}
    )
    fake_rawpy.ColorSpace = type("C", (), {"sRGB": "sRGB"})
    monkeypatch.setattr(io_handler, "rawpy", fake_rawpy)
    real_skimage = io_handler.ski.io.imread
    monkeypatch.setattr(
        io_handler.ski.io,
        "imread",
        lambda fname, **k: calls["skimage"].append(fname) or real_skimage(fname, **k),
    )
    # Metadata extraction would hand the placeholder bytes to exiftool/rawpy;
    # it is out of scope for routing.
    monkeypatch.setattr(Image, "_extract_raw_metadata", classmethod(lambda cls, p: {}))
    return calls


@pytest.mark.parametrize("suffix", RAW_SUFFIXES)
def test_raw_suffixes_are_decoded_by_rawpy(recorded, tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"plate{suffix}"
    path.write_bytes(b"not decoded: rawpy is faked")

    image = Image.imread(path)

    assert recorded["rawpy"] == [str(path)]
    assert recorded["skimage"] == []
    assert image.rgb[:].shape == (8, 8, 3)


def test_other_suffixes_still_use_skimage(recorded, tmp_path: Path) -> None:
    import tifffile

    path = tmp_path / "plate.tiff"
    tifffile.imwrite(path, np.zeros((8, 8, 3), dtype=np.uint8))

    Image.imread(path)

    assert recorded["skimage"] and recorded["rawpy"] == []


def test_raw_without_rawpy_names_the_missing_package(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(io_handler, "rawpy", None)
    path = tmp_path / "plate.nef"
    path.write_bytes(b"x")

    with pytest.raises(UnsupportedFileTypeError, match="rawpy"):
        Image.imread(path)


def test_a_callers_rawpy_params_are_not_mutated(recorded, tmp_path: Path) -> None:
    path = tmp_path / "plate.dng"
    path.write_bytes(b"x")
    params = {"use_auto_wb": True, "gamma": (2.2, 4.5)}

    Image.imread(path, rawpy_params=params)

    assert params == {"use_auto_wb": True, "gamma": (2.2, 4.5)}


def test_a_real_raw_file_decodes_to_a_plausible_16_bit_plate() -> None:
    """Review R9: the rawpy branch had never run on a real file."""
    pytest.importorskip("rawpy")
    image = Image.imread(FIXTURE)

    rgb = image.rgb[:]
    assert rgb.shape == (64, 64, 3)
    gray = image.gray[:]
    colony = float(gray[18:22, 18:22].mean())
    agar = float(gray[2:6, 50:60].mean())
    assert colony > 3 * agar > 0, (colony, agar)
    raw_rgb = np.asarray(rgb)
    if raw_rgb.dtype == np.uint16:
        median = float(np.median(raw_rgb)) / 65535
        assert 0.01 < median < 0.99, median
