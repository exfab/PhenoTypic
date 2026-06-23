"""Lightweight EXIF capture-time reader (no full image decode)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage

from phenotypic.gui.browse._capture_time import read_capture_time

# Committed fixture: an 8x8 JPEG whose EXIF DateTimeOriginal is
# "2024:01:02 03:04:05" (authored once; see the authoring note below).
_FIXTURE = Path(__file__).parent / "fixtures" / "with_datetimeoriginal.jpg"


def test_reads_datetimeoriginal() -> None:
    assert read_capture_time(_FIXTURE) == "2024:01:02 03:04:05"


def test_returns_none_without_exif(tmp_path: Path) -> None:
    img = tmp_path / "plain.png"
    PILImage.new("RGB", (8, 8), (0, 0, 0)).save(img, format="PNG")
    assert read_capture_time(img) is None
