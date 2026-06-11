import numpy as np
from PIL import Image as PILImage

from phenotypic.gui.browse._metadata import _extract_exif, read


def test_extract_exif_from_exifread_keys():
    imported = {
        "EXIF DateTimeOriginal": "2024:03:01 14:22:05",
        "Image Make": "NIKON CORPORATION",
        "Image Model": "NIKON D850",
        "EXIF ExposureTime": "1/200",
    }
    assert _extract_exif(imported) == {
        "captured": "2024:03:01 14:22:05",
        "make": "NIKON CORPORATION",
        "model": "NIKON D850",
    }


def test_extract_exif_empty_when_absent():
    assert _extract_exif({}) == {}


def test_read_dims_and_size_no_exif(tmp_path):
    src = tmp_path / "plate.png"
    PILImage.fromarray(np.zeros((12, 20, 3), dtype=np.uint8)).save(src)
    info = read(src)
    assert info["width"] == 20 and info["height"] == 12
    assert info["bytes"] == src.stat().st_size
    assert info["exif"] == {}
