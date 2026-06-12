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


def test_extract_exif_prefers_capture_time_and_body_over_decoys():
    # Decoy keys are ordered BEFORE the real ones to catch naive first-match:
    #   - "Image DateTime" (write time) must lose to "EXIF DateTimeOriginal".
    #   - "EXIF LensModel" must not be picked as the camera body "model".
    imported = {
        "Image DateTime": "2024:03:02 09:00:00",  # file/scan write time (decoy)
        "EXIF DateTimeOriginal": "2024:03:01 14:22:05",  # true capture time
        "EXIF LensModel": "50mm f/1.8",  # lens, not the body (decoy)
        "Image Model": "NIKON D850",  # camera body
        "Image Make": "NIKON CORPORATION",
    }
    assert _extract_exif(imported) == {
        "captured": "2024:03:01 14:22:05",
        "make": "NIKON CORPORATION",
        "model": "NIKON D850",
    }


def test_read_dims_and_size_no_exif(tmp_path):
    src = tmp_path / "plate.png"
    PILImage.fromarray(np.zeros((12, 20, 3), dtype=np.uint8)).save(src)
    info = read(src)
    assert info["width"] == 20 and info["height"] == 12
    assert info["bytes"] == src.stat().st_size
    assert info["exif"] == {}
