from phenotypic.gui._config import IMAGE_EXTS, RAW_IMAGE_EXTS
from phenotypic.gui.builder._directory_browser import IMAGE_EXTS as BUILDER_IMAGE_EXTS


def test_image_exts_cover_standard_and_raw():
    assert {".png", ".tif", ".tiff", ".jpg", ".jpeg"} <= IMAGE_EXTS
    assert {".cr2", ".cr3", ".nef", ".arw", ".dng"} <= IMAGE_EXTS
    # Bare ``.raw`` is no longer decodable by core and so is not listed.
    assert ".raw" not in IMAGE_EXTS


def test_raw_subset_of_image_exts():
    assert RAW_IMAGE_EXTS <= IMAGE_EXTS
    assert ".png" not in RAW_IMAGE_EXTS
    # CR3 joined the decode set; bare ``.raw`` is excluded from both.
    assert ".cr3" in RAW_IMAGE_EXTS and ".cr3" in IMAGE_EXTS
    assert ".raw" not in RAW_IMAGE_EXTS


def test_builder_reexports_the_same_object():
    # Back-compat: builder must keep exporting the identical frozenset.
    assert BUILDER_IMAGE_EXTS is IMAGE_EXTS
