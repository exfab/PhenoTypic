from phenotypic.gui._config import IMAGE_EXTS, RAW_IMAGE_EXTS
from phenotypic.gui.builder._directory_browser import IMAGE_EXTS as BUILDER_IMAGE_EXTS


def test_image_exts_cover_standard_and_raw():
    assert {".png", ".tif", ".tiff", ".jpg", ".jpeg"} <= IMAGE_EXTS
    assert {".raw", ".nef", ".cr2", ".arw", ".dng"} <= IMAGE_EXTS


def test_raw_subset_of_image_exts():
    assert RAW_IMAGE_EXTS <= IMAGE_EXTS
    assert ".png" not in RAW_IMAGE_EXTS


def test_builder_reexports_the_same_object():
    # Back-compat: builder must keep exporting the identical frozenset.
    assert BUILDER_IMAGE_EXTS is IMAGE_EXTS
