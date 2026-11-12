"""``Image.imread`` reads EXIF without logging exifread's no-EXIF warning.

``exifread.process_file`` logs ``PNG file does not have exif data.`` at
WARNING for every image without an EXIF block. With no logging configured,
that line reached stderr on every plate load, including every docs notebook
cell that loads a sample plate. Every ``process_file`` call now runs inside
``phenotypic.sdk_._exif_logging.quiet_missing_exif``, which drops that one
record for the duration of the call; EXIF is still read where present, and
exifread's other warnings still pass. The Browse probe's use of the same
scope is pinned in ``tests/gui/browse/test_source_probe.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import exifread
import pytest

import phenotypic
import phenotypic.data
from phenotypic.sdk_._exif_logging import quiet_missing_exif

DATA_DIR = Path(phenotypic.data.__file__).parent
PNG_WITHOUT_EXIF = DATA_DIR / "early_colony.png"


def _exifread_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "exifread"]


def test_exifread_itself_logs_the_missing_exif_warning(caplog) -> None:
    """Control: the fixture has no EXIF and exifread does log for it.

    Without this, the test below would pass vacuously if exifread stopped
    logging, or if the fixture gained an EXIF block.
    """
    caplog.set_level(logging.WARNING)
    with PNG_WITHOUT_EXIF.open("rb") as handle:
        assert exifread.process_file(handle, details=True) == {}
    assert _exifread_records(caplog) == ["PNG file does not have exif data."]


def test_imread_of_a_png_without_exif_logs_no_exifread_record(caplog) -> None:
    caplog.set_level(logging.WARNING)
    phenotypic.Image.imread(PNG_WITHOUT_EXIF)
    assert _exifread_records(caplog) == []


@pytest.mark.parametrize(
    ("filename", "tag"),
    [
        ("StandardDay1.jpg", "Image Orientation"),
        ("snp-imager-samples/RhodotorulaYeastCenterCrop.png", "Image Make"),
    ],
)
def test_imread_still_reads_exif_where_present(filename, tag) -> None:
    image = phenotypic.Image.imread(DATA_DIR / filename)
    assert tag in image._metadata.imported


def test_quiet_scope_drops_only_exif_not_found_and_is_removed(caplog) -> None:
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("exifread")
    filters_before = list(logger.filters)

    with quiet_missing_exif():
        logger.warning(exifread.ExifNotFound("inside the scope"))
        logger.warning("Possibly corrupted field %s", "Make")
    logger.warning(exifread.ExifNotFound("after the scope"))

    assert logger.filters == filters_before
    assert _exifread_records(caplog) == [
        "Possibly corrupted field Make",
        "after the scope",
    ]


def test_quiet_scope_is_removed_when_the_block_raises(caplog) -> None:
    """A ``process_file`` that raises must not leave the filter installed."""
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("exifread")
    filters_before = list(logger.filters)

    with pytest.raises(RuntimeError, match="corrupt header"):
        with quiet_missing_exif():
            raise RuntimeError("corrupt header")
    logger.warning(exifread.ExifNotFound("after the failed read"))

    assert logger.filters == filters_before
    assert _exifread_records(caplog) == ["after the failed read"]
