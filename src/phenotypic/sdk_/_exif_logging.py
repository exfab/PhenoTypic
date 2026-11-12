"""Scoped quieting of exifread's missing-EXIF warning.

``exifread.process_file`` catches its own ``ExifNotFound`` and logs it at
WARNING on the ``exifread`` logger before returning ``{}``. That happens for
any image without an EXIF block, which is most PNGs. With no logging
configured, :data:`logging.lastResort` prints
``PNG file does not have exif data.`` to stderr on every read. A missing
EXIF block is the ordinary case for a plate image, so every PhenoTypic call
to ``process_file`` runs inside :func:`quiet_missing_exif`.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

#: The single logger every exifread module logs through.
EXIFREAD_LOGGER = "exifread"


class _DropExifNotFound(logging.Filter):
    """Drop records whose message is an ``ExifNotFound`` exception.

    exifread logs the exception instance itself (``logger.warning(err)``), so
    the test is on the type, not the wording. Every other exifread record --
    a corrupted field or an IFD loop in an image that does carry EXIF --
    still passes.
    """

    def __init__(self, exif_not_found: type[BaseException]) -> None:
        super().__init__()
        self._exif_not_found = exif_not_found

    def filter(self, record: logging.LogRecord) -> bool:
        return not isinstance(record.msg, self._exif_not_found)


@contextlib.contextmanager
def quiet_missing_exif() -> Iterator[None]:
    """Drop exifread's missing-EXIF record for the duration of the block.

    Adds a fresh filter to the ``exifread`` logger and removes it on exit,
    including when the block raises. A fresh instance per call means
    concurrent readers each remove only their own. The logger's level is
    never changed. exifread is imported here rather than at module level, so
    importing this module costs nothing.
    """
    from exifread import ExifNotFound

    logger = logging.getLogger(EXIFREAD_LOGGER)
    drop = _DropExifNotFound(ExifNotFound)
    logger.addFilter(drop)
    try:
        yield
    finally:
        logger.removeFilter(drop)
