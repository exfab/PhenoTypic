"""Shared serialization boundary for canonical pipeline publication."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ._file_locking import exclusive_path_lock
from ._io_constants import pipeline_publication_lock_path


@contextmanager
def pipeline_publication_lock(
    config_path: Path,
    *,
    timeout: float = 30.0,
) -> Iterator[None]:
    """Serialize a complete read/check/write pipeline transaction.

    Callers must hold this lock across both their generation or staleness
    check and the final atomic replacement. Atomic rename prevents partial
    files; this lock prevents a valid concurrent generation from being
    silently overwritten after a check-then-replace race.

    Args:
        config_path: Canonical pipeline configuration being published.
        timeout: Maximum seconds to wait for a competing publisher.

    Yields:
        ``None`` while the shared publication lock is held.
    """
    with exclusive_path_lock(
        pipeline_publication_lock_path(Path(config_path)),
        timeout=timeout,
    ):
        yield


__all__ = ["pipeline_publication_lock"]
