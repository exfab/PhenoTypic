"""Small cross-platform interprocess lock for artifact publication."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

if sys.platform == "win32":  # pragma: win32 cover
    import msvcrt
else:  # pragma: win32 no cover
    import fcntl


class ArtifactLockTimeout(TimeoutError):
    """Raised when an artifact publication lock cannot be acquired."""


@contextmanager
def exclusive_path_lock(
    lock_path: Path,
    *,
    timeout: float = 30.0,
) -> Iterator[None]:
    """Acquire an exclusive interprocess lock anchored at ``lock_path``.

    Args:
        lock_path: Stable lock-file path shared by competing publishers.
        timeout: Maximum number of seconds to wait.

    Yields:
        ``None`` while the exclusive lock is held.

    Raises:
        ArtifactLockTimeout: If acquisition exceeds ``timeout``.
    """
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        _acquire(handle, timeout=timeout)
        try:
            yield
        finally:
            _release(handle)


@contextmanager
def exclusive_file_lock(
    handle: BinaryIO,
    *,
    timeout: float = 30.0,
) -> Iterator[None]:
    """Lock an already-open file without reopening its pathname.

    Callers that establish stronger pathname guarantees (for example with an
    anchored ``openat`` and ``O_NOFOLLOW``) can retain those guarantees while
    using the same cross-platform locking implementation as
    :func:`exclusive_path_lock`.

    Args:
        handle: Open binary handle whose descriptor is the lock authority.
        timeout: Maximum number of seconds to wait.

    Yields:
        ``None`` while the exclusive lock is held.

    Raises:
        ArtifactLockTimeout: If acquisition exceeds ``timeout``.
    """
    _acquire(handle, timeout=timeout)
    try:
        yield
    finally:
        _release(handle)


def _acquire(handle: BinaryIO, *, timeout: float) -> None:
    started = time.monotonic()
    if sys.platform == "win32":  # pragma: win32 cover
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        mode = msvcrt.LK_NBLCK
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), mode, 1)
                return
            except OSError as exc:
                if time.monotonic() - started >= timeout:
                    raise ArtifactLockTimeout(
                        f"Could not acquire artifact lock {handle.name!r}"
                    ) from exc
                time.sleep(0.01)

    while True:  # pragma: win32 no cover
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if time.monotonic() - started >= timeout:
                raise ArtifactLockTimeout(
                    f"Could not acquire artifact lock {handle.name!r}"
                ) from exc
            time.sleep(0.01)


def _release(handle: BinaryIO) -> None:
    if sys.platform == "win32":  # pragma: win32 cover
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: win32 no cover
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = [
    "ArtifactLockTimeout",
    "exclusive_file_lock",
    "exclusive_path_lock",
]
