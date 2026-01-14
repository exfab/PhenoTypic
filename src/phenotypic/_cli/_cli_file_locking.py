"""
Cross-platform file locking utilities for the PhenoTypic CLI.

Provides atomic file read/write operations with proper locking to prevent
race conditions in parallel processing environments (local joblib workers and
distributed SLURM jobs).

This module implements file locking using platform-specific mechanisms:
- Unix/Linux/macOS: fcntl.flock() (BSD-style file locking)
- Windows: msvcrt.locking() (Windows file locking)

The locking is designed to work reliably on HPC filesystems (NFS, Lustre)
where multiple processes may be reading/writing the same files concurrently.

Examples:
    >>> from pathlib import Path
    >>> from phenotypic._cli._cli_file_locking import atomic_append
    >>>
    >>> # Safely append to event log from parallel workers
    >>> atomic_append(Path("events.log"), "timestamp|dataset|image|completed|\\n")
    >>>
    >>> # Safely read event log while workers are writing
    >>> def parse_log(path):
    ...     return path.read_text().splitlines()
    >>> events = atomic_read(event_log_path, parse_events)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, TypeVar, Optional
from contextlib import contextmanager

# Platform-specific imports
if sys.platform == "win32":
    import msvcrt

    WINDOWS = True
else:
    import fcntl

    WINDOWS = False

T = TypeVar('T')


class FileLockTimeout(Exception):
    """Raised when file lock acquisition times out."""
    pass


@contextmanager
def file_lock(file_handle, timeout: float = 30.0, shared: bool = False):
    """
    Cross-platform file locking context manager.

    Acquires a file lock before yielding the file handle, ensuring exclusive
    or shared access depending on the operation. Works on Unix/Linux/macOS
    (fcntl) and Windows (msvcrt).

    Args:
        file_handle: Open file object
        timeout: Maximum seconds to wait for lock (default: 30.0)
        shared: If True, acquire shared lock (read-only); if False, exclusive lock

    Yields:
        File handle with acquired lock

    Raises:
        FileLockTimeout: If lock cannot be acquired within timeout

    Examples:
        # Exclusive lock for writing
        with open(path, 'a') as f:
            with file_lock(f):
                f.write("data\n")

        # Shared lock for reading
        with open(path, 'r') as f:
            with file_lock(f, shared=True):
                data = f.read()
    """
    start_time = time.time()

    if WINDOWS:
        # Windows locking
        mode = msvcrt.LK_NBLCK if not shared else msvcrt.LK_NBRLCK

        while True:
            try:
                # Lock first byte of file
                msvcrt.locking(file_handle.fileno(), mode, 1)
                break
            except OSError:
                if time.time() - start_time > timeout:
                    raise FileLockTimeout(
                            f"Could not acquire lock on {file_handle.name} "
                            f"after {timeout}s"
                    )
                time.sleep(0.01)  # 10ms retry interval

        try:
            yield file_handle
        finally:
            # Unlock
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)

    else:
        # Unix locking (fcntl)
        lock_type = fcntl.LOCK_SH if shared else fcntl.LOCK_EX

        while True:
            try:
                fcntl.flock(file_handle.fileno(), lock_type | fcntl.LOCK_NB)
                break
            except (IOError, OSError):
                if time.time() - start_time > timeout:
                    raise FileLockTimeout(
                            f"Could not acquire lock on {file_handle.name} "
                            f"after {timeout}s"
                    )
                time.sleep(0.01)  # 10ms retry interval

        try:
            yield file_handle
        finally:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


def atomic_read(
        file_path: Path,
        reader: Callable[[Path], T],
        timeout: float = 30.0
) -> T:
    """
    Read file with shared lock to ensure consistency.

    Args:
        file_path: Path to file to read
        reader: Function that takes Path and returns parsed data
        timeout: Maximum seconds to wait for lock

    Returns:
        Result from reader function

    Example:
        >>> def parse_events(path):
        ...     return path.read_text().splitlines()
        >>> lines = atomic_read(event_log, parse_events)
    """
    if not file_path.exists():
        return reader(file_path)  # Let reader handle missing file

    with open(file_path, 'r', encoding='utf-8') as f:
        with file_lock(f, timeout=timeout, shared=True):
            # Re-seek to ensure we read from start after lock acquired
            f.seek(0)
            # Read entire content while locked
            content = f.read()

    # Parse outside lock to minimize lock time
    # Write content to temp location and parse
    import tempfile

    with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.tmp', encoding='utf-8'
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = reader(tmp_path)
    finally:
        tmp_path.unlink()

    return result


def atomic_append(
        file_path: Path,
        content: str,
        timeout: float = 30.0
) -> None:
    """
    Append to file with exclusive lock.

    Ensures thread-safe and process-safe appends in parallel processing
    environments. Uses platform-specific locking mechanisms.

    Args:
        file_path: Path to file
        content: Content to append (should include newline if needed)
        timeout: Maximum seconds to wait for lock (default: 30.0)

    Raises:
        FileLockTimeout: If lock cannot be acquired within timeout

    Example:
        >>> atomic_append(event_log, "2024-01-01|dataset|image|completed|\\n")
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'a', encoding='utf-8') as f:
        with file_lock(f, timeout=timeout, shared=False):
            f.write(content)
