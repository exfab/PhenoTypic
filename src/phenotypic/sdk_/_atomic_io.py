"""Crash-safe file writes — write to a temp sibling, then ``os.replace``.

A reader of a half-written JSON/parquet marker (a killed worker, a full disk
mid-write) must never observe a truncated file. These helpers write the full
payload to a temporary file **in the same directory** as the target, ``fsync``
it, then atomically rename it over the target via :func:`os.replace` (atomic on
POSIX; a best-effort replace on Windows). On any failure the temp file is
removed, so a pre-existing target is left untouched and no ``.tmp`` debris
lingers.

The sibling-directory placement matters: :func:`os.replace` is only atomic when
the source and destination are on the same filesystem, which a same-directory
temp guarantees (a ``/tmp`` temp could land on a different mount and degrade to
a non-atomic copy). Mirrors the callable-based
:func:`phenotypic._cli._cli_output_manager._atomic_write`, but with a plain
``text``/``bytes`` payload for the tune writers.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union


def _atomic_replace(target: Path, data: bytes) -> None:
    """Write ``data`` to a same-dir temp file, fsync, then replace ``target``.

    The shared core of :func:`atomic_write_text` and :func:`atomic_write_bytes`:
    the parent directory is created if missing, the bytes are written to a
    ``NamedTemporaryFile`` beside the target, flushed to disk, and atomically
    renamed over the target. Any exception removes the temp file before
    re-raising, so a pre-existing ``target`` is never clobbered by a partial
    write and no ``.tmp`` debris is left behind.

    Args:
        target: The final destination path.
        data: The exact bytes to write.

    Raises:
        OSError: Propagated from the write/replace after the temp file is
            cleaned up.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Union[str, None] = None
    try:
        handle = tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = handle.name
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def atomic_write_text(
    path: Union[str, Path], text: str, *, encoding: str = "utf-8"
) -> None:
    """Atomically write ``text`` to ``path`` (temp sibling + ``os.replace``).

    A drop-in replacement for ``Path(path).write_text(text)`` that never leaves a
    half-written file: a concurrent reader sees either the old contents or the
    complete new ones, and an exception mid-write leaves any pre-existing file
    intact with no ``.tmp`` debris.

    Args:
        path: The destination file path.
        text: The full text payload to write.
        encoding: The text encoding (default ``"utf-8"``).

    Raises:
        OSError: If the write or rename fails (the temp file is removed first).
    """
    _atomic_replace(Path(path), text.encode(encoding))


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    """Atomically write ``data`` to ``path`` (temp sibling + ``os.replace``).

    The bytes counterpart of :func:`atomic_write_text` for binary payloads
    (e.g. a serialized parquet buffer). Same crash-safety guarantees: an
    all-or-nothing replace and no partial/leftover temp file on failure.

    Args:
        path: The destination file path.
        data: The full binary payload to write.

    Raises:
        OSError: If the write or rename fails (the temp file is removed first).
    """
    _atomic_replace(Path(path), bytes(data))
