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
a non-atomic copy). The module supports plain text/bytes payloads, deterministic
JSON documents, pandas-style parquet writers, and callback-based writers such as
Polars or matplotlib.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Union

PARQUET_WRITE_OPTIONS: dict[str, Any] = {
    "compression": "zstd",
    "compression_level": 3,
}


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


def atomic_write_with_writer(
    path: Union[str, Path],
    writer: Callable[[str], None],
) -> None:
    """Atomically write ``path`` using a callback that receives a temp path.

    Args:
        path: Final destination path.
        writer: Callable that writes complete output to a temporary path string.

    Raises:
        OSError: Propagated from the writer or rename after temp cleanup.
    """
    target = Path(path)
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
        handle.close()
        writer(tmp_path)
        with open(tmp_path, "r+b") as fh:
            os.fsync(fh.fileno())
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


def atomic_write_json(
    path: Union[str, Path],
    payload: Mapping[str, Any] | list[Any],
    *,
    indent: int = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> None:
    """Atomically write a JSON payload with deterministic formatting.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable mapping or list.
        indent: Indentation passed to :func:`json.dumps`.
        sort_keys: Whether mapping keys are sorted for deterministic output.
        ensure_ascii: Whether non-ASCII characters are escaped.
    """
    atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
        )
        + "\n",
    )


def atomic_write_parquet(
    path: Union[str, Path],
    frame: Any,
    **kwargs: Any,
) -> None:
    """Atomically write a pandas-like frame with shared parquet defaults.

    Args:
        path: Destination parquet path.
        frame: Object exposing ``to_parquet(path, **kwargs)``.
        **kwargs: Per-call parquet writer overrides.
    """
    write_options = {"index": False, **PARQUET_WRITE_OPTIONS, **kwargs}
    atomic_write_with_writer(
        path,
        lambda tmp_path: frame.to_parquet(tmp_path, **write_options),
    )
