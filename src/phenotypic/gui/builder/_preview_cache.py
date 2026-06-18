"""Disk-backed preview cache for the builder node-preview modal.

One directory per (session, scope). Each scope dir holds full-resolution
per-node HDF snapshots (written by ``apply_with_intermediates(...,
full_layers=True)``), a ``manifest.json`` mapping block_id -> file/layers,
and (lazily) staged PNGs + DZI tile pyramids. The cache lives under the
system temp dir and is wiped on launch + ``atexit``.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

__all__ = [
    "preview_cache_root",
    "init_cache",
    "wipe_cache",
    "scope_hash",
    "scope_dir",
    "wipe_scope",
    "read_manifest",
    "write_manifest",
]

_CACHE_SUBPATH = ("phenotypic", "pipeline-preview")
_atexit_registered = False


def preview_cache_root() -> Path:
    """Cache root (recomputed each call so ``$TMPDIR`` changes are honoured)."""
    return Path(tempfile.gettempdir()).joinpath(*_CACHE_SUBPATH)


def wipe_cache() -> None:
    """Best-effort recursive delete of the cache root. Never raises."""
    shutil.rmtree(preview_cache_root(), ignore_errors=True)


def init_cache() -> None:
    """Wipe stale previews on launch and register an atexit cleanup (idempotent)."""
    global _atexit_registered
    wipe_cache()
    preview_cache_root().mkdir(parents=True, exist_ok=True)
    if not _atexit_registered:
        atexit.register(wipe_cache)
        _atexit_registered = True


def scope_hash(scope_path: list[str]) -> str:
    """Stable hash of a scope_path (list of container block_ids)."""
    return hashlib.sha1("/".join(scope_path).encode("utf-8")).hexdigest()


def scope_dir(session_id: str, scope_path: list[str]) -> Path:
    """Per-(session, scope) directory, created if missing."""
    d = preview_cache_root() / session_id / scope_hash(scope_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def wipe_scope(session_id: str, scope_path: list[str]) -> None:
    """Remove a single scope's cache dir (best-effort)."""
    shutil.rmtree(
        preview_cache_root() / session_id / scope_hash(scope_path),
        ignore_errors=True,
    )


def _manifest_path(session_id: str, scope_path: list[str]) -> Path:
    return scope_dir(session_id, scope_path) / "manifest.json"


def read_manifest(session_id: str, scope_path: list[str]) -> Optional[dict]:
    """Return the scope manifest dict, or None if absent/unreadable."""
    path = _manifest_path(session_id, scope_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(session_id: str, scope_path: list[str], manifest: dict) -> None:
    """Write the scope manifest atomically."""
    path = _manifest_path(session_id, scope_path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest))
    tmp.replace(path)
