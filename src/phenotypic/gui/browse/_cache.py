"""Persistent, revision-addressed artifact cache for Browse."""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from phenotypic.gui._config import (
    BROWSE_CACHE_ACCESS_TOUCH_SECONDS,
    BROWSE_CACHE_HIGH_WATER_BYTES,
    BROWSE_CACHE_LOW_WATER_BYTES,
    BROWSE_CACHE_STAGING_GRACE_SECONDS,
    BROWSE_CACHE_SUBDIR,
    SANDBOX_GUI_DIRNAME,
)
from phenotypic.gui.browse._source_probe import SourceRevision
from phenotypic.sdk_._file_locking import (
    ArtifactLockTimeout,
    exclusive_path_lock,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BrowseCache",
    "CacheEntry",
    "CacheLocation",
    "CacheUsage",
    "resolve_cache_location",
]

_READY_MARKER = "READY"
_PREVIEW_READY_MARKER = "PREVIEW_READY"
_METADATA_FILENAME = "entry.json"
_NORMALIZED_STEM = "normalized"


@dataclass(frozen=True)
class CacheLocation:
    """Selected cache root and its durability tier."""

    root: Path
    tier: Literal["sandbox", "user", "temporary"]
    persistent: bool


@dataclass(frozen=True)
class CacheEntry:
    """Paths belonging to one immutable source revision."""

    key: str
    root: Path

    @property
    def preview(self) -> Path:
        return self.root / "preview.png"

    @property
    def preview_marker(self) -> Path:
        return self.root / _PREVIEW_READY_MARKER

    @property
    def dzi_dir(self) -> Path:
        return self.root / "dzi"

    @property
    def dzi_manifest(self) -> Path:
        return self.dzi_dir / f"{_NORMALIZED_STEM}.dzi"

    @property
    def normalized_png(self) -> Path:
        """Staging-only normalized image name consumed by the DZI tiler."""
        return self.root / f"{_NORMALIZED_STEM}.png"

    @property
    def ready_marker(self) -> Path:
        return self.root / _READY_MARKER

    @property
    def metadata(self) -> Path:
        return self.root / _METADATA_FILENAME

    @property
    def preview_ready(self) -> bool:
        return self.preview.is_file() and self.preview_marker.is_file()

    @property
    def dzi_ready(self) -> bool:
        return self.dzi_manifest.is_file() and self.ready_marker.is_file()


@dataclass(frozen=True)
class CacheUsage:
    """Cache size and completed-entry count."""

    bytes: int
    entries: int


def resolve_cache_location(sandbox_root: Path) -> CacheLocation:
    """Choose sandbox, user, then temporary cache storage.

    Candidates are proven writable using an atomic create/unlink probe. A
    temporary fallback is removed at process exit; persistent candidates are
    never registered for automatic deletion.
    """
    sandbox = Path(sandbox_root).resolve(strict=False)
    candidates: tuple[tuple[Path, Literal["sandbox", "user"]], ...] = (
        (sandbox / SANDBOX_GUI_DIRNAME / BROWSE_CACHE_SUBDIR, "sandbox"),
        (_user_cache_base() / _sandbox_hash(sandbox), "user"),
    )
    for candidate, tier in candidates:
        try:
            _prove_writable(candidate)
        except OSError:
            logger.warning(
                "Browse %s cache is unavailable; trying fallback", tier
            )
            continue
        return CacheLocation(candidate, tier, True)

    temporary = Path(tempfile.mkdtemp(prefix="phenotypic-browse-"))
    atexit.register(shutil.rmtree, temporary, True)
    logger.warning(
        "Browse is using a temporary cache; prepared images will not persist"
    )
    return CacheLocation(temporary, "temporary", False)


class BrowseCache:
    """Persistent cache with atomic publication, locking, and bounded LRU."""

    def __init__(
        self,
        location: CacheLocation,
        *,
        high_water_bytes: int = BROWSE_CACHE_HIGH_WATER_BYTES,
        low_water_bytes: int = BROWSE_CACHE_LOW_WATER_BYTES,
    ) -> None:
        if low_water_bytes < 0 or high_water_bytes <= low_water_bytes:
            raise ValueError("cache water marks must satisfy 0 <= low < high")
        self.location = location
        self.root = location.root
        self.high_water_bytes = high_water_bytes
        self.low_water_bytes = low_water_bytes
        self.entries_root = self.root / "entries"
        self.staging_root = self.root / "staging"
        self.locks_root = self.root / "locks"
        for path in (
            self.entries_root,
            self.staging_root,
            self.locks_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.cleanup_incomplete()

    @classmethod
    def for_sandbox(
        cls,
        sandbox_root: Path,
        *,
        high_water_bytes: int = BROWSE_CACHE_HIGH_WATER_BYTES,
        low_water_bytes: int = BROWSE_CACHE_LOW_WATER_BYTES,
    ) -> BrowseCache:
        """Create a cache at the best available persistence tier."""
        return cls(
            resolve_cache_location(sandbox_root),
            high_water_bytes=high_water_bytes,
            low_water_bytes=low_water_bytes,
        )

    def entry(self, revision: SourceRevision | str) -> CacheEntry:
        """Return fixed-length paths for a revision or cache key."""
        key = revision if isinstance(revision, str) else revision.cache_key
        if len(key) != 64 or any(
            char not in "0123456789abcdef" for char in key
        ):
            raise ValueError("invalid Browse cache key")
        return CacheEntry(key, self.entries_root / key[:2] / key)

    @contextmanager
    def entry_lock(
        self, revision: SourceRevision | str, *, timeout: float = 30.0
    ) -> Iterator[None]:
        """Hold the cross-process publication lock for one revision."""
        entry = self.entry(revision)
        with exclusive_path_lock(
            self.locks_root / f"{entry.key}.lock", timeout=timeout
        ):
            yield

    @contextmanager
    def staging_entry(self, revision: SourceRevision) -> Iterator[CacheEntry]:
        """Yield a same-filesystem staging entry and always clean it up."""
        directory = Path(
            tempfile.mkdtemp(
                prefix=f"{revision.cache_key}.",
                dir=self.staging_root,
            )
        )
        staged = CacheEntry(revision.cache_key, directory)
        try:
            yield staged
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def publish_preview(
        self, revision: SourceRevision, source: Path
    ) -> CacheEntry:
        """Atomically publish a preview before the full DZI is ready."""
        entry = self.entry(revision)
        entry.root.mkdir(parents=True, exist_ok=True)
        entry.preview_marker.unlink(missing_ok=True)
        _replace_file(Path(source), entry.preview)
        self._write_metadata(entry, revision)
        _atomic_text(entry.preview_marker, "ready\n")
        self.touch(entry)
        return entry

    def publish_dzi(
        self, revision: SourceRevision, staged: CacheEntry
    ) -> CacheEntry:
        """Publish a complete DZI directory, then its readiness marker."""
        if not staged.dzi_manifest.is_file():
            raise FileNotFoundError("staged DZI manifest is missing")
        entry = self.entry(revision)
        entry.root.mkdir(parents=True, exist_ok=True)
        # Readers use this marker as the publication boundary. Remove an old
        # marker before replacing any artifact and publish it again last.
        entry.ready_marker.unlink(missing_ok=True)
        if entry.dzi_dir.exists():
            shutil.rmtree(entry.dzi_dir)
        os.replace(staged.dzi_dir, entry.dzi_dir)
        self._write_metadata(entry, revision)
        _atomic_text(entry.ready_marker, "ready\n")
        staged.normalized_png.unlink(missing_ok=True)
        self.touch(entry)
        return entry

    def touch(self, entry: CacheEntry) -> None:
        """Update the LRU timestamp at most once per configured interval."""
        marker = (
            entry.ready_marker
            if entry.ready_marker.exists()
            else entry.preview_marker
        )
        if marker.exists():
            try:
                if (
                    time.time() - marker.stat().st_mtime
                    < BROWSE_CACHE_ACCESS_TOUCH_SECONDS
                ):
                    return
                marker.touch()
            except OSError:
                logger.debug(
                    "could not update Browse cache access time", exc_info=True
                )

    def usage(self) -> CacheUsage:
        """Return recursive bytes and completed/preview entry count."""
        entries = list(self._iter_entries())
        return CacheUsage(
            sum(_tree_size(entry.root) for entry in entries),
            len(entries),
        )

    def prune(
        self, *, protected: set[str] | frozenset[str] = frozenset()
    ) -> CacheUsage:
        """Prune unlocked LRU entries when usage exceeds the high-water mark."""
        usage = self.usage()
        if usage.bytes <= self.high_water_bytes:
            return usage
        candidates = self._prune_candidates(protected)
        total = usage.bytes
        for entry in candidates:
            if total <= self.low_water_bytes:
                break
            try:
                with self.entry_lock(entry.key, timeout=0.0):
                    size = _tree_size(entry.root)
                    shutil.rmtree(entry.root, ignore_errors=True)
                    total -= size
            except ArtifactLockTimeout:
                continue
        return self.usage()

    def clear(
        self, *, protected: set[str] | frozenset[str] = frozenset()
    ) -> CacheUsage:
        """Remove all unlocked entries except explicitly protected revisions."""
        for entry in self._iter_entries():
            if entry.key in protected:
                continue
            try:
                with self.entry_lock(entry.key, timeout=0.0):
                    shutil.rmtree(entry.root, ignore_errors=True)
            except ArtifactLockTimeout:
                continue
        return self.usage()

    def cleanup_incomplete(self) -> None:
        """Delete abandoned staging and entries with no readiness marker."""
        cutoff = time.time() - BROWSE_CACHE_STAGING_GRACE_SECONDS
        for child in self.staging_root.iterdir():
            try:
                if child.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            cache_key = child.name.split(".", maxsplit=1)[0]
            try:
                with self.entry_lock(cache_key, timeout=0.0):
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
            except ArtifactLockTimeout:
                # Another process owns this revision.
                continue
            except ValueError:
                # ``staging_root`` is cache-owned, so malformed abandoned
                # children are safe to remove even though they have no lock.
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        for entry in self._iter_entries(include_incomplete=True):
            if entry.preview_marker.exists() and not entry.preview.is_file():
                entry.preview_marker.unlink(missing_ok=True)
            if (
                entry.ready_marker.exists()
                and not entry.dzi_manifest.is_file()
            ):
                entry.ready_marker.unlink(missing_ok=True)
            if (
                not entry.preview_marker.exists()
                and not entry.ready_marker.exists()
            ):
                try:
                    with self.entry_lock(entry.key, timeout=0.0):
                        shutil.rmtree(entry.root, ignore_errors=True)
                except ArtifactLockTimeout:
                    continue

    def _iter_entries(
        self, *, include_incomplete: bool = False
    ) -> Iterator[CacheEntry]:
        if not self.entries_root.exists():
            return
        for prefix in self.entries_root.iterdir():
            if not prefix.is_dir():
                continue
            for path in prefix.iterdir():
                if not path.is_dir():
                    continue
                entry = CacheEntry(path.name, path)
                if (
                    include_incomplete
                    or entry.preview_marker.exists()
                    or entry.ready_marker.exists()
                ):
                    yield entry

    def _prune_candidates(
        self, protected: set[str] | frozenset[str]
    ) -> list[CacheEntry]:
        entries = [
            entry
            for entry in self._iter_entries()
            if entry.key not in protected
        ]
        newest_by_source: dict[str, tuple[int, str]] = {}
        metadata_by_key: dict[str, dict[str, object]] = {}
        for entry in entries:
            metadata = _read_json(entry.metadata)
            metadata_by_key[entry.key] = metadata
            source_id = str(metadata.get("source_id", ""))
            raw_mtime_ns = metadata.get("mtime_ns", 0)
            mtime_ns = raw_mtime_ns if isinstance(raw_mtime_ns, int) else 0
            if (
                source_id
                and mtime_ns >= newest_by_source.get(source_id, (-1, ""))[0]
            ):
                newest_by_source[source_id] = (mtime_ns, entry.key)

        def _sort_key(entry: CacheEntry) -> tuple[int, int]:
            metadata = metadata_by_key[entry.key]
            source_id = str(metadata.get("source_id", ""))
            newest = newest_by_source.get(source_id, (0, entry.key))[1]
            obsolete_rank = 0 if source_id and newest != entry.key else 1
            marker = (
                entry.ready_marker
                if entry.ready_marker.exists()
                else entry.preview_marker
            )
            try:
                accessed = marker.stat().st_mtime_ns
            except OSError:
                accessed = 0
            return obsolete_rank, accessed

        return sorted(entries, key=_sort_key)

    def _write_metadata(
        self, entry: CacheEntry, revision: SourceRevision
    ) -> None:
        payload = {
            "cache_key": revision.cache_key,
            "source_id": revision.source_id,
            "relative_path": revision.relative_path,
            "size_bytes": revision.size_bytes,
            "mtime_ns": revision.mtime_ns,
            "ctime_ns": revision.ctime_ns,
            "width": revision.width,
            "height": revision.height,
            "tile_size": revision.tile_size,
            "overlap": revision.overlap,
            "render_schema": revision.render_schema,
        }
        _atomic_text(
            entry.metadata, json.dumps(payload, sort_keys=True) + "\n"
        )


def _prove_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".write-test-{uuid.uuid4().hex}"
    try:
        with probe.open("xb") as handle:
            handle.write(b"ok")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        probe.unlink(missing_ok=True)


def _sandbox_hash(sandbox: Path) -> str:
    import hashlib

    return hashlib.sha256(str(sandbox).encode("utf-8")).hexdigest()[:24]


def _user_cache_base() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":  # pragma: win32 cover
        base = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "phenotypic" / BROWSE_CACHE_SUBDIR


def _replace_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _tree_size(root: Path) -> int:
    total = 0
    try:
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}
