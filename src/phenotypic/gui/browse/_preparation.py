"""Bounded, priority-aware background preparation for Browse artifacts."""

from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Literal

from PIL import Image as PILImage
from PIL import ImageOps

from phenotypic.gui._config import THREAD_NAME_PREFIX
from phenotypic.gui.browse._cache import BrowseCache, CacheEntry
from phenotypic.gui.browse._source_probe import SourceRevision, probe_source
from phenotypic.gui.browse._source_render import normalize_to_png

logger = logging.getLogger(__name__)

PreparationPhase = Literal[
    "queued",
    "previewing",
    "normalizing",
    "preview_ready",
    "tiling",
    "ready",
    "failed",
    "cancelled",
]
PreparationScope = Literal["selected", "nearby", "dataset", "preview"]

__all__ = [
    "BrowsePreparationManager",
    "PreparationHandle",
    "PreparationPhase",
    "PreparationPriority",
    "PreparationSnapshot",
    "SourceChangedError",
]


class SourceChangedError(RuntimeError):
    """Raised when the source no longer matches a requested revision."""


class PreparationPriority(IntEnum):
    """Lower values are prepared first."""

    SELECTED = 0
    NEARBY = 100
    DATASET = 1000


@dataclass(frozen=True)
class PreparationSnapshot:
    """Immutable UI-facing view of preparation state."""

    cache_key: str
    relative_path: str
    phase: PreparationPhase
    preview_ready: bool
    dzi_ready: bool
    error_code: str | None
    queued_at: float
    started_at: float | None
    finished_at: float | None
    queue_ms: float
    preview_ms: float
    normalization_ms: float
    dzi_ms: float


@dataclass
class _PreparationRecord:
    revision: SourceRevision
    phase: PreparationPhase = "queued"
    error_code: str | None = None
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    preview_ms: float = 0.0
    normalization_ms: float = 0.0
    dzi_ms: float = 0.0
    source_change_retries: int = 0
    preview_event: threading.Event = field(default_factory=threading.Event)
    dzi_event: threading.Event = field(default_factory=threading.Event)
    complete_event: threading.Event = field(default_factory=threading.Event)


class PreparationHandle:
    """Waitable handle shared by all requests for the same revision."""

    def __init__(
        self, manager: BrowsePreparationManager, cache_key: str
    ) -> None:
        self._manager = manager
        self.cache_key = cache_key

    @property
    def preview_ready(self) -> threading.Event:
        return self._manager._record_for(self.cache_key).preview_event

    @property
    def dzi_ready(self) -> threading.Event:
        return self._manager._record_for(self.cache_key).dzi_event

    @property
    def complete(self) -> threading.Event:
        return self._manager._record_for(self.cache_key).complete_event

    def snapshot(self) -> PreparationSnapshot:
        """Return a thread-safe immutable state snapshot."""
        return self._manager.snapshot(self.cache_key)


NormalizeCallable = Callable[[Path, Path], Path]
TileCallable = Callable[..., Path]


class BrowsePreparationManager:
    """Deduplicate and serialize selected, nearby, and dataset preparation.

    The single worker is deliberate: preparation is speculative and must not
    turn a navigation burst into simultaneous full-resolution decodes. A newly
    selected image jumps ahead of queued speculation, while an active native
    operation finishes before the priority change takes effect.
    """

    def __init__(
        self,
        cache: BrowseCache,
        *,
        normalize: NormalizeCallable = normalize_to_png,
        tile: TileCallable | None = None,
        preview_size: int = 1024,
        start_worker: bool = True,
    ) -> None:
        if preview_size < 1:
            raise ValueError("preview_size must be positive")
        self.cache = cache
        self._normalize = normalize
        self._tile = tile or _default_tile
        self._preview_size = preview_size
        self._condition = threading.Condition()
        self._records: dict[str, _PreparationRecord] = {}
        self._requesters: dict[
            str, dict[tuple[str, PreparationScope], int]
        ] = {}
        self._generations: dict[tuple[str, PreparationScope], int] = {}
        self._speculation_enabled: dict[str, bool] = {}
        self._pinned_selections: dict[str, str] = {}
        self._queued_priority: dict[str, int] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._sequence = itertools.count()
        self._closed = False
        self._active_key: str | None = None
        self._timings: deque[PreparationSnapshot] = deque(maxlen=256)
        self._worker: threading.Thread | None = None
        if start_worker:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name=f"{THREAD_NAME_PREFIX}-browse-prepare",
                daemon=True,
            )
            self._worker.start()

    def replace_selected(
        self,
        client_id: str,
        generation: int,
        revision: SourceRevision,
    ) -> PreparationHandle:
        """Replace one tab's selected revision and schedule it first."""
        with self._condition:
            if not self._replace_scope(client_id, "selected", generation):
                return self._stale_handle(revision)
            return self._request(
                revision,
                client_id=client_id,
                scope="selected",
                generation=generation,
                priority=int(PreparationPriority.SELECTED),
            )

    def request_preview(self, revision: SourceRevision) -> PreparationHandle:
        """Request a deduplicated preview without scheduling DZI generation."""
        with self._condition:
            return self._request(
                revision,
                client_id=f"preview:{revision.cache_key}",
                scope="preview",
                generation=0,
                priority=int(PreparationPriority.NEARBY),
            )

    def pin_selection(self, client_id: str, cache_key: str) -> None:
        """Protect one tab's displayed revision from automatic eviction."""
        with self._condition:
            self._pinned_selections[client_id] = cache_key

    def replace_nearby(
        self,
        client_id: str,
        generation: int,
        revisions: Iterable[SourceRevision],
    ) -> list[PreparationHandle]:
        """Replace directional neighbors for one tab, preserving input order."""
        with self._condition:
            revisions = list(revisions)
            if not self._replace_scope(client_id, "nearby", generation):
                return [self._stale_handle(revision) for revision in revisions]
            return [
                self._request(
                    revision,
                    client_id=client_id,
                    scope="nearby",
                    generation=generation,
                    priority=int(PreparationPriority.NEARBY) + index,
                )
                for index, revision in enumerate(revisions)
            ]

    def prepare_dataset(
        self,
        client_id: str,
        generation: int,
        revisions: Iterable[SourceRevision],
    ) -> list[PreparationHandle]:
        """Replace one tab's explicit dataset queue at lowest priority."""
        with self._condition:
            revisions = list(revisions)
            if not self._replace_scope(client_id, "dataset", generation):
                return [self._stale_handle(revision) for revision in revisions]
            return [
                self._request(
                    revision,
                    client_id=client_id,
                    scope="dataset",
                    generation=generation,
                    priority=int(PreparationPriority.DATASET) + index,
                )
                for index, revision in enumerate(revisions)
            ]

    def stop_dataset(self, client_id: str) -> None:
        """Cancel queued dataset work for one tab.

        An active normalization or native tiling call is allowed to finish;
        cancellation is observed before the next stage or task.
        """
        with self._condition:
            generation = self._generations.get((client_id, "dataset"), 0) + 1
            self._replace_scope(client_id, "dataset", generation)

    def set_speculation_enabled(self, client_id: str, enabled: bool) -> None:
        """Pause/resume this tab's nearby and dataset requests."""
        with self._condition:
            self._speculation_enabled[client_id] = enabled
            self._condition.notify_all()

    def snapshot(
        self, revision_or_key: SourceRevision | str
    ) -> PreparationSnapshot:
        """Return current state for a revision."""
        key = (
            revision_or_key
            if isinstance(revision_or_key, str)
            else revision_or_key.cache_key
        )
        with self._condition:
            record = self._records[key]
            return PreparationSnapshot(
                cache_key=key,
                relative_path=record.revision.relative_path,
                phase=record.phase,
                preview_ready=record.preview_event.is_set(),
                dzi_ready=record.dzi_event.is_set(),
                error_code=record.error_code,
                queued_at=record.queued_at,
                started_at=record.started_at,
                finished_at=record.finished_at,
                queue_ms=(
                    max(0.0, (record.started_at - record.queued_at) * 1000)
                    if record.started_at is not None
                    else 0.0
                ),
                preview_ms=record.preview_ms,
                normalization_ms=record.normalization_ms,
                dzi_ms=record.dzi_ms,
            )

    def snapshots(self) -> list[PreparationSnapshot]:
        """Return all known records, suitable for a progress response."""
        with self._condition:
            keys = list(self._records)
        return [self.snapshot(key) for key in keys]

    def recent_timings(self) -> tuple[PreparationSnapshot, ...]:
        """Return the bounded local timing window; no data is transmitted."""
        with self._condition:
            return tuple(self._timings)

    def protected_keys(self) -> frozenset[str]:
        """Return selected, queued, and active revisions protected from eviction."""
        with self._condition:
            keys = set(self._requesters)
            keys.update(self._pinned_selections.values())
            if self._active_key:
                keys.add(self._active_key)
            return frozenset(keys)

    def close(self, *, timeout: float = 5.0) -> None:
        """Stop accepting work and join the worker after its current stage."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    def _record_for(self, key: str) -> _PreparationRecord:
        with self._condition:
            return self._records[key]

    def _replace_scope(
        self, client_id: str, scope: PreparationScope, generation: int
    ) -> bool:
        generation_key = (client_id, scope)
        current = self._generations.get(generation_key, -1)
        if generation < current:
            return False
        self._generations[generation_key] = generation
        requester = (client_id, scope)
        for key, requesters in list(self._requesters.items()):
            if requester in requesters:
                del requesters[requester]
                if not requesters:
                    self._cancel_if_queued(key)
        return True

    def _stale_handle(self, revision: SourceRevision) -> PreparationHandle:
        """Return a non-scheduled handle for an out-of-order generation."""
        key = revision.cache_key
        record = self._records.get(key)
        if record is None:
            record = _PreparationRecord(revision, phase="cancelled")
            record.finished_at = time.monotonic()
            record.complete_event.set()
            self._records[key] = record
        return PreparationHandle(self, key)

    def _request(
        self,
        revision: SourceRevision,
        *,
        client_id: str,
        scope: PreparationScope,
        generation: int,
        priority: int,
    ) -> PreparationHandle:
        key = revision.cache_key
        entry = self.cache.entry(revision)
        record = self._records.get(key)
        if record is None or record.complete_event.is_set():
            record = _PreparationRecord(revision)
            self._records[key] = record
        if entry.preview_ready:
            record.preview_event.set()
            if scope == "preview":
                record.phase = "preview_ready"
                record.complete_event.set()
                record.finished_at = time.monotonic()
                self.cache.touch(entry)
                return PreparationHandle(self, key)
        if entry.dzi_ready:
            record.phase = "ready"
            record.dzi_event.set()
            record.complete_event.set()
            record.finished_at = time.monotonic()
            self.cache.touch(entry)
            return PreparationHandle(self, key)

        self._requesters.setdefault(key, {})[(client_id, scope)] = generation
        prior_priority = self._queued_priority.get(key)
        if key != self._active_key and (
            prior_priority is None or priority < prior_priority
        ):
            self._queued_priority[key] = priority
            heapq.heappush(self._heap, (priority, next(self._sequence), key))
        record.phase = "queued"
        self._condition.notify_all()
        return PreparationHandle(self, key)

    def _cancel_if_queued(self, key: str) -> None:
        self._requesters.pop(key, None)
        self._queued_priority.pop(key, None)
        record = self._records.get(key)
        if (
            record is not None
            and key != self._active_key
            and record.phase == "queued"
        ):
            record.phase = "cancelled"
            record.finished_at = time.monotonic()
            record.complete_event.set()
        self._condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                selected = self._next_allowed_locked()
                while selected is None and not self._closed:
                    self._condition.wait()
                    selected = self._next_allowed_locked()
                if self._closed:
                    return
                assert selected is not None
                key, record = selected
                self._active_key = key
                self._queued_priority.pop(key, None)
                record.started_at = time.monotonic()
            try:
                self._prepare(key, record)
            except SourceChangedError:
                logger.info(
                    "Browse source changed during preparation: %s",
                    record.revision.relative_path,
                )
                self._retry_changed_source(key, record)
            except Exception:  # noqa: BLE001 - worker must survive individual failures
                logger.exception(
                    "Browse preparation failed for %s",
                    record.revision.relative_path,
                )
                self._fail(record, "preparation_failed")
            finally:
                with self._condition:
                    self._active_key = None
                    current_record = self._records.get(key)
                    if current_record is not record and self._requesters.get(key):
                        priorities = {
                            "selected": int(PreparationPriority.SELECTED),
                            "nearby": int(PreparationPriority.NEARBY),
                            "preview": int(PreparationPriority.NEARBY),
                            "dataset": int(PreparationPriority.DATASET),
                        }
                        priority = min(
                            priorities[scope]
                            for _client_id, scope in self._requesters[key]
                        )
                        self._queued_priority[key] = priority
                        heapq.heappush(
                            self._heap,
                            (priority, next(self._sequence), key),
                        )
                    elif record.phase in {
                        "ready",
                        "preview_ready",
                        "failed",
                        "cancelled",
                    }:
                        self._requesters.pop(key, None)
                        snapshot = self.snapshot(key)
                        self._timings.append(snapshot)
                        logger.info(
                            "Browse preparation timing: revision=%s "
                            "dimensions=%sx%s phase=%s queue_ms=%.2f "
                            "preview_ms=%.2f normalization_ms=%.2f dzi_ms=%.2f",
                            key[:12],
                            record.revision.width,
                            record.revision.height,
                            snapshot.phase,
                            snapshot.queue_ms,
                            snapshot.preview_ms,
                            snapshot.normalization_ms,
                            snapshot.dzi_ms,
                        )
                    self._condition.notify_all()

    def _next_allowed_locked(self) -> tuple[str, _PreparationRecord] | None:
        deferred: list[tuple[int, int, str]] = []
        selected: tuple[str, _PreparationRecord] | None = None
        while self._heap:
            priority, sequence, key = heapq.heappop(self._heap)
            if self._queued_priority.get(key) != priority:
                continue
            if not self._requesters.get(key):
                self._queued_priority.pop(key, None)
                continue
            if priority >= int(
                PreparationPriority.NEARBY
            ) and not self._speculation_allowed(key):
                deferred.append((priority, sequence, key))
                continue
            selected = key, self._records[key]
            break
        for item in deferred:
            heapq.heappush(self._heap, item)
        return selected

    def _retry_changed_source(
        self, key: str, record: _PreparationRecord
    ) -> None:
        """Re-probe and queue one replacement revision after a source mutation."""
        with self._condition:
            requesters = dict(self._requesters.get(key, {}))
        self._fail(record, "source_changed")
        if record.source_change_retries >= 1 or not requesters:
            return
        relative_parts = Path(record.revision.relative_path).parts
        if not relative_parts:
            return
        sandbox_root = record.revision.source_path.parents[
            len(relative_parts) - 1
        ]
        try:
            replacement = probe_source(
                record.revision.source_path,
                sandbox_root=sandbox_root,
                relative_path=record.revision.relative_path,
                tile_size=record.revision.tile_size,
                overlap=record.revision.overlap,
            )
        except Exception:  # noqa: BLE001 - a second unstable probe is terminal
            return
        with self._condition:
            for (client_id, scope), generation in requesters.items():
                priority = {
                    "selected": int(PreparationPriority.SELECTED),
                    "nearby": int(PreparationPriority.NEARBY),
                    "preview": int(PreparationPriority.NEARBY),
                    "dataset": int(PreparationPriority.DATASET),
                }[scope]
                handle = self._request(
                    replacement,
                    client_id=client_id,
                    scope=scope,
                    generation=generation,
                    priority=priority,
                )
                replacement_record = self._record_for(handle.cache_key)
                replacement_record.source_change_retries = 1
                if scope == "selected":
                    self._pinned_selections[client_id] = replacement.cache_key

    def _speculation_allowed(self, key: str) -> bool:
        requesters = self._requesters.get(key, {})
        return any(
            self._speculation_enabled.get(client_id, True)
            for client_id, _scope in requesters
        )

    def _prepare(self, key: str, record: _PreparationRecord) -> None:
        revision = record.revision
        entry = self.cache.entry(revision)
        if entry.dzi_ready:
            self._mark_ready(record, entry)
            return
        with self.cache.entry_lock(revision, timeout=120.0):
            if entry.dzi_ready:
                self._mark_ready(record, entry)
                return
            if not revision.matches_disk():
                raise SourceChangedError
            with self.cache.staging_entry(revision) as staged:
                if not entry.preview_ready:
                    self._set_phase(record, "previewing")
                    preview_started = time.perf_counter()
                    if _render_header_preview(
                        revision.source_path,
                        staged.preview,
                        max_size=self._preview_size,
                    ):
                        entry = self.cache.publish_preview(
                            revision, staged.preview
                        )
                        record.preview_event.set()
                    record.preview_ms += (
                        time.perf_counter() - preview_started
                    ) * 1000
                if entry.preview_ready and self._preview_only(key):
                    self._mark_preview_only(record, entry)
                    return
                if self._cancelled(key):
                    self._mark_cancelled(record)
                    return

                self._set_phase(record, "normalizing")
                normalization_started = time.perf_counter()
                self._normalize(revision.source_path, staged.normalized_png)
                record.normalization_ms += (
                    time.perf_counter() - normalization_started
                ) * 1000
                if not revision.matches_disk():
                    raise SourceChangedError

                if not entry.preview_ready:
                    _render_header_preview(
                        staged.normalized_png,
                        staged.preview,
                        max_size=self._preview_size,
                    )
                    entry = self.cache.publish_preview(
                        revision, staged.preview
                    )
                record.preview_event.set()
                self._set_phase(record, "preview_ready")
                if self._preview_only(key):
                    self._mark_preview_only(record, entry)
                    return
                if self._cancelled(key):
                    self._mark_cancelled(record)
                    return

                self._set_phase(record, "tiling")
                staged.dzi_dir.mkdir(parents=True, exist_ok=True)
                dzi_started = time.perf_counter()
                self._tile(
                    staged.normalized_png,
                    staged.dzi_dir,
                    tile_size=revision.tile_size,
                    overlap=revision.overlap,
                )
                record.dzi_ms += (time.perf_counter() - dzi_started) * 1000
                if not revision.matches_disk():
                    raise SourceChangedError
                entry = self.cache.publish_dzi(revision, staged)
        # Signal completion only after the publication lock is released.
        # ``complete_event`` is the caller's "this entry is published and
        # actionable" signal, and a caller that acts on it immediately (e.g.
        # ``BrowseCache.clear``, which takes the same lock with ``timeout=0``)
        # would otherwise be refused and silently skip the entry.
        self._mark_ready(record, entry)
        self.cache.prune(protected=set(self.protected_keys()))

    def _cancelled(self, key: str) -> bool:
        with self._condition:
            return not bool(self._requesters.get(key)) or self._closed

    def _preview_only(self, key: str) -> bool:
        """Return whether all live requesters only need the preview milestone."""
        with self._condition:
            requesters = self._requesters.get(key, {})
            return bool(requesters) and all(
                scope == "preview" for _client_id, scope in requesters
            )

    def _set_phase(
        self, record: _PreparationRecord, phase: PreparationPhase
    ) -> None:
        with self._condition:
            record.phase = phase

    def _mark_ready(
        self, record: _PreparationRecord, entry: CacheEntry
    ) -> None:
        with self._condition:
            record.phase = "ready"
            record.preview_event.set()
            record.dzi_event.set()
            record.complete_event.set()
            record.finished_at = time.monotonic()
        self.cache.touch(entry)

    def _mark_preview_only(
        self, record: _PreparationRecord, entry: CacheEntry
    ) -> None:
        """Complete a preview-only request without creating a DZI pyramid."""
        with self._condition:
            record.phase = "preview_ready"
            record.preview_event.set()
            record.complete_event.set()
            record.finished_at = time.monotonic()
        self.cache.touch(entry)

    def _mark_cancelled(self, record: _PreparationRecord) -> None:
        with self._condition:
            record.phase = "cancelled"
            record.complete_event.set()
            record.finished_at = time.monotonic()

    def _fail(self, record: _PreparationRecord, code: str) -> None:
        with self._condition:
            record.phase = "failed"
            record.error_code = code
            record.complete_event.set()
            record.finished_at = time.monotonic()


def _render_header_preview(
    source: Path, destination: Path, *, max_size: int
) -> bool:
    """Best-effort bounded preview decode using Pillow's draft path."""
    try:
        with PILImage.open(source) as image:
            image.draft("RGB", (max_size, max_size))
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail(
                (max_size, max_size), PILImage.Resampling.LANCZOS
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            preview.save(destination, format="PNG", optimize=True)
        return True
    except Exception:  # noqa: BLE001 - faithful normalized fallback follows
        destination.unlink(missing_ok=True)
        return False


def _default_tile(png_path: Path, output_dir: Path, **kwargs: int) -> Path:
    from phenotypic.gui.results_viewer._dzi_tiler import tile

    return tile(
        png_path,
        output_dir,
        tile_size=kwargs["tile_size"],
        overlap=kwargs["overlap"],
    )
