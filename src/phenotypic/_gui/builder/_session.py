"""Per-session in-memory caches for images and pre-baked previews.

The Dash builder needs server-side storage for two things that are too big or
to round-trip through ``dcc.Store`` JSON:

- The currently loaded :class:`phenotypic.Image` (or :class:`GridImage`).
- The map of ``<step-or-block id> -> rendered preview`` produced after
  :meth:`ImagePipeline.apply_with_intermediates`.  Legacy
  ``_LegacyBuilderState`` callers key by 8-char ``StepNode.node_id``;
  DAG callers key by 32-char ``BlockNode.block_id``.

Both live in a single :class:`IntermediatesCache` keyed by a per-tab uuid that
the front end persists in a ``dcc.Store(storage_type='session')``. The cache
is bounded — at most ``max_sessions`` distinct sessions, with eviction in
FIFO order on the session-id list, and at most ``max_per_session``
intermediates per session (LRU on the inner dict).

**Cached preview payload contract** — each intermediate slot holds one of:

- :class:`bytes` — raw PNG bytes pre-baked by ``render_node_preview`` (one
  per ops node).
- :class:`pandas.DataFrame` — the measurement / post-measurement output table.
- :class:`PreviewRenderError` — sentinel describing a failed render so the
  inspector can surface the message inline.

This is a hard departure from the pre-pre-bake design which stored full
``Image`` / ``GridImage`` instances. Dropping the source images after
rendering shrinks worst-case resident memory by ~100×.

Concurrency is handled with a single ``threading.Lock``. This is a deliberate
single-process design — the SSH-tunneled HPCC use case does not require
multi-process replication, and Dash's thread-per-request model is well within
the lock's contention budget.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, TypeAlias, Union

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import pandas as pd  # type: ignore[import-untyped]

    from phenotypic import Image


@dataclass(frozen=True)
class PreviewRenderError:
    """Sentinel marking an intermediate slot whose render failed.

    Stored in :class:`IntermediatesCache` in place of PNG bytes when
    ``render_node_preview`` raises during a preview run, so the inspector
    can show the message inline without crashing the whole preview.

    Attributes:
        message: Human-readable rendering failure (typically the formatted
            exception). Surfaced verbatim in the inspector's warning div.
    """

    message: str


CachedPreview = Union[bytes, "pd.DataFrame", PreviewRenderError]
"""Anything :meth:`IntermediatesCache.set_intermediate` may store."""

PreviewKey: TypeAlias = tuple[str, str, str, int]
"""Stable identity of one published preview in the single-process cache."""


def _store_bounded_preview(
    previews: "OrderedDict[str, CachedPreview]",
    node_id: str,
    value: CachedPreview,
    *,
    limit: int,
) -> None:
    """Insert one preview at the MRU end and enforce ``limit``."""
    previews.pop(node_id, None)
    previews[node_id] = value
    while len(previews) > limit:
        previews.popitem(last=False)


def _touch_preview(
    previews: "OrderedDict[str, CachedPreview]",
    node_id: str,
) -> CachedPreview:
    """Move an existing preview to the MRU end and return it."""
    value = previews.pop(node_id)
    previews[node_id] = value
    return value


@dataclass(frozen=True)
class PreviewGenerationReservation:
    """Serializable identity reserved before a preview computation starts."""

    session_id: str
    pipeline_revision: str
    request_id: str
    request_sequence: int


@dataclass
class PreviewGenerationWriter:
    """Private staging area for one all-or-nothing preview publication.

    A writer is deliberately detached from :class:`SessionData`. Rendering
    callbacks populate it while the currently published generation remains
    readable. Only :meth:`IntermediatesCache.publish_preview_generation`
    swaps the complete staged map into the live session.

    Attributes:
        session_id: Per-tab session owning the generation.
        pipeline_revision: Canonical semantic revision baked by the caller.
        request_sequence: Cache-global monotonic token assigned before preview
            work.
        request_id: Server-issued request identity bound to the reservation.
        max_intermediates: Per-session cache bound applied while staging.
        intermediates: Ordered staged payloads, never directly exposed live.
    """

    session_id: str
    pipeline_revision: str
    request_sequence: int
    request_id: str
    max_intermediates: int
    image: Optional["Image"] = None
    image_path: Optional[str] = None
    image_staged: bool = False
    intermediates: "OrderedDict[str, CachedPreview]" = field(
        default_factory=OrderedDict
    )

    def set_image(
        self,
        session_id: str,
        image: "Image",
        path: Optional[str],
    ) -> None:
        """Stage the source image under the same publication fence."""

        if session_id != self.session_id:
            raise ValueError("preview writer used with a different session")
        self.image = image
        self.image_path = path
        self.image_staged = True

    def set_intermediate(
        self, session_id: str, node_id: str, value: CachedPreview
    ) -> None:
        """Stage one preview payload using the cache-compatible write API."""
        if session_id != self.session_id:
            raise ValueError("preview writer used with a different session")
        _store_bounded_preview(
            self.intermediates,
            node_id,
            value,
            limit=self.max_intermediates,
        )

    def known_intermediate_keys(self, session_id: str) -> list[str]:
        """Return staged node ids for cache-compatible callers."""
        if session_id != self.session_id:
            return []
        return list(self.intermediates)


@dataclass
class SessionData:
    """Mutable per-session state.

    Attributes:
        image: The currently loaded :class:`phenotypic.Image` (or
            :class:`GridImage`); ``None`` when nothing is loaded yet.
        image_path: Path string from which ``image`` was loaded (informational).
            ``None`` when ``image`` came from the synthetic plate fallback.
        intermediates: ``<step-or-block id> -> CachedPreview`` map of
            per-step pre-baked previews. Each value is :class:`bytes`
            (PNG) for ops nodes, a :class:`pandas.DataFrame` for
            measurement / post nodes, or :class:`PreviewRenderError`
            when rendering failed. Eviction order is LRU (oldest
            accessed first). Maintained as an :class:`OrderedDict` so
            :meth:`set_intermediate` can move accessed keys to the end
            without copying.  Keys are 8-char ``StepNode.node_id``
            strings on the legacy path and 32-char ``BlockNode.block_id``
            strings on the DAG path; the cache does not enforce a
            scheme, so a process restart (or feature-flag flip) silently
            invalidates the cache by rotating the keys.
        pipeline_revision: Semantic revision of the published generation.
        preview_generation: Monotonic count of successful publications.
        preview_request_sequence: Cache-global monotonic request token currently
            authorized to publish for this session.
    """

    image: Optional["Image"] = None
    image_path: Optional[str] = None
    intermediates: "OrderedDict[str, CachedPreview]" = field(
        default_factory=OrderedDict
    )
    pipeline_revision: Optional[str] = None
    preview_generation: int = 0
    preview_request_sequence: int = 0
    preview_request_id: Optional[str] = None
    preview_request_revision: Optional[str] = None
    preview_request_claimed: bool = False


class IntermediatesCache:
    """Bounded LRU cache of per-session image + intermediates state.

    Designed for a single-process Dash deployment behind an SSH tunnel.
    Concurrency is coarse-grained: every public method takes a single lock,
    which is fine at the request rates this app sees in practice.

    Args:
        max_sessions: Cap on the number of distinct session ids tracked.
            FIFO eviction on the *insertion* order — the oldest session is
            dropped when a new one would push the count over this cap.
        max_per_session: Cap on the number of intermediates retained for any
            single session. LRU eviction on the inner :class:`OrderedDict`.
    """

    def __init__(self, max_sessions: int = 4, max_per_session: int = 16) -> None:
        self._max_sessions = max_sessions
        self._max_per_session = max_per_session
        # OrderedDict so we can FIFO-evict the oldest session.
        self._sessions: "OrderedDict[str, SessionData]" = OrderedDict()
        # Cache-global so clearing, evicting, and recreating a session id can
        # never re-authorize an older in-flight writer by reusing its token.
        self._preview_request_sequence = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_session(self, session_id: str) -> SessionData:
        """Return the :class:`SessionData` for *session_id*, creating it lazily.

        Caller must hold ``self._lock``.
        """

        data = self._sessions.get(session_id)
        if data is None:
            data = SessionData()
            self._sessions[session_id] = data
            self._evict_oldest_sessions()
        return data

    def _evict_oldest_sessions(self) -> None:
        """Drop the oldest sessions until count <= ``self._max_sessions``."""

        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_image(
        self, session_id: str
    ) -> "tuple[Optional[Image], Optional[str]]":
        """Return ``(image, path)`` for *session_id*, or ``(None, None)``.

        Returning a tuple of immutable references avoids exposing the
        mutable :class:`SessionData` to callers that would otherwise mutate
        it outside the cache lock.

        Args:
            session_id: Per-tab uuid stored in ``STORE_SESSION_ID``.

        Returns:
            ``(image, image_path)`` for the session. Both elements are
            ``None`` when nothing has been loaded yet.
        """

        with self._lock:
            data = self._sessions.get(session_id)
            if data is None:
                return (None, None)
            return (data.image, data.image_path)

    def set_image(
        self,
        session_id: str,
        image: "Image | None",
        path: Optional[str],
    ) -> None:
        """Replace the cached image for *session_id*.

        Clearing the image (passing ``None``) also clears any intermediates
        derived from it, since their node-id labels would no longer match.

        Args:
            session_id: Per-tab uuid.
            image: New :class:`Image` to cache, or ``None`` to clear.
            path: Path string to remember alongside the image (informational).
        """

        with self._lock:
            data = self._ensure_session(session_id)
            data.image = image
            data.image_path = path
            if image is None:
                data.intermediates.clear()
                data.pipeline_revision = None

    def set_intermediate(
        self, session_id: str, node_id: str, value: CachedPreview
    ) -> None:
        """Store *value* as the pre-baked preview for *node_id*.

        If *node_id* already exists, it's moved to the most-recently-used end.

        Args:
            session_id: Per-tab uuid.
            node_id: ``StepNode.node_id`` produced by ``_state``.
            value: PNG :class:`bytes` (ops node), :class:`pandas.DataFrame`
                (measurement / post node), or :class:`PreviewRenderError`
                (rendering failure marker).
        """

        with self._lock:
            data = self._ensure_session(session_id)
            _store_bounded_preview(
                data.intermediates,
                node_id,
                value,
                limit=self._max_per_session,
            )

    def begin_preview_generation(
        self, session_id: str, pipeline_revision: str
    ) -> PreviewGenerationWriter:
        """Reserve and immediately claim a writer for compatibility callers."""

        reservation = self.reserve_preview_generation(
            session_id,
            pipeline_revision,
            request_id=f"direct-{self._preview_request_sequence + 1}",
        )
        writer = self.claim_preview_generation(reservation)
        if writer is None:  # pragma: no cover - same locked cache owns both calls
            raise RuntimeError("new preview reservation could not be claimed")
        return writer

    def reserve_preview_generation(
        self,
        session_id: str,
        pipeline_revision: str,
        *,
        request_id: str,
    ) -> PreviewGenerationReservation:
        """Fence older work before publishing a browser running state."""

        with self._lock:
            data = self._ensure_session(session_id)
            self._preview_request_sequence += 1
            data.preview_request_sequence = self._preview_request_sequence
            data.preview_request_id = request_id
            data.preview_request_revision = pipeline_revision
            data.preview_request_claimed = False
            return PreviewGenerationReservation(
                session_id=session_id,
                pipeline_revision=pipeline_revision,
                request_id=request_id,
                request_sequence=self._preview_request_sequence,
            )

    def claim_preview_generation(
        self,
        reservation: PreviewGenerationReservation,
    ) -> Optional[PreviewGenerationWriter]:
        """Claim an exact reservation once, returning a detached writer."""

        with self._lock:
            data = self._sessions.get(reservation.session_id)
            if (
                data is None
                or data.preview_request_sequence != reservation.request_sequence
                or data.preview_request_id != reservation.request_id
                or data.preview_request_revision != reservation.pipeline_revision
                or data.preview_request_claimed
            ):
                return None
            data.preview_request_claimed = True
            return PreviewGenerationWriter(
                session_id=reservation.session_id,
                pipeline_revision=reservation.pipeline_revision,
                request_id=reservation.request_id,
                request_sequence=reservation.request_sequence,
                max_intermediates=self._max_per_session,
            )

    def publish_preview_generation(
        self, writer: PreviewGenerationWriter
    ) -> Optional[int]:
        """Atomically replace a session's live previews with a complete bake.

        Args:
            writer: Fully populated staging writer returned by
                :meth:`begin_preview_generation`.

        Returns:
            Monotonically increasing generation number for the session, or
            ``None`` when a newer request superseded this writer.
        """
        with self._lock:
            data = self._sessions.get(writer.session_id)
            if (
                data is None
                or writer.request_sequence != data.preview_request_sequence
                or writer.request_id != data.preview_request_id
                or writer.pipeline_revision != data.preview_request_revision
                or not data.preview_request_claimed
            ):
                return None
            if writer.image_staged:
                data.image = writer.image
                data.image_path = writer.image_path
            data.intermediates = OrderedDict(writer.intermediates)
            data.pipeline_revision = writer.pipeline_revision
            data.preview_generation += 1
            return data.preview_generation

    def get_preview(self, key: PreviewKey) -> Optional[CachedPreview]:
        """Return a preview only when its full published key is current."""
        session_id, node_id, pipeline_revision, generation = key
        with self._lock:
            data = self._sessions.get(session_id)
            if (
                data is None
                or data.pipeline_revision != pipeline_revision
                or data.preview_generation != generation
                or node_id not in data.intermediates
            ):
                return None
            return _touch_preview(data.intermediates, node_id)

    def preview_descriptor(
        self, session_id: str
    ) -> Optional[tuple[str, int]]:
        """Return ``(pipeline_revision, generation)`` for the live snapshot."""
        with self._lock:
            data = self._sessions.get(session_id)
            if data is None or data.pipeline_revision is None:
                return None
            return (data.pipeline_revision, data.preview_generation)

    def get_intermediate(
        self, session_id: str, node_id: str
    ) -> Optional[CachedPreview]:
        """Return the cached intermediate for *node_id*, or ``None``.

        Touches the LRU order so frequently-viewed nodes survive eviction.

        Args:
            session_id: Per-tab uuid.
            node_id: ``StepNode.node_id`` to look up.

        Returns:
            The cached value, or ``None`` if the session or node has no
            recorded intermediate.
        """

        with self._lock:
            data = self._sessions.get(session_id)
            if data is None or node_id not in data.intermediates:
                return None
            return _touch_preview(data.intermediates, node_id)

    def clear(self, session_id: str) -> None:
        """Drop the entire cache entry for *session_id*.

        Args:
            session_id: Per-tab uuid to forget.
        """

        with self._lock:
            self._sessions.pop(session_id, None)

    def known_intermediate_keys(self, session_id: str) -> list[str]:
        """Snapshot the cached intermediate keys for *session_id*.

        Useful for populating ``STORE_INTERMEDIATE_KEYS`` after a preview run
        so the canvas can mark nodes that have a hot intermediate.

        Args:
            session_id: Per-tab uuid.

        Returns:
            Ordered list of node-ids currently cached (oldest -> newest).
        """

        with self._lock:
            data = self._sessions.get(session_id)
            if data is None:
                return []
            return list(data.intermediates.keys())


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_GLOBAL_CACHE: Optional[IntermediatesCache] = None
_GLOBAL_LOCK = threading.Lock()


def get_cache() -> IntermediatesCache:
    """Return the process-wide :class:`IntermediatesCache` singleton.

    Lazy-initialised on first call. Thread-safe.

    Returns:
        The shared :class:`IntermediatesCache`. All callbacks should route
        through this rather than constructing their own instance.
    """

    global _GLOBAL_CACHE
    with _GLOBAL_LOCK:
        if _GLOBAL_CACHE is None:
            _GLOBAL_CACHE = IntermediatesCache()
        return _GLOBAL_CACHE


__all__ = [
    "SessionData",
    "IntermediatesCache",
    "PreviewGenerationReservation",
    "PreviewGenerationWriter",
    "PreviewKey",
    "PreviewRenderError",
    "CachedPreview",
    "get_cache",
]
