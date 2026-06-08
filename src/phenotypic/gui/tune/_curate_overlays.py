"""Non-blocking overlay orchestration for the Curate view (B4).

The Curate render callback must NEVER block a Werkzeug worker thread on a heavy
``pipeline.apply``. This module is the non-blocking seam the builder preview
cache (`builder/_session` + `_bake_preview_cache`) established, adapted for tune
overlays:

* :func:`request_overlay` submits a candidate / difference render to the
  process-wide :class:`~phenotypic.gui.tune._overlays.OverlayCache` and stashes
  the :class:`~concurrent.futures.Future` in a **module-level** dict keyed
  ``(session_id, trial, plate, mode)`` — it returns immediately, never calling
  ``future.result()``.
* :func:`overlay_ready` / :func:`take_overlay` are what the
  ``dcc.Interval`` readiness poll calls to check + fetch the rendered array
  once the future resolves.
* :func:`overlay_figure` wraps a rendered RGB array in a Plotly ``go.Image``
  trace with the linked-zoom axes the clientside callback mirrors.

The base pipeline is read once from ``deliverables/tuning_spec.json`` (optuna-free)
and memoized per run.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import plotly.graph_objects as go

from phenotypic.gui.tune._figures import transparent_layout

if TYPE_CHECKING:  # pragma: no cover - type-only
    from phenotypic import ImagePipeline
    from phenotypic.gui.tune._run_root import TuneRunRoot

logger = logging.getLogger(__name__)

#: A pending-overlay key — ``(session_id, trial_number, plate_name, mode)``. The
#: ``session_id`` namespaces a browser tab so two users never share a future.
PendingKey = tuple[str, int, str, str]


def candidate_key(session: str, trial: int, plate: str) -> PendingKey:
    """The pending key for a candidate overlay (``mode == "candidate"``).

    Single-sources the submit/poll handshake: the render path
    (:mod:`._callbacks`) submits under this key and the readiness poll fetches
    under the same one, so the two can never drift on the key shape.
    """
    return (session, int(trial), str(plate), "candidate")


def difference_key(
    session: str, a_trial: int, b_trial: int, plate: str
) -> PendingKey:
    """The pending key for an A-vs-B difference overlay (``mode == "difference"``).

    The plate slot encodes ``"<plate>|<b_trial>"`` so the same A trial diffed
    against a different B (or on a different plate) is a distinct cache entry.
    Submit and poll both build the key here so they can't disagree.
    """
    return (session, int(a_trial), f"{plate}|{b_trial}", "difference")


def cache_key_for(key: PendingKey) -> tuple[int, str, str]:
    """Project a session-namespaced :data:`PendingKey` to its OverlayCache key.

    The per-tab ``_PENDING`` registry namespaces futures by ``session_id`` so two
    browser tabs never share a render, but the process-wide
    :class:`~phenotypic.gui.tune._overlays.OverlayCache` is keyed only by
    ``(trial, plate, mode)`` (the render is identical regardless of which tab
    asked). Single-sources that projection so :func:`request_overlay` (submit)
    and the readiness poll's self-heal (:meth:`OverlayCache.peek`) can never
    drift on the cache-key shape.

    Args:
        key: A ``(session_id, trial, plate, mode)`` pending key.

    Returns:
        The ``(trial, plate, mode)`` OverlayCache key.
    """
    return (key[1], key[2], key[3])

#: Hard cap on the module-level pending-future registry. ``_PENDING`` only needs
#: to bridge IN-FLIGHT renders — the rendered array is already memoized in the
#: per-run :class:`OverlayCache` (mem + disk LRU), so a key dropped here is not a
#: lost render, just a future the readiness poll won't re-resolve. Without a cap
#: every plate change / re-pin / A-B toggle (× a fresh ``session_id`` per tab)
#: leaks a future holding an overlay array forever. Sized like the overlay LRU.
_PENDING_CAP: int = 64

#: Module-level pending-future registry, MRU-ordered and bounded to
#: :data:`_PENDING_CAP`. Guarded by :data:`_PENDING_LOCK` because Werkzeug
#: serves Dash callbacks from many threads.
_PENDING: "OrderedDict[PendingKey, Future[np.ndarray]]" = OrderedDict()
_PENDING_LOCK = threading.Lock()

#: Hard cap on the cached-base-pipeline registry. Matches the ``_PENDING`` /
#: :class:`OverlayCache` capping pattern: each distinct run path opened in a
#: long-lived hub process would otherwise pin its base pipeline forever. 16 is
#: ample (a user rarely audits more than a handful of runs in one session) while
#: bounding resident memory.
_BASE_PIPELINES_CAP: int = 16

#: Cached base pipelines, keyed by run path (read once from tuning_spec.json).
#: MRU-ordered and bounded to :data:`_BASE_PIPELINES_CAP` (oldest evicted),
#: guarded by :data:`_BASE_LOCK`.
_BASE_PIPELINES: "OrderedDict[str, Optional[ImagePipeline]]" = OrderedDict()
_BASE_LOCK = threading.Lock()


def read_base_pipeline(root: "TuneRunRoot") -> "Optional[ImagePipeline]":
    """Read + memoize the run's base :class:`ImagePipeline` (optuna-free).

    The candidate overlay overlays a trial's params onto this base via
    ``build_pipeline``. The base is embedded in ``deliverables/tuning_spec.json``
    (``TuningSpec.pipeline``); importing ``TuningSpec`` never drags in optuna.
    Returns ``None`` when no spec file exists (a legacy / parquet-only run with
    no recoverable base) — the caller then surfaces a "base unavailable" note
    rather than rendering.

    Args:
        root: The validated tune output handle.

    Returns:
        The base pipeline, or ``None`` when it cannot be read.
    """
    key = str(root.path)
    with _BASE_LOCK:
        if key in _BASE_PIPELINES:
            _BASE_PIPELINES.move_to_end(key)  # touch → MRU
            return _BASE_PIPELINES[key]
    base = _read_base_pipeline_uncached(root)
    with _BASE_LOCK:
        _BASE_PIPELINES[key] = base
        _BASE_PIPELINES.move_to_end(key)
        # Hard LRU cap: evict the oldest until bounded.
        while len(_BASE_PIPELINES) > _BASE_PIPELINES_CAP:
            _BASE_PIPELINES.popitem(last=False)
    return base


def _read_base_pipeline_uncached(root: "TuneRunRoot") -> "Optional[ImagePipeline]":
    """Load the base pipeline from the run's tuning_spec.json, or ``None``."""
    from phenotypic.tools_ import resolve_tuning_spec_path

    spec_path = resolve_tuning_spec_path(root.path)
    if not spec_path.exists():
        return None
    try:
        from phenotypic.tune._spec import TuningSpec

        spec = TuningSpec.model_validate_json(spec_path.read_text())
        return spec.pipeline
    except Exception:  # noqa: BLE001 - render must degrade, never raise
        logger.warning("Could not read base pipeline at %s", spec_path, exc_info=True)
        return None


def request_overlay(
    cache: Any,
    key: PendingKey,
    render_fn,  # type: ignore[no-untyped-def]
) -> None:
    """Submit ``render_fn`` for ``key`` and stash the future (non-blocking).

    Routes through the :class:`OverlayCache` singleton (which short-circuits on
    a cache hit) and records the returned future under ``key`` so the readiness
    poll can fetch it. **Never** blocks on the result. Re-submitting an
    in-flight key is a no-op (the existing future is kept).

    The registry is bounded to :data:`_PENDING_CAP`: before inserting, every
    already-resolved future for the SAME ``(session_id, plate)`` prefix is
    dropped (the readiness poll will refetch them from the
    :class:`OverlayCache`), then the LRU entry is evicted if still over cap. A
    dropped future never loses a render — the array lives in the cache.

    Args:
        cache: The process-wide :class:`OverlayCache`.
        key: The ``(session_id, trial, plate, mode)`` pending key.
        render_fn: A zero-arg callable returning the overlay RGB array.
    """
    cache_key = cache_key_for(key)  # OverlayKey: (trial, plate, mode)
    with _PENDING_LOCK:
        existing = _PENDING.get(key)
        if existing is not None and not existing.done():
            _PENDING.move_to_end(key)  # keep the in-flight key fresh
            return
        # Opportunistically drop resolved stale futures for this (session, plate)
        # so a user flipping A/B on one plate doesn't accrete done futures.
        session_id, _trial, plate, _mode = key
        for stale in [
            k
            for k, fut in _PENDING.items()
            if k != key and k[0] == session_id and k[2] == plate and fut.done()
        ]:
            _PENDING.pop(stale, None)
        _PENDING[key] = cache.submit(cache_key, render_fn)
        _PENDING.move_to_end(key)
        # Hard LRU cap: evict the oldest until bounded.
        while len(_PENDING) > _PENDING_CAP:
            _PENDING.popitem(last=False)


def overlay_ready(key: PendingKey) -> bool:
    """Return ``True`` when ``key``'s overlay future has resolved.

    ``future.done()`` is evaluated INSIDE :data:`_PENDING_LOCK` (the B4 TOCTOU
    fix): reading it after releasing the lock let a concurrent ``request_overlay``
    drop/replace the future between the check and the use, so the poll could see
    a stale done-state for a future no longer in the registry.
    """
    with _PENDING_LOCK:
        future = _PENDING.get(key)
        if future is None:
            return False
        _PENDING.move_to_end(key)  # touch → MRU so the poll can't evict it
        return future.done()


def take_overlay(key: PendingKey) -> "Optional[np.ndarray]":
    """Return ``key``'s rendered array (and drop the future), or ``None``.

    Called by the readiness poll once :func:`overlay_ready` is ``True``. A
    render that raised resolves to ``None`` (the poll then shows an error
    figure) so a bad candidate never wedges the poll.

    Args:
        key: The pending key.

    Returns:
        The rendered RGB array, or ``None`` when not ready or the render failed.
    """
    with _PENDING_LOCK:
        future = _PENDING.get(key)
        if future is None or not future.done():
            return None
        _PENDING.pop(key, None)
    try:
        return future.result()
    except Exception:  # noqa: BLE001 - a failed render shows the error figure
        logger.warning("Overlay render failed for %s", key, exc_info=True)
        return None


def overlay_figure(array: "np.ndarray") -> go.Figure:
    """Wrap a rendered RGB overlay array in a Plotly ``go.Image`` figure.

    The figure's axes are configured for the clientside linked-zoom mirror:
    ``scaleanchor`` ties the y-axis to the x-axis so a pan/zoom stays
    proportional, and the matched constrain keeps both side-by-side graphs in
    register when their ranges are synced.

    Args:
        array: An ``(H, W, 3)`` uint8 RGB overlay array.

    Returns:
        A :class:`plotly.graph_objects.Figure` carrying one ``go.Image`` trace.
    """
    fig = go.Figure(go.Image(z=np.asarray(array)))
    fig.update_layout(
        **transparent_layout(
            margin={"l": 8, "r": 8, "t": 8, "b": 8},
            dragmode="zoom",
        )
    )
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", constrain="domain")
    return fig


def clear_pending_for_session(session_id: str) -> None:
    """Drop every pending future for ``session_id`` (test / teardown hook)."""
    with _PENDING_LOCK:
        for key in [k for k in _PENDING if k[0] == session_id]:
            _PENDING.pop(key, None)


def load_plate_grid(image_source: str, plate_name: str, *, sandbox: Any = None):  # type: ignore[no-untyped-def]
    """Load ``<image_source>/<plate_name>`` as a :class:`GridImage`.

    Imported lazily (heavy) and kept here so the render closure built by the
    callback stays small. Raises on a missing file — the caller's render future
    captures the exception and the poll shows the error figure.

    Defense-in-depth: when ``sandbox`` is given the final load path is
    re-confined through :meth:`SandboxRoot.resolve`, so even a ``plate_name``
    from a less-trusted source can't escape the sandbox via ``..`` traversal or
    an absolute path. (Today the plate names come from ``iterdir()`` of an
    already-resolved in-sandbox directory, so this is belt-and-suspenders — but
    it keeps the load path safe for any future caller.) An out-of-sandbox path
    raises ``ValueError`` (from ``sandbox.resolve``), which the render future
    captures → the poll shows the error figure.

    Args:
        image_source: The selected Image Source directory.
        plate_name: The plate file name under it.
        sandbox: Optional :class:`~phenotypic.gui.shell._sandbox.SandboxRoot`
            used to re-confine the resolved load path.

    Returns:
        A :class:`~phenotypic.GridImage` for the plate.

    Raises:
        ValueError: When ``sandbox`` is given and the resolved path escapes it.
    """
    from phenotypic import GridImage
    from phenotypic.gui.tune._image_source import plate_image_path

    path = plate_image_path(image_source, plate_name)
    if sandbox is not None:
        # Re-confine through the sandbox boundary (raises ValueError on escape).
        path = sandbox.resolve(str(path))
    return GridImage.imread(path)


__all__ = [
    "PendingKey",
    "candidate_key",
    "difference_key",
    "cache_key_for",
    "read_base_pipeline",
    "request_overlay",
    "overlay_ready",
    "take_overlay",
    "overlay_figure",
    "clear_pending_for_session",
    "load_plate_grid",
]
