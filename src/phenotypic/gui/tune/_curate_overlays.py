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
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import plotly.graph_objects as go

from phenotypic.gui._design import FONT_FAMILY_BODY

if TYPE_CHECKING:  # pragma: no cover - type-only
    from phenotypic import ImagePipeline
    from phenotypic.gui.tune._run_root import TuneRunRoot

logger = logging.getLogger(__name__)

#: A pending-overlay key — ``(session_id, trial_number, plate_name, mode)``. The
#: ``session_id`` namespaces a browser tab so two users never share a future.
PendingKey = tuple[str, int, str, str]

#: Module-level pending-future registry. Guarded by :data:`_PENDING_LOCK`
#: because Werkzeug serves Dash callbacks from many threads.
_PENDING: "dict[PendingKey, Future[np.ndarray]]" = {}
_PENDING_LOCK = threading.Lock()

#: Cached base pipelines, keyed by run path (read once from tuning_spec.json).
_BASE_PIPELINES: "dict[str, Optional[ImagePipeline]]" = {}
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
            return _BASE_PIPELINES[key]
    base = _read_base_pipeline_uncached(root)
    with _BASE_LOCK:
        _BASE_PIPELINES[key] = base
    return base


def _read_base_pipeline_uncached(root: "TuneRunRoot") -> "Optional[ImagePipeline]":
    """Load the base pipeline from the run's tuning_spec.json, or ``None``."""
    from phenotypic.tools_ import tuning_spec_path

    spec_path = tuning_spec_path(root.path)
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

    Args:
        cache: The process-wide :class:`OverlayCache`.
        key: The ``(session_id, trial, plate, mode)`` pending key.
        render_fn: A zero-arg callable returning the overlay RGB array.
    """
    cache_key = (key[1], key[2], key[3])  # OverlayKey: (trial, plate, mode)
    with _PENDING_LOCK:
        existing = _PENDING.get(key)
        if existing is not None and not existing.done():
            return
        _PENDING[key] = cache.submit(cache_key, render_fn)


def overlay_ready(key: PendingKey) -> bool:
    """Return ``True`` when ``key``'s overlay future has resolved."""
    with _PENDING_LOCK:
        future = _PENDING.get(key)
    return future is not None and future.done()


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
        font={"family": FONT_FAMILY_BODY},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        dragmode="zoom",
    )
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", constrain="domain")
    return fig


def clear_pending_for_session(session_id: str) -> None:
    """Drop every pending future for ``session_id`` (test / teardown hook)."""
    with _PENDING_LOCK:
        for key in [k for k in _PENDING if k[0] == session_id]:
            _PENDING.pop(key, None)


def load_plate_grid(image_source: str, plate_name: str):  # type: ignore[no-untyped-def]
    """Load ``<image_source>/<plate_name>`` as a :class:`GridImage`.

    Imported lazily (heavy) and kept here so the render closure built by the
    callback stays small. Raises on a missing file — the caller's render future
    captures the exception and the poll shows the error figure.

    Args:
        image_source: The selected Image Source directory.
        plate_name: The plate file name under it.

    Returns:
        A :class:`~phenotypic.GridImage` for the plate.
    """
    from phenotypic import GridImage
    from phenotypic.gui.tune._image_source import plate_image_path

    return GridImage(str(plate_image_path(image_source, plate_name)))


__all__ = [
    "PendingKey",
    "read_base_pipeline",
    "request_overlay",
    "overlay_ready",
    "take_overlay",
    "overlay_figure",
    "clear_pending_for_session",
    "load_plate_grid",
]
