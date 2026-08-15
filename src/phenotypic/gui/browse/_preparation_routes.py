"""Same-origin preparation APIs and process-local Browse job bookkeeping."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import dash
from flask import Blueprint, Response, jsonify, request

from phenotypic.gui.browse import _source_render
from phenotypic.gui.browse._cache import BrowseCache
from phenotypic.gui.browse._preparation import BrowsePreparationManager
from phenotypic.gui.browse._source_probe import (
    SourceProbeError,
    SourceRevision,
    probe_source,
)
from phenotypic.gui.shell._sandbox import SandboxRoot

__all__ = ["BrowsePreparationApi", "register", "resolve_revision"]


@dataclass
class _DatasetJob:
    client_id: str
    generation: int
    keys: tuple[str, ...]
    stopping: bool = False


@dataclass
class BrowsePreparationApi:
    """Coordinate tab-scoped preparation requests and cache actions."""

    sandbox: SandboxRoot
    cache: BrowseCache
    manager: BrowsePreparationManager
    _jobs: dict[str, _DatasetJob] = field(default_factory=dict, init=False)
    _selected: dict[str, str] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def select(
        self,
        client_id: str,
        generation: int,
        revision: SourceRevision,
    ):
        """Schedule the selected revision and remember it for cache protection."""
        with self._lock:
            self._selected[client_id] = revision.cache_key
        self.manager.pin_selection(client_id, revision.cache_key)
        return self.manager.replace_selected(client_id, generation, revision)

    def replace_nearby(
        self,
        client_id: str,
        generation: int,
        revisions: Sequence[SourceRevision],
    ) -> None:
        """Replace one tab's directional speculative queue."""
        self.manager.replace_nearby(client_id, generation, revisions)

    def start_dataset(
        self,
        client_id: str,
        generation: int,
        revisions: Sequence[SourceRevision],
    ) -> dict[str, Any]:
        """Start or replace one tab's explicit dataset job."""
        handles = self.manager.prepare_dataset(
            client_id, generation, revisions
        )
        with self._lock:
            self._jobs[client_id] = _DatasetJob(
                client_id=client_id,
                generation=generation,
                keys=tuple(handle.cache_key for handle in handles),
            )
        return self.status(client_id)

    def stop_dataset(self, client_id: str) -> dict[str, Any]:
        """Cancel queued dataset items for one tab."""
        self.manager.stop_dataset(client_id)
        with self._lock:
            job = self._jobs.get(client_id)
            if job is not None:
                job.stopping = True
        return self.status(client_id)

    def status(self, client_id: str) -> dict[str, Any]:
        """Return client-safe progress, cache usage, and backend details."""
        from phenotypic.gui.results_viewer._dzi_tiler import DZI_BACKEND_INFO

        with self._lock:
            job = self._jobs.get(client_id)
        snapshots = []
        if job is not None:
            for key in job.keys:
                try:
                    snapshots.append(self.manager.snapshot(key))
                except KeyError:
                    continue
        ready = sum(snapshot.dzi_ready for snapshot in snapshots)
        failed = sum(
            snapshot.phase in {"failed", "cancelled"} for snapshot in snapshots
        )
        total = len(job.keys) if job is not None else 0
        terminal = ready + failed >= total and total > 0
        if job is None:
            state = "idle"
            message = "Images prepare as you browse."
        elif total == 0:
            state = "complete"
            message = "No images to prepare."
        elif job.stopping and not terminal:
            state = "stopping"
            message = "Stopping after current image."
        elif terminal:
            state = "failed" if failed else "complete"
            message = f"Prepared {ready} of {total}; {failed} failed."
        else:
            state = "running"
            message = f"Preparing dataset: {ready} of {total} ready."
        usage = self.cache.usage()
        backend: str = DZI_BACKEND_INFO.name
        if DZI_BACKEND_INFO.version:
            backend = f"{backend} {DZI_BACKEND_INFO.version}"
        return {
            "state": state,
            "ready": ready,
            "total": total,
            "failed": failed,
            "message": message,
            "cache_usage": {
                "bytes": usage.bytes,
                "entries": usage.entries,
                "tier": self.cache.location.tier,
            },
            "backend": backend,
        }

    def clear(self, *, current_revision: str | None = None) -> dict[str, Any]:
        """Clear unlocked artifacts while preserving selected and active work."""
        protected = set(self.manager.protected_keys())
        with self._lock:
            protected.update(self._selected.values())
        if current_revision:
            protected.add(current_revision)
        before = self.cache.usage()
        after = self.cache.clear(protected=protected)
        return {
            "removed_bytes": max(0, before.bytes - after.bytes),
            "removed_entries": max(0, before.entries - after.entries),
            "remaining_bytes": after.bytes,
            "remaining_entries": after.entries,
        }


def resolve_revision(
    sandbox: SandboxRoot,
    token: str,
    expected_revision: str | None = None,
) -> SourceRevision:
    """Resolve an opaque token in the sandbox and probe its current revision."""
    try:
        relative = _source_render.decode_token(token)
        source = sandbox.resolve(relative)
    except Exception as exc:  # noqa: BLE001 - fixed client-safe route errors
        raise FileNotFoundError("invalid source token") from exc
    if not source.is_file():
        raise FileNotFoundError("source is not a file")
    try:
        revision = probe_source(
            source,
            sandbox_root=sandbox.root,
            relative_path=Path(relative).as_posix(),
        )
    except SourceProbeError as exc:
        raise FileNotFoundError("source cannot be inspected") from exc
    if (
        expected_revision is not None
        and revision.cache_key != expected_revision
    ):
        raise SourceProbeError("stale source revision")
    return revision


def register(app: dash.Dash, api: BrowsePreparationApi) -> None:
    """Register JSON preparation and cache-control routes."""
    bp = Blueprint("browse_preparation", __name__, url_prefix="/api/browse")

    @bp.post("/nearby")
    def nearby() -> Response:
        parsed = _parse_common(_json_body())
        if parsed is None:
            return _error("invalid request", 400)
        client_id, generation, items = parsed
        revisions = _resolve_items(api.sandbox, items)
        if revisions is None:
            return _error("invalid or stale image", 409)
        api.replace_nearby(client_id, generation, revisions)
        return jsonify({"queued": len(revisions)})

    @bp.post("/dataset/start")
    def dataset_start() -> Response:
        parsed = _parse_common(_json_body())
        if parsed is None:
            return _error("invalid request", 400)
        client_id, generation, items = parsed
        revisions = _resolve_items(api.sandbox, items)
        if revisions is None:
            return _error("invalid or stale image", 409)
        return jsonify(api.start_dataset(client_id, generation, revisions))

    @bp.get("/dataset/status")
    def dataset_status() -> Response:
        client_id = request.args.get("client_id", "")
        if not _valid_client_id(client_id):
            return _error("invalid request", 400)
        return jsonify(api.status(client_id))

    @bp.post("/dataset/stop")
    def dataset_stop() -> Response:
        body = _json_body()
        client_id = body.get("client_id") if body else None
        if not isinstance(client_id, str) or not _valid_client_id(client_id):
            return _error("invalid request", 400)
        return jsonify(api.stop_dataset(client_id))

    @bp.post("/cache/clear")
    def cache_clear() -> Response:
        body = _json_body()
        if body is None:
            return _error("invalid request", 400)
        client_id = body.get("client_id")
        current = body.get("current_revision")
        if not isinstance(client_id, str) or not _valid_client_id(client_id):
            return _error("invalid request", 400)
        if current is not None and not _valid_revision(current):
            return _error("invalid request", 400)
        return jsonify(api.clear(current_revision=current))

    app.server.register_blueprint(bp)


def _json_body() -> Mapping[str, Any] | None:
    body = request.get_json(silent=True)
    return body if isinstance(body, Mapping) else None


def _parse_common(
    body: Mapping[str, Any] | None,
) -> tuple[str, int, Sequence[Mapping[str, Any]]] | None:
    if body is None:
        return None
    client_id = body.get("client_id")
    generation = body.get("generation")
    items = body.get("items")
    if (
        not isinstance(client_id, str)
        or not _valid_client_id(client_id)
        or not isinstance(generation, int)
        or generation < 0
        or not isinstance(items, list)
        or len(items) > 100_000
        or not all(isinstance(item, Mapping) for item in items)
    ):
        return None
    return client_id, generation, items


def _resolve_items(
    sandbox: SandboxRoot,
    items: Sequence[Mapping[str, Any]],
) -> list[SourceRevision] | None:
    revisions: list[SourceRevision] = []
    for item in items:
        token = item.get("token")
        revision = item.get("revision")
        if not isinstance(token, str) or not _valid_revision(revision):
            return None
        try:
            revisions.append(resolve_revision(sandbox, token, revision))
        except (FileNotFoundError, SourceProbeError):
            return None
    return revisions


def _valid_client_id(value: str) -> bool:
    return 0 < len(value) <= 128 and all(
        char.isalnum() or char in "-_.:" for char in value
    )


def _valid_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _error(message: str, status: int) -> Response:
    response = jsonify({"error": message})
    response.status_code = status
    return response
