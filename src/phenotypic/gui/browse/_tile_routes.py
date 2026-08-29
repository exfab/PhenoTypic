"""Revision-addressed Browse preview, DZI manifest, and tile routes."""

from __future__ import annotations

import json
import os
import re
import stat
import time
from pathlib import Path
from typing import BinaryIO

import dash
from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
    send_file,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from phenotypic.gui._config import BROWSE_TILES_PREFIX
from phenotypic.gui.browse import _source_render
from phenotypic.gui.browse._preparation_routes import (
    BrowsePreparationApi,
    resolve_revision,
)
from phenotypic.gui.browse._cache import BrowseCache
from phenotypic.gui.browse._preparation import BrowsePreparationManager
from phenotypic.gui.browse._source_probe import SourceProbeError
from phenotypic.gui.browse._source_item import is_source_store
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.sdk_ import store_publication_token

_TILE_NAME_RE = re.compile(r"^\d+_\d+\.png$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_REVISION_RE = re.compile(r"^[a-f0-9]{64}$")
_IMMUTABLE_CACHE = "private, max-age=31536000, immutable"

__all__ = ["register"]


class _MutableStoreUnsupported(RuntimeError):
    """A third-party store has no O(1) immutable publication generation."""


class _UnreadableImageStore(RuntimeError):
    """A published store is not a readable PhenoTypic image store."""


class _UnsafeStoreAccess(RuntimeError):
    """The platform cannot bind store access to held directory identities."""


_SAFE_STORE_IO = (
    os.name == "posix"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_NONBLOCK")
    and os.open in os.supports_dir_fd
)
_MAX_STORE_METADATA_BYTES = 16 * 1024 * 1024


def _store_member_parts(member: str) -> tuple[str, ...]:
    """Return canonical URL path components or raise before filesystem I/O."""
    if (
        not member
        or member.startswith("/")
        or "\\" in member
        or "\x00" in member
        or "\ufffd" in member
    ):
        raise ValueError("invalid store member")
    try:
        member.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError("invalid store member") from exc
    parts = tuple(member.split("/"))
    if any(
        part in {"", ".", ".."}
        or any(ord(character) < 32 or ord(character) == 127 for character in part)
        for part in parts
    ):
        raise ValueError("invalid store member")
    return parts


def _open_store_root(store: Path) -> int:
    """Open a store directory without following a swapped root symlink."""
    if not _SAFE_STORE_IO:
        raise _UnsafeStoreAccess(
            "this platform cannot safely serve Zarr store members"
        )
    root_fd = os.open(
        store,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise OSError("store root is not a directory")
    except BaseException:
        os.close(root_fd)
        raise
    return root_fd


def _open_regular_store_member(
    root_fd: int,
    parts: tuple[str, ...],
) -> BinaryIO:
    """Open a regular member through held, no-follow directory descriptors."""
    directory_fd = os.dup(root_fd)
    member_fd: int | None = None
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        for component in parts[:-1]:
            child_fd = os.open(
                component,
                directory_flags,
                dir_fd=directory_fd,
            )
            if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                os.close(child_fd)
                raise OSError("store path component is not a directory")
            os.close(directory_fd)
            directory_fd = child_fd
        member_fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        identity = os.fstat(member_fd)
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise OSError("store member is not a single-link regular file")
        stream = os.fdopen(member_fd, "rb")
        member_fd = None
        return stream
    finally:
        if member_fd is not None:
            os.close(member_fd)
        os.close(directory_fd)


def _read_store_json(root_fd: int, parts: tuple[str, ...]) -> dict:
    """Read one bounded JSON metadata member through anchored descriptors."""
    try:
        with _open_regular_store_member(root_fd, parts) as handle:
            size = os.fstat(handle.fileno()).st_size
            if size > _MAX_STORE_METADATA_BYTES:
                raise ValueError("store metadata is too large")
            payload = json.load(handle)
    except (
        json.JSONDecodeError,
        OSError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise _UnreadableImageStore("store metadata is malformed") from exc
    if not isinstance(payload, dict):
        raise _UnreadableImageStore("store metadata is malformed")
    return payload


def _is_ngff_image_group(
    root_fd: int,
    parts: tuple[str, ...],
    *,
    require_image_label: bool,
) -> bool:
    """Return whether a declared path is a Zarr v3 NGFF multiscale group."""
    from phenotypic.sdk_ import ngff_

    try:
        payload = _read_store_json(root_fd, (*parts, ngff_.STORE_ROOT_JSON))
        if payload.get("zarr_format") != 3 or payload.get("node_type") != "group":
            return False
        attributes = payload.get("attributes")
        ome = attributes.get("ome") if isinstance(attributes, dict) else None
        if not isinstance(ome, dict) or ome.get("version") != ngff_.NGFF_VERSION:
            return False
        if require_image_label and not isinstance(ome.get("image-label"), dict):
            return False
        multiscales = ome.get("multiscales")
        if not isinstance(multiscales, list) or not multiscales:
            return False
        first = multiscales[0]
        datasets = first.get("datasets") if isinstance(first, dict) else None
        if not isinstance(datasets, list) or not datasets:
            return False
        first_dataset = datasets[0]
        dataset = first_dataset.get("path") if isinstance(first_dataset, dict) else None
        dataset_parts = _store_member_parts(dataset) if isinstance(dataset, str) else ()
        if not dataset_parts:
            return False
        array = _read_store_json(
            root_fd,
            (*parts, *dataset_parts, ngff_.STORE_ROOT_JSON),
        )
        return array.get("zarr_format") == 3 and array.get("node_type") == "array"
    except (_UnreadableImageStore, ValueError):
        return False


def _image_store_prefixes(root_fd: int) -> frozenset[tuple[str, ...]]:
    """Return validated declared image paths, excluding reserved namespaces."""
    from phenotypic.sdk_ import ngff_

    root = _read_store_json(root_fd, (ngff_.STORE_ROOT_JSON,))
    try:
        attributes = root["attributes"]
        block = attributes[ngff_.PhenotypicAttr.ROOT]
        if (
            not isinstance(block, dict)
            or block.get(ngff_.PhenotypicAttr.STORE_SCHEMA_VERSION)
            != ngff_.STORE_SCHEMA_VERSION
        ):
            raise TypeError("store schema is not readable")
        series = block[ngff_.PhenotypicAttr.SERIES]
        labels = block.get(ngff_.PhenotypicAttr.LABELS, {})
        if not isinstance(series, dict) or not isinstance(labels, dict):
            raise TypeError("image-series maps are malformed")
    except (KeyError, TypeError) as exc:
        raise _UnreadableImageStore(str(exc)) from exc

    reserved = {
        ngff_.OME_GROUP,
        ngff_.STORE_ROOT_JSON,
        ngff_.TABLES_GROUP,
    }
    prefixes: set[tuple[str, ...]] = set()
    for declarations, require_image_label in (
        (series, False),
        (labels, True),
    ):
        for member in declarations.values():
            if not isinstance(member, str):
                continue
            try:
                parts = _store_member_parts(member)
            except ValueError:
                continue
            if any(part in reserved or part.startswith(".") for part in parts):
                continue
            if _is_ngff_image_group(
                root_fd,
                parts,
                require_image_label=require_image_label,
            ):
                prefixes.add(parts)
    return frozenset(prefixes)


def _is_authorized_image_member(
    parts: tuple[str, ...],
    prefixes: frozenset[tuple[str, ...]],
) -> bool:
    """Return whether a member is required by a validated image declaration."""
    from phenotypic.sdk_ import ngff_

    if ngff_.TABLES_GROUP in parts:
        return False
    if parts == (ngff_.STORE_ROOT_JSON,) or parts[0] == ngff_.OME_GROUP:
        return True
    for prefix in prefixes:
        if parts[: len(prefix)] == prefix:
            return True
        ancestor = parts[:-1]
        if (
            parts[-1] == ngff_.STORE_ROOT_JSON
            and ancestor
            and len(ancestor) < len(prefix)
            and prefix[: len(ancestor)] == ancestor
        ):
            return True
    return False


def register(
    app: dash.Dash,
    api: BrowsePreparationApi | SandboxRoot,
) -> None:
    """Mount revision-addressed assets and legacy Timeline-compatible DZI URLs."""
    if isinstance(api, SandboxRoot):
        sandbox = api
        cache = BrowseCache.for_sandbox(sandbox.root)
        api = BrowsePreparationApi(
            sandbox=sandbox,
            cache=cache,
            manager=BrowsePreparationManager(
                cache,
                normalize=lambda source,
                destination: _source_render.normalize_to_png(
                    source,
                    destination,
                ),
            ),
        )
    asset_bp = Blueprint("browse_assets", __name__, url_prefix="/assets")
    legacy_bp = Blueprint(
        "browse_tiles", __name__, url_prefix=BROWSE_TILES_PREFIX
    )

    def _revision(token: str, key: str | None = None):
        if not _TOKEN_RE.fullmatch(token):
            raise FileNotFoundError
        if key is not None and not _REVISION_RE.fullmatch(key):
            raise FileNotFoundError
        return resolve_revision(api.sandbox, token, key)

    def _selection_identity() -> tuple[str, int]:
        client_id = request.args.get("client_id", "asset-route")
        if not (0 < len(client_id) <= 128):
            client_id = "asset-route"
        try:
            generation = max(0, int(request.args.get("generation", "0")))
        except ValueError:
            generation = 0
        return client_id, generation

    def _prepare(token: str, key: str | None = None):
        lookup_started = time.perf_counter()
        revision = _revision(token, key)
        entry = api.cache.entry(revision)
        cache_hit = entry.dzi_ready
        lookup_ms = (time.perf_counter() - lookup_started) * 1000
        client_id, generation = _selection_identity()
        handle = api.select(client_id, generation, revision)
        return revision, entry, handle, cache_hit, lookup_ms

    def _published_store_revision(token: str, key: str):
        """Resolve one immutable PhenoTypic store without a recursive rescan."""
        if not _TOKEN_RE.fullmatch(token) or not _REVISION_RE.fullmatch(key):
            raise FileNotFoundError
        try:
            relative = _source_render.decode_token(token)
            source = api.sandbox.resolve(relative)
        except Exception as exc:  # noqa: BLE001 - fixed route error
            raise FileNotFoundError from exc
        if not is_source_store(source):
            raise FileNotFoundError
        try:
            publication = store_publication_token(source)
        except OSError as exc:
            raise SourceProbeError("unstable store root") from exc
        if publication is None:
            raise _MutableStoreUnsupported
        revision = resolve_revision(api.sandbox, token, key)
        if revision.store_revision != publication:
            raise SourceProbeError("stale source revision")
        return revision

    @asset_bp.get("/<token>/<revision>/zarr/<path:member>")
    def zarr_member(token: str, revision: str, member: str) -> Response:
        """Serve one byte member from an immutable process-store generation."""
        try:
            parts = _store_member_parts(member)
        except (TypeError, UnicodeError, ValueError):
            return _error("invalid store member", 404)
        try:
            source = _published_store_revision(token, revision)
        except _MutableStoreUnsupported:
            return _error(
                "mutable third-party Zarr store has no root-last publication "
                "token; Browse refuses an unsafe multi-request image view",
                422,
            )
        except FileNotFoundError:
            return _error("invalid or unknown image store", 404)
        except SourceProbeError:
            return _error("source image changed", 409)

        try:
            root_fd = _open_store_root(source.source_path)
        except _UnsafeStoreAccess as exc:
            return _error(str(exc), 422)
        except (OSError, RuntimeError, TypeError, ValueError):
            return _error("store member not found", 404)
        try:
            try:
                publication = store_publication_token(
                    source.source_path,
                    root_dir_fd=root_fd,
                )
            except OSError:
                return _error("source image changed", 409)
            if publication != source.store_revision:
                return _error("source image changed", 409)
            try:
                image_prefixes = _image_store_prefixes(root_fd)
            except _UnreadableImageStore as exc:
                return _error(str(exc), 422)
            if not _is_authorized_image_member(parts, image_prefixes):
                return _error("invalid store member", 404)
            try:
                handle = _open_regular_store_member(root_fd, parts)
            except (OSError, RuntimeError, TypeError, ValueError):
                return _error("store member not found", 404)
            try:
                try:
                    publication = store_publication_token(
                        source.source_path,
                        root_dir_fd=root_fd,
                    )
                except OSError:
                    handle.close()
                    return _error("source image changed", 409)
                if publication != source.store_revision:
                    handle.close()
                    return _error("source image changed", 409)
                size = os.fstat(handle.fileno()).st_size
                response = send_file(
                    handle,
                    conditional=False,
                    download_name=parts[-1],
                )
                response.content_length = size
                response.make_conditional(
                    request,
                    accept_ranges=True,
                    complete_length=size,
                )
                response.headers["Cache-Control"] = _IMMUTABLE_CACHE
                response.call_on_close(handle.close)
                return response
            except BaseException:
                handle.close()
                raise
        except (OSError, RuntimeError, TypeError, UnicodeError, ValueError):
            return _error("store member not found", 404)
        finally:
            os.close(root_fd)

    @asset_bp.get("/<token>/<revision>/zarr/")
    def empty_zarr_member(token: str, revision: str) -> Response:
        """Reject an empty member before Dash's catch-all can serve HTML."""
        del token, revision
        return _error("invalid store member", 404)

    @asset_bp.get("/<token>/<revision>/preview.png")
    def preview(token: str, revision: str) -> Response:
        try:
            source, entry, handle, _cache_hit, lookup_ms = _prepare(
                token, revision
            )
        except FileNotFoundError:
            return _error("invalid or unknown image", 404)
        except SourceProbeError:
            return _error("source image changed", 409)
        wait_started = time.perf_counter()
        handle.preview_ready.wait(timeout=120.0)
        wait_ms = (time.perf_counter() - wait_started) * 1000
        if not source.matches_disk():
            return _error("source image changed", 409)
        if not entry.preview_ready:
            snapshot = handle.snapshot()
            status = 422 if snapshot.phase == "failed" else 500
            return _error("source preview unavailable", status)
        response = send_from_directory(
            entry.root,
            entry.preview.name,
            mimetype="image/png",
        )
        snapshot = handle.snapshot()
        return _asset_headers(
            response,
            lookup_ms,
            wait_ms,
            normalization_ms=snapshot.normalization_ms,
            dzi_ms=snapshot.dzi_ms,
        )

    @asset_bp.get("/<token>/<revision>/preview-if-ready.png")
    def preview_if_ready(token: str, revision: str) -> Response:
        started = time.perf_counter()
        try:
            source = _revision(token, revision)
        except FileNotFoundError:
            return _error("invalid or unknown image", 404)
        except SourceProbeError:
            return _error("source image changed", 409)
        entry = api.cache.entry(source)
        lookup_ms = (time.perf_counter() - started) * 1000
        if not entry.preview_ready:
            return _error("preview not prepared", 404)
        response = send_from_directory(
            entry.root,
            entry.preview.name,
            mimetype="image/png",
        )
        return _asset_headers(response, lookup_ms, 0.0)

    @asset_bp.get("/<token>/<revision>/image.dzi")
    def manifest(token: str, revision: str) -> Response:
        try:
            source, entry, handle, _cache_hit, lookup_ms = _prepare(
                token, revision
            )
        except FileNotFoundError:
            return _error("invalid or unknown image", 404)
        except SourceProbeError:
            return _error("source image changed", 409)
        wait_started = time.perf_counter()
        handle.complete.wait(timeout=600.0)
        wait_ms = (time.perf_counter() - wait_started) * 1000
        if not source.matches_disk():
            return _error("source image changed", 409)
        if not entry.dzi_ready:
            snapshot = handle.snapshot()
            status = 409 if snapshot.error_code == "source_changed" else 422
            return _error("source image cannot be prepared", status)
        response = send_from_directory(
            entry.dzi_dir,
            entry.dzi_manifest.name,
            mimetype="application/xml",
        )
        snapshot = handle.snapshot()
        return _asset_headers(
            response,
            lookup_ms,
            wait_ms,
            normalization_ms=snapshot.normalization_ms,
            dzi_ms=snapshot.dzi_ms,
        )

    @asset_bp.get("/<token>/<revision>/image_files/<int:level>/<filename>")
    def tile(token: str, revision: str, level: int, filename: str) -> Response:
        try:
            source = _revision(token, revision)
        except FileNotFoundError:
            return _error("invalid or unknown image", 404)
        except SourceProbeError:
            return _error("source image changed", 409)
        if secure_filename(
            filename
        ) != filename or not _TILE_NAME_RE.fullmatch(filename):
            return _error("invalid tile filename", 404)
        entry = api.cache.entry(source)
        tile_dir = entry.dzi_dir / "normalized_files" / str(level)
        if not entry.dzi_ready or not tile_dir.is_dir():
            return _error("tile cache missing", 404)
        response = send_from_directory(
            tile_dir, filename, mimetype="image/png"
        )
        response.headers["Cache-Control"] = _IMMUTABLE_CACHE
        return response

    @legacy_bp.get("/<token>.dzi")
    def legacy_manifest(token: str) -> Response:
        try:
            _source, entry, handle, _cache_hit, lookup_ms = _prepare(token)
        except (FileNotFoundError, SourceProbeError):
            return _error("invalid or unknown image", 404)
        wait_started = time.perf_counter()
        handle.complete.wait(timeout=600.0)
        wait_ms = (time.perf_counter() - wait_started) * 1000
        if not entry.dzi_ready:
            return _error(
                "source image cannot be rendered on this platform", 422
            )
        response = send_from_directory(
            entry.dzi_dir,
            entry.dzi_manifest.name,
            mimetype="application/xml",
        )
        response.headers["Cache-Control"] = "private, no-cache"
        snapshot = handle.snapshot()
        response.headers["Server-Timing"] = _server_timing(
            lookup_ms,
            wait_ms,
            normalization_ms=snapshot.normalization_ms,
            dzi_ms=snapshot.dzi_ms,
        )
        return response

    @legacy_bp.get("/<token>_files/<int:level>/<filename>")
    def legacy_tile(token: str, level: int, filename: str) -> Response:
        try:
            source = _revision(token)
        except (FileNotFoundError, SourceProbeError):
            return _error("invalid or unknown image", 404)
        if secure_filename(
            filename
        ) != filename or not _TILE_NAME_RE.fullmatch(filename):
            return _error("invalid tile filename", 404)
        entry = api.cache.entry(source)
        tile_dir = entry.dzi_dir / "normalized_files" / str(level)
        if not entry.dzi_ready or not tile_dir.is_dir():
            return _error("tile cache missing", 404)
        return send_from_directory(tile_dir, filename, mimetype="image/png")

    app.server.register_blueprint(asset_bp)
    app.server.register_blueprint(legacy_bp)


def _asset_headers(
    response: Response,
    lookup_ms: float,
    wait_ms: float,
    *,
    normalization_ms: float = 0.0,
    dzi_ms: float = 0.0,
) -> Response:
    response.headers["Cache-Control"] = _IMMUTABLE_CACHE
    response.headers["Server-Timing"] = _server_timing(
        lookup_ms,
        wait_ms,
        normalization_ms=normalization_ms,
        dzi_ms=dzi_ms,
    )
    return response


def _server_timing(
    lookup_ms: float,
    wait_ms: float,
    *,
    normalization_ms: float = 0.0,
    dzi_ms: float = 0.0,
) -> str:
    return (
        f"cache;dur={lookup_ms:.2f}, queue;dur={wait_ms:.2f}, "
        f"normalize;dur={normalization_ms:.2f}, dzi;dur={dzi_ms:.2f}"
    )


def _error(message: str, status: int) -> Response:
    response = jsonify({"error": message})
    response.status_code = status
    return response
