"""Raw OME-Zarr store bytes, with HTTP Range, for the browser pixel client.

The results viewer's deep-zoom surfaces read store chunks **in the browser**
rather than asking the server to render a DZI pyramid. This blueprint is the
byte source for that: it hands out files from inside a per-image
``*.ome.zarr`` directory, and nothing else.

Three properties are load-bearing and each exists for a measured reason.

**Range is not a nicety.** The store is sharded: a level-0 ``rgb`` shard on a
4000x3000 plate is a single 34.4 MiB file, and zarrita's sharding codec
issues *two* ranged GETs per cold tile -- a suffix read of the shard index,
then the inner chunk. Measured in the phase-0 spike, a cold tile costs
1,049,381 B with Range and 72,090,062 B without: **68.7x**. That is why the
bytes go out through :func:`flask.send_file`, which negotiates Range, rather
than a hand-built ``Response`` over ``read_bytes()``.

``conditional=True`` is passed explicitly, but note it is *already*
``send_file``'s default in Flask 3.1 -- the plan's framing of it as "the
single flag this phase exists to get right" does not hold. Dropping the
keyword changes nothing; the real failure mode is not calling ``send_file``
at all, which is what the Range tests actually pin.

**The readable-root restriction binds the resolved path.** It is enforced in
the shared :func:`~phenotypic.gui._shared.tiles.resolve_within_root`, and the
set it is given is derived per store from the store's own
``attributes.phenotypic`` block -- never a literal ``{rgb, gray,
detect_mat}``. ``_write_store_part`` appends ``"original"`` to
``series_names`` whenever the image carries one, so a hard-coded set makes
the Layers panel offer a series this route would 404. The set is what keeps
``tables/measurements/table.parquet`` -- the authoritative per-object
measurements, which now live *inside* the store -- off the wire.

**Every URL carries a generation token.** ``promote_store`` republishes by
renaming the whole store directory, and this route resolves fresh per request
holding no handle. Without a token a client can combine metadata from promote
*N* with chunks from *N+1*: harmless for a run-store re-promote, a decode
error or plausibly-wrong pixels for a builder preview where re-running a node
changes the extent. Because the token is a path *segment*, a new promote
yields a new base URL and the mix is structurally impossible rather than
merely unlikely.

**What the route still exposes.** The root ``zarr.json`` is mandatory -- the
client bootstraps from it -- and carries ``attributes.phenotypic.metadata``:
the ``protected``, ``public`` and ``imported`` sections plus ``work_id``.
``OME/METADATA.ome.xml`` carries the same ``Metadata_*`` sections. The
narrowing keeps the measurements table off the wire; it does **not** make the
route metadata-free. Combined with the no-authentication assumption of the
documented Open OnDemand recipe (``--host 0.0.0.0``), anything that can reach
the node's port can read a run's image metadata. This is not new -- the DZI
and crop routes already serve pixels the same way -- but it is now written
down.

The error contract is narrow on purpose, because zarrita's fetch store
returns ``undefined`` on **404** and *throws* on every other non-2xx status:

* absent store, unknown chunk, or a Zarr v2 metadata probe -> **404**
  (transient during a promote, and "absent" is what a sparse store means)
* stale generation token -> **409**, never 404 or 410. 404 reads as "chunk
  missing" and gets retried forever, and 410 is heuristically cacheable under
  RFC 9110 -- a cacheable "gone" for a chunk URL behind the documented
  reverse proxy would be poison.
* a store this build cannot decode -> **422** with the store's own message,
  matching what ``crop_colony`` already does. 404 would say "no such image"
  when the truth is a run-wide decode failure, and the two surfaces must
  agree.
"""

from __future__ import annotations

from collections.abc import Mapping
import functools
import json
import logging
import os
from pathlib import Path

import dash
from flask import Blueprint, Response, abort, request, send_file

from phenotypic.gui._config import VIEWER_ZARR_PREFIX
from phenotypic.gui._shared.tiles import (
    StoreUnreadable,
    _readable_block,
    is_safe_path_component,
    is_zarr_v2_metadata_probe,
    resolve_within_root,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.sdk_ import paths_fingerprint
from phenotypic.sdk_ import ngff_

logger = logging.getLogger(__name__)

#: How many (store, generation) pairs to keep resolved. Both
#: :func:`store_generation_token` and :func:`readable_roots_for` run on every
#: chunk request -- thousands per pan -- and each reads and parses the root
#: ``zarr.json``. The cache key costs one ``stat``.
_STORE_METADATA_CACHE_SIZE = 256


def _root_identity(root_json: Path) -> tuple[int, int, int, int]:
    """Identify one generation of a store's root ``zarr.json``.

    Keyed on inode, size and **both** timestamps rather than ``st_mtime_ns``
    alone. A promote writes a fresh root file, so the inode moves even on a
    filesystem whose timestamp resolution is coarser than the gap between two
    republishes -- which shared cluster storage can be. A cache that keyed on
    mtime alone would defeat the very invalidation the token provides.

    Args:
        root_json: Path to the store's root ``zarr.json``.

    Returns:
        A tuple identifying this generation of the file.

    Raises:
        OSError: If the root does not exist -- a promote in flight.
    """
    stat = os.stat(root_json)
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@functools.lru_cache(maxsize=_STORE_METADATA_CACHE_SIZE)
def _generation_token_for(
    root_json: str, identity: tuple[int, int, int, int]
) -> str:
    """Digest one generation of a root ``zarr.json``. Memoized on *identity*."""
    digest = paths_fingerprint([Path(root_json)]).removeprefix("sha256:")[:16]
    identity_token = "-".join(f"{value:x}" for value in identity)
    return f"{digest}-{identity_token}"


def store_generation_token(store: Path) -> str:
    """Return a short opaque token identifying one promote of ``store``.

    Construction inherited from the retired
    ``_tile_routes._store_content_token``, which was the DZI cache's key
    until the Plate surface stopped rendering server-built pyramids: the
    root ``zarr.json``'s content fingerprint **and** its
    ``st_mtime_ns``. Neither
    alone is enough -- a re-promote can reproduce byte-identical metadata
    while the pixels underneath differ, and only the mtime moves; a
    metadata-only change that lands in the same tick moves only the bytes.

    The token deliberately keys on the root ``zarr.json`` **only**. An
    in-place nested-chunk rewrite moves neither the store directory's
    ``st_mtime_ns`` nor the root, so the token does not move and the URL
    stays valid -- which is correct, because the route holds no cache and
    serves the new bytes.

    Args:
        store: Path to a ``*.ome.zarr`` directory.

    Returns:
        A URL-safe token identifying this published generation.

    Raises:
        OSError: If the root ``zarr.json`` does not exist -- the routine
            signal that a promote is in flight.
    """
    root_json = store / ngff_.STORE_ROOT_JSON
    return _generation_token_for(str(root_json), _root_identity(root_json))


@functools.lru_cache(maxsize=_STORE_METADATA_CACHE_SIZE)
def _readable_roots_for(
    store: str, identity: tuple[int, int, int, int]
) -> frozenset[str]:
    """Derive the readable first-path-components. Memoized on *identity*."""
    store_path = Path(store)
    block = _readable_block(store_path)
    series = block.get(ngff_.PhenotypicAttr.SERIES)
    labels = block.get(ngff_.PhenotypicAttr.LABELS, {})
    if not isinstance(series, Mapping) or not isinstance(labels, Mapping):
        raise KeyError("image-series maps are malformed")

    reserved = {
        ngff_.OME_GROUP,
        ngff_.STORE_ROOT_JSON,
        ngff_.TABLES_GROUP,
    }
    roots: set[str] = set()
    for declarations, require_image_label in (
        (series, False),
        (labels, True),
    ):
        for member in declarations.values():
            parts = _declared_store_parts(member)
            if (
                not parts
                or any(part in reserved or part.startswith(".") for part in parts)
                or not _is_ngff_image_group(
                    store_path,
                    parts,
                    require_image_label=require_image_label,
                )
            ):
                continue
            roots.add(parts[0])
    roots.add(ngff_.OME_GROUP)
    return frozenset(roots)


def _declared_store_parts(member: object) -> tuple[str, ...]:
    """Return safe POSIX components for one store-declared image path."""
    if (
        not isinstance(member, str)
        or not member
        or member.startswith("/")
        or "\\" in member
        or "\x00" in member
    ):
        return ()
    parts = tuple(member.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        return ()
    return parts


def _is_ngff_image_group(
    store: Path,
    parts: tuple[str, ...],
    *,
    require_image_label: bool,
) -> bool:
    """Return whether a declared path is a Zarr-v3 NGFF image group."""
    try:
        group = json.loads(
            (store.joinpath(*parts) / ngff_.STORE_ROOT_JSON).read_text(
                encoding="utf-8"
            )
        )
        if group.get("zarr_format") != 3 or group.get("node_type") != "group":
            return False
        attributes = group.get("attributes")
        ome = attributes.get("ome") if isinstance(attributes, Mapping) else None
        if not isinstance(ome, Mapping) or ome.get("version") != ngff_.NGFF_VERSION:
            return False
        if require_image_label and not isinstance(ome.get("image-label"), Mapping):
            return False
        multiscales = ome.get("multiscales")
        if not isinstance(multiscales, list) or not multiscales:
            return False
        first = multiscales[0]
        datasets = first.get("datasets") if isinstance(first, Mapping) else None
        if not isinstance(datasets, list) or not datasets:
            return False
        first_dataset = datasets[0]
        level = first_dataset.get("path") if isinstance(first_dataset, Mapping) else None
        level_parts = _declared_store_parts(level)
        if not level_parts:
            return False
        array = json.loads(
            (
                store.joinpath(*parts, *level_parts)
                / ngff_.STORE_ROOT_JSON
            ).read_text(encoding="utf-8")
        )
        return array.get("zarr_format") == 3 and array.get("node_type") == "array"
    except (AttributeError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


def readable_roots_for(store: Path) -> frozenset[str]:
    """Return the first-path-components a pixel client may read from ``store``.

    Derived from the store's own ``attributes.phenotypic`` block, so a series
    the writer legitimately added (``original``) is readable without editing
    this function -- and ``tables/``, which holds the per-object measurement
    parquet, never is.

    Reuses ``_readable_block`` rather than a raw ``json.loads`` so a store
    written by a newer build raises :class:`StoreUnreadable` here exactly as
    it does in the crop path. That is what keeps Plate and Colony agreeing
    about a store this build cannot decode.

    ``labels`` is resolved through ``phenotypic.labels`` rather than assumed
    to sit under ``rgb``: the key is **optional** (a store with no label
    image omits it entirely) and an rgb-less store puts the label under
    ``gray``.

    Args:
        store: Path to a ``*.ome.zarr`` directory.

    Returns:
        The readable first-component allow-list, including ``"OME"``.

    Raises:
        OSError: If the root ``zarr.json`` does not exist.
        KeyError: If the root exists but carries no ``phenotypic`` block.
        StoreUnreadable: If the store's schema version is not this build's.
    """
    root_json = store / ngff_.STORE_ROOT_JSON
    return _readable_roots_for(str(store), _root_identity(root_json))


def send_generation_file(
    store: Path,
    tail: str,
    token: str,
    *,
    allowed_roots: frozenset[str],
) -> Response:
    """Open and serve one file only if it belongs to ``token``'s generation.

    The file handle is opened before the final token check. If an atomic
    promotion lands before the open, that check rejects the new file; if it
    lands after the check, the already-open handle remains pinned to the old
    inode. This closes the validation-to-open race without holding a lock
    across response streaming.

    Args:
        store: Canonical store path.
        tail: Store-relative file requested by the client.
        token: Generation token carried by the request URL.
        allowed_roots: Store-derived readable root names.

    Returns:
        A conditional response with byte-range support.
    """
    resolved = resolve_within_root(
        store, tail, allowed_roots=allowed_roots
    )
    handle = resolved.open("rb")
    try:
        if store_generation_token(store) != token:
            abort(409)
        size = os.fstat(handle.fileno()).st_size
        response = send_file(
            handle,
            conditional=False,
            download_name=resolved.name,
        )
        response.content_length = size
        response.make_conditional(
            request,
            accept_ranges=True,
            complete_length=size,
        )
        response.call_on_close(handle.close)
        return response
    except BaseException:
        handle.close()
        raise


def zarr_store_url(
    url_prefix: str, dataset: str, stem: str, token: str
) -> str:
    """Build the browser-visible base URL of one store generation.

    The token is a path segment rather than a query parameter so that a
    re-promote yields a *different* base URL. Every relative key the client
    resolves against it -- ``zarr.json``, ``rgb/0/c.0.0.0`` -- therefore
    belongs to one generation by construction, and a torn read across a
    promote cannot be assembled.

    Args:
        url_prefix: Browser-visible mount prefix, with a trailing slash
            (``"/"`` standalone, ``"/results/"`` under the hub, plus any
            reverse-proxy prefix).
        dataset: Dataset name.
        stem: Image stem, without the ``.ome.zarr`` suffix.
        token: Value of :func:`store_generation_token` for this store.

    Returns:
        The store root URL, without a trailing slash.
    """
    prefix = url_prefix.rstrip("/")
    return f"{prefix}{VIEWER_ZARR_PREFIX}/{dataset}/{stem}.ome.zarr/{token}"


def register_zarr_routes(app: dash.Dash, output_root: OutputRoot) -> None:
    """Mount the raw OME-Zarr byte route on ``app.server``.

    Exposes ``GET /zarr/<dataset>/<stem>.ome.zarr/<token>/<path...>``,
    serving one file out of the per-image store with HTTP Range support.

    Args:
        app: The Dash application whose Flask server should be extended.
        output_root: Validated handle on the CLI output directory. Captured
            by closure and used to resolve each image's store directory.
    """
    bp = Blueprint(
        "results_viewer_zarr", __name__, url_prefix=VIEWER_ZARR_PREFIX
    )

    @bp.route("/<dataset>/<stem>.ome.zarr/<token>/<path:tail>")
    def store_bytes(
        dataset: str, stem: str, token: str, tail: str
    ) -> Response:
        """Serve one file from inside a per-image store."""
        if not is_safe_path_component(dataset) or not is_safe_path_component(
            stem
        ):
            abort(400)
        # A v3 store holds none of these, but a zarr client probes for all
        # four beside every ``zarr.json``. They must read as ABSENT (404);
        # the leading-dot rule would otherwise make them 400s, which the
        # client throws on rather than treating as absent.
        if is_zarr_v2_metadata_probe(tail):
            abort(404)

        store = output_root.store_path(dataset, stem)
        if store is None or not store.is_dir():
            abort(404)

        # Both calls read the root ``zarr.json``, which a concurrent promote
        # can rename away between the ``is_dir()`` above and here.
        # Unguarded, the routine promote path yields a 500 where 404 is
        # meant -- and with ``--debug`` plus the documented ``--host
        # 0.0.0.0``, an unhandled exception is the Werkzeug interactive
        # debugger.
        try:
            expected = store_generation_token(store)
            roots = readable_roots_for(store)
        except StoreUnreadable as exc:
            # 422, NOT 404 -- matching what ``crop_colony`` already does. A
            # store this build cannot decode is a run-wide, actionable
            # condition; 404 would tell the user "no such image", which is
            # false and hides it.
            logger.error("Unreadable store for %s/%s: %s", dataset, stem, exc)
            abort(422, description=str(exc))
        except (OSError, KeyError):
            # Root gone (promote in flight) or carrying no ``phenotypic``
            # block. ``require_readable_store`` raises FileNotFoundError,
            # KeyError AND ValueError -- KeyError is NOT an OSError, so it
            # must be named or a store with no block yields a 500.
            abort(404)

        if token != expected:
            abort(409)

        return send_generation_file(
            store,
            tail,
            token,
            allowed_roots=roots,
        )

    app.server.register_blueprint(bp)
    logger.debug(
        "Registered results viewer zarr byte route under %s for root=%s",
        VIEWER_ZARR_PREFIX,
        output_root.root,
    )
