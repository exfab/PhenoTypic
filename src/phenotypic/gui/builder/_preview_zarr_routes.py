"""Raw OME-Zarr bytes for the builder's per-node preview pane.

The preview pane reads store chunks **in the browser** through the shared Viv
facade, exactly as the results Plate does. This blueprint is its byte source.

It cannot reuse the results viewer's ``/zarr/`` route: that one resolves a
store through ``OutputRoot.store_path``, and a preview store lives under the
builder's scratch sandbox with no ``OutputRoot`` behind it. **The resolver is
shared; the routes are not** -- both call
:func:`~phenotypic.gui._shared.tiles.resolve_within_root`, and each keeps its
own resolution and guard regime (spec section 7).

**Session scoping here is a capability URL, not authentication.**
:func:`_validate_scope` validates the *shape* of ``session_id``,
``scope_hash`` and ``block_id`` and nothing more -- nothing binds a request to
the session that issued the id. Isolation rests entirely on ``session_id``
being ``uuid.uuid4().hex`` (``builder/_callbacks.py``): 122 bits, carried in
the URL **path**, where it reaches access logs, the Open OnDemand reverse
proxy's logs, browser history and ``Referer``. Spec section 7 records this as
an **accepted** risk (user ruling, 2026-08-26) rather than a mitigated one:
the entropy is adequate and the exposure matches the ``/preview-tiles/`` route
this replaced, so tightening one of the two would diverge them for no
behaviour change. The id is a secret; the cache tree is created ``0o700``
(``_preview_cache.init_cache``) so the filesystem does not hand it out for
free.

The error contract is the results route's, for the same reason: zarrita's
fetch store returns ``undefined`` on **404** and *throws* on every other
non-2xx status.

* absent scope, node, store or chunk, and every Zarr v2 metadata probe -> **404**
* stale generation token -> **409**, never 410 -- 410 is heuristically
  cacheable under RFC 9110, and a cacheable "gone" for a chunk URL behind the
  documented reverse proxy would be poison
* a store this build cannot decode -> **422** with the store's own message
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import dash
from flask import Blueprint, Response, abort, send_file

from phenotypic.gui._shared.tiles import (
    StoreUnreadable,
    is_safe_path_component,
    is_zarr_v2_metadata_probe,
    json_error,
    resolve_within_root,
)
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.results_viewer._zarr_routes import (
    readable_roots_for,
    store_generation_token,
)
from phenotypic.sdk_ import ngff_

logger = logging.getLogger(__name__)

#: URL prefix the preview byte blueprint mounts under.
PREVIEW_ZARR_PREFIX = "/preview-zarr"

#: ``scope_hash`` is a sha1 hexdigest.
_HASH_RE = re.compile(r"^[0-9a-f]{40}$")

__all__ = [
    "PREVIEW_ZARR_PREFIX",
    "preview_zarr_url",
    "register_preview_zarr_routes",
]


def _validate_scope(
    session_id: str, scope_hash: str, block_id: str
) -> Optional[Response]:
    """Shape-validate the session / scope / block triple.

    **This is not authentication.** Nothing here binds the request to a
    session -- see the capability-URL note in the module docstring. It rejects
    path components that could escape the cache tree, and a scope hash that is
    not a sha1 hexdigest, and that is all it claims to do.

    Args:
        session_id: Browser session id from the URL.
        scope_hash: Scope hash from the URL.
        block_id: DAG block id from the URL.

    Returns:
        ``None`` when the triple is well-formed, otherwise a 404 response.
    """
    if (
        is_safe_path_component(session_id)
        and bool(_HASH_RE.match(scope_hash))
        and is_safe_path_component(block_id)
    ):
        return None
    return json_error("invalid preview request", 404)


def _store_for_block(
    session_id: str, scope_hash: str, block_id: str
) -> Optional[Path]:
    """Resolve one node's preview store, or ``None`` if it is not there.

    The manifest's ``nodes`` is a **dict keyed by block_id** -- there is no
    ``"blocks"`` list and no ``"block_id"`` field (``_preview_cache``'s
    ``_build_manifest``). Reading it as a list 404s every request.

    Args:
        session_id: Browser session id (already shape-validated).
        scope_hash: Scope hash (already shape-validated).
        block_id: DAG block id (already shape-validated).

    Returns:
        The store directory, or ``None`` when the scope, the node or the
        store's root ``zarr.json`` is absent.
    """
    manifest = pc.read_manifest_by_hash(session_id, scope_hash)
    if not manifest:
        return None
    node = manifest.get("nodes", {}).get(block_id)
    if node is None:
        return None
    store = pc._scope_path_by_hash(session_id, scope_hash) / node["store"]
    # The ROOT, not the directory: the promote writes it last, so an
    # interrupted write reads as ABSENT rather than as partial -- the same
    # disposition ``_describe_store_node`` gives it when building the manifest.
    return store if (store / ngff_.STORE_ROOT_JSON).is_file() else None


def preview_zarr_url(
    url_prefix: str,
    session_id: str,
    scope_hash: str,
    block_id: str,
    token: str,
) -> str:
    """Build the browser-visible base URL of one node store generation.

    The token is a path **segment**, so a re-run of the node yields a
    different base URL and every key the client resolves against it belongs to
    one generation by construction.

    Args:
        url_prefix: Mount-point prefix the browser sees (``"/"`` standalone,
            ``"/builder/"`` under the hub).
        session_id: Browser session id.
        scope_hash: ``_preview_cache.scope_hash`` of the active scope path.
        block_id: DAG block id of the previewed node.
        token: :func:`store_generation_token` for that node's store.

    Returns:
        The store root URL, without a trailing slash.
    """
    prefix = url_prefix.rstrip("/")
    return (
        f"{prefix}{PREVIEW_ZARR_PREFIX}/{session_id}/{scope_hash}"
        f"/{block_id}/{token}"
    )


def register_preview_zarr_routes(app: dash.Dash) -> None:
    """Mount the preview store byte route on ``app.server``.

    Exposes
    ``GET /preview-zarr/<session_id>/<scope_hash>/<block_id>/<token>/<path...>``,
    serving one file out of a node's preview store with HTTP Range support.

    Args:
        app: The Dash application whose Flask server should be extended.
    """
    bp = Blueprint(
        "builder_preview_zarr", __name__, url_prefix=PREVIEW_ZARR_PREFIX
    )

    @bp.route("/<session_id>/<scope_hash>/<block_id>/<token>/<path:tail>")
    def preview_store_bytes(
        session_id: str, scope_hash: str, block_id: str, token: str, tail: str
    ) -> Response:
        """Serve one file from inside a node's preview store."""
        invalid = _validate_scope(session_id, scope_hash, block_id)
        if invalid is not None:
            return invalid
        # A v3 store holds none of these, but a zarr client probes all four
        # beside every ``zarr.json``. They must read as ABSENT (404); the
        # leading-dot rule would otherwise make them 400s, which the client
        # throws on rather than treating as absent.
        if is_zarr_v2_metadata_probe(tail):
            abort(404)

        store = _store_for_block(session_id, scope_hash, block_id)
        if store is None:
            abort(404)

        # Guarded for the same reason as the results route: a promote renames
        # the store directory, and ``compute_scope`` rewrites a whole scope in
        # place on any parameter edit, so both of these can raise on the
        # ROUTINE path.
        try:
            expected = store_generation_token(store)
            roots = readable_roots_for(store)
        except StoreUnreadable as exc:
            logger.error(
                "Unreadable preview store for %s/%s: %s",
                scope_hash,
                block_id,
                exc,
            )
            abort(422, description=str(exc))
        except (OSError, KeyError):
            # Root gone (a promote in flight, or a scope being recomputed) or
            # carrying no ``phenotypic`` block. ``require_readable_store``
            # raises KeyError as well as FileNotFoundError, and KeyError is
            # NOT an OSError -- unnamed, it would surface as a 500.
            abort(404)

        if token != expected:
            abort(409)

        return send_file(
            resolve_within_root(store, tail, allowed_roots=roots),
            conditional=True,
        )

    app.server.register_blueprint(bp)
    logger.debug(
        "Registered builder preview zarr byte route under %s",
        PREVIEW_ZARR_PREFIX,
    )
