"""Pure helpers for the node-preview modal callbacks (unit-testable).

The pane renders through the shared Viv facade, so what these helpers hand the
client is a **source spec** -- ``build_source_spec``'s dict, extended with the
layer the radio selected -- not a rendered tile URL.

**The spec is rebuilt on every request, and that is load-bearing.** A preview
store is rewritten *in place* under the same ``scope_hash`` whenever a node's
parameters change (``compute_scope`` wipes and recomputes the scope), so the
generation token moves on exactly the case this pane exists to serve. Cached in
a ``dcc.Store`` and replayed, the token would be stale and the byte route would
answer 409 forever after the first parameter edit.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from phenotypic.gui._shared.tiles import StoreUnreadable
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._preview_zarr_routes import preview_zarr_url
from phenotypic.gui.results_viewer._store_source import build_source_spec
from phenotypic.gui.results_viewer._zarr_routes import store_generation_token
from phenotypic.sdk_ import ngff_

logger = logging.getLogger(__name__)

_LAYER_LABELS = {
    "rgb": "RGB", "gray": "Gray", "detect_mat": "Detect",
    "objmap": "Objmap", "overlay": "Overlay",
}
_LAYER_ORDER = ("rgb", "gray", "detect_mat", "objmap", "overlay")

#: Channels that show the label image. ``overlay`` draws it over the pixel
#: series; ``objmap`` shows it alone, with the image layer hidden.
_LABEL_CHANNELS = frozenset({"objmap", "overlay"})

#: Series the label channels anchor on, most preferred first. The label group
#: itself is not a series and cannot be an image layer, so ``objmap`` and
#: ``overlay`` still open one -- ``detect_mat`` because that is the base the
#: retired PNG overlay composited onto.
_LABEL_CHANNEL_BASE = ("detect_mat", "gray", "rgb")


def _default_channel(class_name: str, available: list[str]) -> str:
    from phenotypic.gui.builder._image_renderer import _registry_info_for

    # Mirror render_node_preview: Enhancer->detect_mat, Detector/Refiner->overlay.
    cat = None
    try:
        info = _registry_info_for(class_name)
        cat = getattr(info, "category", None)
    except Exception:  # noqa: BLE001
        cat = None
    if cat == "Enhancer" and "detect_mat" in available:
        return "detect_mat"
    if cat in {"Detector", "Refiner"} and "overlay" in available:
        return "overlay"
    return available[0] if available else "rgb"


def build_channel_spec(store: Path, base_url: str, channel: str) -> dict:
    """Build the facade source spec for one node store and one channel.

    Reuses :func:`~phenotypic.gui.results_viewer._store_source.build_source_spec`
    rather than assembling a second vocabulary -- it is written at a
    store-path-plus-base-URL signature precisely so this caller exists -- and
    then narrows it to what the layer radio picked:

    * ``rgb`` / ``gray`` / ``detect_mat`` -- that series, no label layer.
    * ``overlay`` -- a pixel series with the label drawn over it.
    * ``objmap`` -- the label alone; ``imageVisible`` is ``False`` and the
      client hides the image layer.

    ``labelPath`` stays whatever the store recorded under
    ``phenotypic.labels.objmap`` and is **never** constructed as
    ``f"{series}/labels/objmap"``: backend section 1.1 forbids hard-coding it,
    and the key is absent entirely on a store with no label image.

    Args:
        store: Path to the node's ``*.ome.zarr`` preview store.
        base_url: Browser-visible base URL of this store generation.
        channel: One of :data:`_LAYER_ORDER`.

    Returns:
        The facade spec, plus the surface's own ``channel`` and
        ``imageVisible`` keys (which the facade ignores).

    Raises:
        OSError: If the store's root ``zarr.json`` does not exist.
        KeyError: If the root exists but carries no ``phenotypic`` block.
        StoreUnreadable: If the store's schema version is not this build's.
        ValueError: If the store declares neither ``rgb`` nor ``gray``.
    """
    spec = build_source_spec(store, base_url)
    series = list(spec["series"])
    if channel in _LABEL_CHANNELS:
        # ``next(...)`` over a preference order rather than a literal: an
        # rgb-less or detect_mat-less store must still show its label.
        spec["seriesPath"] = next(
            (name for name in _LABEL_CHANNEL_BASE if name in series),
            spec["seriesPath"],
        )
    elif channel in series:
        spec["seriesPath"] = channel
        spec["labelPath"] = None
    spec["channel"] = channel
    spec["imageVisible"] = channel != "objmap"
    return spec


def build_preview_payload(
    *, session_id: str, state_data: Any, block_id: str, scope_path: list[str],
    image_path: Optional[str], nrows: Any, ncols: Any, url_prefix: str,
) -> dict:
    """Compute the scope, then build layer options + a source spec for one node."""
    from phenotypic.gui.builder._state import state_from_json

    state = state_from_json(state_data)
    manifest = pc.compute_scope(session_id, state, list(scope_path),
                                image_path, nrows, ncols)
    if manifest.get("error"):
        return {"error": manifest["error"], "options": [], "value": None,
                "source_spec": None, "title": "Preview",
                "caption": manifest["error"]}

    node = manifest["nodes"].get(block_id)
    if node is None:
        return {"error": "node not previewable", "options": [], "value": None,
                "source_spec": None, "title": "Preview",
                "caption": "Node not previewable"}

    available = [c for c in ("rgb", "gray", "detect_mat") if c in node["layers"]]
    if node.get("num_objects", 0) > 0 and "objmap" in node["layers"]:
        available += ["objmap", "overlay"]
    available = [c for c in _LAYER_ORDER if c in available]

    # class_name for the default-channel rule.
    block = _find_block(state.root, block_id)
    default = _default_channel(getattr(block, "class_name", ""), available)
    h, w = node.get("shape", [0, 0])
    options = [{"label": _LAYER_LABELS[c], "value": c} for c in available]
    return {
        "error": None,
        "options": options,
        "value": default,
        "source_spec": resolve_source_spec(
            session_id=session_id, scope_path=list(scope_path),
            block_id=block_id, channel=default, url_prefix=url_prefix,
            manifest=manifest,
        ),
        "title": getattr(block, "label", None) or getattr(block, "class_name", "Preview"),
        "caption": f"{w}×{h} · {default}",
    }


def resolve_source_spec(
    *, session_id: str, scope_path: list[str], block_id: str, channel: str,
    url_prefix: str, manifest: Optional[dict] = None,
) -> Optional[dict]:
    """Resolve one node's current source spec, or ``None`` if unreadable.

    Called on every layer switch as well as on open, so the generation token
    is always read from the store as it is **now** -- see the module
    docstring.

    Args:
        session_id: Browser session id.
        scope_path: Active scope path (list of container block ids).
        block_id: DAG block id of the previewed node.
        channel: Layer the radio selected.
        url_prefix: The builder's mount-point prefix.
        manifest: Already-read manifest, when the caller has one.

    Returns:
        The facade spec, or ``None`` when the scope, the node or the store is
        absent or this build cannot decode it.
    """
    if manifest is None:
        manifest = pc.read_manifest(session_id, list(scope_path)) or {}
    node = manifest.get("nodes", {}).get(block_id)
    if node is None:
        return None
    shash = pc.scope_hash(list(scope_path))
    store = pc._scope_path_by_hash(session_id, shash) / node["store"]
    if not (store / ngff_.STORE_ROOT_JSON).is_file():
        return None
    try:
        token = store_generation_token(store)
        base_url = preview_zarr_url(
            url_prefix, session_id, shash, block_id, token
        )
        return build_channel_spec(store, base_url, channel)
    except (OSError, KeyError, StoreUnreadable, ValueError):
        # A scope recompute renames the store directory out from under this
        # read, and `KeyError` is not an `OSError`, so both are named. The
        # pane shows its caption; the next callback re-resolves.
        logger.debug("preview source spec unavailable for %s", block_id,
                     exc_info=True)
        return None


def _find_block(scope: Any, block_id: str) -> Any:
    for b in scope.blocks:
        if b.block_id == block_id:
            return b
        if b.nested is not None:
            found = _find_block(b.nested, block_id)
            if found is not None:
                return found
    return None
