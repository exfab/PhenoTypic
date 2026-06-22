"""Pure helpers for the node-preview modal callbacks (unit-testable)."""
from __future__ import annotations

from typing import Any, Optional

from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._preview_tiles import preview_dzi_url

_LAYER_LABELS = {
    "rgb": "RGB", "gray": "Gray", "detect_mat": "Detect",
    "objmap": "Objmap", "overlay": "Overlay",
}
_LAYER_ORDER = ("rgb", "gray", "detect_mat", "objmap", "overlay")


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


def build_preview_payload(
    *, session_id: str, state_data: Any, block_id: str, scope_path: list[str],
    image_path: Optional[str], nrows: Any, ncols: Any, url_prefix: str,
) -> dict:
    """Compute the scope, then build layer options + DZI url for one node."""
    from phenotypic.gui.builder._state import state_from_json

    state = state_from_json(state_data)
    manifest = pc.compute_scope(session_id, state, list(scope_path),
                                image_path, nrows, ncols)
    if manifest.get("error"):
        return {"error": manifest["error"], "options": [], "value": None,
                "dzi_url": None, "title": "Preview", "caption": manifest["error"]}

    node = manifest["nodes"].get(block_id)
    if node is None:
        return {"error": "node not previewable", "options": [], "value": None,
                "dzi_url": None, "title": "Preview", "caption": "Node not previewable"}

    available = [c for c in ("rgb", "gray", "detect_mat") if c in node["layers"]]
    if node.get("num_objects", 0) > 0 and "objmap" in node["layers"]:
        available += ["objmap", "overlay"]
    available = [c for c in _LAYER_ORDER if c in available]

    # class_name for the default-channel rule.
    block = _find_block(state.root, block_id)
    default = _default_channel(getattr(block, "class_name", ""), available)
    shash = pc.scope_hash(list(scope_path))
    h, w = node.get("shape", [0, 0])
    options = [{"label": _LAYER_LABELS[c], "value": c} for c in available]
    return {
        "error": None,
        "options": options,
        "value": default,
        "dzi_url": preview_dzi_url(url_prefix, session_id, shash, block_id, default),
        "title": getattr(block, "label", None) or getattr(block, "class_name", "Preview"),
        "caption": f"{w}×{h} · {default}",
    }


def _find_block(scope: Any, block_id: str) -> Any:
    for b in scope.blocks:
        if b.block_id == block_id:
            return b
        if b.nested is not None:
            found = _find_block(b.nested, block_id)
            if found is not None:
                return found
    return None
