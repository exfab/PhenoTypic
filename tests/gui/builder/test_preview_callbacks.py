"""Preview compute delegate: layer options + a Viv source spec for a node."""
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._preview_callbacks import build_preview_payload
from phenotypic.gui.builder._state import (
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
    state_to_json,
)


def _image_edge(src, tgt):
    return Edge(edge_id=_new_block_id(), source_block_id=src, source_port="out",
                target_block_id=tgt, target_port="in", kind="image")


def test_build_payload_lists_available_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    scope = _DagBuilderScope()
    inp = scope.blocks[0]
    det = BlockNode(block_id=_new_block_id(), class_name="OtsuDetector", params={})
    scope.blocks.append(det)
    scope.edges.append(_image_edge(inp.block_id, det.block_id))
    state = _DagBuilderState(root=scope)

    payload = build_preview_payload(
        session_id="sess-preview-01",
        state_data=state_to_json(state),
        block_id=det.block_id,
        scope_path=[],
        image_path=None, nrows=None, ncols=None, url_prefix="/",
    )
    assert payload["error"] is None
    layer_values = {opt["value"] for opt in payload["options"]}
    assert {"rgb", "gray", "detect_mat", "objmap", "overlay"} & layer_values
    assert payload["value"] in layer_values
    spec = payload["source_spec"]
    # The spec crosses to ``setSource`` unmodified, so these are the facade's
    # own key names -- not a second vocabulary built here.
    assert spec["storeUrl"].startswith("/preview-zarr/")
    assert spec["seriesPath"] in {"rgb", "gray", "detect_mat"}
    # The token is a PATH SEGMENT of ``storeUrl``, so a recompute yields a
    # different base URL rather than a reusable one with a stale query.
    assert spec["storeUrl"].endswith(spec["token"])
