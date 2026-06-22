"""Preview tile blueprint: stage HDF layer -> DZI; reject bad components."""
import numpy as np
import pytest
from phenotypic import Image
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder import _preview_tiles as pt
from phenotypic.gui.builder._app import create_app


def _seed_scope_hdf(session_id, block_id):
    # NOTE: caller must monkeypatch pc.preview_cache_root AND create the app
    # FIRST — create_app runs init_preview_cache() which wipes the cache root.
    # Seeding after app creation keeps the fixture alive.
    sdir = pc.scope_dir(session_id, [])
    img = Image(arr=np.zeros((48, 64, 3), dtype=np.uint8))
    hdf = sdir / "base_00.h5"
    img.save2hdf5(hdf)
    manifest = {"fingerprint": "fp", "scope_key": "", "error": None,
                "nodes": {block_id: {"hdf": "base_00.h5",
                                     "layers": ["rgb", "gray", "detect_mat", "objmap"],
                                     "shape": [48, 64], "num_objects": 0}}}
    pc.write_manifest(session_id, [], manifest)
    return pc.scope_hash([])


def _seed_scope_hdf_no_objmap(session_id, block_id):
    """Seed a scope with an HDF that has no objmap layer (legacy flat layout).

    Uses ``save_intermediate_layers`` with only the non-objmap layers so
    that ``Image.load_layer_hdf5(hdf, "objmap")`` raises ``KeyError``,
    exercising the 404 path for overlay/objmap channels.
    """
    sdir = pc.scope_dir(session_id, [])
    img = Image(arr=np.zeros((48, 64, 3), dtype=np.uint8))
    hdf = sdir / "no_objmap.h5"
    img.save_intermediate_layers(hdf, layers=("rgb", "gray", "detect_mat"))
    manifest = {"fingerprint": "fp", "scope_key": "", "error": None,
                "nodes": {block_id: {"hdf": "no_objmap.h5",
                                     "layers": ["rgb", "gray", "detect_mat"],
                                     "shape": [48, 64], "num_objects": 0}}}
    pc.write_manifest(session_id, [], manifest)
    return pc.scope_hash([])


def test_preview_dzi_served(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    app = create_app(image_root=tmp_path)  # wipes the (empty) tmp cache root
    sid = "previewsess0001"
    blk = "b" * 32
    shash = _seed_scope_hdf(sid, blk)  # seed AFTER app creation survives the wipe
    client = app.server.test_client()
    resp = client.get(f"/preview-tiles/{sid}/{shash}/{blk}/gray.dzi")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "<Image" in body and "deepzoom" in body.lower()


def test_preview_rejects_bad_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    app = create_app(image_root=tmp_path)
    sid = "previewsess0001"
    blk = "b" * 32
    shash = _seed_scope_hdf(sid, blk)
    client = app.server.test_client()
    resp = client.get(f"/preview-tiles/{sid}/{shash}/{blk}/bogus.dzi")
    assert resp.status_code == 404


def test_overlay_on_no_objmap_node_returns_404(tmp_path, monkeypatch):
    """Requesting overlay for a node whose HDF lacks objmap returns 404, not 500."""
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    app = create_app(image_root=tmp_path)
    sid = "previewsess0002"
    blk = "c" * 32
    shash = _seed_scope_hdf_no_objmap(sid, blk)
    client = app.server.test_client()
    resp = client.get(f"/preview-tiles/{sid}/{shash}/{blk}/overlay.dzi")
    assert resp.status_code == 404


def test_stage_channel_png_does_not_publish_partial_final_file(
    tmp_path,
    monkeypatch,
):
    hdf_path = tmp_path / "node.h5"
    hdf_path.write_bytes(b"hdf")
    final_path = tmp_path / "tiles_src" / "blk__gray.png"

    monkeypatch.setattr(
        pt,
        "_channel_to_rgb_uint8",
        lambda *_args: np.zeros((2, 2, 3), dtype=np.uint8),
    )

    class BrokenImage:
        def save(self, path, format):  # noqa: A002
            path.write_bytes(b"partial")
            raise RuntimeError("simulated write failure")

    monkeypatch.setattr(pt.PILImage, "fromarray", lambda *_args, **_kwargs: BrokenImage())

    with pytest.raises(RuntimeError, match="simulated write failure"):
        pt.stage_channel_png(tmp_path, "blk", "gray", hdf_path)

    assert not final_path.exists()
