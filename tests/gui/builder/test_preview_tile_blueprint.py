"""Preview tile blueprint: stage HDF layer -> DZI; reject bad components."""
import numpy as np
from phenotypic import Image
from phenotypic.gui.builder._app import create_app
from phenotypic.gui.builder import _preview_cache as pc


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
