import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic.gui.browse import _source_render as sr


def test_token_round_trip_is_slash_free():
    rel = "plates/batch7/day3/A1_scan.nef"
    token = sr.encode_token(rel)
    assert "/" not in token and "=" not in token
    assert sr.decode_token(token) == rel


def test_cache_base_under_tempdir(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    assert sr.browse_cache_base() == tmp_path / "phenotypic" / "browse"
    assert sr.cache_png_path("tok") == tmp_path / "phenotypic" / "browse" / "tok.png"


def test_normalize_standard_png(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    src = tmp_path / "src.png"
    PILImage.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(src)
    out = sr.normalize_to_png(src, sr.cache_png_path("t1"))
    assert out.exists()
    arr = np.asarray(PILImage.open(out).convert("RGB"))
    assert arr.dtype == np.uint8 and arr.shape == (8, 8, 3)


def test_normalize_is_mtime_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    src = tmp_path / "src.png"
    PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(src)
    out = sr.normalize_to_png(src, sr.cache_png_path("t2"))
    first_mtime = out.stat().st_mtime_ns
    out2 = sr.normalize_to_png(src, sr.cache_png_path("t2"))  # cache hit
    assert out2.stat().st_mtime_ns == first_mtime


def test_raw_unavailable_raises_typed(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    raw = tmp_path / "shot.nef"
    raw.write_bytes(b"not really a raw file")

    def _boom(*a, **k):
        raise ImportError("rawpy not installed")

    monkeypatch.setattr(sr.Image, "imread", _boom)
    with pytest.raises(sr.SourceRenderUnavailable):
        sr.normalize_to_png(raw, sr.cache_png_path("t3"))


def test_wipe_and_init_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    base = sr.browse_cache_base()
    base.mkdir(parents=True)
    (base / "stale.png").write_bytes(b"x")
    sr.init_cache()
    assert base.is_dir()
    assert not (base / "stale.png").exists()
