import numpy as np
import pytest
import tifffile
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


def test_normalize_16bit_tiff_full_scale_no_stretch(monkeypatch, tmp_path):
    # A 16-bit TIFF must downcast on the FIXED dtype range (65535 -> 255),
    # with NO per-image min/max stretch. The discriminating fixture is a
    # *uniform half-scale* image: a fixed-range downcast yields 128, whereas
    # any per-image stretch on a uniform array could not.
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))

    full = tmp_path / "full16.tiff"
    tifffile.imwrite(full, np.full((4, 4, 3), 65535, dtype=np.uint16))
    out_full = np.asarray(
        PILImage.open(sr.normalize_to_png(full, sr.cache_png_path("f16"))).convert(
            "RGB"
        )
    )
    assert out_full.dtype == np.uint8
    assert out_full.min() == 255 and out_full.max() == 255  # 65535 -> 255

    half = tmp_path / "half16.tiff"
    tifffile.imwrite(half, np.full((4, 4, 3), 32768, dtype=np.uint16))
    out_half = np.asarray(
        PILImage.open(sr.normalize_to_png(half, sr.cache_png_path("h16"))).convert(
            "RGB"
        )
    )
    # 32768 / 257 ≈ 127.5 → 128 under skimage's fixed-range img_as_ubyte.
    assert out_half.min() == 128 and out_half.max() == 128


def test_normalize_standard_decode_failure_reraises(monkeypatch, tmp_path):
    # A decode failure on a STANDARD format (here .tiff) must re-raise the
    # original error verbatim — only RAW extensions map to the typed
    # SourceRenderUnavailable.
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    tiff = tmp_path / "broken.tiff"
    tiff.write_bytes(b"not really a tiff")

    def _boom(*a, **k):
        raise ValueError("corrupt tiff")

    monkeypatch.setattr(sr.Image, "imread", _boom)
    with pytest.raises(ValueError, match="corrupt tiff"):
        sr.normalize_to_png(tiff, sr.cache_png_path("bt"))


def test_init_cache_registers_atexit_once(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    # Reset the module's one-shot guard so this test owns the registration.
    monkeypatch.setattr(sr, "_atexit_registered", False)
    calls: list[object] = []
    monkeypatch.setattr(sr.atexit, "register", lambda fn: calls.append(fn))

    sr.init_cache()
    sr.init_cache()  # second call must NOT re-register

    assert calls == [sr.wipe_cache]


def test_wipe_and_init_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    base = sr.browse_cache_base()
    base.mkdir(parents=True)
    (base / "stale.png").write_bytes(b"x")
    sr.init_cache()
    assert base.is_dir()
    assert not (base / "stale.png").exists()
