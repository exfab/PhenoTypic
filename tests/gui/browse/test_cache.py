import os
import time
from pathlib import Path

from PIL import Image as PILImage

from phenotypic.gui.browse._cache import BrowseCache, CacheLocation
from phenotypic.gui.browse._source_probe import probe_source
from phenotypic.gui._config import BROWSE_CACHE_STAGING_GRACE_SECONDS


def _cache(
    tmp_path: Path, *, high: int = 10_000, low: int = 8_000
) -> BrowseCache:
    return BrowseCache(
        CacheLocation(tmp_path / "cache", "sandbox", True),
        high_water_bytes=high,
        low_water_bytes=low,
    )


def _revision(tmp_path: Path, name: str = "plate.png"):
    source = tmp_path / "sources" / name
    source.parent.mkdir(exist_ok=True)
    PILImage.new("RGB", (4, 4), "red").save(source)
    return probe_source(source, sandbox_root=tmp_path / "sources")


def _publish(cache: BrowseCache, revision, payload_bytes: int = 10):
    with cache.entry_lock(revision):
        with cache.staging_entry(revision) as staged:
            staged.preview.write_bytes(b"p" * payload_bytes)
            cache.publish_preview(revision, staged.preview)
            staged.normalized_png.write_bytes(b"n")
            staged.dzi_dir.mkdir()
            staged.dzi_manifest.write_text("<Image/>", encoding="utf-8")
            (staged.dzi_dir / "payload.bin").write_bytes(b"x" * payload_bytes)
            return cache.publish_dzi(revision, staged)


def test_publication_exposes_preview_then_dzi(tmp_path):
    cache = _cache(tmp_path)
    revision = _revision(tmp_path)
    with cache.entry_lock(revision):
        with cache.staging_entry(revision) as staged:
            staged.preview.write_bytes(b"preview")
            entry = cache.publish_preview(revision, staged.preview)
            assert entry.preview_ready
            assert not entry.dzi_ready
            staged.dzi_dir.mkdir()
            staged.dzi_manifest.write_text("<Image/>", encoding="utf-8")
            entry = cache.publish_dzi(revision, staged)
    assert entry.dzi_ready


def test_restart_reuses_completed_entry_and_cleans_staging(tmp_path):
    cache = _cache(tmp_path)
    revision = _revision(tmp_path)
    entry = _publish(cache, revision)
    abandoned = cache.staging_root / "abandoned"
    abandoned.mkdir()
    old = time.time() - BROWSE_CACHE_STAGING_GRACE_SECONDS - 1
    os.utime(abandoned, (old, old))

    reopened = _cache(tmp_path)

    assert reopened.entry(revision).dzi_ready
    assert not abandoned.exists()
    assert entry.root.exists()


def test_clear_respects_protected_entries(tmp_path):
    cache = _cache(tmp_path)
    first = _revision(tmp_path, "one.png")
    second = _revision(tmp_path, "two.png")
    _publish(cache, first)
    _publish(cache, second)

    remaining = cache.clear(protected={first.cache_key})

    assert cache.entry(first).dzi_ready
    assert not cache.entry(second).root.exists()
    assert remaining.entries == 1


def test_prune_reaches_low_water(tmp_path):
    cache = _cache(tmp_path, high=150, low=80)
    first = _revision(tmp_path, "one.png")
    second = _revision(tmp_path, "two.png")
    _publish(cache, first, payload_bytes=100)
    _publish(cache, second, payload_bytes=100)

    usage = cache.prune(protected={second.cache_key})

    assert not cache.entry(first).root.exists()
    assert cache.entry(second).dzi_ready
    # A protected entry can itself exceed the low-water target.
    assert usage.entries == 1
