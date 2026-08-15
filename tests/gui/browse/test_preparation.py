import shutil
import threading
import time
from pathlib import Path

from PIL import Image as PILImage

from phenotypic.gui.browse._cache import BrowseCache, CacheLocation
from phenotypic.gui.browse._preparation import BrowsePreparationManager
from phenotypic.gui.browse._source_probe import probe_source


def _revision(root: Path, name: str):
    source = root / "sources" / name
    source.parent.mkdir(exist_ok=True)
    PILImage.new("RGB", (8, 8), name.removesuffix(".png")).save(source)
    return probe_source(source, sandbox_root=root / "sources")


def _fake_tile(png_path, output_dir, **_kwargs):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / f"{png_path.stem}.dzi"
    manifest.write_text("<Image/>", encoding="utf-8")
    return manifest


def test_duplicate_requests_share_one_normalization(tmp_path):
    cache = BrowseCache(CacheLocation(tmp_path / "cache", "sandbox", True))
    revision = _revision(tmp_path, "red.png")
    calls = 0

    def normalize(source, destination):
        nonlocal calls
        calls += 1
        shutil.copyfile(source, destination)
        return destination

    manager = BrowsePreparationManager(
        cache, normalize=normalize, tile=_fake_tile
    )
    try:
        first = manager.replace_selected("tab-a", 1, revision)
        second = manager.replace_selected("tab-b", 1, revision)
        assert first.complete.wait(5)
        assert second.complete.is_set()
        assert first.snapshot().phase == "ready"
        assert calls == 1
    finally:
        manager.close()


def test_cleared_ready_revision_resets_events_and_rebuilds(tmp_path):
    cache = BrowseCache(CacheLocation(tmp_path / "cache", "sandbox", True))
    revision = _revision(tmp_path, "red.png")
    manager = BrowsePreparationManager(cache, tile=_fake_tile)
    try:
        first = manager.replace_selected("tab", 1, revision)
        assert first.complete.wait(5)
        cache.clear()

        second = manager.replace_selected("tab", 2, revision)
        assert not second.complete.is_set()
        assert not second.dzi_ready.is_set()
        assert second.complete.wait(5)
        assert second.snapshot().phase == "ready"
        assert cache.entry(revision).dzi_ready
    finally:
        manager.close()


def test_preview_request_does_not_generate_dzi(tmp_path):
    cache = BrowseCache(CacheLocation(tmp_path / "cache", "sandbox", True))
    revision = _revision(tmp_path, "red.png")
    tile_calls = 0

    def tile(*args, **kwargs):
        nonlocal tile_calls
        tile_calls += 1
        return _fake_tile(*args, **kwargs)

    manager = BrowsePreparationManager(cache, tile=tile)
    try:
        handle = manager.request_preview(revision)
        assert handle.preview_ready.wait(5)
        assert handle.complete.wait(5)
        assert handle.snapshot().phase == "preview_ready"
        assert tile_calls == 0
        assert not cache.entry(revision).dzi_ready
    finally:
        manager.close()


def test_selected_request_after_preview_completion_is_not_stranded(tmp_path):
    cache = BrowseCache(CacheLocation(tmp_path / "cache", "sandbox", True))
    revision = _revision(tmp_path, "red.png")
    manager = BrowsePreparationManager(cache, tile=_fake_tile)
    try:
        preview = manager.request_preview(revision)
        assert preview.complete.wait(5)
        selected = manager.replace_selected("tab", 1, revision)
        assert selected.complete.wait(5)
        assert selected.snapshot().phase == "ready"
        assert cache.entry(revision).dzi_ready
    finally:
        manager.close()


def test_source_change_reprobes_and_retries_once(tmp_path):
    cache = BrowseCache(CacheLocation(tmp_path / "cache", "sandbox", True))
    revision = _revision(tmp_path, "red.png")
    calls = 0

    def normalize(source, destination):
        nonlocal calls
        calls += 1
        shutil.copyfile(source, destination)
        if calls == 1:
            PILImage.new("RGB", (9, 9), "blue").save(source)
        return destination

    manager = BrowsePreparationManager(
        cache, normalize=normalize, tile=_fake_tile
    )
    try:
        original = manager.replace_selected("tab", 1, revision)
        assert original.complete.wait(5)
        assert original.snapshot().error_code == "source_changed"
        deadline = time.monotonic() + 5
        replacement = None
        while time.monotonic() < deadline:
            replacement = next(
                (
                    snapshot
                    for snapshot in manager.snapshots()
                    if snapshot.cache_key != revision.cache_key
                    and snapshot.phase == "ready"
                ),
                None,
            )
            if replacement is not None:
                break
            time.sleep(0.01)
        assert replacement is not None
        assert calls == 2
    finally:
        manager.close()


def test_selected_overtakes_queued_dataset_work(tmp_path):
    cache = BrowseCache(CacheLocation(tmp_path / "cache", "sandbox", True))
    active = _revision(tmp_path, "red.png")
    queued = _revision(tmp_path, "blue.png")
    selected = _revision(tmp_path, "green.png")
    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def normalize(source, destination):
        order.append(source.name)
        if source.name == "red.png":
            entered.set()
            assert release.wait(5)
        shutil.copyfile(source, destination)
        return destination

    manager = BrowsePreparationManager(
        cache, normalize=normalize, tile=_fake_tile
    )
    try:
        handles = manager.prepare_dataset("tab", 1, [active, queued])
        assert entered.wait(5)
        selected_handle = manager.replace_selected("tab", 1, selected)
        release.set()
        assert selected_handle.complete.wait(5)
        assert handles[1].complete.wait(5)
        assert order == ["red.png", "green.png", "blue.png"]
    finally:
        manager.close()


def test_stop_dataset_cancels_queued_but_preserves_completed_preview(tmp_path):
    cache = BrowseCache(CacheLocation(tmp_path / "cache", "sandbox", True))
    first = _revision(tmp_path, "red.png")
    second = _revision(tmp_path, "blue.png")
    entered = threading.Event()
    release = threading.Event()

    def normalize(source, destination):
        if source.name == "red.png":
            entered.set()
            assert release.wait(5)
        shutil.copyfile(source, destination)
        return destination

    manager = BrowsePreparationManager(
        cache, normalize=normalize, tile=_fake_tile
    )
    try:
        first_handle, second_handle = manager.prepare_dataset(
            "tab", 1, [first, second]
        )
        assert entered.wait(5)
        manager.stop_dataset("tab")
        release.set()
        assert first_handle.complete.wait(5)
        assert second_handle.complete.wait(5)
        assert second_handle.snapshot().phase == "cancelled"
        assert cache.entry(first).preview_ready
    finally:
        manager.close()


def test_speculation_pause_does_not_block_selected(tmp_path):
    cache = BrowseCache(CacheLocation(tmp_path / "cache", "sandbox", True))
    nearby = _revision(tmp_path, "red.png")
    selected = _revision(tmp_path, "blue.png")
    manager = BrowsePreparationManager(cache, tile=_fake_tile)
    try:
        manager.set_speculation_enabled("tab", False)
        near_handle = manager.replace_nearby("tab", 1, [nearby])[0]
        selected_handle = manager.replace_selected("tab", 1, selected)
        assert selected_handle.complete.wait(5)
        assert not near_handle.complete.is_set()
        manager.set_speculation_enabled("tab", True)
        assert near_handle.complete.wait(5)
    finally:
        manager.close()


def test_stale_generation_is_not_rescheduled(tmp_path):
    cache = BrowseCache(CacheLocation(tmp_path / "cache", "sandbox", True))
    current = _revision(tmp_path, "red.png")
    stale = _revision(tmp_path, "blue.png")
    manager = BrowsePreparationManager(cache, tile=_fake_tile)
    try:
        current_handle = manager.replace_selected("tab", 2, current)
        assert current_handle.complete.wait(5)
        stale_handle = manager.replace_selected("tab", 1, stale)
        assert stale_handle.complete.is_set()
        assert stale_handle.snapshot().phase == "cancelled"
        assert not cache.entry(stale).root.exists()
    finally:
        manager.close()
