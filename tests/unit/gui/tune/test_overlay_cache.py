"""Unit tests for the overlay disk-LRU cache + background render (B-i, task B3).

``OverlayCache`` is the worker the Curate Dash surface (B-ii) submits overlay
renders to: it runs ``render_fn`` on a background thread, memoizes the result
array to disk as ``.npy``, serves a second request for the same key from the
cache (no re-render), and evicts the least-recently-used entry past ``capacity``.
The LRU ordering is guarded by a ``threading.RLock`` because Werkzeug serves
Dash callbacks from multiple threads.
"""
from __future__ import annotations

import threading

import numpy as np

from phenotypic.gui._config import THREAD_NAME_PREFIX
from phenotypic.gui.tune._overlays import OverlayCache


class _CountingRender:
    """A render stub that records how many times it was invoked, and on which thread."""

    def __init__(self, fill: int = 7) -> None:
        self.calls = 0
        self.thread_names: list[str] = []
        self._fill = fill
        self._lock = threading.Lock()

    def __call__(self) -> np.ndarray:
        with self._lock:
            self.calls += 1
            self.thread_names.append(threading.current_thread().name)
        return np.full((4, 4, 3), self._fill, dtype=np.uint8)


def test_get_or_render_renders_once_then_hits_cache(tmp_path) -> None:
    cache = OverlayCache(tmp_path, capacity=64)
    render = _CountingRender()
    key = (3, "plateA", "candidate")

    first = cache.get_or_render(key, render)
    second = cache.get_or_render(key, render)

    assert render.calls == 1  # second request is a cache hit, no re-render
    assert np.array_equal(first, second)
    assert first.shape == (4, 4, 3)


def test_get_or_render_runs_on_background_thread(tmp_path) -> None:
    cache = OverlayCache(tmp_path, capacity=64)
    render = _CountingRender()
    cache.get_or_render((1, "p", "candidate"), render)
    # The render ran on a pool worker named with the tune-overlay prefix, not
    # the calling thread.
    assert render.thread_names
    assert all(
        name.startswith(f"{THREAD_NAME_PREFIX}-overlay")
        for name in render.thread_names
    )


def test_lru_evicts_oldest_past_capacity(tmp_path) -> None:
    cache = OverlayCache(tmp_path, capacity=2)
    r0, r1, r2 = _CountingRender(0), _CountingRender(1), _CountingRender(2)
    k0, k1, k2 = (0, "p", "m"), (1, "p", "m"), (2, "p", "m")

    cache.get_or_render(k0, r0)
    cache.get_or_render(k1, r1)
    cache.get_or_render(k2, r2)  # over capacity -> k0 (oldest) evicted

    # k1 and k2 are still hot: no re-render.
    cache.get_or_render(k1, r1)
    cache.get_or_render(k2, r2)
    assert r1.calls == 1
    assert r2.calls == 1

    # k0 was evicted: re-accessing it re-renders.
    cache.get_or_render(k0, r0)
    assert r0.calls == 2


def test_lru_touch_keeps_recently_used_alive(tmp_path) -> None:
    cache = OverlayCache(tmp_path, capacity=2)
    r0, r1, r2 = _CountingRender(0), _CountingRender(1), _CountingRender(2)
    k0, k1, k2 = (0, "p", "m"), (1, "p", "m"), (2, "p", "m")

    cache.get_or_render(k0, r0)
    cache.get_or_render(k1, r1)
    cache.get_or_render(k0, r0)  # touch k0 -> k1 becomes the LRU victim
    cache.get_or_render(k2, r2)  # evicts k1, not k0

    cache.get_or_render(k0, r0)  # still hot
    assert r0.calls == 1
    cache.get_or_render(k1, r1)  # evicted -> re-render
    assert r1.calls == 2


def test_submit_and_poll_readiness(tmp_path) -> None:
    cache = OverlayCache(tmp_path, capacity=4)
    render = _CountingRender()
    key = (5, "plateB", "difference")

    future = cache.submit(key, render)
    result = future.result(timeout=30)  # block until the worker finishes
    assert result.shape == (4, 4, 3)
    assert cache.is_ready(key)
    assert np.array_equal(cache.result(key), result)


def test_disk_persistence_survives_new_cache_instance(tmp_path) -> None:
    key = (9, "plateC", "candidate")
    first_cache = OverlayCache(tmp_path, capacity=4)
    rendered = first_cache.get_or_render(key, _CountingRender(fill=5))

    # A fresh cache over the same dir reuses the persisted .npy — render_fn that
    # would raise is never invoked.
    def _boom() -> np.ndarray:
        raise AssertionError("render_fn must not run on a disk hit")

    reopened = OverlayCache(tmp_path, capacity=4)
    again = reopened.get_or_render(key, _boom)
    assert np.array_equal(rendered, again)
