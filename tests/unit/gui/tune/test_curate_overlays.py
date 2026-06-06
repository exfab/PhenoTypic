"""Unit tests for the non-blocking Curate overlay orchestration (B4).

The render path must never block a Werkzeug worker on a heavy ``apply``:
:func:`request_overlay` submits to the :class:`OverlayCache` and stashes the
future, returning immediately; :func:`overlay_ready` / :func:`take_overlay` are
what the readiness poll uses to swap the spinner for the real figure. Also pins
the process-wide :func:`get_overlay_cache` singleton + its run-scoped cache dir.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np


def test_get_overlay_cache_is_a_per_run_singleton(tmp_path: Path) -> None:
    from phenotypic.gui.tune._overlays import get_overlay_cache, overlay_cache_dir

    a = get_overlay_cache(tmp_path)
    b = get_overlay_cache(tmp_path)
    assert a is b  # same run → same cache
    # The cache dir lives under the run's .pht-tune-cache machine-state tree.
    assert overlay_cache_dir(tmp_path).is_relative_to(tmp_path)
    assert ".pht-tune-cache" in str(overlay_cache_dir(tmp_path))

    other = get_overlay_cache(tmp_path / "other_run")
    assert other is not a  # distinct run → distinct cache


def test_request_overlay_is_non_blocking(tmp_path: Path) -> None:
    from phenotypic.gui.tune import _curate_overlays as ov
    from phenotypic.gui.tune._overlays import get_overlay_cache

    cache = get_overlay_cache(tmp_path)
    gate = threading.Event()

    def _slow_render() -> np.ndarray:
        gate.wait(timeout=5.0)  # block until the test releases it
        return np.zeros((2, 2, 3), dtype=np.uint8)

    key = ("sess", 0, "plate.tif", "candidate")
    started = time.monotonic()
    ov.request_overlay(cache, key, _slow_render)
    # The submit returned immediately even though the render is still blocked.
    assert time.monotonic() - started < 1.0
    assert not ov.overlay_ready(key)  # render still gated → not ready
    assert ov.take_overlay(key) is None  # not ready → nothing to take

    gate.set()
    # Poll for readiness (the worker resolves the future off-thread).
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not ov.overlay_ready(key):
        time.sleep(0.02)
    assert ov.overlay_ready(key)
    array = ov.take_overlay(key)
    assert array is not None
    assert array.shape == (2, 2, 3)
    # Taking again returns None (the future was dropped on the first take).
    assert ov.take_overlay(key) is None
    ov.clear_pending_for_session("sess")


def test_take_overlay_swallows_render_failure(tmp_path: Path) -> None:
    from phenotypic.gui.tune import _curate_overlays as ov
    from phenotypic.gui.tune._overlays import get_overlay_cache

    cache = get_overlay_cache(tmp_path / "fail_run")

    def _boom() -> np.ndarray:
        raise RuntimeError("bad candidate")

    key = ("sess", 7, "plate.tif", "candidate")
    ov.request_overlay(cache, key, _boom)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not ov.overlay_ready(key):
        time.sleep(0.02)
    assert ov.overlay_ready(key)
    # A failed render resolves to None (the poll then shows an error figure),
    # never re-raising into the callback.
    assert ov.take_overlay(key) is None
    ov.clear_pending_for_session("sess")


def test_pending_dict_stays_bounded(tmp_path: Path) -> None:
    # The module-level _PENDING dict bridges only in-flight renders (the array
    # is already memoized in OverlayCache). A user changing plate / re-pinning /
    # toggling A/B leaves stale keys whose Futures hold rendered arrays — over a
    # session that is an unbounded leak. request_overlay must cap _PENDING and
    # evict LRU so it can never grow without bound.
    from phenotypic.gui.tune import _curate_overlays as ov
    from phenotypic.gui.tune._curate_overlays import _PENDING, _PENDING_CAP
    from phenotypic.gui.tune._overlays import get_overlay_cache

    cache = get_overlay_cache(tmp_path / "bounded_run")

    def _trivial() -> np.ndarray:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    n = _PENDING_CAP * 3
    for i in range(n):
        # Distinct keys (distinct trial), never drained.
        ov.request_overlay(cache, ("sess", i, "plate.tif", "candidate"), _trivial)

    assert len(_PENDING) <= _PENDING_CAP, (
        f"_PENDING grew to {len(_PENDING)} > cap {_PENDING_CAP}"
    )
    ov.clear_pending_for_session("sess")


def test_overlay_figure_wraps_rgb_array() -> None:
    from phenotypic.gui.tune._curate_overlays import overlay_figure

    array = np.zeros((4, 6, 3), dtype=np.uint8)
    fig = overlay_figure(array)
    # One go.Image trace; the y-axis is scale-anchored to x for linked zoom.
    assert len(fig.data) == 1
    assert fig.data[0].type == "image"
    assert fig.layout.yaxis.scaleanchor == "x"


def test_load_plate_grid_rejects_out_of_sandbox(tmp_path: Path) -> None:
    # Defense-in-depth: even though plate names come from iterdir() today, the
    # final load path is re-confined through the sandbox so a future caller
    # sourcing plate_name from less-trusted input can't escape via traversal.
    from phenotypic.gui.shell import SandboxRoot
    from phenotypic.gui.tune._curate_overlays import load_plate_grid

    sandbox = SandboxRoot.from_path(tmp_path)
    # A ``..`` traversal in the plate name escapes the image source / sandbox.
    import pytest

    with pytest.raises(ValueError):
        load_plate_grid(str(tmp_path), "../../etc/passwd", sandbox=sandbox)
