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


def _wait_ready(ov, key, timeout: float = 5.0) -> None:
    """Block until ``key``'s overlay future has resolved (test helper)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not ov.overlay_ready(key):
        time.sleep(0.02)
    assert ov.overlay_ready(key)


def test_base_pipelines_registry_stays_bounded(tmp_path: Path) -> None:
    # _BASE_PIPELINES memoizes one base pipeline per distinct run path. In a
    # long-lived hub process each opened run would pin its base forever, so the
    # registry must be a capped LRU (like _PENDING / OverlayCache) — never grow
    # without bound. Drive read_base_pipeline across far more than the cap worth
    # of distinct run dirs (each has no tuning_spec.json → memoizes None, which
    # still occupies a slot) and assert the cap holds.
    from phenotypic.gui.tune import _curate_overlays as ov
    from phenotypic.gui.tune._curate_overlays import (
        _BASE_PIPELINES,
        _BASE_PIPELINES_CAP,
    )
    from phenotypic.gui.tune._run_root import TuneRunRoot

    _BASE_PIPELINES.clear()
    n = _BASE_PIPELINES_CAP * 3
    for i in range(n):
        run_dir = tmp_path / f"run_{i}"
        run_dir.mkdir()
        root = TuneRunRoot(
            path=run_dir,
            trials_path=None,
            storage_url=None,
            study_name="tune",
            directions=None,
            images_dir=None,
            best_pipeline_path=run_dir / "best_pipeline.json",
        )
        ov.read_base_pipeline(root)

    assert len(_BASE_PIPELINES) <= _BASE_PIPELINES_CAP, (
        f"_BASE_PIPELINES grew to {len(_BASE_PIPELINES)} > cap {_BASE_PIPELINES_CAP}"
    )
    _BASE_PIPELINES.clear()


def test_cache_key_for_strips_session_namespace() -> None:
    # The per-tab PendingKey is (session, trial, plate, mode); the process-wide
    # OverlayCache key drops the session (the render is identical regardless of
    # tab). request_overlay (submit) and the poll self-heal (peek) MUST agree on
    # this projection, so it is single-sourced here.
    from phenotypic.gui.tune import _curate_overlays as ov

    assert ov.cache_key_for(("sessABC", 3, "plate.tif", "candidate")) == (
        3,
        "plate.tif",
        "candidate",
    )
    diff_key = ov.difference_key("sessABC", 1, 2, "plate.tif")
    # The cache key is exactly what request_overlay would submit under.
    assert ov.cache_key_for(diff_key) == (1, "plate.tif|2", "difference")


def test_overlay_cache_peek_is_non_consuming(tmp_path: Path) -> None:
    # peek() returns the cached array WITHOUT consuming a future (the self-heal
    # read the poll falls back to). It must be idempotent — the array lives in
    # the OverlayCache independent of the _PENDING future registry.
    from phenotypic.gui.tune._overlays import OverlayCache

    cache = OverlayCache(tmp_path / "peek_cache", capacity=4)
    key = (0, "plate.tif", "candidate")
    assert cache.peek(key) is None  # nothing rendered yet

    rendered = cache.get_or_render(key, lambda: np.full((3, 3, 3), 5, np.uint8))
    first = cache.peek(key)
    second = cache.peek(key)
    assert first is not None and second is not None
    assert np.array_equal(first, rendered)
    assert np.array_equal(first, second)  # repeatable, non-destructive


def test_poll_self_heals_from_cache_after_sibling_resubmit_drops_future(
    tmp_path: Path,
) -> None:
    # B4 regression: pin trial A (slot A renders), then pin trial B. Pinning B
    # re-submits the batch, whose stale-drop loop drops A's already-resolved
    # future from _PENDING (same session+plate). Pre-fix, the poll then returned
    # no_update for slot A forever → a permanent "rendering…" spinner only a full
    # reload cleared. The rendered A array still lives in the OverlayCache, so the
    # poll must self-heal: peek the cache and render the figure.
    from phenotypic.gui.tune import _curate_overlays as ov
    from phenotypic.gui.tune._callbacks import _poll_curate_overlays
    from phenotypic.gui.tune._overlays import get_overlay_cache
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tools_ import trials_parquet_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    # A discoverable (legacy parquet) run so the poll resolves the SAME per-run
    # OverlayCache singleton the submit used.
    run_dir = tmp_path / "selfheal_run"
    run_dir.mkdir()
    parquet = trials_parquet_path(run_dir)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(
        trials=[
            Trial(number=0, params={"0.sigma": 1.0}, score=0.3, terms={}, n_images=2),
            Trial(number=1, params={"0.sigma": 2.0}, score=0.6, terms={}, n_images=2),
        ]
    ).to_parquet(parquet)
    TuneRunRoot.discover(run_dir)  # precondition: the run dir is discoverable
    run_root_data = {"path": str(run_dir)}
    cache = get_overlay_cache(run_dir)

    session, plate = "sess-b4", "plate.tif"
    key_a = ov.candidate_key(session, 0, plate)
    key_b = ov.candidate_key(session, 1, plate)

    # 1) Pin A: submit + let it resolve.
    ov.request_overlay(cache, key_a, lambda: np.full((4, 4, 3), 10, np.uint8))
    _wait_ready(ov, key_a)

    # 2) Pin B: submitting B drops A's resolved future from _PENDING (the
    #    stale-drop on same session+plate) — exactly the wedge.
    ov.request_overlay(cache, key_b, lambda: np.full((4, 4, 3), 20, np.uint8))
    _wait_ready(ov, key_b)
    from phenotypic.gui.tune._curate_overlays import _PENDING

    assert key_a not in _PENDING  # A's future was dropped by B's submit
    assert not ov.overlay_ready(key_a)  # so tier-1 (take) can't resolve it

    # 3) The poll self-heals slot A from the cache (peek), not no_update.
    fig_a, fig_b, _fig_diff = _poll_curate_overlays(
        ov,
        pinned={"a": 0, "b": 1},
        plate=plate,
        mode="side",
        session_id=session,
        run_root_data=run_root_data,
    )
    from dash import no_update

    assert fig_a is not no_update, "slot A wedged on the spinner (no self-heal)"
    assert fig_a.data[0].type == "image"  # the real overlay figure
    # Slot B resolves through the normal take path.
    assert fig_b is not no_update
    assert fig_b.data[0].type == "image"

    ov.clear_pending_for_session(session)


def test_poll_returns_no_update_when_nothing_cached_or_pending(tmp_path: Path) -> None:
    # The self-heal must not over-reach: with no pending future AND no cached
    # array, the poll keeps no_update (the render is genuinely still in flight or
    # never submitted) rather than fabricating a figure.
    from phenotypic.gui.tune import _curate_overlays as ov
    from phenotypic.gui.tune._callbacks import _poll_curate_overlays
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tools_ import trials_parquet_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    parquet = trials_parquet_path(run_dir)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(
        trials=[Trial(number=0, params={}, score=0.1, terms={}, n_images=1)]
    ).to_parquet(parquet)
    TuneRunRoot.discover(run_dir)

    from dash import no_update

    fig_a, fig_b, fig_diff = _poll_curate_overlays(
        ov,
        pinned={"a": 0, "b": None},
        plate="never_submitted.tif",
        mode="side",
        session_id="sess-empty",
        run_root_data={"path": str(run_dir)},
    )
    assert fig_a is no_update
    assert fig_b is no_update  # b not pinned → key is None → no_update
    assert fig_diff is no_update


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
