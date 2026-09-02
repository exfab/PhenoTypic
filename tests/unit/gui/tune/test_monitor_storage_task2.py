"""Bounded Tune Monitor storage reads and out-of-band fANOVA refreshes."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace


def _root(tmp_path: Path):
    return SimpleNamespace(
        path=tmp_path,
        trials_path=None,
        storage_url=f"journal://{tmp_path}/journal.log",
        study_name="tune_cost_v1",
        directions=None,
        images_dir=None,
        best_pipeline_path=tmp_path / "best_pipeline.json",
    )


def test_connect_bound_fits_inside_whole_read_bound() -> None:
    from phenotypic.gui.tune._callbacks import (
        _LIVE_CONNECT_TIMEOUT_S,
        _LIVE_READ_TIMEOUT_S,
    )

    assert 2.0 <= _LIVE_CONNECT_TIMEOUT_S < _LIVE_READ_TIMEOUT_S


def test_completed_importance_refresh_is_reused_for_same_trial_count() -> None:
    from phenotypic.gui.tune._callbacks import _ImportanceCache

    cache = _ImportanceCache()
    calls: list[int] = []

    def _compute() -> dict[str, float]:
        calls.append(1)
        return {"threshold": 1.0}

    key = ("journal:///output/journal.log", "tune_cost_v1")
    assert cache.read(key, 4, _compute) is None
    assert cache.wait_for_refresh(10.0)
    assert cache.read(key, 4, _compute) == {"threshold": 1.0}
    assert cache.read(key, 4, _compute) == {"threshold": 1.0}
    assert calls == [1]


def test_only_one_importance_refresh_can_be_in_flight() -> None:
    from phenotypic.gui.tune._callbacks import _ImportanceCache

    cache = _ImportanceCache()
    calls: list[int] = []
    started = threading.Event()
    gate = threading.Event()

    def _slow() -> dict[str, float]:
        calls.append(1)
        started.set()
        gate.wait(timeout=30.0)
        return {"threshold": 1.0}

    key = ("journal:///output/journal.log", "tune_cost_v1")
    try:
        assert cache.read(key, 0, _slow) is None
        assert started.wait(10.0)
        for n_trials in range(1, 6):
            assert cache.read(key, n_trials, _slow) is None
        assert calls == [1]
    finally:
        gate.set()
        cache.wait_for_refresh(10.0)


class _RecordingPool:
    def __init__(self) -> None:
        self.submissions = 0
        self.futures: list[Future] = []

    def submit(self, _callback, *_args) -> Future:
        self.submissions += 1
        future: Future = Future()
        self.futures.append(future)
        return future


def test_importance_refresh_remains_globally_bounded_across_run_switch(
    monkeypatch,
) -> None:
    from phenotypic.gui.tune import _callbacks

    pool = _RecordingPool()
    monkeypatch.setattr(_callbacks, "_LIVE_IMPORTANCE_POOL", pool)
    cache = _callbacks._ImportanceCache()
    first = ("journal:///a/journal.log", "tune_cost_v1")
    second = ("journal:///b/journal.log", "tune_cost_v1")

    assert cache.read(first, 1, lambda: {"a": 1.0}) is None
    assert cache.read(second, 1, lambda: {"b": 1.0}) is None
    assert pool.submissions == 1

    pool.futures[0].set_result({"a": 1.0})
    assert cache.read(second, 1, lambda: {"b": 1.0}) is None
    assert pool.submissions == 2
    pool.futures[1].set_result({"b": 1.0})
    assert cache.read(second, 1, lambda: None) == {"b": 1.0}


def test_failed_importance_refresh_is_absorbed_and_run_switch_clears_value() -> None:
    from phenotypic.gui.tune._callbacks import _ImportanceCache

    cache = _ImportanceCache()
    first = ("journal:///a/journal.log", "tune_cost_v1")
    second = ("journal:///b/journal.log", "tune_cost_v1")

    assert cache.read(first, 2, lambda: {"threshold": 1.0}) is None
    assert cache.wait_for_refresh(10.0)
    assert cache.read(first, 2, lambda: None) == {"threshold": 1.0}

    assert cache.read(second, 2, lambda: 1 / 0) is None
    assert cache.wait_for_refresh(10.0)
    assert cache.read(second, 2, lambda: None) is None


class _CountingStore:
    def __init__(
        self,
        trials: list[object],
        *,
        read_gate: threading.Event | None = None,
        importance_gate: threading.Event | None = None,
        terminal_trials: list[object] | None = None,
    ) -> None:
        self._trials = trials
        self._read_gate = read_gate
        self._importance_gate = importance_gate
        self._terminal_trials = terminal_trials
        self.reads = 0
        self.importance_reads = 0

    @property
    def trials(self) -> list[object]:
        self.reads += 1
        if self._read_gate is not None:
            self._read_gate.wait(timeout=30.0)
        return list(self._trials)

    def best(self) -> object | None:
        return self._trials[0] if self._trials else None

    def pareto_front(self) -> list[object]:
        return []

    def terminal_trials(self) -> list[object]:
        if self._terminal_trials is None:
            return list(self._trials)
        return list(self._terminal_trials)

    def param_importances(self) -> dict[str, float]:
        self.importance_reads += 1
        if self._importance_gate is not None:
            self._importance_gate.wait(timeout=30.0)
        return {"threshold": 1.0}


def test_whole_storage_read_is_bounded_and_degrades_without_joining(
    tmp_path: Path, monkeypatch
) -> None:
    from phenotypic.gui.tune import _callbacks

    monkeypatch.setattr(_callbacks.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_callbacks, "_LIVE_READ_TIMEOUT_S", 0.1)
    gate = threading.Event()
    store = _CountingStore([object()], read_gate=gate)
    fallback = object()
    monkeypatch.setattr(_callbacks, "_open_live_study", lambda _root: store)
    monkeypatch.setattr(_callbacks, "_load_journal", lambda _root: fallback)

    try:
        got, note = _callbacks.read_study_for_monitor(_root(tmp_path))
        assert got is fallback
        assert note
        assert not gate.is_set()
    finally:
        gate.set()


def test_slow_fanova_is_outside_read_deadline_and_snapshot_is_detached(
    tmp_path: Path, monkeypatch
) -> None:
    from phenotypic.gui.tune import _callbacks
    from phenotypic.tune._study_store import Trial

    _callbacks._IMPORTANCES.clear()
    monkeypatch.setattr(_callbacks.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_callbacks, "_LIVE_READ_TIMEOUT_S", 0.5)
    gate = threading.Event()
    trial = Trial(
        number=7,
        params={"threshold": 0.5},
        score=0.1,
        terms={},
        n_images=3,
    )
    store = _CountingStore([trial], importance_gate=gate)
    monkeypatch.setattr(_callbacks, "_open_live_study", lambda _root: store)

    try:
        snapshot, note = _callbacks.read_study_for_monitor(_root(tmp_path))
        assert note == ""
        assert [item.number for item in snapshot.trials] == [7]
        assert snapshot.best() is trial
        assert snapshot.param_importances() is None
        for _ in range(3):
            snapshot.trials
            snapshot.best()
            snapshot.pareto_front()
        assert store.reads == 1
    finally:
        gate.set()
        _callbacks._IMPORTANCES.wait_for_refresh(10.0)
        _callbacks._IMPORTANCES.clear()


def test_live_read_timeouts_coalesce_one_pending_storage_future(
    tmp_path: Path, monkeypatch
) -> None:
    from phenotypic.gui.tune import _callbacks

    pool = _RecordingPool()
    fallback = object()
    monkeypatch.setattr(_callbacks.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_callbacks, "_LIVE_READ_TIMEOUT_S", 0.01)
    monkeypatch.setattr(_callbacks, "_LIVE_OPEN_POOL", pool)
    monkeypatch.setattr(_callbacks, "_load_journal", lambda _root: fallback)

    try:
        for _ in range(2):
            got, note = _callbacks.read_study_for_monitor(_root(tmp_path))
            assert got is fallback
            assert note
        assert pool.submissions == 1
    finally:
        pool.futures[0].set_result(object())
        coalescer = getattr(_callbacks, "_LIVE_READS", None)
        if coalescer is not None:
            coalescer.clear()


def test_terminal_state_progression_refreshes_importances(
    tmp_path: Path, monkeypatch
) -> None:
    from phenotypic.gui.tune import _callbacks
    from phenotypic.tune._study_store import Trial

    _callbacks._IMPORTANCES.clear()
    trial = Trial(
        number=7,
        params={"threshold": 0.5},
        score=0.1,
        terms={},
        n_images=3,
    )
    store = _CountingStore([trial], terminal_trials=[])
    monkeypatch.setattr(_callbacks, "_open_live_study", lambda _root: store)

    try:
        _callbacks._snapshot_live_study(_root(tmp_path))
        assert _callbacks._IMPORTANCES.wait_for_refresh(10.0)
        _callbacks._snapshot_live_study(_root(tmp_path))
        assert store.importance_reads == 1

        store._terminal_trials = [trial]
        _callbacks._snapshot_live_study(_root(tmp_path))
        assert _callbacks._IMPORTANCES.wait_for_refresh(10.0)
        _callbacks._snapshot_live_study(_root(tmp_path))
        assert store.importance_reads == 2
    finally:
        _callbacks._IMPORTANCES.wait_for_refresh(10.0)
        _callbacks._IMPORTANCES.clear()
