"""C1 regression: the live-study read must be bounded and non-re-blocking.

The OQ4 poll opens a live ``OptunaStudyStore`` whose constructor connects to
the storage eagerly. Against a dead / unreachable Postgres that connect can
hang ~30 s (libpq's default) — so the 3 s poll must (a) bound the connect at
the source (a libpq ``connect_timeout`` merged into the storage URL) and
(b) never re-join a still-connecting worker (so even a slow connect can't
freeze the poll). These tests pin both halves.
"""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import pytest

from phenotypic.gui.tune._run_root import TuneRunRoot

_OPTUNA_PRESENT = importlib.util.find_spec("optuna") is not None


def _journal_with_live_url(path: Path, storage_url: str) -> TuneRunRoot:
    """A run that has a finished trials.parquet AND a (live) storage URL.

    The poll attempts the live read first; on timeout it must degrade to this
    journal, so the test asserts the fallback store is the parquet one.
    """
    from phenotypic.sdk_ import trials_parquet_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    store = JournalStudyStore(
        trials=[
            Trial(number=0, params={"thresh": 0.1}, score=0.3, terms={}, n_images=3),
            Trial(number=1, params={"thresh": 0.2}, score=0.6, terms={}, n_images=3),
        ]
    )
    parquet = trials_parquet_path(path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(parquet)

    # Use the current cost-era study name so the poll exercises the generic
    # timeout-degradation path (``_NOTE_LIVE_UNREACHABLE``). A legacy ``"tune"``
    # study would short-circuit to the cost-cutover note before the live open is
    # even attempted (Phase 2/4 legacy-study branch), which is a different code
    # path than the C1 non-blocking-degradation behavior this test pins.
    from phenotypic.tune._tune_cli._run import _STUDY_NAME

    return TuneRunRoot(
        path=path,
        trials_path=parquet,
        storage_url=storage_url,
        study_name=_STUDY_NAME,
        directions=None,
        images_dir=None,
        best_pipeline_path=path / "best_pipeline.json",
    )


@pytest.mark.skipif(not _OPTUNA_PRESENT, reason="live path is gated on the tune extra")
def test_slow_live_open_degrades_to_journal_without_joining_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A still-blocked live open degrades to the parquet fallback, non-blocking.

    Behavioral (not wall-clock) pin of the C1 fix: when the live-store
    constructor is still blocked, the poll must (a) return the parquet store +
    the unreachable note rather than hang, and (b) NOT re-join the worker (the
    bug the ``with``-managed executor's ``__exit__`` re-introduced).

    Determinism: the slow open is gated on a :class:`threading.Event` the test
    releases in ``finally`` (no fixed ``time.sleep`` that zombies teardown), and
    the ``future.result`` timeout is monkeypatched to a small, deterministic
    value so the bound is exercised without timing the wall clock. The proof of
    "didn't join the worker" is that the gate is still un-released at the moment
    the poll returns. RED before the fix: the executor's ``__exit__`` would block
    on ``gate.wait()`` until the ``finally`` released it, and the assertion that
    the gate is still held on return would fail.
    """
    from phenotypic.gui.tune import _callbacks

    # A small, deterministic connect-timeout ceiling (the SUT reads this module
    # global in ``future.result(timeout=...)``). It is a ceiling, not a sleep —
    # the poll returns as soon as the bounded wait elapses.
    monkeypatch.setattr(_callbacks, "_LIVE_CONNECT_TIMEOUT_S", 0.2)

    gate = threading.Event()
    entered = threading.Event()

    def _blocked_open(_root: object) -> object:
        entered.set()  # the worker actually ran (the open was attempted)
        gate.wait(timeout=30.0)  # held open until the test releases it
        raise AssertionError("should never return — the poll must abandon this")

    monkeypatch.setattr(_callbacks, "_open_live_study", _blocked_open)

    root = _journal_with_live_url(
        tmp_path, "postgresql+psycopg://nope@10.255.255.1:54399/x"
    )

    # Run the poll on a helper thread so the test can assert it RETURNS while the
    # worker's gate is still held. A re-joining implementation (the C1 bug) could
    # only return by waiting out the 30 s gate, so ``done.wait`` below would not
    # fire — a deterministic, fast-failing behavioral signal that waits on a
    # completion event, never a fixed sleep.
    result: dict[str, object] = {}
    done = threading.Event()

    def _drive() -> None:
        try:
            store, note = _callbacks.read_study_for_monitor(root)
            result["store"], result["note"] = store, note
        finally:
            done.set()

    worker = threading.Thread(target=_drive, name="poll-driver")
    worker.start()
    try:
        # The poll degraded and returned promptly WHILE the live open is still
        # blocked (gate un-released) — i.e. it did not join the worker.
        assert done.wait(timeout=10.0), "poll did not return — it joined the worker"
        assert not gate.is_set(), "the gate was released — the test, not the SUT"

        store = result["store"]
        note = result["note"]
        assert store is not None
        assert [t.score for t in store.trials] == [0.3, 0.6]  # the parquet fallback
        assert "couldn't reach the live study" in note
    finally:
        gate.set()  # release the orphaned worker so it exits cleanly (no zombie)
        entered.wait(timeout=5.0)
        worker.join(timeout=5.0)


def test_postgres_url_gets_connect_timeout_applied() -> None:
    from phenotypic.gui.tune._callbacks import _ensure_connect_timeout

    out = _ensure_connect_timeout("postgresql+psycopg://user@host:5432/db")
    assert "connect_timeout=" in out
    # The bound matches the module's timeout constant.
    from phenotypic.gui.tune._callbacks import _LIVE_CONNECT_TIMEOUT_S

    assert f"connect_timeout={int(_LIVE_CONNECT_TIMEOUT_S)}" in out


def test_postgres_url_preserves_user_connect_timeout() -> None:
    from phenotypic.gui.tune._callbacks import _ensure_connect_timeout

    out = _ensure_connect_timeout(
        "postgresql+psycopg://user@host:5432/db?connect_timeout=9"
    )
    assert "connect_timeout=9" in out
    assert out.count("connect_timeout") == 1  # not doubled


def test_sqlite_url_passed_through_unchanged() -> None:
    from phenotypic.gui.tune._callbacks import _ensure_connect_timeout

    url = "sqlite:////tmp/study.db"
    assert _ensure_connect_timeout(url) == url
    assert "connect_timeout" not in _ensure_connect_timeout(url)
