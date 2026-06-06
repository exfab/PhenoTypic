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
import time
from pathlib import Path

import pytest

from phenotypic.gui.tune._run_root import TuneRunRoot

_OPTUNA_PRESENT = importlib.util.find_spec("optuna") is not None


def _journal_with_live_url(path: Path, storage_url: str) -> TuneRunRoot:
    """A run that has a finished trials.parquet AND a (live) storage URL.

    The poll attempts the live read first; on timeout it must degrade to this
    journal, so the test asserts the fallback store is the parquet one.
    """
    from phenotypic.tools_ import trials_parquet_path
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

    return TuneRunRoot(
        path=path,
        trials_path=parquet,
        storage_url=storage_url,
        study_name="tune",
        directions=None,
        images_dir=None,
        best_pipeline_path=path / "best_pipeline.json",
    )


@pytest.mark.skipif(not _OPTUNA_PRESENT, reason="live path is gated on the tune extra")
def test_slow_live_open_does_not_block_poll_past_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long-hanging live open must return the parquet fallback in < timeout+eps.

    Proves the bound: even if the live-store constructor blocks far longer than
    the timeout, the poll returns within ~timeout (not the full block), with the
    parquet store + the unreachable note. RED before the fix (the
    ``with``-managed executor re-joins the worker on ``__exit__`` and blocks the
    full sleep). The sleep (well above the 3 s timeout, but short enough not to
    zombie test teardown via the executor's at-exit join) is interruptible by
    the orphaned worker finishing on its own.
    """
    from phenotypic.gui.tune import _callbacks

    # Comfortably exceeds _LIVE_CONNECT_TIMEOUT_S (3 s) and the < timeout+2 s
    # assertion bound (5 s) without leaving a 30 s zombie worker at teardown.
    _HANG_SECONDS = 8.0

    def _hang(_root: object) -> object:
        time.sleep(_HANG_SECONDS)
        raise AssertionError("should never return — the poll must abandon this")

    monkeypatch.setattr(_callbacks, "_open_live_study", _hang)

    root = _journal_with_live_url(
        tmp_path, "postgresql+psycopg://nope@10.255.255.1:54399/x"
    )

    started = time.monotonic()
    store, note = _callbacks.read_study_for_monitor(root)
    elapsed = time.monotonic() - started

    assert elapsed < _callbacks._LIVE_CONNECT_TIMEOUT_S + 2.0, (
        f"poll blocked {elapsed:.1f}s — the connect timeout did not bound it"
    )
    assert store is not None
    assert [t.score for t in store.trials] == [0.3, 0.6]  # the parquet fallback
    assert "couldn't reach the live study" in note


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
