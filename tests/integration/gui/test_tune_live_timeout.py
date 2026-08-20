"""C1 regression: the live-study read must be bounded and non-re-blocking.

The OQ4 poll opens a live ``OptunaStudyStore`` whose constructor connects to
the storage eagerly. Against a dead / unreachable Postgres that connect can
hang ~30 s (libpq's default) — so the 3 s poll must (a) bound the connect at
the source (a libpq ``connect_timeout`` merged into the storage URL) and
(b) never re-join a still-connecting worker (so even a slow connect can't
freeze the poll). These tests pin both halves.

C7b/B3 extends the same requirement past the constructor. The ``journal://``
backend a ``--slurm`` fleet defaults to has no URL to hang a driver timeout on,
and its "local file" lives on GPFS, where a read blocks on the metadata server —
so the worker-thread bound is its only bound, and it has to cover the reads the
poll renders, not just the open. The last three tests pin that.
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

    # A small, deterministic live-read ceiling (the SUT reads this module global
    # in ``future.result(timeout=...)``). It is a ceiling, not a sleep — the
    # poll returns as soon as the bounded wait elapses.
    monkeypatch.setattr(_callbacks, "_LIVE_READ_TIMEOUT_S", 0.2)

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


# ---------------------------------------------------------------------------
# C7b / B3 — the bound must cover the READS, not just the open
# ---------------------------------------------------------------------------


class _CountingStore:
    """A live store whose reads are observable (and optionally blocking).

    Stands in for ``OptunaStudyStore``: the poll's whole live surface is
    ``trials`` / ``best`` / ``pareto_front`` / ``param_importances``, and each of
    those re-reads the backing storage on the real thing — a ``JournalStorage``
    re-``stat``s and re-reads its append-only log every sync.
    """

    def __init__(
        self,
        trials: "list[object]",
        *,
        gate: "threading.Event | None" = None,
        importances_raise: bool = False,
        importances_gate: "threading.Event | None" = None,
    ) -> None:
        self._trials = trials
        self._gate = gate
        self._importances_raise = importances_raise
        self._importances_gate = importances_gate
        self.reads = 0

    @property
    def trials(self) -> "list[object]":
        self.reads += 1
        if self._gate is not None:
            self._gate.wait(timeout=30.0)  # a GPFS read that never comes back
        return list(self._trials)

    def best(self) -> "object | None":
        return self._trials[0] if self._trials else None

    def pareto_front(self) -> "list[object]":
        return []

    def param_importances(self) -> "dict[str, float] | None":
        if self._importances_raise:
            raise RuntimeError("fANOVA is degenerate")
        if self._importances_gate is not None:
            # A slow fANOVA, not a failing one: the real cost is seconds of
            # compute, and at 400 trials it is longer than a whole poll tick.
            self._importances_gate.wait(timeout=30.0)
        return {"thresh": 1.0}


def _live_trials() -> "list[object]":
    from phenotypic.tune._study_store import Trial

    return [Trial(number=7, params={"thresh": 0.5}, score=0.1, terms={}, n_images=3)]


@pytest.fixture(autouse=True)
def _isolated_importance_cache():
    """Empty the process-wide importance cache around every test in this file.

    ``_IMPORTANCES`` deliberately outlives a poll tick (that is the whole point
    of it), which also means it outlives a test. Clearing on both sides keeps
    one test's fANOVA out of the next one's snapshot.
    """
    from phenotypic.gui.tune import _callbacks

    _callbacks._IMPORTANCES.clear()
    yield
    _callbacks._IMPORTANCES.wait_for_refresh(30.0)
    _callbacks._IMPORTANCES.clear()


@pytest.mark.skipif(not _OPTUNA_PRESENT, reason="live path is gated on the tune extra")
def test_a_hanging_store_read_degrades_instead_of_freezing_the_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The open succeeds; the *read* hangs — and the poll still returns.

    This is the journal backend's hazard specifically: its URL carries no
    driver-level timeout, and its "local file" sits on GPFS, where a read blocks
    on the metadata server. Bounding only the constructor left every subsequent
    ``store.trials`` running unbounded on the Dash callback thread.

    Same determinism contract as the connect-timeout test above: a gated event,
    a monkeypatched ceiling, and the proof of non-joining is that the gate is
    still held when the poll returns.
    """
    from phenotypic.gui.tune import _callbacks

    monkeypatch.setattr(_callbacks, "_LIVE_READ_TIMEOUT_S", 0.2)

    gate = threading.Event()
    store = _CountingStore(_live_trials(), gate=gate)
    monkeypatch.setattr(_callbacks, "_open_live_study", lambda _root: store)

    root = _journal_with_live_url(tmp_path, f"journal:///{tmp_path}/journal.log")

    result: dict[str, object] = {}
    done = threading.Event()

    def _drive() -> None:
        try:
            got, note = _callbacks.read_study_for_monitor(root)
            result["store"], result["note"] = got, note
        finally:
            done.set()

    worker = threading.Thread(target=_drive, name="poll-driver")
    worker.start()
    try:
        assert done.wait(timeout=10.0), "poll did not return — the read was unbounded"
        assert not gate.is_set(), "the gate was released — the test, not the SUT"
        # The parquet fallback, not the half-read live store.
        assert [t.score for t in result["store"].trials] == [0.3, 0.6]
        # A ``journal://`` URL that overran its bound: neither "unreachable" nor
        # a network problem, so the note must say *timed out* and must not send
        # the user to check ~/.pgpass (there is no server in this configuration).
        assert result["note"] == _callbacks._NOTE_LIVE_TIMEOUT
        assert "pgpass" not in result["note"]
    finally:
        gate.set()
        worker.join(timeout=5.0)


@pytest.mark.skipif(not _OPTUNA_PRESENT, reason="live path is gated on the tune extra")
def test_the_returned_snapshot_never_touches_storage_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rendering the poll's outputs must not re-read the study.

    The poll reads ``trials`` several times over (the objective figure, the gap
    badge, the table). Each of those was a fresh storage read outside the bound;
    the snapshot makes them all reads of plain data captured inside it.
    """
    from phenotypic.gui.tune import _callbacks

    store = _CountingStore(_live_trials())
    monkeypatch.setattr(_callbacks, "_open_live_study", lambda _root: store)

    root = _journal_with_live_url(tmp_path, f"journal:///{tmp_path}/journal.log")
    snapshot, note = _callbacks.read_study_for_monitor(root)

    assert note == ""
    assert [t.number for t in snapshot.trials] == [7]  # the live study, not parquet
    for _ in range(3):
        snapshot.trials
        snapshot.best()
        snapshot.pareto_front()
        snapshot.param_importances()

    assert store.reads == 1, "the poll re-read the live storage after the bound"


@pytest.mark.skipif(not _OPTUNA_PRESENT, reason="live path is gated on the tune extra")
def test_a_degenerate_importance_model_costs_only_the_importance_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fANOVA failing is a figure problem, not a reason to abandon the live read."""
    from phenotypic.gui.tune import _callbacks

    store = _CountingStore(_live_trials(), importances_raise=True)
    monkeypatch.setattr(_callbacks, "_open_live_study", lambda _root: store)

    root = _journal_with_live_url(tmp_path, f"journal:///{tmp_path}/journal.log")
    snapshot, note = _callbacks.read_study_for_monitor(root)

    assert note == ""
    assert [t.number for t in snapshot.trials] == [7]
    assert snapshot.param_importances() is None

    # …and still None once the out-of-band refresh has actually run and failed:
    # the raise is absorbed by the cache, not left to poison the next live read.
    assert _callbacks._IMPORTANCES.wait_for_refresh(30.0)
    again, again_note = _callbacks.read_study_for_monitor(root)
    assert again_note == ""
    assert [t.number for t in again.trials] == [7]
    assert again.param_importances() is None


@pytest.mark.skipif(not _OPTUNA_PRESENT, reason="live path is gated on the tune extra")
def test_a_slow_importance_model_no_longer_costs_the_whole_live_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCK-2: a *slow* fANOVA must not spend the read budget it dwarfs.

    Measured on a real ``journal://`` study, ``param_importances`` costs 2.0 s
    at 200 trials and 4.9 s at 400 — against a 3.0 s bound that every other read
    in the snapshot fits inside a hundred times over. Sharing the budget did not
    degrade the odd tick; past ~250 trials it degraded *every* tick, to a
    parquet fallback that mid-run does not exist yet, so the Monitor read "No
    trials yet." for the whole campaign.

    RED before the fix: the gated ``param_importances`` holds the worker past
    the bound, so the poll returns the parquet fallback and a degrade note.
    """
    from phenotypic.gui.tune import _callbacks

    monkeypatch.setattr(_callbacks, "_LIVE_READ_TIMEOUT_S", 0.5)

    gate = threading.Event()
    store = _CountingStore(_live_trials(), importances_gate=gate)
    monkeypatch.setattr(_callbacks, "_open_live_study", lambda _root: store)

    root = _journal_with_live_url(tmp_path, f"journal:///{tmp_path}/journal.log")
    try:
        snapshot, note = _callbacks.read_study_for_monitor(root)
        assert note == "", "a slow fANOVA degraded the whole live read"
        # The LIVE trials, not the parquet fallback's [0.3, 0.6].
        assert [t.number for t in snapshot.trials] == [7]
        assert snapshot.best() is not None
        # Only the figure is late.
        assert snapshot.param_importances() is None
    finally:
        gate.set()


# ---------------------------------------------------------------------------
# C7b / B3 — the snapshot against a REAL store, not a stand-in
# ---------------------------------------------------------------------------
#
# Every test above monkeypatches ``_open_live_study``, so none of them ever
# observes what ``_snapshot_live_study`` actually copies out of a live
# ``OptunaStudyStore``. Mutation-testing found what that hides: replacing
# ``_read_importances`` with ``return None``, ``store.best()`` with ``None``, or
# ``store.pareto_front()`` with ``[]`` each left the whole tune + tune-GUI suite
# green — i.e. the importance chart empty forever and the gap badge stuck at
# ``--``, undetected. These two tests read a real journal-backed study and
# compare the snapshot field by field against it.


def _real_run_root(path: Path, *, directions: "list[str] | None"):
    """Build a real ``journal://`` study through ask/tell; return (root, store).

    Real ask/suggest/tell — not ``store.append`` — because fANOVA needs the
    native sampler distributions that only the ask path records, and because the
    Pareto front is Optuna's own non-domination over per-objective ``values``.
    """
    from types import SimpleNamespace

    import optuna

    from phenotypic.tune._study._optuna_store import OptunaStudyStore
    from phenotypic.tune._study._storage import journal_url_for_path
    from phenotypic.tune._tune_cli._run import _STUDY_NAME
    from phenotypic.tune.strategy._optuna_support import set_trial_user_attrs

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    url = journal_url_for_path(path / "journal.log")
    store = OptunaStudyStore(
        storage_url=url, study_name=_STUDY_NAME, directions=directions
    )
    for index in range(8):
        trial = store.study.ask()
        params = {
            "thresh": trial.suggest_float("thresh", 0.0, 1.0),
            "radius": trial.suggest_int("radius", 1, 15, step=2),
            "method": trial.suggest_categorical("method", ["otsu", "li", "yen"]),
        }
        primary = (index % 5) / 10.0
        secondary = (7 - index) / 10.0
        objectives = (
            {"dice": primary, "sizecv": secondary} if directions else None
        )
        set_trial_user_attrs(
            trial,
            params=params,
            result=SimpleNamespace(
                score=primary,
                terms={"primary": primary},
                n_images=3,
                objectives=objectives,
                gap=0.02,
                suspicious=False,
            ),
        )
        if directions:
            store.study.tell(trial, [primary, secondary])
        else:
            store.study.tell(trial, primary)

    root = TuneRunRoot(
        path=path,
        trials_path=None,
        storage_url=url,
        study_name=_STUDY_NAME,
        directions=directions,
        images_dir=None,
        best_pipeline_path=path / "best_pipeline.json",
    )
    return root, store


@pytest.mark.skipif(not _OPTUNA_PRESENT, reason="live path is gated on the tune extra")
def test_the_snapshot_carries_what_the_real_live_store_holds(tmp_path: Path) -> None:
    """Every field of the snapshot must equal the live store's own answer.

    RED for each of the three surviving mutants: ``_best = None`` (the gap badge
    and the winner card), ``_read_importances -> None`` (the fANOVA chart), and
    a snapshot that drops ``trials``.
    """
    from phenotypic.gui.tune import _callbacks

    root, store = _real_run_root(tmp_path, directions=None)

    snapshot = _callbacks._snapshot_live_study(root)

    live_trials = store.trials
    assert [t.number for t in snapshot.trials] == [t.number for t in live_trials]
    assert [t.score for t in snapshot.trials] == [t.score for t in live_trials]

    live_best = store.best()
    assert live_best is not None, "fixture produced no rankable trial"
    assert snapshot.best() is not None, "the snapshot dropped the study's best trial"
    assert snapshot.best().number == live_best.number
    assert snapshot.best().score == live_best.score

    # The single-objective back-compat contract, verified THROUGH the snapshot.
    assert snapshot.pareto_front() == []

    # Importances are computed out of band, so they land on a later tick.
    assert _callbacks._IMPORTANCES.wait_for_refresh(60.0)
    refreshed = _callbacks._snapshot_live_study(root)
    live_importances = store.param_importances()
    assert live_importances, "fixture produced no fANOVA model"
    snapshot_importances = refreshed.param_importances()
    assert snapshot_importances, "the snapshot dropped the study's importances"
    # fANOVA fits an unseeded random forest, so two evaluations of the same
    # study agree on the knobs and on the normalization, not to the last digit.
    assert set(snapshot_importances) == set(live_importances)
    assert sum(snapshot_importances.values()) == pytest.approx(1.0, abs=1e-6)
    for knob, weight in snapshot_importances.items():
        assert weight == pytest.approx(live_importances[knob], abs=0.15), knob


@pytest.mark.skipif(not _OPTUNA_PRESENT, reason="live path is gated on the tune extra")
def test_the_snapshot_carries_the_real_pareto_front(tmp_path: Path) -> None:
    """A multi-objective study's front must survive the snapshot.

    The single-objective case above can never catch ``_pareto_front = []`` — the
    contract there *is* ``[]``. This is the lane that does.
    """
    from phenotypic.gui.tune import _callbacks

    root, store = _real_run_root(tmp_path, directions=["minimize", "minimize"])

    snapshot = _callbacks._snapshot_live_study(root)

    live_front = store.pareto_front()
    assert live_front, "fixture produced no Pareto front"
    assert [t.number for t in snapshot.pareto_front()] == [
        t.number for t in live_front
    ]
    assert [t.objectives for t in snapshot.pareto_front()] == [
        t.objectives for t in live_front
    ]
