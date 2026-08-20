"""P1 / C7a — the ``journal://`` scheme dispatch across every construction site.

Optuna's string resolver hands *every* storage URL to ``RDBStorage``, so there
is no hook for a pseudo-scheme: ``journal:///x`` dies with
``NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:journal``. That is
the fact this module pins from both sides —

1. the negative control (:func:`test_optuna_string_resolver_rejects_journal_url`)
   proves the raw URL genuinely cannot be passed to optuna, so every "we
   dispatch here" test below is testing something real rather than a redundant
   wrapper; and
2. each construction site is driven with a ``journal://`` URL and asserted to
   produce a live ``JournalStorage``.

Sites covered (spec §7 P1's table):

===========================================  =========================================
``_optuna_store.py`` create + load paths     :func:`test_store_creates_journal_backed_study`,
                                             :func:`test_store_loads_journal_backed_study_read_only`
``_run.py`` ``_open_store``                  :func:`test_open_store_dispatches_journal_url`
``_run.py`` ``_submit_slurm_fleet``          :func:`test_slurm_default_is_the_journal_backend`
``_worker.py`` ``build_worker_store``        :func:`test_worker_store_opens_the_journal_study`
``strategy/_optuna.py`` no-``.study``        :func:`test_strategy_fallback_dispatches_journal_url`
===========================================  =========================================

Plus the routing policy the dispatch exists to serve: ``--slurm`` defaults to
``journal://`` while a local run keeps ``sqlite://``, explicit URLs still win at
every precedence level, and a SQLite URL that *reaches* a fleet is refused (H1).
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_ import _io_constants as io
from phenotypic.tune import (
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
)
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.tune._study._storage import (
    build_optuna_storage,
    is_journal_url,
    is_sqlite_url,
    journal_path_from_url,
    journal_url_for_path,
)
from phenotypic.tune.score import Scorer
from phenotypic.tune.strategy import OptunaConfig
from phenotypic.tune.strategy._config import PHENOTYPIC_TUNE_STORAGE_URL_ENV

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")

_STUDY = "tune_cost_v1"


class _ConstScorer(Scorer):
    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


def _optuna_spec() -> TuningSpec:
    return TuningSpec(
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=OptunaConfig(n_trials=2),
        budget=Budget(),
    )


class _FakeExecutor:
    """A ``SlurmExecutor`` stand-in that records its kwargs and submits nothing."""

    captured: dict = {}

    def __init__(self, **kwargs):
        type(self).captured = dict(kwargs)

    def run(self, work, items):
        return None


# ---------------------------------------------------------------------------
# The negative control: the raw URL really is unusable
# ---------------------------------------------------------------------------


def test_optuna_string_resolver_rejects_journal_url(tmp_path):
    """``journal://`` is a pseudo-scheme — optuna's own resolvers cannot take it.

    This is what makes every dispatch assertion below load-bearing rather than a
    test of a pass-through wrapper: delete the dispatch and the sites do not
    "fall back to RDB", they raise.
    """
    import optuna

    url = journal_url_for_path(tmp_path / "journal.log")

    with pytest.raises(Exception) as rdb:
        optuna.storages.RDBStorage(url)
    assert "journal" in str(rdb.value)

    with pytest.raises(Exception) as generic:
        optuna.storages.get_storage(url)
    assert "journal" in str(generic.value)


# ---------------------------------------------------------------------------
# The dispatcher itself
# ---------------------------------------------------------------------------


def test_build_optuna_storage_dispatches_on_scheme(tmp_path):
    journal = build_optuna_storage(journal_url_for_path(tmp_path / "j.log"))
    assert type(journal).__name__ == "JournalStorage"

    rdb = build_optuna_storage(f"sqlite:///{tmp_path / 'study.db'}")
    assert type(rdb).__name__ == "RDBStorage"


def test_journal_backend_carries_the_nfs_safe_symlink_lock(tmp_path):
    """The lock must be ``JournalFileSymlinkLock``, not ``JournalFileOpenLock``.

    Symlink creation is atomic on NFS; the ``O_EXCL`` semantics the open-lock
    relies on are not reliably provided there. Both locks satisfy the same
    interface and both pass a happy-path concurrency test on a POSIX-coherent
    mount, so nothing else in the suite can tell them apart — assert the type.
    """
    storage = build_optuna_storage(journal_url_for_path(tmp_path / "j.log"))
    backend = storage._backend
    lock = backend._lock
    assert type(lock).__name__ == "JournalFileSymlinkLock"


def test_journal_url_round_trips_through_the_path_helpers(tmp_path):
    path = tmp_path / ".pht-tune-cache" / "journal.log"
    url = journal_url_for_path(path)
    assert url.startswith("journal:///")
    assert journal_path_from_url(url) == path
    assert is_journal_url(url) and not is_sqlite_url(url)


def test_journal_backend_creates_its_parent_directory(tmp_path):
    """``.pht-tune-cache/`` may not exist when a fleet submission resolves the URL.

    ``JournalFileBackend`` creates the log file but not its parent, so without
    the ``mkdir`` in :func:`build_optuna_storage` a first ``--slurm`` submission
    would die on ``FileNotFoundError`` before writing anything.
    """
    target = tmp_path / "never" / "created" / "journal.log"
    assert not target.parent.exists()

    build_optuna_storage(journal_url_for_path(target))

    assert target.parent.is_dir()


def test_build_optuna_storage_rejects_a_journal_url_with_no_path():
    with pytest.raises(ValueError, match="no path"):
        build_optuna_storage("journal://")


# ---------------------------------------------------------------------------
# Site: _optuna_store.py — the create and the load paths
# ---------------------------------------------------------------------------


def test_store_creates_journal_backed_study(tmp_path):
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = journal_url_for_path(tmp_path / "journal.log")
    store = OptunaStudyStore(storage_url=url, study_name=_STUDY)

    assert type(store.study._storage).__name__ == "JournalStorage"
    assert store.study.study_name == _STUDY
    assert store.storage_url == url


def test_store_loads_journal_backed_study_read_only(tmp_path):
    """The ``create=False`` monitor path dispatches too.

    ``optuna.load_study`` resolves a storage *string* through the same RDB-only
    resolver, so this is a second, independent dispatch site — not covered by
    the create path.
    """
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = journal_url_for_path(tmp_path / "journal.log")
    OptunaStudyStore(storage_url=url, study_name=_STUDY)

    reopened = OptunaStudyStore(storage_url=url, study_name=_STUDY, create=False)
    assert type(reopened.study._storage).__name__ == "JournalStorage"
    assert reopened.study.study_name == _STUDY


def test_journal_study_persists_and_resumes_trials(tmp_path):
    """A reopened journal study restores the trials — the whole point of P1.

    Type assertions alone would pass against a backend that persists nothing.
    """
    from phenotypic.tune._study._optuna_store import OptunaStudyStore
    from phenotypic.tune._study_store import Trial

    url = journal_url_for_path(tmp_path / "journal.log")
    store = OptunaStudyStore(storage_url=url, study_name=_STUDY)
    store.append(
        Trial(number=0, params={"a": 1}, score=0.25, terms={"Count": 0.25}, n_images=1)
    )
    store.append(
        Trial(number=1, params={"a": 2}, score=0.75, terms={"Count": 0.75}, n_images=1)
    )

    reopened = OptunaStudyStore(storage_url=url, study_name=_STUDY, create=False)
    assert [t.score for t in reopened.trials] == [0.25, 0.75]
    assert [t.params["a"] for t in reopened.trials] == [1, 2]


def test_multi_objective_journal_study_keeps_its_directions(tmp_path):
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = journal_url_for_path(tmp_path / "journal.log")
    store = OptunaStudyStore(
        storage_url=url, study_name=_STUDY, directions=["minimize", "minimize"]
    )
    assert len(store.study.directions) == 2


def test_sqlite_still_gets_rdb_storage_and_wal(tmp_path):
    """The dispatch must not disturb the unchanged local path.

    WAL is what lets concurrent readers/writers share ``study.db``; it is set
    only on the RDB branch, so a mis-dispatched SQLite URL would silently lose
    it.
    """
    import sqlalchemy

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    db = tmp_path / "study.db"
    OptunaStudyStore(storage_url=f"sqlite:///{db}", study_name=_STUDY)

    engine = sqlalchemy.create_engine(f"sqlite:///{db}")
    with engine.connect() as conn:
        mode = conn.execute(sqlalchemy.text("PRAGMA journal_mode")).scalar()
    assert str(mode).lower() == "wal"


# ---------------------------------------------------------------------------
# Site: _run.py — _open_store
# ---------------------------------------------------------------------------


def test_open_store_dispatches_journal_url(tmp_path):
    from phenotypic.tune._tune_cli._run import _open_store

    url = journal_url_for_path(tmp_path / "journal.log")
    store = _open_store(
        OptunaConfig(n_trials=2),
        tmp_path,
        storage_url=url,
        resume_path=tmp_path / "trials.parquet",
    )
    assert type(store.study._storage).__name__ == "JournalStorage"


# ---------------------------------------------------------------------------
# Site: _worker.py — build_worker_store
# ---------------------------------------------------------------------------


def test_worker_store_opens_the_journal_study(tmp_path):
    """Every SLURM worker opens the shared journal — the entire point of P1."""
    from phenotypic.tune._study._optuna_store import OptunaStudyStore
    from phenotypic.tune._tune_cli._worker import build_worker_store

    url = journal_url_for_path(tmp_path / "journal.log")
    OptunaStudyStore(storage_url=url, study_name=_STUDY)  # submitter pre-create

    worker_store = build_worker_store(storage_url=url, study_name=_STUDY)
    assert type(worker_store.study._storage).__name__ == "JournalStorage"
    assert worker_store.study.study_name == _STUDY


def test_two_worker_stores_share_one_journal_study(tmp_path):
    """Two workers on one URL see each other's trials (they share a study)."""
    from phenotypic.tune._study_store import Trial
    from phenotypic.tune._tune_cli._worker import build_worker_store

    url = journal_url_for_path(tmp_path / "journal.log")
    a = build_worker_store(storage_url=url, study_name=_STUDY)
    b = build_worker_store(storage_url=url, study_name=_STUDY)

    a.append(
        Trial(number=0, params={"a": 1}, score=0.5, terms={"Count": 0.5}, n_images=1)
    )
    b.append(
        Trial(number=1, params={"a": 2}, score=0.25, terms={"Count": 0.25}, n_images=1)
    )

    reader = build_worker_store(storage_url=url, study_name=_STUDY)
    assert sorted(t.score for t in reader.trials) == [0.25, 0.5]


# ---------------------------------------------------------------------------
# Site: strategy/_optuna.py — the no-`.study` fallback
# ---------------------------------------------------------------------------


def test_strategy_fallback_dispatches_journal_url(tmp_path):
    """A store with no ``.study`` makes the strategy open the URL itself.

    That branch used to hand the raw string to ``create_study``, which is
    exactly where a ``journal://`` URL would raise.
    """
    from phenotypic.tune.strategy._optuna import OptunaStrategy

    class _StoreWithoutStudy:
        pass

    url = journal_url_for_path(tmp_path / "journal.log")
    strategy = OptunaStrategy(
        SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        n_trials=2,
        storage_url=url,
        store=_StoreWithoutStudy(),
        study_name=_STUDY,
    )
    assert type(strategy._study._storage).__name__ == "JournalStorage"


def test_strategy_without_storage_url_stays_in_memory(tmp_path):
    """``None`` still means an in-memory study — the dispatch must not force one."""
    from phenotypic.tune.strategy._optuna import OptunaStrategy

    class _StoreWithoutStudy:
        pass

    strategy = OptunaStrategy(
        SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        n_trials=2,
        storage_url=None,
        store=_StoreWithoutStudy(),
    )
    assert type(strategy._study._storage).__name__ == "InMemoryStorage"


# ---------------------------------------------------------------------------
# Routing policy: which default a run lands on
# ---------------------------------------------------------------------------


def test_local_default_is_sqlite_and_slurm_default_is_journal(tmp_path):
    from phenotypic.tune._tune_cli._run import _resolve_storage_url

    local = _resolve_storage_url(None, tmp_path)
    fleet = _resolve_storage_url(None, tmp_path, slurm=True)

    assert is_sqlite_url(local)
    assert is_journal_url(fleet)
    assert journal_path_from_url(fleet) == io.tune_cache_journal_path(tmp_path)


def test_the_journal_default_is_per_run_so_studies_cannot_pool(tmp_path):
    """Each output dir gets its own journal — H2 stops being reachable by default.

    A shared server URL (the old distributed answer) puts two concurrent studies
    on one database under one hardcoded study name, silently pooling their
    trials.
    """
    from phenotypic.tune._tune_cli._run import _resolve_storage_url

    one = _resolve_storage_url(None, tmp_path / "run_a", slurm=True)
    two = _resolve_storage_url(None, tmp_path / "run_b", slurm=True)

    assert is_journal_url(one) and is_journal_url(two)
    assert journal_path_from_url(one) == io.tune_cache_journal_path(tmp_path / "run_a")
    assert journal_path_from_url(two) == io.tune_cache_journal_path(tmp_path / "run_b")


def test_explicit_url_still_wins_over_the_slurm_default(tmp_path, monkeypatch):
    """Only the *default* moves. All three higher precedence levels are intact."""
    from phenotypic.tune._tune_cli._run import _resolve_storage_url

    monkeypatch.delenv(PHENOTYPIC_TUNE_STORAGE_URL_ENV, raising=False)
    explicit = "postgresql+psycopg://host/db"

    assert _resolve_storage_url(explicit, tmp_path, slurm=True) == explicit
    assert (
        _resolve_storage_url(None, tmp_path, spec_storage_url=explicit, slurm=True)
        == explicit
    )

    monkeypatch.setenv(PHENOTYPIC_TUNE_STORAGE_URL_ENV, explicit)
    assert _resolve_storage_url(None, tmp_path, slurm=True) == explicit


# ---------------------------------------------------------------------------
# H1: a SQLite URL must never reach a fleet
# ---------------------------------------------------------------------------


def _run_slurm(tmp_path, monkeypatch, *, storage_url, spec=None):
    from phenotypic.tune._tune_cli._run import run_tuning

    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run.SlurmExecutor", _FakeExecutor
    )
    spec = spec if spec is not None else _optuna_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    out = tmp_path / "out"
    run_tuning(
        spec,
        images=[],
        output_dir=out,
        slurm=True,
        spec_path=spec_path,
        images_dir=tmp_path / "imgs",
        storage_url=storage_url,
    )
    return out


def test_slurm_refuses_an_explicit_sqlite_url(tmp_path, monkeypatch):
    """H1 — ``--slurm`` used to submit straight into the corruption case."""
    with pytest.raises(ValueError, match="SQLite"):
        _run_slurm(
            tmp_path, monkeypatch, storage_url=f"sqlite:///{tmp_path / 'study.db'}"
        )


def test_slurm_refuses_a_sqlite_url_from_the_environment(tmp_path, monkeypatch):
    """The env var reaches the fleet by the same route the flag does.

    ``tune_distributed_hpcc.md`` instructs exporting this variable, so an
    operator can arrive at SQLite without ever typing ``--storage-url``. Guard
    the RESOLVED backend, not the flag.
    """
    from phenotypic.tune._tune_cli._run import run_tuning

    monkeypatch.setenv(
        PHENOTYPIC_TUNE_STORAGE_URL_ENV, f"sqlite:///{tmp_path / 'env.db'}"
    )
    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run.SlurmExecutor", _FakeExecutor
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_optuna_spec().model_dump_json())

    with pytest.raises(ValueError, match="SQLite"):
        run_tuning(
            _optuna_spec(),
            images=[],
            output_dir=tmp_path / "out",
            slurm=True,
            spec_path=spec_path,
            images_dir=tmp_path / "imgs",
        )


def test_the_sqlite_refusal_lands_before_any_artifact(tmp_path, monkeypatch):
    """The guard sits in the pre-artifact validator, not at the submit branch.

    Raising after the spec echo and the run marker have landed would leave a
    half-written output directory the GUI classifies as a live tune run.
    """
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="SQLite"):
        _run_slurm(
            tmp_path, monkeypatch, storage_url=f"sqlite:///{tmp_path / 'study.db'}"
        )

    assert not io.tuning_spec_path(out).exists()
    assert not io.tune_cache_run_marker_path(out).exists()


def test_local_sqlite_run_is_untouched_by_the_guard(tmp_path, monkeypatch):
    """The refusal is scoped to ``--slurm``; a local SQLite run still works."""
    from phenotypic.tune._tune_cli._run import _open_store

    monkeypatch.delenv(PHENOTYPIC_TUNE_STORAGE_URL_ENV, raising=False)
    # `run_tuning` has already created the cache dir by the time it opens a
    # store; calling `_open_store` directly has not.
    io.tune_cache_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    store = _open_store(
        OptunaConfig(n_trials=2),
        tmp_path,
        storage_url=None,
        resume_path=tmp_path / "trials.parquet",
    )
    assert is_sqlite_url(store.storage_url)


# ---------------------------------------------------------------------------
# End-to-end: a fleet submission with no --storage-url
# ---------------------------------------------------------------------------


def test_slurm_default_is_the_journal_backend(tmp_path, monkeypatch):
    """The submitter pre-creates a journal-backed study and hands workers its URL.

    Covers ``_submit_slurm_fleet``: the pre-create call, the URL the executor
    forwards to every worker, and the marker the GUI reads.
    """
    import optuna

    monkeypatch.delenv(PHENOTYPIC_TUNE_STORAGE_URL_ENV, raising=False)
    out = _run_slurm(tmp_path, monkeypatch, storage_url=None)

    expected = journal_url_for_path(io.tune_cache_journal_path(out))
    assert _FakeExecutor.captured["storage_url"] == expected

    marker = json.loads(io.tune_cache_run_marker_path(out).read_text())
    assert marker["storage_url"] == expected

    # The journal exists and already holds the pre-created study, so no worker
    # races to materialize it.
    assert io.tune_cache_journal_path(out).exists()
    study = optuna.load_study(
        study_name="tune_cost_v1", storage=build_optuna_storage(expected)
    )
    assert study.study_name == "tune_cost_v1"


# ---------------------------------------------------------------------------
# Site: gui/tune/_callbacks.py — the read-only Monitor open
# ---------------------------------------------------------------------------


def test_monitor_reads_a_live_journal_backed_study(tmp_path):
    """The Monitor's ``create=False`` open reaches the live journal study.

    ``read_study_for_monitor`` swallows every open failure and degrades to the
    parquet journal with a note, so a broken dispatch here would look like a
    quiet, permanent "couldn't reach the live study" rather than a crash. Assert
    the LIVE store and the empty note, not just "something came back".
    """
    from phenotypic.gui.tune._callbacks import read_study_for_monitor
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tune._study._optuna_store import OptunaStudyStore
    from phenotypic.tune._study_store import Trial

    url = journal_url_for_path(tmp_path / ".pht-tune-cache" / "journal.log")
    live = OptunaStudyStore(storage_url=url, study_name=_STUDY)
    live.append(
        Trial(number=0, params={"a": 1}, score=0.125, terms={"Count": 0.125}, n_images=1)
    )

    root = TuneRunRoot(
        path=tmp_path,
        trials_path=None,
        storage_url=url,
        study_name=_STUDY,
        directions=None,
        images_dir=None,
        best_pipeline_path=tmp_path / "best_pipeline.json",
    )

    store, note = read_study_for_monitor(root)

    assert note == ""
    assert [t.score for t in store.trials] == [0.125]
