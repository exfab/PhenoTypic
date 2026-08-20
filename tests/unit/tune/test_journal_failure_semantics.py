"""P1 / C7b — what a ``journal://`` study does when the filesystem misbehaves.

C7a routed every construction site to the journal backend; this module pins the
four failure semantics that routing exposed, each of which the RDB-era code got
for free and the journal backend does not:

* **B1** — :func:`is_transient_storage_error` / ``retry_on_transient_db_error``.
  The retry matched ``sqlalchemy.exc.OperationalError`` alone, a class the
  journal backend never raises, so the fleet's resilience switched itself off
  the moment ``journal://`` became the ``--slurm`` default. The tests below
  drive real ``OSError``\\ s through the real ``JournalFileBackend`` rather than
  asserting on the wrapper in isolation — a retry test that never raises what it
  claims to retry proves nothing.
* **B2** — the ``create=False`` open must not create. It did: the backend
  ``open(path, "ab")``-s its log into existence, so pointing the GUI Monitor at
  a not-yet-started run manufactured ``journal.log`` and its parent tree, after
  which an absent study read as a present one.
* **B4** — ``journal.log`` never compacts. The growth model behind
  ``_JOURNAL_SIZE_WARN_BYTES`` is re-derived here from the real
  ``ask`` → ``set_trial_user_attrs`` → ``tell`` path, so the sizing claim fails
  a test if optuna's record format changes, instead of aging quietly in a
  comment.

(B3 — bounding the Monitor's reads — lives with the rest of the Monitor's
timeout behaviour in ``tests/integration/gui/test_tune_live_timeout.py``.)
"""
from __future__ import annotations

import errno
import importlib.util
import inspect
import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from phenotypic.tune.strategy._optuna_support import (
    _JOURNAL_TORN_LINE_MSG,
    _TRANSIENT_ERRNOS,
    is_transient_storage_error,
    retry_on_transient_db_error,
    set_trial_user_attrs,
)

_OPTUNA = importlib.util.find_spec("optuna") is not None
requires_optuna = pytest.mark.skipif(
    not _OPTUNA, reason="journal storage is gated on the tune extra"
)

_STUDY = "tune_cost_v1"


def _journal_url(tmp_path: Path, name: str = "journal.log") -> str:
    from phenotypic.tune._study._storage import journal_url_for_path

    return journal_url_for_path(tmp_path / name)


# ---------------------------------------------------------------------------
# B1 — the transient predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["EIO", "ESTALE", "EAGAIN", "EBUSY", "EINTR", "ETIMEDOUT"]
)
def test_a_shared_filesystem_hiccup_is_transient(name: str) -> None:
    """The errnos a GPFS/NFS mount fails a call with are retryable."""
    code = getattr(errno, name)
    assert code in _TRANSIENT_ERRNOS
    assert is_transient_storage_error(OSError(code, "transient"))


@pytest.mark.parametrize("name", ["ENOSPC", "EDQUOT", "EACCES", "EROFS", "ENOENT"])
def test_an_operator_problem_is_not_transient(name: str) -> None:
    """A full disk / bad permissions / deleted output dir must fail loudly.

    Retrying these is a busy-wait on a condition no amount of waiting clears.
    """
    assert not is_transient_storage_error(OSError(getattr(errno, name), "fatal"))


def test_the_ambiguous_stolen_lock_is_not_transient() -> None:
    """``JournalFileSymlinkLock.release``'s ``RuntimeError`` is never retried.

    It is raised from the lock context manager's ``finally``, so it means "the
    lock was forcibly stolen after the grace period" — and the append it guarded
    may already have landed. Retrying the ask/tell would then write the record
    twice. A dead worker is recoverable; a doubled record is not.
    """
    assert not is_transient_storage_error(RuntimeError("Error: did not possess lock"))


def test_a_programming_error_is_not_transient() -> None:
    assert not is_transient_storage_error(ValueError("bad knob"))
    assert not is_transient_storage_error(KeyError("Record does not exist."))


def test_a_torn_journal_read_is_transient() -> None:
    """A record another worker was mid-append on; re-reading is always safe."""
    assert is_transient_storage_error(json.JSONDecodeError("x", "{", 0))
    assert is_transient_storage_error(ValueError(_JOURNAL_TORN_LINE_MSG))


@requires_optuna
def test_journal_torn_line_message_is_still_optunas() -> None:
    """Pin the message string against the installed optuna's own source.

    Optuna raises a bare ``ValueError`` for a newline-less trailing record, so
    its message is the only thing that distinguishes it from a programming bug.
    A rewording upstream must fail here rather than silently switch that arm of
    the predicate off.
    """
    from optuna.storages.journal import JournalFileBackend

    source = inspect.getsource(JournalFileBackend.read_logs)
    assert f'ValueError("{_JOURNAL_TORN_LINE_MSG}")' in source


@requires_optuna
def test_a_real_torn_append_raises_something_the_predicate_accepts(
    tmp_path: Path,
) -> None:
    """Drive the real reader over a real torn log — not a hand-built exception.

    A writer interrupted mid-record leaves a newline-less tail; its next record
    joins that tail into one invalid-JSON line. That is the failure a concurrent
    reader actually sees, and the predicate has to recognize *it*, not the
    exception a test author imagined.
    """
    from optuna.storages.journal import JournalFileBackend

    log = tmp_path / "journal.log"
    record = json.dumps({"op_code": 0, "worker_id": "w"}, separators=(",", ":"))
    log.write_bytes((record + "\n").encode() + record[:20].encode())

    backend = JournalFileBackend(str(log))
    assert len(list(backend.read_logs(0))) == 1  # the torn tail is simply skipped

    with log.open("ab") as handle:  # the writer's remaining bytes, then one more
        handle.write((record + "\n").encode() * 2)

    with pytest.raises(ValueError) as caught:
        list(backend.read_logs(0))
    assert is_transient_storage_error(caught.value)


# ---------------------------------------------------------------------------
# B1 — the retry loop over the real backend
# ---------------------------------------------------------------------------


def test_retry_absorbs_a_transient_error_and_then_succeeds() -> None:
    calls: list[int] = []

    def _flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise OSError(errno.EIO, "Input/output error")
        return "ok"

    assert retry_on_transient_db_error(_flaky, attempts=3) == "ok"
    assert len(calls) == 3


def test_retry_gives_up_after_the_attempt_budget() -> None:
    calls: list[int] = []

    def _always_broken() -> str:
        calls.append(1)
        raise OSError(errno.ESTALE, "Stale file handle")

    with pytest.raises(OSError):
        retry_on_transient_db_error(_always_broken, attempts=3)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "exc",
    [
        OSError(errno.ENOSPC, "No space left on device"),
        RuntimeError("Error: did not possess lock"),
        ValueError("bad knob"),
    ],
    ids=["full-disk", "stolen-lock", "bug"],
)
def test_retry_never_repeats_a_non_transient_failure(exc: Exception) -> None:
    """Called exactly once — the wrapper must not paper over these."""
    calls: list[int] = []

    def _fails() -> str:
        calls.append(1)
        raise exc

    with pytest.raises(type(exc)):
        retry_on_transient_db_error(_fails, attempts=3)
    assert len(calls) == 1


@requires_optuna
def test_a_transient_journal_append_failure_no_longer_kills_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end B1 claim, through a live ``JournalStorage``.

    The failure is injected into ``JournalFileBackend.append_logs`` — the call
    that actually touches the shared filesystem — and the retried operation is a
    real ``study.ask`` against a real journal-backed study. Before the predicate
    became backend-aware this ``OSError`` propagated on its first occurrence and
    took the worker with it.
    """
    from optuna.storages.journal import JournalFileBackend

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    store = OptunaStudyStore(storage_url=_journal_url(tmp_path), study_name=_STUDY)

    real_append = JournalFileBackend.append_logs
    failures: list[int] = []

    def _flaky_append(self, logs):  # type: ignore[no-untyped-def]
        if not failures:
            failures.append(1)
            raise OSError(errno.EIO, "Input/output error")
        return real_append(self, logs)

    monkeypatch.setattr(JournalFileBackend, "append_logs", _flaky_append)

    trial = retry_on_transient_db_error(store.study.ask)

    assert failures == [1]  # the injected failure really fired
    assert trial.number == 0
    assert len(store.study.get_trials(deepcopy=False)) == 1


# ---------------------------------------------------------------------------
# B2 — a read-only open must not create the thing it reads
# ---------------------------------------------------------------------------


@requires_optuna
def test_read_only_open_does_not_create_a_missing_journal(tmp_path: Path) -> None:
    """``create=False`` on an absent journal raises, leaving no trace on disk."""
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    cache = tmp_path / ".pht-tune-cache"
    url = _journal_url(cache)

    with pytest.raises(FileNotFoundError):
        OptunaStudyStore(storage_url=url, study_name=_STUDY, create=False)

    assert not cache.exists(), "the read-only open materialized the run directory"


@requires_optuna
def test_read_only_open_does_not_create_a_missing_sqlite(tmp_path: Path) -> None:
    """The same guard covers SQLite, whose file SQLAlchemy also creates on open."""
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    db = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        OptunaStudyStore(
            storage_url=f"sqlite:///{db}", study_name=_STUDY, create=False
        )
    assert not db.exists()


@requires_optuna
def test_read_only_open_still_reaches_an_existing_journal_study(
    tmp_path: Path,
) -> None:
    """The guard refuses absence, never a study that is really there."""
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = _journal_url(tmp_path)
    OptunaStudyStore(storage_url=url, study_name=_STUDY).study.ask()

    reopened = OptunaStudyStore(storage_url=url, study_name=_STUDY, create=False)
    assert len(reopened.study.get_trials(deepcopy=False)) == 1


def test_backing_file_is_none_for_a_server_or_memory_url() -> None:
    """Nothing to guard where nothing gets created — and no connect to probe."""
    from phenotypic.tune._study._optuna_store import (
        backing_file_for_url,
        require_existing_backing_store,
    )

    assert backing_file_for_url("postgresql+psycopg://user@host:5432/db") is None
    assert backing_file_for_url("sqlite:///:memory:") is None
    # Neither raises: a read-only open of these is allowed to proceed.
    require_existing_backing_store("postgresql+psycopg://user@host:5432/db")
    require_existing_backing_store("sqlite:///:memory:")


# ---------------------------------------------------------------------------
# B4 — growth is bounded, stated, and watched
# ---------------------------------------------------------------------------

#: A representative tuning workload for the growth measurement: a dozen knobs
#: with realistically long operation-qualified names, six score terms, and a
#: quarter of trials pruned after three ASHA rungs.
_GROWTH_TRIALS = 20
_GROWTH_KNOBS = 12
_GROWTH_TERMS = 6


@requires_optuna
def test_journal_growth_per_trial_stays_near_the_measured_rate(
    tmp_path: Path,
) -> None:
    """Re-derive ``_JOURNAL_SIZE_WARN_BYTES``'s growth model from the real path.

    The measured rate (~5.9 KB and ~20 log records per trial, optuna 4.9.0) is
    what makes a 200-trial × 30-min campaign a 1.2 MB log, which is in turn why
    64 MiB is the right place to speak up and why no compaction is offered. The
    band is wide enough to survive a minor record-format change and narrow
    enough that an order-of-magnitude one — the only kind that would invalidate
    those conclusions — fails here.
    """
    import optuna

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log = tmp_path / "journal.log"
    store = OptunaStudyStore(
        storage_url=_journal_url(tmp_path), study_name=_STUDY
    )
    rng = random.Random(0)

    for index in range(_GROWTH_TRIALS):
        trial = store.study.ask()
        params: dict[str, object] = {}
        for knob in range(_GROWTH_KNOBS):
            name = f"EnhanceStep{knob}_operations[{knob}].threshold_parameter_{knob}"
            if knob % 3 == 0:
                params[name] = trial.suggest_float(name, 0.0, 1.0)
            elif knob % 3 == 1:
                params[name] = trial.suggest_int(name, 1, 51, step=2)
            else:
                params[name] = trial.suggest_categorical(
                    name, ["otsu", "li", "yen", "triangle"]
                )
        pruned = index % 4 == 0
        if pruned:
            for rung in range(3):
                trial.report(rng.random(), rung)
        result = SimpleNamespace(
            score=rng.random(),
            terms={f"term_{term}": rng.random() for term in range(_GROWTH_TERMS)},
            n_images=24,
            objectives=None,
            gap=rng.random() * 0.1,
            suspicious=False,
        )
        set_trial_user_attrs(trial, params=params, result=result)
        if pruned:
            store.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            store.study.tell(trial, result.score)

    bytes_per_trial = log.stat().st_size / _GROWTH_TRIALS
    records_per_trial = sum(1 for _ in log.open("rb")) / _GROWTH_TRIALS

    assert 3_000 <= bytes_per_trial <= 9_000, bytes_per_trial
    assert records_per_trial <= 30, records_per_trial


@requires_optuna
def test_an_oversized_journal_warns_on_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Past the bound, every open says so — and names the remedy."""
    from phenotypic.tune._study import _optuna_store
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = _journal_url(tmp_path)
    OptunaStudyStore(storage_url=url, study_name=_STUDY)  # writes a few records

    monkeypatch.setattr(_optuna_store, "_JOURNAL_SIZE_WARN_BYTES", 1)
    with caplog.at_level(logging.WARNING, logger=_optuna_store.__name__):
        OptunaStudyStore(storage_url=url, study_name=_STUDY, create=False)

    assert any("never compacts" in record.message for record in caplog.records)


@requires_optuna
def test_a_normal_sized_journal_is_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A real campaign's log is three orders of magnitude under the bound."""
    from phenotypic.tune._study import _optuna_store
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = _journal_url(tmp_path)
    with caplog.at_level(logging.WARNING, logger=_optuna_store.__name__):
        OptunaStudyStore(storage_url=url, study_name=_STUDY)

    assert not any("never compacts" in record.message for record in caplog.records)
