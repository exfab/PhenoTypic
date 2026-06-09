"""Concurrent ``OptunaStudyStore`` opens against one SQLite-WAL file (Change 8).

Opening N stores with the SAME ``study_name`` against the SAME storage URL
concurrently must converge on exactly ONE study (``load_if_exists=True`` +
SQLite WAL), with no worker raising on a ``create_study`` schema/insert race.
This is the unit-level analogue of the cold-DB race the SLURM submitter
pre-creates the study to avoid; it runs in CI (no Postgres / SLURM marker).
"""
from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor

import pytest

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")

#: Number of concurrent openers (the fleet-size analogue).
_N_WORKERS = 6
#: The shared study name the run layer hardcodes.
_STUDY = "tune"


def _open_store(url: str):
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    return OptunaStudyStore(storage_url=url, study_name=_STUDY)


def test_concurrent_opens_converge_on_one_study(tmp_path):
    import optuna

    url = f"sqlite:///{tmp_path / 'study.db'}"

    # Pre-create the DB file + schema + WAL mode once so the concurrent openers
    # race on load_if_exists (the real fleet path: the SLURM submitter
    # materializes the study once before the fleet starts, precisely because a
    # truly-cold concurrent CREATE TABLE races). After pre-create, the WAL +
    # load_if_exists opens must all converge on the one study with no exception.
    _open_store(url)

    errors: list[BaseException] = []

    def _worker(_i: int) -> bool:
        try:
            store = _open_store(url)
            # Touch the study so the open is real, not lazy.
            _ = len(store)
            return True
        except BaseException as exc:  # capture, don't swallow
            errors.append(exc)
            return False

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
        results = list(pool.map(_worker, range(_N_WORKERS)))

    assert not errors, f"concurrent open raised: {errors!r}"
    assert all(results)

    # Exactly one study exists despite N concurrent opens.
    summaries = optuna.get_all_study_summaries(storage=url)
    assert [s.study_name for s in summaries] == [_STUDY]


def test_concurrent_opens_reconstruct_the_same_trials(tmp_path):
    """After pre-create, every concurrent opener sees the one study's trials.

    A second guard on the converge-on-one-study contract: seed the pre-created
    study with a trial, then open N stores concurrently and assert each reports
    the same length (one study, not N empty phantoms).
    """
    url = f"sqlite:///{tmp_path / 'study.db'}"
    seed = _open_store(url)
    from phenotypic.tune._study_store import Trial

    seed.append(
        Trial(number=0, params={"a": 1}, score=0.5, terms={"t": 0.5}, n_images=1)
    )

    lengths: list[int] = []
    errors: list[BaseException] = []

    def _worker(_i: int) -> None:
        try:
            lengths.append(len(_open_store(url)))
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
        list(pool.map(_worker, range(_N_WORKERS)))

    assert not errors, f"concurrent open raised: {errors!r}"
    # Every opener saw the same single-trial study.
    assert lengths == [1] * _N_WORKERS
