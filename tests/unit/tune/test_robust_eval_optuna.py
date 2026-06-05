"""4.5p1 A3 — ``Trial.gap`` + ``Trial.suspicious`` survive the Optuna round-trip.

The Optuna-backed ``StudyStore`` stashes our non-native ``Trial`` fields in the
trial's ``user_attrs``; the robust-eval ``gap`` / ``suspicious`` signals must ride
along so a reopened study reconstructs the exact records. Gated on the ``tune``
extra (``skipif`` when optuna is absent), like ``test_optuna_study_store.py``.
"""
from __future__ import annotations

import importlib.util

import pytest

from phenotypic.tune._study_store import Trial

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")


def _store(tmp_path, name="s"):
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = f"sqlite:///{tmp_path / 'study.db'}"
    return OptunaStudyStore(storage_url=url, study_name=name)


def _trial(number, score, *, gap=None, suspicious=False):
    return Trial(
        number=number, params={"a": number}, score=score,
        terms={"Count": score}, n_images=2, gap=gap, suspicious=suspicious,
    )


def test_gap_suspicious_survive_optuna_round_trip(tmp_path):
    store = _store(tmp_path, name="robust")
    store.append(_trial(0, 0.6, gap=0.12, suspicious=True))
    store.append(_trial(1, 0.9))  # neutral sibling

    reopened = _store(tmp_path, name="robust")
    by_number = {t.number: t for t in reopened.trials}
    assert by_number[0].gap == 0.12
    assert by_number[0].suspicious is True
    assert by_number[1].gap is None
    assert by_number[1].suspicious is False
