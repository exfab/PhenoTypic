"""4.5p1 A2 — ``Trial.gap`` + ``Trial.suspicious`` reach the journal.

Mirrors ``test_study_store_objectives.py``: the robust-eval signals ride the
journal as two new **last** columns (``gap`` nullable float, ``suspicious``
bool). Back-compat is the contract — a legacy ``trials.parquet`` with only the
existing eight columns must still load (``gap=None``, ``suspicious=False`` for
every trial).
"""
from __future__ import annotations

import json

import pandas as pd

from phenotypic.tune._study_store import StudyStore, Trial


def _trial(n: int, score: float, *, gap=None, suspicious=False, **params) -> Trial:
    return Trial(
        number=n, params=params, score=score,
        terms={"Count": score}, n_images=2, gap=gap, suspicious=suspicious,
    )


def test_trial_gap_suspicious_default():
    trial = _trial(0, 0.5, a=1)
    assert trial.gap is None
    assert trial.suspicious is False


def test_gap_suspicious_round_trip_parquet(tmp_path):
    store = StudyStore()
    store.append(_trial(0, 0.6, gap=0.12, suspicious=True, a=1))
    store.append(_trial(1, 0.9, a=2))  # neutral sibling
    path = tmp_path / "trials.parquet"
    store.to_parquet(path)

    back = StudyStore.from_parquet(path)
    by_number = {t.number: t for t in back.trials}
    assert by_number[0].gap == 0.12
    assert by_number[0].suspicious is True
    assert by_number[1].gap is None
    assert by_number[1].suspicious is False


def test_legacy_parquet_without_gap_columns_loads(tmp_path):
    # A pre-4.5p1 journal predating gap/suspicious must still load → neutral.
    legacy = pd.DataFrame(
        [
            {
                "number": 0, "score": 0.3, "n_images": 2,
                "failed": False, "pruned": False,
                "params_json": json.dumps({"a": 1}, sort_keys=True),
                "terms_json": json.dumps({"Count": 0.3}, sort_keys=True),
                "objectives_json": None,
            },
            {
                "number": 1, "score": 0.9, "n_images": 2,
                "failed": False, "pruned": False,
                "params_json": json.dumps({"a": 2}, sort_keys=True),
                "terms_json": json.dumps({"Count": 0.9}, sort_keys=True),
                "objectives_json": None,
            },
        ]
    )
    path = tmp_path / "legacy.parquet"
    legacy.to_parquet(path, index=False)

    back = StudyStore.from_parquet(path)
    assert len(back) == 2
    assert all(t.gap is None for t in back.trials)
    assert all(t.suspicious is False for t in back.trials)
    assert back.best().score == 0.9


def test_gap_columns_appended_last(tmp_path):
    store = StudyStore()
    store.append(_trial(0, 0.6, gap=0.12, suspicious=True, a=1))
    path = tmp_path / "trials.parquet"
    store.to_parquet(path)
    cols = list(pd.read_parquet(path).columns)
    assert cols[-2:] == ["gap", "suspicious"]
