"""4.0c — ``Trial.objectives`` + the ``objectives_json`` journal column (plan §0a).

The multi-objective sidecar reaches the journal: ``Trial`` carries an optional
``objectives`` dict, persisted as an ``objectives_json`` column. Back-compat is the
contract — scalar-only trials round-trip with a ``null`` ``objectives_json``, and a
**legacy parquet with no ``objectives_json`` column at all** must still load (every
trial's ``objectives`` defaults to ``None``).
"""
from __future__ import annotations

import json

import pandas as pd

from phenotypic.tune._study_store import StudyStore, Trial


def _trial(n: int, score: float, *, objectives=None, **params) -> Trial:
    return Trial(
        number=n, params=params, score=score,
        terms={"Count": score}, n_images=2, objectives=objectives,
    )


def test_trial_objectives_defaults_to_none():
    assert _trial(0, 0.5, a=1).objectives is None


def test_scalar_only_round_trip_writes_null_objectives_json(tmp_path):
    # Single-objective trials persist objectives_json as null and reload to None.
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1))
    store.append(_trial(1, 0.9, a=2))
    path = tmp_path / "trials.parquet"
    store.to_parquet(path)

    df = pd.read_parquet(path)
    assert "objectives_json" in df.columns
    assert df["objectives_json"].isna().all()

    back = StudyStore.from_parquet(path)
    assert all(t.objectives is None for t in back.trials)


def test_multi_objective_dict_survives_parquet(tmp_path):
    store = StudyStore()
    store.append(_trial(0, 0.6, objectives={"Dice": 0.8, "IoU": 0.4}, a=1))
    store.append(_trial(1, 0.9, a=2))  # scalar-only sibling
    path = tmp_path / "trials.parquet"
    store.to_parquet(path)
    back = StudyStore.from_parquet(path)
    by_number = {t.number: t for t in back.trials}
    assert by_number[0].objectives == {"Dice": 0.8, "IoU": 0.4}
    assert by_number[1].objectives is None


def test_objectives_json_column_holds_json_string(tmp_path):
    store = StudyStore()
    store.append(_trial(0, 0.6, objectives={"Dice": 0.8, "IoU": 0.4}, a=1))
    path = tmp_path / "trials.parquet"
    store.to_parquet(path)
    df = pd.read_parquet(path)
    payload = df.loc[df["number"] == 0, "objectives_json"].iloc[0]
    assert json.loads(payload) == {"Dice": 0.8, "IoU": 0.4}


def test_legacy_parquet_without_objectives_column_loads(tmp_path):
    # A Phase-1/2 journal predating the column must still load → objectives None.
    legacy = pd.DataFrame(
        [
            {
                "number": 0, "score": 0.3, "n_images": 2,
                "failed": False, "pruned": False,
                "params_json": json.dumps({"a": 1}, sort_keys=True),
                "terms_json": json.dumps({"Count": 0.3}, sort_keys=True),
            },
            {
                "number": 1, "score": 0.9, "n_images": 2,
                "failed": False, "pruned": False,
                "params_json": json.dumps({"a": 2}, sort_keys=True),
                "terms_json": json.dumps({"Count": 0.9}, sort_keys=True),
            },
        ]
    )
    path = tmp_path / "legacy.parquet"
    legacy.to_parquet(path, index=False)

    back = StudyStore.from_parquet(path)
    assert len(back) == 2
    assert all(t.objectives is None for t in back.trials)
    # Cost convention (minimize): the lowest-cost trial wins.
    assert back.best().score == 0.3
