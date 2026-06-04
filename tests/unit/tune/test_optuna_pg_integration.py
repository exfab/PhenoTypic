"""Gated Postgres integration for the distributed shared-study pattern (F3).

Skipped unless ``PHENOTYPIC_TEST_PG_URL`` points at a live Postgres server (the
conftest autoskips ``@pytest.mark.postgres``). Proves the distributed
ask-and-tell pattern (optuna-integration §7): two ``OptunaStudyStore`` handles on
the **same study by name + URL** both observe every trial — the shared,
concurrently-writable study underpinning local + SLURM execution.
"""
from __future__ import annotations

import importlib.util
import os
import uuid

import pytest

from phenotypic.tune._study_store import Trial

_OPTUNA = importlib.util.find_spec("optuna") is not None

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed"),
]


def _trial(number: int, score: float) -> Trial:
    return Trial(
        number=number,
        params={"a": number},
        score=score,
        terms={"t": score},
        n_images=2,
    )


def test_distributed_study_over_postgres():
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = os.environ["PHENOTYPIC_TEST_PG_URL"]
    assert url.startswith("postgresql+psycopg://"), (
        "the tune study DB uses the psycopg driver scheme"
    )
    # A unique study name keeps concurrent CI runs from colliding on the server.
    study_name = f"tune_pg_it_{uuid.uuid4().hex[:12]}"

    # Driver A opens the study and writes a trial.
    store_a = OptunaStudyStore(storage_url=url, study_name=study_name)
    store_a.append(_trial(0, 0.5))

    # Driver B opens the SAME study by name + URL — it sees driver A's trial.
    store_b = OptunaStudyStore(storage_url=url, study_name=study_name)
    assert len(store_b) == 1
    assert store_b.trials[0].score == 0.5

    # Driver B writes; driver A sees it too (the shared study is bidirectional).
    store_b.append(_trial(1, 0.9))
    assert len(store_a) == 2
    assert store_a.best().score == 0.9
    assert {t.number for t in store_a.trials} == {0, 1}
