# tests/unit/tune/test_study_persistence_cutover.py
"""Phase 2: persistence hard cutover — no silent maximize, stamp + detector.

Verified optuna 4.9.0 hazard: ``create_study(load_if_exists=True,
direction="minimize")`` against an existing ``maximize`` study does NOT raise; it
silently keeps ``MAXIMIZE``. The name bump (``tune`` → ``tune_cost_v1``) makes
reopening the legacy study impossible by construction. These tests assert: the
new store opens the bumped minimize study; a pre-existing legacy ``"tune"``
maximize study in the SAME storage stays inert; the friendly detector fires; and
a fresh minimize study carries the ``tune_convention`` stamp.
"""
from __future__ import annotations

import importlib.util

import pytest

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")

_CONVENTION_KEY = "tune_convention"
_CONVENTION_VALUE = "minimize-cost-v1"


def _seed_legacy_maximize_study(url: str) -> None:
    """Write a pre-cutover ``"tune"`` MAXIMIZE study into the storage."""
    import optuna

    legacy = optuna.create_study(
        storage=url, study_name="tune", direction="maximize"
    )
    legacy.add_trial(
        optuna.trial.create_trial(
            value=0.95, params={}, distributions={},
            state=optuna.trial.TrialState.COMPLETE,
        )
    )


def test_fresh_study_minimizes_and_is_stamped(tmp_path):
    import optuna

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = f"sqlite:///{tmp_path / 'study.db'}"
    store = OptunaStudyStore(storage_url=url, study_name="tune_cost_v1")
    assert store.study.direction == optuna.study.StudyDirection.MINIMIZE
    assert store.study.user_attrs.get(_CONVENTION_KEY) == _CONVENTION_VALUE


def test_legacy_maximize_study_left_inert(tmp_path, caplog):
    import logging

    import optuna

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = f"sqlite:///{tmp_path / 'study.db'}"
    _seed_legacy_maximize_study(url)

    with caplog.at_level(logging.WARNING):
        store = OptunaStudyStore(storage_url=url, study_name="tune_cost_v1")

    # The new store opened the BUMPED, MINIMIZE study — never the legacy one.
    assert store.study.study_name == "tune_cost_v1"
    assert store.study.direction == optuna.study.StudyDirection.MINIMIZE

    # The legacy study is still present and still MAXIMIZE (inert, not reopened).
    legacy = optuna.load_study(storage=url, study_name="tune")
    assert legacy.direction == optuna.study.StudyDirection.MAXIMIZE

    # Friendly detector fired with an actionable message.
    assert any(
        "pre-cutover" in rec.getMessage().lower()
        or "tune_cost_v1" in rec.getMessage()
        for rec in caplog.records
    )


def test_no_detector_warning_without_legacy_study(tmp_path, caplog):
    import logging

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = f"sqlite:///{tmp_path / 'study.db'}"
    with caplog.at_level(logging.WARNING):
        OptunaStudyStore(storage_url=url, study_name="tune_cost_v1")
    assert not any("pre-cutover" in rec.getMessage().lower() for rec in caplog.records)
