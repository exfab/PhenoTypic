"""The ``StudyStore`` Protocol + resume awareness.

The concrete journal is now ``JournalStudyStore`` (back-compat alias
``StudyStore`` still exported from ``phenotypic.tune``). The engine types its
``store`` against the ``StudyStore`` *Protocol* in ``tune/_study/_protocol.py``
and branches resume on ``is_resumable_in_place()``: a journal replays the
deterministic strategy past recorded trials, an in-place-resumable store (e.g. a
future Optuna RDB) skips the replay.
"""
from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import (
    Budget,
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    Scorer,
    SearchSpace,
    StudyStore,
    TuningEngine,
    TuningSpec,
)
from phenotypic.tune._study._protocol import StudyStore as StudyStoreProtocol
from phenotypic.tune._study_store import JournalStudyStore, Trial


class _ConstScorer(Scorer):
    def score_image(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


def _grid_space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _spec(budget: Budget) -> TuningSpec:
    return TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=_grid_space(),
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=budget,
    )


def test_backcompat_alias_is_the_concrete_journal():
    # The public StudyStore name remains usable and is the concrete journal.
    assert StudyStore is JournalStudyStore
    store = StudyStore()
    assert store.is_resumable_in_place() is False


def test_journal_satisfies_protocol_by_calling_each_method():
    # Structurally satisfy the Protocol by exercising every method, not isinstance.
    store: StudyStoreProtocol = JournalStudyStore()
    store.append(Trial(number=0, params={"a": 1}, score=0.5, terms={"X": 0.5}, n_images=2))
    assert len(store) == 1
    assert [t.number for t in store.trials] == [0]
    best = store.best()
    assert best is not None and best.number == 0
    assert store.is_resumable_in_place() is False
    assert store.completed_count() == 1


def test_completed_count_excludes_failures():
    store = JournalStudyStore()
    store.append(Trial(number=0, params={}, score=1.0, terms={"X": 1.0}, n_images=2))
    store.append(
        Trial(number=1, params={}, score=0.0, terms={}, n_images=2, failed=True)
    )
    store.append(
        Trial(number=2, params={}, score=0.2, terms={"X": 0.2}, n_images=1, pruned=True)
    )
    # Completed = non-failed (a pruned trial is a real, partial evaluation).
    assert store.completed_count() == 2


class _FakeResumableStore:
    """A minimal in-place-resumable store the engine can drive (no replay)."""

    def __init__(self) -> None:
        self._trials: list[Trial] = []
        self.suggest_replayed = 0

    def append(self, trial: Trial) -> None:
        self._trials.append(trial)

    @property
    def trials(self) -> list[Trial]:
        return list(self._trials)

    def __len__(self) -> int:
        return len(self._trials)

    def best(self):
        valid = [t for t in self._trials if not t.failed]
        return max(valid, key=lambda t: t.score) if valid else None

    def is_resumable_in_place(self) -> bool:
        return True

    def completed_count(self) -> int:
        return sum(1 for t in self._trials if not t.failed)


def test_engine_drives_a_fake_resumable_store_without_replay(monkeypatch):
    # When the store is resumable in place, the engine must NOT fast-forward the
    # strategy with len()-many suggest() replays.
    from phenotypic.tune._strategies._config import GridConfig as _GridConfig

    replays = {"count": 0}
    fed = {"first": True}

    class _CountingGrid:
        def suggest(self):
            from phenotypic.tune._strategies._pruning import NoOpChannel
            replays["count"] += 1
            return {"0.ignore_zeros": True}, NoOpChannel()

        def register_result(self, params, result, *, pruned=False) -> None:
            return None

        def is_exhausted(self) -> bool:
            # Allow exactly one fresh suggest beyond setup, then stop.
            if fed["first"]:
                fed["first"] = False
                return False
            return True

    monkeypatch.setattr(
        _GridConfig, "build",
        lambda self, space, store, *, directions=None: _CountingGrid(),
    )

    store = _FakeResumableStore()
    # Pre-seed a prior trial: an in-place store would NOT replay these.
    store.append(Trial(number=0, params={}, score=1.0, terms={"X": 1.0}, n_images=1))
    engine = TuningEngine(_spec(Budget(n_trials=2)), store=store)
    engine.optimize([load_synth_yeast_plate()])
    # Exactly one fresh suggest (the new trial); no replay of the seeded trial.
    assert replays["count"] == 1
