"""The strategy + store share ONE Optuna study (no phantom; bounded resume).

Regression lock for the two-study split: ``OptunaConfig.build`` must bind the
``OptunaStrategy`` to the **store's** named study, so the strategy's native
``ask``/``tell`` trials (sampler-learnable distributions) ARE the persisted
record — not a throwaway auto-named study written alongside the store's
``add_trial`` mirror. Consequences the locks below pin (all hermetic SQLite; the
bug is storage-agnostic, so no Postgres is needed):

* one study per run (no ``no-name-<uuid>`` phantom in the storage),
* a resume of an ``n_trials``-complete study adds **nothing** (the shared
  ``is_exhausted`` counts the persisted trials — the SLURM fleet drains one
  budget the same way), and
* the persisted trials carry native ``distributions`` (proving the strategy is
  the sole writer) plus the ``pheno_params`` user-attr (so the store still
  reconstructs the full :class:`Trial`).
"""
from __future__ import annotations

import importlib.util

import pytest

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.tune import (
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
)
from phenotypic.tune.score import Scorer
from phenotypic.tune.strategy import OptunaConfig
from phenotypic.tune._engine import TuningEngine
from phenotypic.tune._spec import Budget, TuningSpec

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")

#: The shared study name the run layer (``_open_store``) hardcodes.
_STUDY = "tune"


class _ConstScorer(Scorer):
    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


def _space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.BlurGauss.__enabled__",
             domain=Categorical(choices=(True, False))),
        Knob(key="0.sigma", domain=Categorical(choices=(1.0, 2.0)),
             conditional_on=(("0.BlurGauss.__enabled__", True),)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _spec(url: str, n_trials: int) -> TuningSpec:
    return TuningSpec(
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
        search_space=_space(),
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=OptunaConfig(sampler="tpe", n_trials=n_trials, storage_url=url),
        budget=Budget(),
    )


def _store(url: str):
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    return OptunaStudyStore(storage_url=url, study_name=_STUDY)


def _run(url: str, n_trials: int, images: list) -> None:
    """Mirror the run layer's wiring: a shared-named store + the engine loop."""
    TuningEngine(_spec(url, n_trials), store=_store(url)).optimize(images)


def test_optuna_run_writes_a_single_study_no_phantom(tmp_path):
    import optuna

    url = f"sqlite:///{tmp_path / 'study.db'}"
    _run(url, 4, [load_synth_yeast_plate()])

    summaries = optuna.get_all_study_summaries(storage=url)
    # Exactly one study — the strategy bound to the store's "tune", not a
    # throwaway auto-named one written alongside it.
    assert [s.study_name for s in summaries] == [_STUDY]
    assert _store(url).completed_count() == 4


def test_optuna_resume_honors_total_budget(tmp_path):
    url = f"sqlite:///{tmp_path / 'study.db'}"
    images = [load_synth_yeast_plate()]

    _run(url, 4, images)
    assert len(_store(url)) == 4
    # Re-running the same n_trials against the n_trials-complete shared study
    # adds nothing (is_exhausted counts the persisted trials) — NOT 8.
    _run(url, 4, images)
    assert len(_store(url)) == 4


def test_shared_study_trials_carry_native_distributions(tmp_path):
    import optuna

    url = f"sqlite:///{tmp_path / 'study.db'}"
    _run(url, 4, [load_synth_yeast_plate()])

    study = optuna.load_study(study_name=_STUDY, storage=url)
    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    assert complete, "expected at least one COMPLETE trial"
    # Native distributions ⇒ the strategy's ask/tell is the writer (sampler can
    # learn); the pheno_params user-attr ⇒ the store still reconstructs the
    # full materialized combo (including Fixed knobs).
    assert all(t.distributions for t in complete)
    assert all("pheno_params" in t.user_attrs for t in complete)


def test_config_build_binds_strategy_to_store_study(tmp_path):
    # Unit-level seam: build() must pull the study name + URL off the store so
    # the strategy opens the SAME study (load_if_exists), not its own.
    url = f"sqlite:///{tmp_path / 'study.db'}"
    strat = OptunaConfig(sampler="tpe", n_trials=3).build(_space(), _store(url))
    assert strat._study_name == _STUDY
    assert strat._storage_url == url
