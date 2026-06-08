from __future__ import annotations

import json

import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tools_ import _io_constants as io
from phenotypic.tune import (
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    ReferenceFreeScorer,
    QCScorer,
    SearchSpace,
)
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.tune._tune_cli._run import run_tuning


def _spec(tmp_path) -> TuningSpec:
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["Synthetic96PlateWithObjects"] * 96,
         "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=QCScorer(check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"])),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def test_run_tuning_writes_deliverables(tmp_path):
    out = tmp_path / "run"
    best = run_tuning(_spec(tmp_path), [load_synth_yeast_plate()], out)

    assert io.best_pipeline_path(out).exists()
    assert io.tuning_spec_path(out).exists()
    assert io.param_importance_path(out).exists()
    assert io.trials_parquet_path(out).exists()
    # the written best pipeline reloads as a runnable ImagePipeline
    winner = ImagePipeline.from_json(io.best_pipeline_path(out).read_text())
    assert "OtsuDetector" in winner.get_ops()
    # importance covers the tuned knob
    imp = json.loads(io.param_importance_path(out).read_text())
    assert "1.ignore_zeros" in imp
    assert best is not None


def test_cli_main_invokes_run(tmp_path, monkeypatch):
    from phenotypic.tune import __main__ as cli

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    out = tmp_path / "out"

    # patch image loading (no PNG fixtures needed)
    monkeypatch.setattr(cli, "_load_images", lambda _p: [load_synth_yeast_plate()])
    cli.main([str(spec_path), "-i", str(tmp_path), "-o", str(out)])

    assert io.best_pipeline_path(out).exists()


# --- H2: run-subparser flags + journal export ---------------------------------

import importlib.util  # noqa: E402

import pytest  # noqa: E402

from phenotypic.tune import OptunaConfig, RandomConfig  # noqa: E402
from phenotypic.tune._tune_cli._run import resolve_strategy  # noqa: E402

_OPTUNA = importlib.util.find_spec("optuna") is not None


def test_resolve_strategy_grid_and_random():
    assert isinstance(resolve_strategy("grid", n_trials=None, storage_url=None), GridConfig)
    rnd = resolve_strategy("random", n_trials=7, storage_url=None)
    assert isinstance(rnd, RandomConfig)
    assert rnd.n_trials == 7


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_resolve_strategy_tpe_builds_optuna_config():
    cfg = resolve_strategy("tpe", n_trials=12, storage_url="sqlite:///x.db")
    assert isinstance(cfg, OptunaConfig)
    assert cfg.sampler == "tpe"
    assert cfg.n_trials == 12
    assert cfg.storage_url == "sqlite:///x.db"


def test_resolve_strategy_optuna_without_extra_raises_actionable(monkeypatch):
    # Without the tune extra, requesting an Optuna sampler raises an actionable
    # error pointing at `uv sync --extras tune` — never a bare KeyError.
    import phenotypic.tune._strategies._optuna_support as support

    def _boom():
        raise ImportError("Optuna is required ... uv sync --extras tune")

    monkeypatch.setattr(support, "_require_optuna", _boom)
    with pytest.raises(ImportError, match="tune"):
        resolve_strategy("nsga2", n_trials=5, storage_url=None)


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_optuna_strategy_run_writes_trials_parquet(tmp_path):
    # An Optuna-strategy run selects the OptunaStudyStore (study.db) AND exports
    # trials.parquet at finalize, so deliverables/ stay backend-agnostic.
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = tmp_path / "optuna_run"
    best = run_tuning(
        _spec(tmp_path),
        [load_synth_yeast_plate()],
        out,
        strategy="tpe",
        n_trials=4,
    )
    assert io.tune_cache_study_db_path(out).exists()  # the Optuna study DB
    assert io.trials_parquet_path(out).exists()  # exported at the output root
    assert io.best_pipeline_path(out).exists()
    assert best is not None


def test_cli_storage_url_env_fallback(tmp_path, monkeypatch):
    # --storage-url is omitted but $PHENOTYPIC_TUNE_STORAGE_URL is set: the run
    # routes the env URL into the resolved Optuna strategy.
    from phenotypic.tune import __main__ as cli
    from phenotypic.tune._strategies._config import PHENOTYPIC_TUNE_STORAGE_URL_ENV

    captured = {}

    def _fake_run_tuning(spec, images, output_dir, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(cli, "run_tuning", _fake_run_tuning)
    monkeypatch.setattr(cli, "_load_images", lambda _p: [load_synth_yeast_plate()])
    monkeypatch.setenv(PHENOTYPIC_TUNE_STORAGE_URL_ENV, "sqlite:///env.db")

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    cli.main([str(spec_path), "-i", str(tmp_path), "-o", str(tmp_path / "o"),
              "--strategy", "tpe", "--n-trials", "3"])

    assert captured["strategy"] == "tpe"
    assert captured["storage_url"] == "sqlite:///env.db"


def test_open_store_uses_env_url_for_local_run(tmp_path, monkeypatch):
    # An env-var-driven LOCAL run (storage_url=None, slurm=False, env set): the
    # engine MUST open the env Postgres URL, not the local sqlite study.db. This
    # guards the marker-vs-engine divergence — _open_store and the run.json
    # marker must agree on the URL via the single _resolve_storage_url fallback.
    import phenotypic.tune._study._optuna_store as store_mod
    from phenotypic.tune import OptunaConfig
    from phenotypic.tune._strategies._config import PHENOTYPIC_TUNE_STORAGE_URL_ENV
    from phenotypic.tune._tune_cli._run import _open_store, _resolve_storage_url

    env_url = "postgresql://host:5432/tune"
    monkeypatch.setenv(PHENOTYPIC_TUNE_STORAGE_URL_ENV, env_url)

    opened = {}

    class _FakeOptunaStudyStore:
        def __init__(self, *, storage_url, study_name, directions=None):
            opened["url"] = storage_url

    monkeypatch.setattr(store_mod, "OptunaStudyStore", _FakeOptunaStudyStore)

    out = tmp_path / "run"
    strategy = OptunaConfig(sampler="tpe", n_trials=3)
    _open_store(
        strategy,
        out,
        storage_url=None,
        resume_path=out / "trials.parquet",
        directions=None,
    )

    # The engine opened the env URL (NOT sqlite:///…study.db).
    assert opened["url"] == env_url
    # …and the marker's resolver agrees — single source of truth.
    assert _resolve_storage_url(None, out) == env_url


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_spec_storage_url_wins_over_env_without_cli_url(tmp_path, monkeypatch):
    import phenotypic.tune._study._optuna_store as store_mod
    from phenotypic.tune._strategies._config import PHENOTYPIC_TUNE_STORAGE_URL_ENV

    env_url = "sqlite:///env.db"
    spec_url = f"sqlite:///{tmp_path / 'spec.db'}"
    monkeypatch.setenv(PHENOTYPIC_TUNE_STORAGE_URL_ENV, env_url)

    opened = {}

    class _FakeOptunaStudyStore:
        def __init__(self, *, storage_url, study_name, directions=None):
            opened["url"] = storage_url
            self.trials = []

        def is_resumable_in_place(self):
            return True

        def pareto_front(self):
            return []

        def best(self):
            return None

        def param_importances(self):
            return None

        def to_parquet(self, _path):
            pass

    class _FakeEngine:
        def __init__(self, spec, store):
            pass

        def optimize(self, images):
            return None

        def best_pipeline(self):
            return None

    monkeypatch.setattr(store_mod, "OptunaStudyStore", _FakeOptunaStudyStore)
    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run.TuningEngine", _FakeEngine
    )
    spec = _spec(tmp_path).model_copy(
        update={
            "strategy": OptunaConfig(
                sampler="tpe", n_trials=1, storage_url=spec_url
            )
        }
    )

    run_tuning(spec, [load_synth_yeast_plate()], tmp_path / "out")
    assert opened["url"] == spec_url


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_cli_storage_url_wins_over_spec_url(tmp_path, monkeypatch):
    import phenotypic.tune._study._optuna_store as store_mod

    spec_url = f"sqlite:///{tmp_path / 'spec.db'}"
    cli_url = f"sqlite:///{tmp_path / 'cli.db'}"
    opened = {}

    class _FakeOptunaStudyStore:
        def __init__(self, *, storage_url, study_name, directions=None):
            opened["url"] = storage_url
            self.trials = []

        def is_resumable_in_place(self):
            return True

        def pareto_front(self):
            return []

        def best(self):
            return None

        def param_importances(self):
            return None

    class _FakeEngine:
        def __init__(self, spec, store):
            pass

        def optimize(self, images):
            return None

        def best_pipeline(self):
            return None

    monkeypatch.setattr(store_mod, "OptunaStudyStore", _FakeOptunaStudyStore)
    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run.TuningEngine", _FakeEngine
    )
    spec = _spec(tmp_path).model_copy(
        update={
            "strategy": OptunaConfig(
                sampler="tpe", n_trials=1, storage_url=spec_url
            )
        }
    )

    run_tuning(
        spec,
        [load_synth_yeast_plate()],
        tmp_path / "out",
        storage_url=cli_url,
    )

    assert opened["url"] == cli_url


def test_n_trials_overrides_existing_random_strategy(tmp_path, monkeypatch):
    captured = {}

    class _FakeEngine:
        def __init__(self, spec, store):
            captured["n_trials"] = spec.strategy.n_trials

        def optimize(self, images):
            return None

        def best_pipeline(self):
            return None

    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run.TuningEngine", _FakeEngine
    )
    spec = _spec(tmp_path).model_copy(
        update={"strategy": RandomConfig(n_trials=100)}
    )

    run_tuning(spec, [load_synth_yeast_plate()], tmp_path / "out", n_trials=5)

    assert captured["n_trials"] == 5


def test_n_trials_override_rejects_grid_strategy(tmp_path):
    with pytest.raises(ValueError, match="--n-trials"):
        run_tuning(
            _spec(tmp_path),
            [load_synth_yeast_plate()],
            tmp_path / "out",
            n_trials=5,
        )


def test_n_trials_rejects_explicit_grid_strategy(tmp_path):
    with pytest.raises(ValueError, match="--n-trials"):
        run_tuning(
            _spec(tmp_path),
            [load_synth_yeast_plate()],
            tmp_path / "out",
            strategy="grid",
            n_trials=5,
        )


def test_unavailable_reference_free_scorer_fails_before_artifacts(tmp_path):
    out = tmp_path / "out"
    spec = _spec(tmp_path).model_copy(update={"scorer": ReferenceFreeScorer()})

    with pytest.raises(ValueError, match="ReferenceFreeScorer"):
        run_tuning(spec, [load_synth_yeast_plate()], out)

    assert not io.tuning_spec_path(out).exists()


def test_cli_screen_flag_toggles_screening(tmp_path, monkeypatch):
    from phenotypic.tune import __main__ as cli

    captured = {}

    def _fake_run_tuning(spec, images, output_dir, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(cli, "run_tuning", _fake_run_tuning)
    monkeypatch.setattr(cli, "_load_images", lambda _p: [load_synth_yeast_plate()])

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    cli.main([str(spec_path), "-i", str(tmp_path), "-o", str(tmp_path / "o"),
              "--screen"])
    assert captured["screen"] is True

    cli.main([str(spec_path), "-i", str(tmp_path), "-o", str(tmp_path / "o2"),
              "--no-screen"])
    assert captured["screen"] is False


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_cli_slurm_flag_uses_slurm_executor(tmp_path, monkeypatch):
    # --slurm routes through the SlurmExecutor worker-fleet submission instead of
    # the local in-process engine run.
    from phenotypic.tune._tune_cli import _run as run_mod

    submitted = {}

    class _FakeSlurmExecutor:
        def __init__(self, **kwargs):
            submitted["kwargs"] = kwargs

        def run(self, work, items):
            submitted["ran"] = True
            return ["9001"]

    monkeypatch.setattr(run_mod, "SlurmExecutor", _FakeSlurmExecutor)

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    out = tmp_path / "slurm_out"
    run_tuning(
        _spec(tmp_path),
        [load_synth_yeast_plate()],
        out,
        strategy="tpe",
        n_trials=4,
        storage_url=f"sqlite:///{out / 'study.db'}",
        slurm=True,
        spec_path=spec_path,
        images_dir=tmp_path,
    )
    assert submitted.get("ran") is True
