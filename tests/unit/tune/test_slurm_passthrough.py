"""SLURM config passthrough + study-name sanitization (Changes 6 & 7).

Script generation is pure string-building (no live ``sbatch``), so these run in
CI without the ``slurm`` marker:

* ``_submit_slurm_fleet`` threads ``--n-workers`` / ``--slurm-partition`` /
  ``--slurm-mem`` / ``--slurm-time`` into ``SlurmExecutor`` and OMITS the
  ``#SBATCH --partition`` directive when the partition is ``None`` (no longer
  hardcoded to ``"batch"``);
* the SLURM job name + ``# Study:`` comment sanitize a study name containing
  characters outside ``[A-Za-z0-9._-]``.
"""
from __future__ import annotations

import importlib.util

import pytest

_OPTUNA = importlib.util.find_spec("optuna") is not None


# --- study-name sanitization (Change 7) ---------------------------------------


def test_sanitize_job_name_replaces_unsafe_chars():
    from phenotypic._execution._slurm import _sanitize_job_name

    assert _sanitize_job_name("my study/v2 (alpha)") == "my-study-v2--alpha-"
    # Safe characters are preserved.
    assert _sanitize_job_name("tune_2024.01-A") == "tune_2024.01-A"


def test_worker_script_sanitizes_study_name_in_job_name_and_comment(tmp_path):
    from phenotypic._execution._slurm import SlurmExecutor

    ex = SlurmExecutor(
        output_dir=tmp_path,
        spec_path=tmp_path / "tuning_spec.json",
        images_dir=tmp_path / "images",
        study_name="bad name/v2",
        storage_url=f"sqlite:///{tmp_path / 'study.db'}",
        n_workers=2,
        slurm_args={},
    )
    content = ex.generate_worker_array_script().read_text()
    lines = content.splitlines()
    # The job-name directive + the # Study: comment use the SANITIZED token.
    assert "#SBATCH --job-name=pht-tune-bad-name-v2" in lines
    assert "# Study: bad-name-v2" in lines
    # The raw (unsafe) name never appears in either of those two surfaces (it
    # *does* legitimately appear in the shlex-quoted --study-name worker arg,
    # because the study really is named that — sanitization is only for the
    # SBATCH directive + comment, which is Change 7's scope).
    job_name_line = next(
        line for line in lines if line.startswith("#SBATCH --job-name=")
    )
    study_comment_line = next(
        line for line in lines if line.startswith("# Study:")
    )
    assert "bad name/v2" not in job_name_line
    assert "bad name/v2" not in study_comment_line


# --- SLURM passthrough (Change 6) ---------------------------------------------


def _patched_fleet_call(tmp_path, monkeypatch, **fleet_kwargs):
    """Call ``_submit_slurm_fleet`` with the study pre-create + executor mocked.

    Returns the kwargs ``SlurmExecutor`` was constructed with so a test can
    assert what got threaded through. ``executor.run`` is a no-op.
    """
    from phenotypic.tune._tune_cli import _run as run_mod

    captured: dict = {}

    class _FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, work, items):
            captured["_run_items"] = list(items)
            return []

    # Don't actually open Optuna / hit the DB during the submit path.
    monkeypatch.setattr(run_mod, "SlurmExecutor", _FakeExecutor)
    monkeypatch.setattr(
        run_mod, "objective_directions", lambda scorer: None
    )

    class _FakeStore:
        def __init__(self, **kwargs):
            pass

    import phenotypic.tune._study._optuna_store as store_mod

    monkeypatch.setattr(store_mod, "OptunaStudyStore", _FakeStore)
    monkeypatch.setattr(
        "phenotypic._cli._cli_utils.get_python_command",
        lambda for_slurm=False: (["python"], None),
    )

    spec = _minimal_spec()
    run_mod._submit_slurm_fleet(
        spec,
        tmp_path,
        storage_url=f"sqlite:///{tmp_path / 'study.db'}",
        spec_path=tmp_path / "spec.json",
        images_dir=tmp_path / "images",
        **fleet_kwargs,
    )
    return captured


def _minimal_spec():
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.tune import Evaluator, Scorer, SearchSpace
    from phenotypic.tune._spec import Budget, TuningSpec
    from phenotypic.tune._strategies._config import OptunaConfig

    class _ConstScorer(Scorer):
        def score_image(self, image, measurements) -> dict[str, float]:
            return {"Count": 1.0}

    return TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(knobs=()),
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=OptunaConfig(sampler="tpe", n_trials=10),
        budget=Budget(),
    )


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_passthrough_threads_flags_into_executor(tmp_path, monkeypatch):
    captured = _patched_fleet_call(
        tmp_path,
        monkeypatch,
        n_workers=3,
        slurm_partition="gpu",
        slurm_mem="16G",
        slurm_time="08:00:00",
    )
    assert captured["n_workers"] == 3
    assert captured["_run_items"] == [0, 1, 2]
    assert captured["slurm_args"] == {
        "slurm_partition": "gpu",
        "slurm_mem": "16G",
        "slurm_time": "08:00:00",
    }


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_partition_omitted_when_none(tmp_path, monkeypatch):
    captured = _patched_fleet_call(tmp_path, monkeypatch)
    # No partition flag => slurm_args carries no partition key (cluster default),
    # NOT the old hardcoded "batch".
    assert "slurm_partition" not in captured["slurm_args"]
    assert captured["slurm_args"] == {}


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_n_workers_defaults_when_unset(tmp_path, monkeypatch):
    # n_trials=10 in the minimal spec → default min(8, 10) == 8.
    captured = _patched_fleet_call(tmp_path, monkeypatch)
    assert captured["n_workers"] == 8
