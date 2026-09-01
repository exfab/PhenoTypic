"""Behavioral coverage for Task 1's standalone journal backend contract."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

import pytest


def _installed_tune_console(python_executable: Path, platform: str) -> Path:
    """Return the generated console-script path for an installed environment."""
    suffix = ".exe" if platform == "win32" else ""
    return python_executable.with_name(f"phenotypic-tune{suffix}")


def test_storage_precedence_uses_absolute_run_local_journal_for_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Slurm Optuna run defaults to its own shared journal, not SQLite."""
    from phenotypic.tune._tune_cli._run import _resolve_storage_url

    monkeypatch.delenv("PHENOTYPIC_TUNE_STORAGE_URL", raising=False)
    output = tmp_path / "relative-output"

    assert _resolve_storage_url(None, output, slurm=True) == (
        f"journal://{output.absolute()}/.pht-tune-cache/journal.log"
    )
    assert _resolve_storage_url(
        "postgresql+psycopg://db.example/tune",
        output,
        spec_storage_url="journal:///ignored.log",
        slurm=True,
    ) == "postgresql+psycopg://db.example/tune"


def test_journal_url_round_trips_and_recovers_a_torn_tail(tmp_path: Path) -> None:
    """A failed partial append cannot be joined to the next journal record."""
    from phenotypic.tune._study._storage import (
        journal_path_from_url,
        journal_url_for_path,
        truncate_torn_journal_tail,
    )

    journal = tmp_path / "journal.log"
    journal.write_bytes(b'{"ok":true}\n{"partial"')
    url = journal_url_for_path(journal)

    assert journal_path_from_url(url) == journal
    assert truncate_torn_journal_tail(journal) == len(b'{"partial"')
    assert journal.read_bytes() == b'{"ok":true}\n'


def test_slurm_key_values_override_legacy_aliases_without_duplicate_directives() -> None:
    """Changing an explicit Slurm alias must affect the one emitted directive."""
    from phenotypic.tune._tune_cli._run import merge_slurm_args

    assert merge_slurm_args(
        {"mem": "32G", "slurm_account": "exfab"},
        partition="batch",
        mem="16G",
        time=None,
        constraint=None,
    ) == {
        "slurm_partition": "batch",
        "slurm_mem": "32G",
        "slurm_account": "exfab",
    }


def test_terminal_winner_excludes_an_unevaluated_running_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that dies after ask cannot become the zero-cost winner."""
    from phenotypic.tune._study._optuna_store import OptunaStudyStore
    from phenotypic.tune._study_store import Trial

    states = SimpleNamespace(COMPLETE="complete", PRUNED="pruned", FAIL="fail")
    monkeypatch.setitem(sys.modules, "optuna", SimpleNamespace(trial=SimpleNamespace(TrialState=states)))
    running = SimpleNamespace(state="running", number=0)
    complete = SimpleNamespace(state="complete", number=1)
    store = OptunaStudyStore.__new__(OptunaStudyStore)
    store._study = SimpleNamespace(
        get_trials=lambda *, deepcopy, states=None: [complete]
        if states is not None
        else [running, complete]
    )
    store._to_trial = lambda frozen: Trial(
        number=frozen.number,
        params={},
        score=0.4 if frozen.number else 0.0,
        terms={},
        n_images=1,
    )

    assert store.completed_count() == 1
    assert store.best() is not None
    assert store.best().number == 1


def test_console_entry_and_bare_spec_share_the_module_cli_contract() -> None:
    """Changing the installed target or bare-spec normalization breaks this API."""
    from phenotypic.tune.__main__ import _normalize_argv

    console = _installed_tune_console(Path(sys.executable), sys.platform)
    console_help = subprocess.run(
        [console, "--help"], text=True, capture_output=True, check=False
    )
    module_help = subprocess.run(
        [sys.executable, "-m", "phenotypic.tune", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert console_help.returncode == module_help.returncode == 0
    assert console_help.stdout == module_help.stdout
    assert _normalize_argv(["tuning_spec.json", "-i", "plates"]) == [
        "run",
        "tuning_spec.json",
        "-i",
        "plates",
    ]


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("linux", "phenotypic-tune"), ("win32", "phenotypic-tune.exe")],
)
def test_installed_console_path_uses_the_platform_entrypoint_suffix(
    platform: str, expected: str
) -> None:
    """Using a POSIX-only script name makes the real console test fail on Windows."""
    assert _installed_tune_console(Path("/venv/bin/python"), platform).name == expected


def test_slurm_aliases_render_once_and_keep_mem_gb_units() -> None:
    """Removing rendered-identity folding emits duplicate SBATCH directives."""
    from phenotypic.sdk_.slurm._sbatch import format_sbatch_directives
    from phenotypic.tune._tune_cli._run import merge_slurm_args

    merged = merge_slurm_args(
        {
            "mem_gb": 32,
            "cpus_per_task": 4,
            "slurm_cpus_per_task": 8,
        },
        partition=None,
        mem="16G",
        time=None,
        constraint=None,
    )
    directives = format_sbatch_directives(
        "task-1", merged, Path("out.log"), Path("err.log")
    ).splitlines()

    assert directives.count("#SBATCH --mem=32G") == 1
    assert directives.count("#SBATCH --cpus-per-task=8") == 1


def test_missing_journal_read_only_open_refuses_to_create_a_store(tmp_path: Path) -> None:
    """Dropping the pre-open guard creates an empty journal during observation."""
    from phenotypic.tune._study._optuna_store import _require_existing_backing_store
    from phenotypic.tune._study._storage import journal_url_for_path

    journal = tmp_path / "missing.log"

    with pytest.raises(FileNotFoundError):
        _require_existing_backing_store(journal_url_for_path(journal))
    assert not journal.exists()


def test_slurm_sqlite_rejection_happens_before_run_artifacts(tmp_path: Path) -> None:
    """Moving Slurm validation after setup would leave a rejected run on disk."""
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.tune import Evaluator, SearchSpace
    from phenotypic.tune.score import Scorer
    from phenotypic.tune.strategy import OptunaConfig
    from phenotypic.tune._spec import Budget, TuningSpec
    from phenotypic.tune._tune_cli._run import run_tuning

    class _ConstScorer(Scorer):
        def _score_terms(self, image, measurements) -> dict[str, float]:
            return {"Count": 0.5}

    output = tmp_path / "rejected"
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(knobs=()),
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=OptunaConfig(
            sampler="tpe", n_trials=1, storage_url=f"sqlite:///{tmp_path / 'study.db'}"
        ),
        budget=Budget(n_trials=1),
    )

    with pytest.raises(ValueError, match="SQLite"):
        run_tuning(
            spec,
            [],
            output,
            slurm=True,
            spec_path=tmp_path / "spec.json",
            images_dir=tmp_path / "images",
        )
    assert not output.exists()
