"""Behavioral coverage for Task 1's standalone journal backend contract."""
from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
import sqlite3
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
        f"journal://{output.absolute()}/.pht-tune-cache/journal.log?v=1"
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


@pytest.mark.parametrize(
    ("component", "encoded_component"),
    [
        ("hash#mark", "hash%23mark"),
        ("query?mark", "query%3Fmark"),
        ("percent%mark", "percent%25mark"),
        ("space name", "space%20name"),
        ("unicodé雪", "unicod%C3%A9%E9%9B%AA"),
        ("all #?% 雪", "all%20%23%3F%25%20%E9%9B%AA"),
    ],
)
def test_journal_url_canonically_encodes_path_data_and_round_trips_once(
    tmp_path: Path,
    component: str,
    encoded_component: str,
) -> None:
    """Raw delimiters or a second unquote must not redirect a journal path."""
    from phenotypic.tune._study._storage import (
        journal_path_from_url,
        journal_url_for_path,
    )

    journal = tmp_path / component / "journal.log"
    expected = (
        f"journal://{tmp_path.as_posix()}/{encoded_component}/journal.log?v=1"
    )

    url = journal_url_for_path(journal)

    assert url == expected
    assert journal_path_from_url(url) == journal


@pytest.mark.parametrize("component", ["%2F", "%25", "%", "%ZZ", "%5C", "%5c"])
def test_unmarked_legacy_journal_url_preserves_raw_percent_text(
    component: str,
) -> None:
    """Adding canonical decoding must not reinterpret a persisted legacy path."""
    from phenotypic.tune._study._storage import journal_path_from_url

    assert journal_path_from_url(
        f"journal:///runs/{component}/journal.log"
    ) == Path(f"/runs/{component}/journal.log")


@pytest.mark.parametrize("component", ["%2F", "%25", "%", "%ZZ", "%5C", "%5c"])
def test_versioned_journal_url_canonically_encodes_literal_percent_names(
    component: str,
) -> None:
    """A generated versioned URL must distinguish literal percent text from escapes."""
    from phenotypic.tune._study._storage import (
        journal_path_from_url,
        journal_url_for_path,
    )

    journal = Path("/runs") / component / "journal.log"
    url = journal_url_for_path(journal)

    assert url == f"journal:///runs/%25{component[1:]}/journal.log?v=1"
    assert journal_path_from_url(url) == journal


@pytest.mark.parametrize(
    "storage_url",
    [
        "journal:///runs/bare%/journal.log?v=1",
        "journal:///runs/bad%ZZ/journal.log?v=1",
        "journal:///runs/bad%FF/journal.log?v=1",
        "journal:///runs/encoded%2Fseparator/journal.log?v=1",
        "journal:///runs/lower%2fseparator/journal.log?v=1",
        "journal:///runs/encoded%41/journal.log?v=1",
        "journal:///runs/encoded%5Cseparator/journal.log?v=1",
        "journal:///runs/lower%5cseparator/journal.log?v=1",
        "journal:///C:/runs/a%5Cb/journal.log?v=1",
        "journal:////server/share/a%5Cb/journal.log?v=1",
    ],
)
def test_versioned_journal_url_rejects_malformed_or_noncanonical_paths(
    storage_url: str,
) -> None:
    """Malformed escapes and alternate spellings must not alias canonical paths."""
    from phenotypic.tune._study._storage import journal_path_from_url

    with pytest.raises(ValueError, match="canonical"):
        journal_path_from_url(storage_url)


def test_versioned_journal_url_rejects_literal_posix_backslash_path() -> None:
    """A POSIX filename backslash must not become a Windows separator elsewhere."""
    from phenotypic.tune._study._storage import journal_url_for_path

    journal = PurePosixPath(r"/runs/a\b/journal.log")

    with pytest.raises(ValueError, match="backslash"):
        journal_url_for_path(journal)


@pytest.mark.parametrize(
    "storage_url",
    [
        "journal:///runs/journal.log?v=2",
        "journal:///runs/journal.log?version=1",
        "journal:///runs/journal.log?v=1&mode=rw",
        "journal:///runs/journal.log?v=1#worker",
    ],
)
def test_journal_url_rejects_unknown_or_composed_version_markers(
    storage_url: str,
) -> None:
    """Only the exact canonical marker may occupy URL metadata components."""
    from phenotypic.tune._study._storage import journal_path_from_url

    with pytest.raises(ValueError, match="query or fragment|version marker"):
        journal_path_from_url(storage_url)


@pytest.mark.parametrize(
    "storage_url",
    [
        "journal:///runs/journal.log?mode=rw",
        "journal:///runs/journal.log#worker-1",
        "journal:///runs/journal.log?",
        "journal:///runs/journal.log#",
        "journal:///runs/journal.log?mode=rw#worker-1",
    ],
)
def test_journal_url_rejects_query_and_fragment_components(storage_url: str) -> None:
    """URL metadata must not be silently discarded into another file identity."""
    from phenotypic.tune._study._storage import journal_path_from_url

    with pytest.raises(ValueError, match="query or fragment"):
        journal_path_from_url(storage_url)


@pytest.mark.parametrize(
    ("journal", "expected_url"),
    [
        (
            PureWindowsPath(r"C:\runs\space #?% 雪\journal.log"),
            "journal:///C:/runs/space%20%23%3F%25%20%E9%9B%AA/journal.log?v=1",
        ),
        (
            PureWindowsPath(r"\\server\share\space #?% 雪\journal.log"),
            "journal:////server/share/space%20%23%3F%25%20%E9%9B%AA/journal.log?v=1",
        ),
    ],
)
def test_journal_url_preserves_windows_absolute_path_identity(
    journal: PureWindowsPath,
    expected_url: str,
) -> None:
    """Drive and UNC paths use Windows semantics even on a POSIX test host."""
    from phenotypic.tune._study._storage import (
        journal_path_from_url,
        journal_url_for_path,
    )

    url = journal_url_for_path(journal)

    assert url == expected_url
    assert PureWindowsPath(journal_path_from_url(url).as_posix()) == journal


def test_repairing_backend_appends_a_readable_record_after_a_torn_tail(
    tmp_path: Path,
) -> None:
    """The private Optuna append contract stays readable after repair + append."""
    pytest.importorskip("optuna")
    from optuna.storages.journal import JournalFileSymlinkLock

    from phenotypic.tune._study._storage import _repairing_journal_backend

    journal = tmp_path / "journal.log"
    text_path = str(journal)
    backend = _repairing_journal_backend()(
        text_path,
        lock_obj=JournalFileSymlinkLock(text_path),
    )
    backend.append_logs([{"record": 1}])
    journal.write_bytes(journal.read_bytes() + b'{"torn":')

    backend.append_logs([{"record": 2}])

    assert list(backend.read_logs(0)) == [{"record": 1}, {"record": 2}]


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
    assert "usage: uv run phenotypic-tune" in console_help.stdout
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


def test_noncreating_rdb_schema_probe_rejects_empty_catalog_without_mutation(
    tmp_path: Path,
) -> None:
    """The read-only external-RDB guard must inspect, not initialize, a schema."""
    from phenotypic.tune._study._storage import build_optuna_storage

    database = tmp_path / "empty-schema.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated (value INTEGER)")
        connection.commit()
    before = database.read_bytes()

    with pytest.raises(RuntimeError, match="Optuna.*schema"):
        build_optuna_storage(
            f"sqlite:///{database}",
            create=False,
        )

    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert tables == {"unrelated"}


def _sqlite_catalog_snapshot(
    database: Path,
) -> tuple[bytes, tuple[str, ...], dict[str, tuple[tuple[object, ...], ...]]]:
    payload = database.read_bytes()
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        tables = tuple(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        )
        rows = {
            table: tuple(connection.execute(f'SELECT * FROM "{table}"').fetchall())
            for table in tables
        }
    return payload, tables, rows


def test_noncreating_rdb_rejects_partial_optuna_catalog_without_mutation(
    tmp_path: Path,
) -> None:
    """A plausible partial schema must not be completed by the Optuna constructor."""
    pytest.importorskip("optuna")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from optuna.storages._rdb import models

    from phenotypic.tune._study._storage import build_optuna_storage

    database = tmp_path / "partial-schema.db"
    url = f"sqlite:///{database}"
    engine = sqlalchemy.create_engine(url)
    for table_name in ("version_info", "studies", "study_directions", "trials"):
        models.BaseModel.metadata.tables[table_name].create(engine)
    engine.dispose()
    before = _sqlite_catalog_snapshot(database)

    storage = None
    failure = None
    try:
        storage = build_optuna_storage(url, create=False)
    except RuntimeError as error:
        failure = error
    finally:
        if storage is not None:
            storage.engine.dispose()
    after = _sqlite_catalog_snapshot(database)

    assert after == before
    assert isinstance(failure, RuntimeError)
    assert "schema" in str(failure).lower()


def test_noncreating_rdb_opens_an_initialized_real_optuna_catalog(tmp_path: Path) -> None:
    """The full-schema guard must keep normal existing SQLite studies loadable."""
    optuna = pytest.importorskip("optuna")
    from phenotypic.tune._study._storage import build_optuna_storage

    database = tmp_path / "initialized.db"
    url = f"sqlite:///{database}"
    created = build_optuna_storage(url, create=True)
    optuna.create_study(storage=created, study_name="existing")
    created.engine.dispose()

    reopened = build_optuna_storage(url, create=False)
    try:
        study = optuna.load_study(storage=reopened, study_name="existing")
        assert study.study_name == "existing"
    finally:
        reopened.engine.dispose()


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
