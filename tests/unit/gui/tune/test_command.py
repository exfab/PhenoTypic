"""Unit tests for the Launch command renderer (Task C1).

:func:`~phenotypic.gui.tune._command.render_launch_command` is the
**source-of-truth** for the ``python -m phenotypic.tune run …`` invocation the
Launch view shows (the clientside callback mirrors it, but this pure function is
the unit-tested truth). The flag spellings are confirmed against the real CLI in
``src/phenotypic/tune/__main__.py``: positional ``spec``, ``-i``/``-o``,
``--strategy``, ``--n-trials``, ``--storage-url``, ``--screen``, ``--slurm``.

The renderer only RENDERS a string — it never spawns a process (the no-re-optimize
lock). Importing it must never drag ``optuna`` into ``sys.modules``.
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

from phenotypic.gui.shell._sandbox import SandboxRoot


def _command_paths(tmp_path: Path) -> tuple[SandboxRoot, Path, Path, Path]:
    spec = tmp_path / "spec with spaces.json.pht-tune"
    spec.write_text("{}", encoding="utf-8")
    images = tmp_path / "plate images"
    images.mkdir()
    output = tmp_path / "tune output"
    return SandboxRoot.from_path(tmp_path), spec, images, output


def test_render_postgres_tpe_run_includes_strategy_trials_and_storage() -> None:
    from phenotypic.gui.tune._command import render_launch_command

    command = render_launch_command(
        "out/deliverables/tuning_spec.json",
        "images",
        "out",
        strategy="tpe",
        n_trials=50,
        storage_url="postgresql+psycopg://tuner@db:5432/tune",
        screen=False,
        slurm=False,
    )
    assert "python -m phenotypic.tune run" in command
    assert "--strategy tpe" in command
    assert "--n-trials 50" in command
    assert "--storage-url postgresql+psycopg://tuner@db:5432/tune" in command
    # The positional spec + input/output flags are present.
    assert "out/deliverables/tuning_spec.json" in command
    assert "-i images" in command
    assert "-o out" in command
    # No toggled-off flags leak in.
    assert "--screen" not in command
    assert "--slurm" not in command


def test_render_local_grid_omits_storage_screen_and_slurm() -> None:
    from phenotypic.gui.tune._command import render_launch_command

    command = render_launch_command(
        "spec.json",
        "imgs",
        "run",
        strategy="grid",
        n_trials=None,
        storage_url=None,
        screen=False,
        slurm=False,
    )
    assert "--strategy grid" in command
    # grid is exhaustive — an omitted n_trials renders no flag.
    assert "--n-trials" not in command
    assert "--storage-url" not in command
    assert "--screen" not in command
    assert "--slurm" not in command


def test_render_grid_suppresses_n_trials_even_when_set() -> None:
    """Grid is exhaustive and ignores ``--n-trials`` — never emit it for grid."""
    from phenotypic.gui.tune._command import render_launch_command

    command = render_launch_command(
        "spec.json",
        "imgs",
        "run",
        strategy="grid",
        n_trials=50,  # a stale budget left in the form must NOT leak into grid
        storage_url=None,
        screen=False,
        slurm=False,
    )
    assert "--strategy grid" in command
    assert "--n-trials" not in command


def test_render_appends_screen_and_slurm_when_toggled() -> None:
    from phenotypic.gui.tune._command import render_launch_command

    command = render_launch_command(
        "spec.json",
        "imgs",
        "run",
        strategy="random",
        n_trials=25,
        storage_url=None,
        n_workers=None,
        slurm_partition=None,
        slurm_mem=None,
        slurm_time=None,
        held_out_fraction=None,
        cv_group=None,
        screen=True,
        slurm=True,
    )
    assert "--strategy random" in command
    assert "--n-trials 25" in command
    assert "--screen" in command
    assert "--slurm" in command


def test_render_includes_run_form_overrides() -> None:
    from phenotypic.gui.tune._command import render_launch_command

    command = render_launch_command(
        "spec.json",
        "imgs",
        "out",
        strategy="tpe",
        n_trials=50,
        storage_url="sqlite:///study.db",
        n_workers=4,
        slurm_partition="batch",
        slurm_mem="8G",
        slurm_time="04:00:00",
        held_out_fraction=0.2,
        cv_group="plate_id",
        screen=False,
        slurm=True,
    )
    assert "--n-workers 4" in command
    assert "--slurm-partition batch" in command
    assert "--slurm-mem 8G" in command
    assert "--slurm-time 04:00:00" in command
    assert "--held-out-fraction 0.2" in command
    assert "--cv-group plate_id" in command


def test_render_quotes_paths_with_spaces() -> None:
    from phenotypic.gui.tune._command import render_launch_command

    command = render_launch_command(
        "my runs/spec.json",
        "plate images",
        "out dir",
        strategy="tpe",
        n_trials=10,
        storage_url=None,
        n_workers=None,
        slurm_partition=None,
        slurm_mem=None,
        slurm_time=None,
        held_out_fraction=None,
        cv_group=None,
        screen=False,
        slurm=False,
    )
    # The rendered command must be a valid, re-parseable shell invocation.
    tokens = shlex.split(command)
    assert "my runs/spec.json" in tokens
    assert "plate images" in tokens
    assert "out dir" in tokens


def test_render_parses_through_the_real_cli_parser() -> None:
    """The rendered command parses cleanly through the REAL argparse CLI.

    Drift defence: instead of asserting flag spellings against a hand-copied
    list (which a CLI rename would silently outpace), feed the rendered tokens
    (minus the ``python -m phenotypic.tune`` prefix) through the actual
    ``phenotypic.tune.__main__._build_parser()`` and assert the parsed namespace
    carries every value. A future flag rename breaks this test.
    """
    from phenotypic.gui.tune._command import render_launch_command
    from phenotypic.tune.__main__ import (
        _build_parser,
        _normalize_argv,
        _resolve_slurm_request,
    )

    command = render_launch_command(
        "spec.json",
        "imgs",
        "out",
        strategy="tpe",
        n_trials=50,
        storage_url="sqlite:///out/study.db",
        screen=True,
        slurm=True,
    )
    tokens = shlex.split(command)
    # The base invocation is exactly ``python -m phenotypic.tune`` — drop it and
    # hand the remainder (starting at the ``run`` subcommand) to the real parser.
    assert tokens[:4] == ["python", "-m", "phenotypic.tune", "run"]
    namespace = _build_parser().parse_args(_normalize_argv(tokens[3:]))

    assert namespace.command == "run"
    assert namespace.spec == "spec.json"
    assert namespace.input == "imgs"
    assert namespace.output == "out"
    assert namespace.strategy == "tpe"
    assert namespace.n_trials == 50
    assert namespace.storage_url == "sqlite:///out/study.db"
    assert namespace.screen is True
    # ``--slurm`` is repeatable and takes an optional KEY=VALUE, so the raw
    # namespace value is a list, not a bool. Assert the MEANING through the
    # function the CLI itself uses: presence in any form requests submission.
    assert _resolve_slurm_request(_build_parser(), namespace.slurm) == (True, {})


def test_render_grid_command_parses_without_n_trials() -> None:
    """A grid command (no ``--n-trials``) still parses; the budget defaults None."""
    from phenotypic.gui.tune._command import render_launch_command
    from phenotypic.tune.__main__ import _build_parser, _normalize_argv

    command = render_launch_command(
        "spec.json",
        "imgs",
        "out",
        strategy="grid",
        n_trials=50,
        storage_url=None,
        screen=False,
        slurm=False,
    )
    tokens = shlex.split(command)
    namespace = _build_parser().parse_args(_normalize_argv(tokens[3:]))
    assert namespace.strategy == "grid"
    # Grid suppresses --n-trials, so the parser falls back to its default (None).
    assert namespace.n_trials is None


def test_render_launch_command_does_not_import_optuna() -> None:
    sys.modules.pop("optuna", None)
    import importlib

    importlib.import_module("phenotypic.gui.tune._command")
    assert "optuna" not in sys.modules


def test_validated_command_owns_actual_display_and_portable_tokens(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.tune._command import build_tune_command

    sandbox, spec, images, output = _command_paths(tmp_path)
    storage = "postgresql+psycopg://user@db/tune"
    command = build_tune_command(
        sandbox=sandbox,
        spec_path=str(spec),
        images_dir=str(images),
        output_dir=str(output),
        strategy="tpe",
        n_trials=9,
        storage_mode="environment",
        storage_environment_name="PHENOTYPIC_STORAGE_URL",
        environ={"PHENOTYPIC_STORAGE_URL": storage},
    )

    assert command.deploy_eligible is True
    assert command.copy_eligible is True
    assert command.argv[0] == sys.executable
    assert command.argv[3:] == command.semantic_tail
    assert command.portable_tokens[:5] == (
        "uv",
        "run",
        "python",
        "-m",
        "phenotypic.tune",
    )
    assert command.portable_tokens[5:] == command.display_tokens[3:]
    assert storage in command.argv
    assert storage not in command.display_command()
    assert storage not in command.portable_command()
    assert storage not in repr(command)
    assert "$PHENOTYPIC_STORAGE_URL" in command.display_command()
    assert "$PHENOTYPIC_STORAGE_URL" in command.portable_command()
    assert command.display_tokens[3:] == tuple(
        "$PHENOTYPIC_STORAGE_URL" if token == storage else token
        for token in command.semantic_tail
    )


def test_inline_password_environment_storage_is_rejected_without_disclosure(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.tune._command import build_tune_command

    sandbox, spec, images, output = _command_paths(tmp_path)
    password = "do-not-disclose"
    storage = f"postgresql+psycopg://user:{password}@db/tune"
    command = build_tune_command(
        sandbox=sandbox,
        spec_path=str(spec),
        images_dir=str(images),
        output_dir=str(output),
        strategy="tpe",
        n_trials=9,
        storage_mode="environment",
        storage_environment_name="PHENOTYPIC_STORAGE_URL",
        environ={"PHENOTYPIC_STORAGE_URL": storage},
    )

    combined = " ".join(
        [
            *command.issues,
            command.display_command(),
            command.portable_command(),
            repr(command),
            *command.argv,
        ]
    )
    assert command.deploy_eligible is False
    assert command.argv == ()
    assert "inline password" in combined
    assert password not in combined


def test_validated_command_disables_copy_for_missing_images_and_env(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.tune._command import build_tune_command

    sandbox, spec, _, output = _command_paths(tmp_path)
    command = build_tune_command(
        sandbox=sandbox,
        spec_path=str(spec),
        images_dir="missing images",
        output_dir=str(output),
        strategy="tpe",
        n_trials=9,
        storage_mode="environment",
        storage_environment_name="SERVER_TUNE_URL",
        environ={},
    )

    assert command.deploy_eligible is False
    assert command.copy_eligible is False
    assert any("Image source" in issue for issue in command.issues)
    assert any("SERVER_TUNE_URL" in issue for issue in command.issues)
    assert command.argv == ()


def test_local_storage_path_is_sandbox_resolved_and_displayable(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.tune._command import build_tune_command

    sandbox, spec, images, output = _command_paths(tmp_path)
    command = build_tune_command(
        sandbox=sandbox,
        spec_path=str(spec),
        images_dir=str(images),
        output_dir=str(output),
        strategy="grid",
        n_trials=99,
        storage_mode="local",
        storage_local_path="state/study.sqlite3",
    )

    storage_url = f"sqlite:///{tmp_path / 'state/study.sqlite3'}"
    assert command.issues == ()
    assert storage_url in command.argv
    assert storage_url in command.display_tokens
    assert "--n-trials" not in command.argv
