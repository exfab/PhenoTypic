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
        screen=True,
        slurm=True,
    )
    assert "--strategy random" in command
    assert "--n-trials 25" in command
    assert "--screen" in command
    assert "--slurm" in command


def test_render_quotes_paths_with_spaces() -> None:
    from phenotypic.gui.tune._command import render_launch_command

    command = render_launch_command(
        "my runs/spec.json",
        "plate images",
        "out dir",
        strategy="tpe",
        n_trials=10,
        storage_url=None,
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
    from phenotypic.tune.__main__ import _build_parser, _normalize_argv

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
    assert namespace.slurm is True


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
