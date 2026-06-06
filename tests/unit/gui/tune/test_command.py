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


def test_render_uses_real_cli_subcommand_and_flag_names() -> None:
    """The rendered tokens match the real argparse CLI (defence against drift)."""
    from phenotypic.gui.tune._command import render_launch_command

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
    # The subcommand is ``run`` and the spec is the bare positional after it.
    run_index = tokens.index("run")
    assert tokens[run_index + 1] == "spec.json"
    # The exact long-flag spellings the CLI parser declares.
    for flag in ("-i", "-o", "--strategy", "--n-trials", "--storage-url",
                 "--screen", "--slurm"):
        assert flag in tokens


def test_render_launch_command_does_not_import_optuna() -> None:
    sys.modules.pop("optuna", None)
    import importlib

    importlib.import_module("phenotypic.gui.tune._command")
    assert "optuna" not in sys.modules
