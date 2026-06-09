"""Render the ``python -m phenotypic.tune run …`` launch command (Task C1).

The Launch view shows the exact CLI invocation a user would run to start (or
resume) a tuning study from the bound run's ``tuning_spec.json``. This module is
the **source-of-truth** for that string: :func:`render_launch_command` builds the
token list and the clientside callback in :mod:`._callbacks` mirrors the same
logic into the live command card. The Python function powers the copy payload and
is the unit-tested truth; the clientside is a convenience mirror kept equivalent.

The renderer **only renders a string** — it never spawns a process (the
no-re-optimize lock). The flag spellings are single-sourced here from the real
argparse CLI in :mod:`phenotypic.tune.__main__` (subcommand ``run``; positional
``spec``; ``-i``/``-o``; ``--strategy``; ``--n-trials``; ``--storage-url``;
``--screen``; ``--slurm``) so the rendered command and the parser cannot drift.

Importing this module must never drag ``optuna`` into ``sys.modules``: it builds
only a shell string and touches no tuning internals.
"""
from __future__ import annotations

import shlex
from typing import Final

from phenotypic.gui.tune._run_argv import tune_run_argv

#: The base invocation every rendered command starts with — the module entry
#: point plus the ``run`` subcommand (confirmed in ``phenotypic.tune.__main__``).
_BASE_COMMAND: Final[tuple[str, ...]] = (
    "python", "-m", "phenotypic.tune", "run",
)

#: The CLI flag spellings, single-sourced from ``phenotypic.tune.__main__``'s
#: ``run`` sub-parser so the rendered command tracks the real parser. ``-i``/``-o``
#: are the short forms the parser declares (``--input``/``--output`` are aliases);
#: the short forms keep the rendered command compact.
_FLAG_INPUT: Final[str] = "-i"
_FLAG_OUTPUT: Final[str] = "-o"
_FLAG_STRATEGY: Final[str] = "--strategy"
_FLAG_N_TRIALS: Final[str] = "--n-trials"
_FLAG_STORAGE_URL: Final[str] = "--storage-url"
_FLAG_SCREEN: Final[str] = "--screen"
_FLAG_SLURM: Final[str] = "--slurm"

#: The exhaustive strategy that ignores ``--n-trials`` (the CLI covers the whole
#: grid regardless of the budget), so the renderer never emits the flag for it.
_GRID_STRATEGY: Final[str] = "grid"


def render_launch_command(
    spec_path: str,
    input_dir: str,
    output_dir: str,
    *,
    strategy: str,
    n_trials: int | None,
    storage_url: str | None,
    n_workers: int | None = None,
    slurm_partition: str | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    held_out_fraction: float | None = None,
    cv_group: str | None = None,
    screen: bool = False,
    slurm: bool = False,
) -> str:
    """Render the exact ``python -m phenotypic.tune run …`` launch command.

    Builds the invocation that runs the bound run's ``tuning_spec.json`` over the
    calibration images. The string is shell-safe — every path / value token is
    ``shlex.quote``-escaped so a path with spaces round-trips through
    ``shlex.split``. Optional flags are appended only when set:

    * ``--n-trials`` only when ``n_trials`` is set **and** the strategy is not
      ``grid`` (grid is exhaustive and ignores the budget, so the flag would be
      misleading — it is suppressed for grid even when a budget is present);
    * ``--storage-url`` only when ``storage_url`` is a non-empty string (a local
      run resolves the storage URL to the run's ``study.db`` and needs no flag);
    * ``--screen`` / ``--slurm`` only when the respective toggle is ``True``
      (store-true flags carrying no value).

    Args:
        spec_path: Path to the ``tuning_spec.json`` (the bare positional arg).
        input_dir: The calibration image directory (the ``-i`` value).
        output_dir: The run output directory (the ``-o`` value).
        strategy: The ``--strategy`` value (``grid``/``random``/``tpe``/…).
        n_trials: The ``--n-trials`` budget, or ``None`` to omit the flag.
        storage_url: The ``--storage-url`` value, or ``None``/``""`` to omit it.
        screen: Whether to append the ``--screen`` store-true flag.
        slurm: Whether to append the ``--slurm`` store-true flag.

    Returns:
        A single shell-safe command string (the tokens joined by spaces).
    """
    tokens = tune_run_argv(
        spec_path=spec_path,
        images_dir=input_dir,
        output_dir=output_dir,
        strategy=strategy,
        n_trials=n_trials,
        storage_url=storage_url,
        n_workers=n_workers,
        slurm_partition=slurm_partition,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
        held_out_fraction=held_out_fraction,
        cv_group=cv_group,
        slurm=slurm,
        screen=screen,
        python="python",
    )
    return " ".join(shlex.quote(token) for token in tokens)


__all__ = ["render_launch_command"]
