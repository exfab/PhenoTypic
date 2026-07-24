"""Pure builders for the ``phenotypic.tune run`` semantic CLI tail and argv."""
from __future__ import annotations

import sys
from collections.abc import Sequence


def tune_run_tail(
    *,
    spec_path: str,
    images_dir: str,
    output_dir: str,
    strategy: str,
    n_trials: int | None,
    storage_url: str | None,
    n_workers: int | None,
    slurm_partition: str | None,
    slurm_mem: str | None,
    slurm_time: str | None,
    held_out_fraction: float | None,
    cv_group: str | None,
    slurm: bool,
    screen: bool,
) -> list[str]:
    """Build the launcher-independent ``run`` argument tail.

    Args:
        spec_path: Tuning spec path passed as the ``run`` positional argument.
        images_dir: Image directory passed to ``-i``.
        output_dir: Output directory passed to ``-o``.
        strategy: CLI ``--strategy`` override.
        n_trials: Trial budget, omitted for exhaustive grid search.
        storage_url: Optional Optuna storage URL.
        n_workers: Optional SLURM worker count.
        slurm_partition: Optional SLURM partition.
        slurm_mem: Optional SLURM memory request.
        slurm_time: Optional SLURM wall time.
        held_out_fraction: Optional robust-eval fraction override.
        cv_group: Optional robust-eval grouping-column override.
        slurm: Whether to append ``--slurm``.
        screen: Whether to append ``--screen``.

    Returns:
        Tokens beginning with the ``run`` subcommand.

    Raises:
        ValueError: If a required path or strategy is empty.
    """
    missing = [
        name
        for name, value in (
            ("spec_path", spec_path),
            ("images_dir", images_dir),
            ("output_dir", output_dir),
            ("strategy", strategy),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "tune_run_tail missing required field(s): " + ", ".join(missing)
        )

    tail = [
        "run",
        spec_path,
        "-i",
        images_dir,
        "-o",
        output_dir,
        "--strategy",
        strategy,
    ]
    if n_trials is not None and strategy != "grid":
        tail += ["--n-trials", str(n_trials)]
    if storage_url:
        tail += ["--storage-url", storage_url]
    if n_workers is not None:
        tail += ["--n-workers", str(n_workers)]
    if slurm_partition:
        tail += ["--slurm-partition", slurm_partition]
    if slurm_mem:
        tail += ["--slurm-mem", slurm_mem]
    if slurm_time:
        tail += ["--slurm-time", slurm_time]
    if held_out_fraction is not None:
        tail += ["--held-out-fraction", str(held_out_fraction)]
    if cv_group:
        tail += ["--cv-group", cv_group]
    if slurm:
        tail.append("--slurm")
    if screen:
        tail.append("--screen")
    return tail


def tune_run_argv_from_tail(
    tail: Sequence[str],
    *,
    python: str | None = None,
) -> list[str]:
    """Prefix one validated semantic tail with the executable module launcher."""
    return [python or sys.executable, "-m", "phenotypic.tune", *tail]


def tune_run_argv(
    *,
    spec_path: str,
    images_dir: str,
    output_dir: str,
    strategy: str,
    n_trials: int | None,
    storage_url: str | None,
    n_workers: int | None,
    slurm_partition: str | None,
    slurm_mem: str | None,
    slurm_time: str | None,
    held_out_fraction: float | None,
    cv_group: str | None,
    slurm: bool,
    screen: bool,
    python: str | None = None,
) -> list[str]:
    """Build the full launch argv for a tune run.

    Args:
        spec_path: Tuning spec path passed as the ``run`` positional argument.
        images_dir: Image directory passed to ``-i``.
        output_dir: Output directory passed to ``-o``.
        strategy: CLI ``--strategy`` override.
        n_trials: Trial budget, omitted for exhaustive grid search.
        storage_url: Optional Optuna storage URL.
        n_workers: Optional SLURM worker count.
        slurm_partition: Optional SLURM partition.
        slurm_mem: Optional SLURM memory request.
        slurm_time: Optional SLURM wall time.
        held_out_fraction: Optional robust-eval fraction override.
        cv_group: Optional robust-eval grouping-column override.
        slurm: Whether to append ``--slurm``.
        screen: Whether to append ``--screen``.
        python: Python executable, defaulting to :data:`sys.executable`.

    Returns:
        Full argv, including Python executable and module entry point.

    Raises:
        ValueError: If ``spec_path``, ``images_dir``, or ``output_dir`` is empty.
    """
    tail = tune_run_tail(
        spec_path=spec_path,
        images_dir=images_dir,
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
    )
    return tune_run_argv_from_tail(tail, python=python)


__all__ = ["tune_run_argv", "tune_run_argv_from_tail", "tune_run_tail"]
