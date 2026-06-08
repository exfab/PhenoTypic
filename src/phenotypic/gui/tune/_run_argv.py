"""Pure builder for the ``python -m phenotypic.tune run`` argv."""
from __future__ import annotations

import sys


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
    missing = [
        name
        for name, value in (
            ("spec_path", spec_path),
            ("images_dir", images_dir),
            ("output_dir", output_dir),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "tune_run_argv missing required field(s): " + ", ".join(missing)
        )

    argv: list[str] = [
        python or sys.executable,
        "-m",
        "phenotypic.tune",
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
        argv += ["--n-trials", str(n_trials)]
    if storage_url:
        argv += ["--storage-url", storage_url]
    if n_workers is not None:
        argv += ["--n-workers", str(n_workers)]
    if slurm_partition:
        argv += ["--slurm-partition", slurm_partition]
    if slurm_mem:
        argv += ["--slurm-mem", slurm_mem]
    if slurm_time:
        argv += ["--slurm-time", slurm_time]
    if held_out_fraction is not None:
        argv += ["--held-out-fraction", str(held_out_fraction)]
    if cv_group:
        argv += ["--cv-group", cv_group]
    if slurm:
        argv.append("--slurm")
    if screen:
        argv.append("--screen")
    return argv


__all__ = ["tune_run_argv"]
