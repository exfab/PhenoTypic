"""Shared pytest-xdist worker-count resolution."""

from __future__ import annotations

from collections.abc import Mapping

SLURM_CPUS_ENV = "SLURM_CPUS_PER_TASK"


def _at_least_one(count: int) -> int:
    """Clamp worker counts to the minimum xdist can run."""
    return max(1, count)


def resolve_xdist_auto_workers(
    env: Mapping[str, str],
    *,
    affinity_count: int | None,
    cpu_count: int | None,
) -> int | None:
    """Resolve the worker count for ``pytest -n auto``.

    Args:
        env: Environment mapping to inspect for scheduler allocation.
        affinity_count: CPU count from the process affinity mask, or ``None``
            when affinity is unavailable.
        cpu_count: Host CPU fallback, or ``None`` when the caller wants xdist's
            default behavior.

    Returns:
        Positive worker count, or ``None`` to defer to pytest-xdist's default.

    Raises:
        ValueError: If ``SLURM_CPUS_PER_TASK`` is set to a non-integer value.
    """
    slurm_cpus = env.get(SLURM_CPUS_ENV)
    if slurm_cpus is not None:
        return _at_least_one(int(slurm_cpus))
    if affinity_count is not None:
        return _at_least_one(affinity_count)
    if cpu_count is not None:
        return _at_least_one(cpu_count)
    return None
