"""Tests for shared pytest-xdist auto-worker resolution."""

from __future__ import annotations

import pytest


def _resolve(
    *,
    env: dict[str, str] | None = None,
    affinity_count: int | None = None,
    cpu_count: int | None = None,
) -> int | None:
    from tests._support.xdist_workers import resolve_xdist_auto_workers

    return resolve_xdist_auto_workers(
        env or {},
        affinity_count=affinity_count,
        cpu_count=cpu_count,
    )


def test_slurm_cpus_take_precedence_over_affinity_and_cpu_count() -> None:
    """SLURM allocation should win over host-level CPU signals."""
    assert _resolve(
        env={"SLURM_CPUS_PER_TASK": "4"},
        affinity_count=8,
        cpu_count=16,
    ) == 4


def test_invalid_slurm_cpus_raise_value_error() -> None:
    """Invalid scheduler metadata should fail loudly."""
    with pytest.raises(ValueError):
        _resolve(
            env={"SLURM_CPUS_PER_TASK": "many"},
            affinity_count=8,
            cpu_count=16,
        )


def test_affinity_count_is_used_when_slurm_is_absent() -> None:
    """Affinity masks reflect scheduler or container CPU grants."""
    assert _resolve(affinity_count=3, cpu_count=16) == 3


def test_zero_counts_are_clamped_to_one() -> None:
    """xdist worker counts must never resolve to zero."""
    assert _resolve(env={"SLURM_CPUS_PER_TASK": "0"}, cpu_count=16) == 1
    assert _resolve(affinity_count=0, cpu_count=16) == 1
    assert _resolve(cpu_count=0) == 1


def test_cpu_count_fallback_is_used_without_affinity() -> None:
    """Platforms without affinity support can still supply a CPU fallback."""
    assert _resolve(cpu_count=6) == 6


def test_none_is_returned_without_any_worker_signal() -> None:
    """Root conftest can preserve xdist's default behavior with no fallback."""
    assert _resolve() is None
