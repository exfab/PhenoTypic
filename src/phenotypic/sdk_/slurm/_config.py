"""SLURM configuration querying and array job chunking logic.

This module provides utilities for querying SLURM cluster configuration
(array size limits, job submission limits) and calculating optimal array
job chunking strategies for large image datasets.
"""

from __future__ import annotations

import getpass
import re
import subprocess
from functools import lru_cache
from typing import List, Optional, Tuple


@lru_cache(maxsize=1)
def get_slurm_array_limit() -> int:
    """Query SLURM for MaxArraySize configuration.

    Uses ``scontrol show config`` to retrieve the maximum number of array
    tasks allowed per job. Falls back to a conservative default if the
    query fails or SLURM is not available.

    Returns:
        Integer limit for array job size (default: 1000).

    Examples:
        >>> limit = get_slurm_array_limit()
        >>> limit >= 1000  # At least the default
        True

    Notes:
        - Result is cached for the session (lru_cache)
        - Default fallback is 1000 (conservative for most clusters)
        - Common SLURM values: 1001, 10000, 100000
    """
    try:
        result = subprocess.run(
            ["scontrol", "show", "config"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode != 0:
            return 1000

        match = re.search(r"MaxArraySize\s*=\s*(\d+)", result.stdout)
        if match:
            limit = int(match.group(1))
            return limit

        return 1000

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return 1000


@lru_cache(maxsize=1)
def get_slurm_max_submit_jobs() -> Optional[int]:
    """Query a conservative SLURM MaxSubmitJobs limit per user.

    Uses ``sacctmgr`` to retrieve the maximum number of jobs a user can
    have in the queue simultaneously. This is typically set by QoS
    (Quality of Service) policies.

    Returns:
        Smallest configured positive QoS or user-association limit, or ``None``
        if no limit is configured or available. Taking the smallest value is
        conservative when the cluster exposes several QoS or association
        records and the caller cannot reliably infer which one the submitted
        job will use.

    Examples:
        >>> limit = get_slurm_max_submit_jobs()
        >>> limit is None or limit > 0
        True

    Notes:
        - Result is cached for the session
        - Returns None if sacctmgr is not available
        - Returns None if no limit is configured (unlimited)
        - QoS and association records can carry different limits; returning the
          largest value can oversize an array for the active policy.
    """
    commands = (
        [
            "sacctmgr",
            "show",
            "qos",
            "format=MaxSubmitJobsPerUser",
            "-P",
        ],
        [
            "sacctmgr",
            "show",
            "assoc",
            "where",
            f"user={getpass.getuser()}",
            "format=MaxSubmitJobs",
            "-P",
        ],
    )
    values: list[int] = []
    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            continue
        if result.returncode != 0:
            continue
        for token in re.split(r"[|\n]", result.stdout):
            token = token.strip()
            if token.isdigit() and int(token) > 0:
                values.append(int(token))

    return min(values) if values else None


def estimate_concurrent_capacity(
    partition: str,
    cpus_per_task: int = 1,
    mem_gb_per_task: float = 4.0,
) -> int:
    """Estimate max concurrent tasks from partition resources via sinfo.

    Queries SLURM's ``sinfo`` for the given partition to determine total
    CPUs, memory, and node count, then estimates how many tasks can run
    concurrently given per-task resource requirements.

    Args:
        partition: SLURM partition name.
        cpus_per_task: CPUs requested per task.
        mem_gb_per_task: Memory in GB per task.

    Returns:
        Estimated number of concurrent tasks. Falls back to 100 if
        sinfo is unavailable.

    Examples:
        >>> capacity = estimate_concurrent_capacity("compute")
        >>> capacity >= 1  # Always at least 1
        True
    """
    try:
        result = subprocess.run(
            ["sinfo", "-p", partition, "-o", "%c %m %D", "--noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode != 0:
            return 100

        total_cpus = 0
        total_mem_gb = 0.0

        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                cpus_per_node = int(parts[0])
                mem_per_node_mb = int(parts[1])
                num_nodes = int(parts[2])
                total_cpus += cpus_per_node * num_nodes
                total_mem_gb += (mem_per_node_mb / 1024.0) * num_nodes
            except (ValueError, IndexError):
                continue

        if total_cpus == 0:
            return 100

        by_cpu = total_cpus // max(cpus_per_task, 1)
        by_mem = int(total_mem_gb // max(mem_gb_per_task, 0.1))
        concurrent = min(by_cpu, by_mem)

        return max(concurrent, 1)

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return 100


def calculate_optimal_array_chunks(
    num_images: int, array_limit: int
) -> List[Tuple[int, int]]:
    """Split images into array job chunks based on SLURM array size limits.

    Calculates the minimum number of array jobs needed to process all
    images while respecting the cluster's MaxArraySize limit. Each chunk
    is represented as a (start, end) tuple of array indices.

    Args:
        num_images: Total number of images to process.
        array_limit: Maximum array size allowed by SLURM.

    Returns:
        List of (start_idx, end_idx) tuples defining each array job chunk.
        Indices are 0-based and end is exclusive (Python slice convention).

    Raises:
        ValueError: If ``array_limit`` is not positive.

    Examples:
        >>> calculate_optimal_array_chunks(500, 1000)
        [(0, 500)]

        >>> calculate_optimal_array_chunks(2500, 1000)
        [(0, 1000), (1000, 2000), (2000, 2500)]

        >>> calculate_optimal_array_chunks(1000, 1000)
        [(0, 1000)]

        >>> calculate_optimal_array_chunks(1001, 1000)
        [(0, 1000), (1000, 1001)]
    """
    if num_images <= 0:
        return []

    if array_limit <= 0:
        raise ValueError(f"array_limit must be positive, got {array_limit}")

    if num_images <= array_limit:
        return [(0, num_images)]

    num_chunks = (num_images + array_limit - 1) // array_limit

    chunks = []
    for i in range(num_chunks):
        start = i * array_limit
        end = min(start + array_limit, num_images)
        chunks.append((start, end))

    return chunks


def validate_array_chunk(
    chunk: Tuple[int, int], num_images: int, array_limit: int
) -> bool:
    """Validate that an array chunk is within acceptable bounds.

    Args:
        chunk: (start, end) tuple to validate.
        num_images: Total number of images.
        array_limit: Maximum array size.

    Returns:
        True if chunk is valid, False otherwise.

    Examples:
        >>> validate_array_chunk((0, 500), 1000, 1000)
        True

        >>> validate_array_chunk((0, 1500), 1000, 1000)
        False

        >>> validate_array_chunk((-1, 100), 1000, 1000)
        False
    """
    start, end = chunk

    if start < 0 or end < 0:
        return False
    if start >= end:
        return False
    if end > num_images:
        return False

    chunk_size = end - start
    if chunk_size > array_limit:
        return False

    return True
