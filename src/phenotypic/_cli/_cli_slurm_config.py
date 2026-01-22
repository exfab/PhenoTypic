"""
SLURM configuration querying and array job chunking logic.

This module provides utilities for querying SLURM cluster configuration
(array size limits, job submission limits) and calculating optimal array
job chunking strategies for large image datasets.
"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from typing import List, Optional, Tuple


@lru_cache(maxsize=1)
def get_slurm_array_limit() -> int:
    """
    Query SLURM for MaxArraySize configuration.

    Uses `scontrol show config` to retrieve the maximum number of array
    tasks allowed per job. Falls back to a conservative default if the
    query fails or SLURM is not available.

    Returns:
        Integer limit for array job size (default: 1000)

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
            # scontrol failed, use default
            return 1000

        # Parse output for MaxArraySize
        # Example line: "MaxArraySize           = 10000"
        match = re.search(r"MaxArraySize\s*=\s*(\d+)", result.stdout)
        if match:
            limit = int(match.group(1))
            # Some systems report MaxArraySize as the max index (0-based)
            # so actual limit might be limit+1, but we'll be conservative
            return limit

        # No MaxArraySize found, use default
        return 1000

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        # scontrol not found, or timeout, or parse error - use default
        return 1000


@lru_cache(maxsize=1)
def get_slurm_max_submit_jobs() -> Optional[int]:
    """
    Query SLURM for MaxSubmitJobs limit per user.

    Uses `sacctmgr` to retrieve the maximum number of jobs a user can
    have in the queue simultaneously. This is typically set by QoS
    (Quality of Service) policies.

    Returns:
        Integer limit or None if not configured/available

    Examples:
        >>> limit = get_slurm_max_submit_jobs()
        >>> limit is None or limit > 0
        True

    Notes:
        - Result is cached for the session
        - Returns None if sacctmgr is not available
        - Returns None if no limit is configured (unlimited)
        - This limit is typically much higher than array limits (e.g., 10000+)
    """
    try:
        result = subprocess.run(
            ["sacctmgr", "show", "qos", "format=MaxSubmitJobsPerUser", "-P"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode != 0:
            return None

        # Parse output (format: "MaxSubmitJobsPerUser\n123\n456\n...")
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return None

        # Get non-header values and find maximum (most permissive)
        values = []
        for line in lines[1:]:  # Skip header
            line = line.strip()
            if line and line.isdigit():
                values.append(int(line))

        if values:
            # Return maximum limit found (most permissive QoS)
            return max(values)

        return None

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def calculate_optimal_array_chunks(
    num_images: int, array_limit: int
) -> List[Tuple[int, int]]:
    """
    Split images into array job chunks based on SLURM array size limits.

    Calculates the minimum number of array jobs needed to process all
    images while respecting the cluster's MaxArraySize limit. Each chunk
    is represented as a (start, end) tuple of array indices.

    Args:
        num_images: Total number of images to process
        array_limit: Maximum array size allowed by SLURM

    Returns:
        List of (start_idx, end_idx) tuples defining each array job chunk.
        Indices are 0-based and end is exclusive (Python slice convention).

    Examples:
        >>> calculate_optimal_array_chunks(500, 1000)
        [(0, 500)]

        >>> calculate_optimal_array_chunks(2500, 1000)
        [(0, 1000), (1000, 2000), (2000, 2500)]

        >>> calculate_optimal_array_chunks(1000, 1000)
        [(0, 1000)]

        >>> calculate_optimal_array_chunks(1001, 1000)
        [(0, 1000), (1000, 1001)]

    Notes:
        - Returns single chunk if num_images <= array_limit
        - Chunks are equal-sized except possibly the last chunk
        - Start indices are inclusive, end indices are exclusive
        - Designed for 0-based indexing (Python/bash arrays)
    """
    if num_images <= 0:
        return []

    if array_limit <= 0:
        raise ValueError(f"array_limit must be positive, got {array_limit}")

    if num_images <= array_limit:
        # Single array job sufficient
        return [(0, num_images)]

    # Calculate number of chunks needed
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
    """
    Validate that an array chunk is within acceptable bounds.

    Args:
        chunk: (start, end) tuple to validate
        num_images: Total number of images
        array_limit: Maximum array size

    Returns:
        True if chunk is valid, False otherwise

    Examples:
        >>> validate_array_chunk((0, 500), 1000, 1000)
        True

        >>> validate_array_chunk((0, 1500), 1000, 1000)
        False

        >>> validate_array_chunk((-1, 100), 1000, 1000)
        False
    """
    start, end = chunk

    # Check bounds
    if start < 0 or end < 0:
        return False
    if start >= end:
        return False
    if end > num_images:
        return False

    # Check chunk size
    chunk_size = end - start
    if chunk_size > array_limit:
        return False

    return True
