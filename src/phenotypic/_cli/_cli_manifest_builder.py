"""Backward-compatibility re-export -- moved to _dashboard._manifest_builder."""
from ._dashboard._manifest_builder import (
    build_manifest,
    detect_silent_failures,
    query_sacct_chunk_states,
    query_sacct_job_states,
)

__all__ = [
    "build_manifest",
    "detect_silent_failures",
    "query_sacct_chunk_states",
    "query_sacct_job_states",
]
