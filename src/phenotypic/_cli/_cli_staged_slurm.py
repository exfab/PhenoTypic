"""SLURM 3-stage chaining for the staged GPU engine (Spec 1 §7).

Stage 1 (CPU array over images) -> Stage 2 (GPU array over shards, resident
model) -> Stage 3 (CPU array over images), wired with ``afterany`` between
stages so a few per-image failures never block the next stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypeVar

_T = TypeVar("_T")

#: Dependency type linking the three stages. ``afterany`` (not ``afterok``) so a
#: handful of per-image failures in one stage never block the next — the staged
#: workers are content-defined and skip already-done images.
STAGE_DEPENDENCY = "afterany"


def partition_shards(items: List[_T], n_shards: int) -> List[List[_T]]:
    """Split *items* into up to *n_shards* near-even contiguous shards (no loss)."""
    n = max(1, n_shards)
    k, r = divmod(len(items), n)
    shards: List[List[_T]] = []
    start = 0
    for i in range(n):
        size = k + (1 if i < r else 0)
        shards.append(items[start:start + size])
        start += size
    return shards


def resolve_stage_slurm_args(
    gpu_slurm_args: Dict[str, Any], cpu_slurm_args: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """GPU-stage (Stage 2) SBATCH args: inherit/delta over the CPU profile.

    Effective args = ``{**cpu_slurm_args, **gpu_slurm_args}``. Then resolve the
    GPU count:

    - absent -> auto-add ``slurm_gpus_per_node=1`` (request one whole GPU);
    - explicit ``0`` -> **omit** the key entirely (a CPU-only run of the GPU
      stage, e.g. the live dispatch test) — ``format_sbatch_directives`` would
      otherwise emit ``--gpus-per-node=0``, which SLURM rejects (OQ4);
    - explicit ``>0`` -> keep as given.

    Shared keys (account, qos, time) set in ``--slurm`` carry over; a separate
    GPU partition/account in ``--gpu-slurm`` overrides.
    """
    args = {**(cpu_slurm_args or {}), **gpu_slurm_args}
    gpus = args.get("slurm_gpus_per_node")
    if gpus == 0:
        args.pop("slurm_gpus_per_node", None)
    elif gpus is None:
        args["slurm_gpus_per_node"] = 1
    return args
