"""Aggregation fan-out over SLURM array tasks and a local process pool (spec §8).

**This is correctness machinery, and it is not a performance optimisation at
any N this project has measured.** P0's S-2 spike read every measurement table
in a real 6,529-store GPFS tree in 169.6 s -- one shard, against a 900 s
budget -- which puts :func:`shard_count` at **1** for the design target of
N = 6,000 and the crossover where a second shard first earns its keep at
**N ~= 34,600**, 5.8x that target (``spikes/RESULTS.md``). Nothing here makes a
realistic run faster.

What the decomposition buys is somewhere legal and ordered for the reserved
``TASK_FINALIZE`` entry to run: the shard-completeness check (CAN-5), the
two-phase partial-failure semantics D-A narrows §8 to, and a single publisher
for the aggregate and run proofs. Those hold whether K is 1 or 2,499.

**Consequence worth stating, because it is a maintenance hazard rather than a
bug:** the K > 1 path is exercised only by its own tests, never by production
need at present scale. A path that never runs is a path that rots, and these
tests are the only thing between it and silent decay.
"""

from __future__ import annotations

import math

#: Target wall-clock for one aggregation shard task.
#:
#: Chosen so a shard fits comfortably inside ``short``'s 2-hour cap with room
#: for scheduler latency and a cold GPFS read. **It is loose by roughly 6x at
#: the design target** -- S-2 measured the whole N = 6,529 tree at 169.6 s in a
#: single shard -- and P0 deliberately declined to retune it, because deciding
#: whether the interesting N is 6,000 or 30,000+ is a design choice a spike
#: must not make on the user's behalf (``spikes/RESULTS.md``, "Open question").
TARGET_TASK_SECONDS = 900

#: Seconds to read and append one image's embedded measurement table.
#:
#: **Measured, not estimated.** S-2, cold page cache, on
#: ``/bigdata/.../ucr_029_e_d_Maresca/data/results/2026-08-11`` -- 6,529 stores
#: carrying ``tables/measurements/table.parquet``, 527 MB total, mean 82.6 KB
#: per table, on GPFS. Four cold rungs (K = 64/16/4/1) agreed within 5%:
#: 0.0320 / 0.0249 / 0.0263 / 0.0258 s per image.
#:
#: **The cold qualifier is load-bearing.** The same K = 64 slice re-read warm
#: measured 0.0014 s per image -- 23x cheaper. A real shard worker starts on a
#: freshly allocated node with none of the tree cached, so sizing from a warm
#: number would undersize every array by over an order of magnitude.
#:
#: **One witness.** This describes one dataset directory with this column set.
#: A tree with many small datasets, far wider tables, or many more objects per
#: image could move it with no code change. P0 surveyed every candidate on
#: ``/bigdata/exfab/anguy344`` and ``/rhome/anguy344/bigdata_exfab`` -- nine
#: trees, four of which looked viable on store count and carried zero
#: measurement tables. A second witness does not exist to be used; it would
#: have to be built by running a full pipeline.
SECONDS_PER_IMAGE_S2 = 0.026


def shard_count(
    *,
    n_images: int,
    seconds_per_image: float,
    max_array_size: int,
) -> int:
    """Return K, the number of aggregation shard tasks.

    The ``- 1`` reserves the ``TASK_FINALIZE`` trigger entry's index. Project
    ``CLAUDE.md`` requires every trigger entry to be counted when sizing chunks
    against ``MaxArraySize``; the failure mode of not doing so is an ``sbatch``
    rejection whose message names neither the trigger nor the formula.

    ``max_array_size`` caps the *index*, not the task count -- with the
    cluster's 2500, the highest legal index is 2499, so ``--array=0-2499``
    is accepted and ``--array=1-2500`` is rejected. K is therefore capped at
    2499 and the array it sizes is ``0-K``: K shards plus one finalizer.

    ``MaxSubmitJobs`` is deliberately absent from the signature. It is 5,000
    on this cluster against a ``MaxArraySize`` of 2,500, so it can never bind;
    and it does not appear in ``scontrol show config`` at all, so a caller
    populating it from that command gets an empty string, and
    ``min(2500, 0) = 0`` silently sizes every shard to zero
    (``spikes/RESULTS.md``, "Trap"). Pass the binding term directly.

    Args:
        n_images: Images to aggregate.
        seconds_per_image: Cold per-image read cost. See
            :data:`SECONDS_PER_IMAGE_S2` for the measured value and the
            conditions it was measured under.
        max_array_size: The cluster's ``MaxArraySize``.

    Returns:
        K in ``[1, max_array_size - 1]``.

    Raises:
        ValueError: ``max_array_size`` is below 2, which leaves no index for a
            shard once the finalizer's is reserved.

    Examples:
        The design target needs one shard -- S-2's verdict, instantiated:

        >>> shard_count(
        ...     n_images=6000, seconds_per_image=0.026, max_array_size=2500
        ... )
        1

        The clamp reserves the finalizer's index:

        >>> shard_count(
        ...     n_images=3_000_000, seconds_per_image=1.0, max_array_size=2500
        ... )
        2499
    """
    if max_array_size < 2:
        raise ValueError(
            f"max_array_size={max_array_size} leaves no index for a shard "
            "once TASK_FINALIZE's is reserved"
        )
    target = math.ceil(n_images * seconds_per_image / TARGET_TASK_SECONDS)
    return max(1, min(target, max_array_size - 1))


def array_spec(shards: int) -> str:
    """Return the ``--array`` directive for *shards* shards plus the finalizer.

    The top index is ``shards`` itself, not ``shards - 1``: index K is the
    reserved ``TASK_FINALIZE`` entry, which lives **inside** the array rather
    than beside it (project ``CLAUDE.md``'s array-auxiliary contract).

    Args:
        shards: K, as returned by :func:`shard_count`.

    Returns:
        A 0-based inclusive range, e.g. ``"0-1"`` for one shard plus the
        finalizer.

    Raises:
        ValueError: *shards* is below 1.

    Examples:
        >>> array_spec(1)
        '0-1'
        >>> array_spec(2499)
        '0-2499'
    """
    if shards < 1:
        raise ValueError(f"shards={shards} must be at least 1")
    return f"0-{shards}"
