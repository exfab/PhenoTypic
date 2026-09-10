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
from collections.abc import Sequence
from pathlib import Path

import click

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


# ---------------------------------------------------------------------------
# Where a fan-out invocation keeps its scratch
# ---------------------------------------------------------------------------


def aggregation_status_dir(output_dir: Path, scheduler_epoch: str | None) -> Path:
    """Return this invocation's per-shard status directory.

    Mirrors recompile's ``<attempt>/status/``. The status files are what index
    K waits on, so they live beside the shards and are cleared with them.
    """
    from phenotypic.sdk_ import aggregation_shard_dir

    return aggregation_shard_dir(output_dir, scheduler_epoch) / "status"


def aggregation_task_manifest_path(
    output_dir: Path, scheduler_epoch: str | None
) -> Path:
    """Return this invocation's task manifest.

    Carries K. **K is carried, never counted** (CAN-5): if index K derived the
    expected shard count by globbing the shard directory, ``len(shard_paths)``
    would be the number of files that happen to exist and "exactly K" would
    compare the list against itself.
    """
    from phenotypic.sdk_ import aggregation_shard_dir

    return (
        aggregation_shard_dir(output_dir, scheduler_epoch)
        / "task_manifest.json"
    )


def shard_parquet_path(
    output_dir: Path, scheduler_epoch: str | None, shard_id: int
) -> Path:
    """Return one shard's Parquet path."""
    from phenotypic.sdk_ import aggregation_shard_dir, shard_parquet_filename

    return aggregation_shard_dir(
        output_dir, scheduler_epoch
    ) / shard_parquet_filename(shard_id)


# ---------------------------------------------------------------------------
# Fan-out start -- the clear, and the manifest that carries K
# ---------------------------------------------------------------------------


def begin_aggregation_fanout(
    output_dir: Path,
    *,
    scheduler_epoch: str | None,
    shards: int,
    dataset_names: Sequence[str],
) -> Path:
    """Empty this invocation's shard directory and write its task manifest.

    **The ordering is the whole point, and getting it wrong converts a
    correctness fix into data loss.** The directory is emptied when fan-out
    *begins* -- before any worker writes -- and never at merge time, because a
    finalizer that cleared before merging would delete the shards it is about
    to merge.

    "Fan-out begins" is the moment the driver starts. On SLURM that is
    **submission**: this runs once, in the submitting process, before the
    array exists, so there is exactly one writer and no race with the workers.
    Locally it is the driver's first statement. Same logical point on both
    paths, as the P2-close ruling requires.

    **Why clearing rather than trusting the namespace.**
    ``_scheduler_epoch`` returns ``None`` for every local run, so consecutive
    local invocations share one directory and a prior run's shards would be
    merged into this run's master -- silently, and in violation of INV-INPUTS.
    Clearing is also strictly stronger than namespacing, which was not obvious
    going in: namespacing leaves every prior run's shards on disk forever,
    accumulating, while emptying closes the hole *and* stops the accretion.

    Args:
        output_dir: Run output root.
        scheduler_epoch: Active SLURM lifecycle generation, or ``None``
            locally.
        shards: K, from :func:`shard_count`.
        dataset_names: Datasets this run finalizes, carried to index K.

    Returns:
        The task manifest path.
    """
    import json
    import shutil

    from phenotypic.sdk_ import aggregation_shard_dir, atomic_write_json

    shard_dir = aggregation_shard_dir(output_dir, scheduler_epoch)
    if shard_dir.is_dir():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    aggregation_status_dir(output_dir, scheduler_epoch).mkdir(
        parents=True, exist_ok=True
    )

    manifest_path = aggregation_task_manifest_path(
        output_dir, scheduler_epoch
    )
    atomic_write_json(
        manifest_path,
        {
            "version": 1,
            "scheduler_epoch": scheduler_epoch,
            "tasks": build_task_list(
                dataset_names=dataset_names, shards=shards
            ),
        },
    )
    if not json.loads(manifest_path.read_text(encoding="utf-8"))["tasks"]:
        raise RuntimeError(
            f"Aggregation task manifest {manifest_path} carries no tasks"
        )
    return manifest_path


def build_task_list(
    *, dataset_names: Sequence[str], shards: int
) -> list[dict[str, object]]:
    """Return K aggregation tasks followed by exactly one finalizer task.

    The task-type vocabulary is recompile's, imported rather than restated:
    two vocabularies for one dispatch pattern is the cardinality problem this
    whole change is about.

    ``expected_non_finalizer_tasks`` reuses the key recompile's finalizer
    already carries (``_cli_recompile_slurm_scripts.py:203``), and it is the
    **carried** K that index K checks the shard files against.

    Note this takes ``shards`` rather than an image count. The sources are
    discovered from ``authorized_measurement_sources`` at shard time; a
    signature that also took ``n_images`` would give one derived value two
    producers, which ``_cli/CLAUDE.md``'s one-producer rule names as a defect.

    Args:
        dataset_names: Datasets this run finalizes.
        shards: K, from :func:`shard_count`.

    Returns:
        ``K + 1`` task dictionaries; the last is the finalizer.
    """
    from ._cli_recompile_slurm_scripts import TASK_FINALIZE, TASK_MEASUREMENTS

    tasks: list[dict[str, object]] = [
        {"task_type": TASK_MEASUREMENTS, "shard_id": shard_id}
        for shard_id in range(shards)
    ]
    tasks.append(
        {
            "task_type": TASK_FINALIZE,
            "dataset_names": [str(name) for name in dataset_names],
            "expected_non_finalizer_tasks": shards,
        }
    )
    return tasks


# ---------------------------------------------------------------------------
# The shard worker
# ---------------------------------------------------------------------------


def shard_sources(
    sources: "dict[Path, str]", *, shard_id: int, shards: int
) -> "dict[Path, str]":
    """Return the slice of *sources* this shard owns.

    Deterministic by sorted path, and **contiguous** -- shard *i* takes a
    block, not every *K*-th entry. A balanced block split (the first
    ``n % K`` shards take one extra) gives shard sizes differing by at most 1,
    so concatenating the shards **in shard order reproduces sorted order
    exactly**.

    ⚠ **That equality is the whole point, and an earlier draft broke it.** The
    finalizer merges by sorted shard filename, so if the assignment is not
    contiguous the master's row order becomes a function of K:

    ```
    ten sources, merged in shard order
      K=1  ->  abcdefghij      K=3  ->  adgjbehcfi   (strided)
      K=2  ->  acegibdfhj      K=4  ->  aeibfjcgdh   (strided)
    ```

    So the same data aggregated at two different K produced two different
    masters, and ``source_set_digest`` certified a byte sequence that depended
    on the worker count. **This was not a local-only defect** -- the SLURM
    path merges by the same sorted glob, and ``shard_count`` crosses from 1 to
    2 at N ~= 34,600, so it was latent there rather than absent.

    **The rationale the strided version gave for itself was false**, and it is
    worth stating because it is why the stride was chosen: *"strided rather
    than blocked so an uneven tail does not land entirely on one worker."*
    That describes **naive** blocking (``chunk = ceil(n/K)``, remainder piled
    on the last shard). It is not true of a balanced split, whose spread is
    identical to the stride's -- measured at ``(n,K)`` of (3,2), (10,2),
    (10,3), (10,4), (7,3) and (6529,8): spread <= 1 for both, and ordered for
    the block split only. The stride bought nothing and cost the invariant.

    **Contiguous also matches the precedent this phase generalises.**
    Recompile's ``_chunk_paths`` (``_cli_recompile_slurm_scripts.py:662``)
    slices contiguously; the strided version diverged from the very pattern it
    claimed to reuse.

    Determinism is what makes the master byte-identical across ``--njobs``: if
    the assignment could vary, two runs of the same data would disagree and
    the aggregate proof would certify nothing.

    **ASSUMPTION, not an enforcement: every shard worker sees the same
    ``sources``.** It holds because the aggregation array is dependent
    (``afterany``) on the last image chunk, so no image is completing while
    shards run. The slices partition one set only while that is true. If that
    dependency ever changes -- an aggregation array submitted alongside image
    work rather than after it -- the slices stop partitioning and an image is
    silently duplicated or dropped, with nothing failing to say so.

    Args:
        sources: Authorized measurement table -> dataset.
        shard_id: This worker's index in ``[0, shards)``.
        shards: K.

    Returns:
        The contiguous subset this shard aggregates. May be empty when K
        exceeds the number of sources -- see :func:`shard_count`'s note on
        planned-versus-successful image counts.
    """
    if not 0 <= shard_id < shards:
        raise ValueError(f"shard_id={shard_id} is outside [0, {shards})")
    ordered = sorted(sources, key=lambda path: str(path))
    base, extra = divmod(len(ordered), shards)
    start = shard_id * base + min(shard_id, extra)
    stop = start + base + (1 if shard_id < extra else 0)
    return {path: sources[path] for path in ordered[start:stop]}


def write_measurement_shard(
    output_dir: Path,
    *,
    scheduler_epoch: str | None,
    shard_id: int,
    shards: int,
    include_dataset_column: bool = True,
) -> list[str]:
    """Aggregate this shard's authorized tables into one Parquet.

    One pass over its images and nothing else: read each store's
    ``tables/measurements/table.parquet``, concatenate, write
    ``shard_NNNN.parquet``. **No store write, no metadata projection, no
    global frame.** D-A removed the metadata half of §8's array task -- per-
    store metadata is written at promote time (P4 Task 2) -- so a shard
    worker aggregates and does not certify.

    Args:
        output_dir: Run output root.
        scheduler_epoch: Active SLURM lifecycle generation, or ``None``.
        shard_id: This worker's index.
        shards: K.
        include_dataset_column: Whether to insert ``Metadata_Dataset`` into
            sources that lack it.

    Returns:
        The source work identities this shard merged, sorted. Index K unions
        these to obtain the **planned** set that the aggregate proof is
        published against.
    """
    from phenotypic.sdk_ import (
        PARQUET_WRITE_OPTIONS,
        atomic_write_with_writer,
    )

    from ._cli_completion import authorized_measurement_sources
    from ._cli_parquet_agg import aggregate_parquet_files
    from ._cli_recompile_worker import _sort_measurement_shard
    from ._measurement_sources import add_metadata_image_name_from_filename

    authorized = authorized_measurement_sources(output_dir)
    if authorized is None:
        raise RuntimeError(
            "Aggregation fan-out requires marker-authorized measurement "
            "sources; this tree has none. A legacy tree must be converted "
            "with `--mode migrate` before it can be aggregated in shards."
        )
    mine = shard_sources(authorized, shard_id=shard_id, shards=shards)
    shard_path = shard_parquet_path(output_dir, scheduler_epoch, shard_id)
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    if not mine:
        # An EMPTY shard is written, deliberately, rather than skipped. K is
        # sized from the run's PLANNED image count, so a failure-heavy run can
        # leave a shard with no sources; index K checks the file COUNT against
        # the carried K, and a skipped file would read there as a dead worker.
        import polars as pl

        atomic_write_with_writer(
            shard_path,
            lambda p: pl.DataFrame().write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )
        return []

    shard_df = aggregate_parquet_files(
        file_paths=list(mine.keys()),
        path_to_dataset=mine,
        include_dataset_column=include_dataset_column,
        keep_filename=True,
    )
    if shard_df is None:
        raise RuntimeError(
            f"Shard {shard_id} found no readable measurements among "
            f"{len(mine)} authorized source(s)"
        )
    shard_df = _sort_measurement_shard(
        add_metadata_image_name_from_filename(shard_df)
    )
    atomic_write_with_writer(
        shard_path,
        lambda p: shard_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
    )
    return sorted(_work_ids_for_sources(output_dir, mine))


def _work_ids_for_sources(
    output_dir: Path, sources: "dict[Path, str]"
) -> list[str]:
    """Return the work identity backing each of *sources*, via its record.

    The proof speaks in ``work_id``s and a shard knows its sources by path, so
    the two are reconciled **here**, in the worker that still has its own view.
    At the merge that view is gone, and re-deriving it from live state would
    reintroduce exactly the live-derivation CAN-5 exists to remove.

    The record is the mapping: a source is
    ``<store>/tables/measurements/table.parquet``, so the store is its
    ``MEASUREMENT_TABLE_RELATIVE_PATH``-th parent and the record is read by
    that store's stem.

    **A source with no readable record raises rather than being skipped.**
    Skipping would silently shrink the planned set, and the aggregate proof
    would then be true of a master that omitted the very image whose record
    could not be read -- the CAN-5 failure, arriving through the code written
    to prevent it.

    Args:
        output_dir: Run output root.
        sources: Authorized measurement table -> dataset.

    Returns:
        One work identity per source, in ``sources`` iteration order.

    Raises:
        RuntimeError: A source has no record, or a record carries no
            ``work_id``.
    """
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, store_stem
    from phenotypic.sdk_._image_record import read_image_record

    depth = len(MEASUREMENT_TABLE_RELATIVE_PATH.parts)
    work_ids: list[str] = []
    for source, dataset in sources.items():
        store = Path(source).parents[depth - 1]
        record = read_image_record(output_dir, dataset, store_stem(store))
        if record is None:
            raise RuntimeError(
                f"No per-image record for {store}; its work identity cannot "
                "be recorded, and a shard that dropped it would publish a "
                "proof asserting an image the master does not carry"
            )
        work_id = record.get("work_id")
        if not isinstance(work_id, str) or not work_id:
            raise RuntimeError(
                f"The per-image record for {store} carries no work_id"
            )
        work_ids.append(work_id)
    return work_ids


# ---------------------------------------------------------------------------
# Ordering: index K runs concurrently with its shards unless it waits
# ---------------------------------------------------------------------------


def write_shard_status(
    output_dir: Path,
    *,
    scheduler_epoch: str | None,
    shard_id: int,
    source_work_ids: Sequence[str],
    status: str = "completed",
) -> Path:
    """Record one shard's outcome and the source set it merged.

    The union of these ``source_work_ids`` is the **planned** set index K
    publishes the aggregate proof against, which is what makes the proof true
    of the master by construction rather than by a live re-derivation that can
    disagree with it (flow-r3 C2).
    """
    from phenotypic.sdk_ import atomic_write_json, task_status_filename

    from ._cli_recompile_slurm_scripts import TASK_MEASUREMENTS

    path = aggregation_status_dir(
        output_dir, scheduler_epoch
    ) / task_status_filename(shard_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        {
            "task_type": TASK_MEASUREMENTS,
            "shard_id": shard_id,
            "status": status,
            "source_work_ids": list(source_work_ids),
        },
    )
    return path


def wait_for_shard_statuses(
    output_dir: Path,
    *,
    scheduler_epoch: str | None,
    expected: int,
    timeout: float = 3600.0,
) -> list[dict]:
    """Block until all *expected* shard statuses exist, then return them.

    **Without this the completeness check reddens every run.** Within one
    SLURM array, index K has no ordering relation to indices ``0..K-1`` -- the
    scheduler starts them together. So index K would find an incomplete shard
    set and raise CAN-5's refusal on a run where nothing is wrong.

    At the measured ``K = 1`` the array is two tasks, which is the shape where
    that race is least likely to be noticed by eye and therefore most in need
    of being handled rather than assumed away.

    Delegates to recompile's ``_wait_for_non_finalizer_statuses`` rather than
    restating the poll: one synchronisation rule, one home.

    Args:
        output_dir: Run output root.
        scheduler_epoch: Active SLURM lifecycle generation, or ``None``.
        expected: The **carried** K, from the task manifest.
        timeout: Seconds to wait before giving up.

    Returns:
        The shard status payloads.
    """
    from ._cli_recompile_worker import _wait_for_non_finalizer_statuses

    return _wait_for_non_finalizer_statuses(
        aggregation_status_dir(output_dir, scheduler_epoch),
        expected,
        timeout=int(timeout),
    )


def collect_shard_paths(
    output_dir: Path, scheduler_epoch: str | None
) -> list[Path]:
    """Return this invocation's shards in deterministic merge order.

    Sorted by filename, which is zero-padded shard id, so **merge order is
    shard order**.

    ⚠ **That alone does not give byte-identical master bytes, and an earlier
    version of this docstring claimed it did.** It said *"a re-run of
    identical inputs produces byte-identical master bytes"* -- true at a
    **fixed K**, and silent on the axis where it failed: identical inputs at a
    *different* K produced a different row order, because the assignment was
    strided while the merge was shard-ordered. Neither statement was wrong;
    their conjunction was.

    The guarantee holds now because :func:`shard_sources` assigns a
    **contiguous** block of the sorted sources to each shard, so shard order
    equals sorted source order and the master is byte-identical **across
    re-runs and across K**. That is a property of the *decomposition*, not of
    this function -- this function only preserves it. **If the assignment ever
    stops being contiguous, this ordering silently stops meaning anything**,
    and nothing here would fail to say so.

    The reverse pointer is in :func:`shard_sources`, deliberately: the defect
    existed only in the composition of two individually-correct functions, and
    neither docstring was the place a reader would look for the other's
    assumption.
    """
    from phenotypic.sdk_ import aggregation_shard_dir

    return sorted(
        aggregation_shard_dir(output_dir, scheduler_epoch).glob(
            "shard_*.parquet"
        )
    )


def planned_work_ids_from_statuses(statuses: Sequence[dict]) -> list[str]:
    """Return the union of the source sets the shards actually merged.

    This is ``planned`` -- the set the aggregate proof is published against.
    It is deliberately **not** ``authorized_measurement_sources`` evaluated at
    merge time: under a rolling input more images can succeed between shard
    planning and the merge, so ``merged`` is a strict subset of ``authorized``
    with nothing wrong, and a proof asserting the live set would describe a
    master that never contained it.
    """
    union: set[str] = set()
    for status in statuses:
        raw = status.get("source_work_ids")
        if isinstance(raw, list):
            union.update(str(work_id) for work_id in raw)
    return sorted(union)


# ---------------------------------------------------------------------------
# What index K asks for before it is allowed to publish (CAN-5)
# ---------------------------------------------------------------------------


def resolve_finalizer_shard_inputs(
    output_dir: Path, scheduler_epoch: str | None
) -> "tuple[list[Path], list[str]] | None":
    """Return ``(shard_paths, planned_work_ids)``, or ``None`` for no fan-out.

    ``None`` means this invocation never fanned out -- no task manifest -- and
    the finalizer aggregates the embedded tables directly, exactly as before
    this phase. That is the ordinary local path and is not an error.

    Otherwise this is the gate CAN-5 asks for, and **both checks are stated so
    that neither is vacuous**:

    1. **The shard files are counted against a K that was CARRIED, not
       counted.** K comes from the task manifest written at fan-out start. Had
       it been derived by globbing the shard directory, ``len(shard_paths)``
       would be the number of files that happen to exist and "exactly K" would
       compare the list against itself -- green on a run missing four shards.
    2. **The merged set is compared against what the shards planned**, not
       against ``authorized_measurement_sources`` evaluated now. Under a
       rolling input more images can succeed between shard time and the merge,
       so ``planned`` is a strict subset of ``authorized`` with nothing wrong;
       asserting equality against a live predicate would fail correct runs.
       What must hold is ``planned <= authorized``.

    **Why this is not belt-and-braces.** The finalizer is ``afterany``, so it
    runs when a shard task dies. ``publish_aggregate_snapshot`` derives its
    source set from the markers, not from what was merged, so a master missing
    four shards' worth of images would otherwise receive a proof asserting the
    *full* success set. Checking at the writer matters because a proof that
    should never have been published is worse than one a reader rejects.

    Args:
        output_dir: Run output root.
        scheduler_epoch: Active SLURM lifecycle generation, or ``None``.

    Returns:
        The shards to merge and the source set to publish the proof against,
        or ``None`` when this invocation did not fan out.

    Raises:
        RuntimeError: A shard is missing, or the shards merged an image that
            is no longer authorized.
    """
    import json

    from ._cli_completion import authorized_measurement_sources

    manifest_path = aggregation_task_manifest_path(
        output_dir, scheduler_epoch
    )
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks") or []
    expected = int(tasks[-1].get("expected_non_finalizer_tasks", 0))

    statuses = wait_for_shard_statuses(
        output_dir, scheduler_epoch=scheduler_epoch, expected=expected
    )
    failed = [s for s in statuses if s.get("status") != "completed"]
    shard_paths = collect_shard_paths(output_dir, scheduler_epoch)

    if failed or len(shard_paths) != expected:
        raise RuntimeError(
            f"Aggregation fan-out is incomplete: {len(shard_paths)} of "
            f"{expected} measurement shard(s) present, {len(failed)} shard "
            "task(s) failed. Nothing was published, because certifying this "
            "master would assert measurements it does not contain. "
            "RECOVERY: re-run the same command. Shards are per-invocation "
            "scratch and are cleared when the next fan-out begins, so the "
            "incomplete set is not in the way and this is not a deadlock."
        )

    planned = planned_work_ids_from_statuses(statuses)
    authorized = authorized_measurement_sources(output_dir)
    if authorized is None:
        raise RuntimeError(
            "Aggregation fan-out produced shards for a tree that no longer "
            "reports marker-authorized sources"
        )
    live = set(_work_ids_for_sources(output_dir, authorized))
    strayed = sorted(set(planned) - live)
    if strayed:
        raise RuntimeError(
            f"{len(strayed)} image(s) the shards merged are no longer "
            f"authorized (first: {strayed[0]}). Publishing would certify a "
            "master containing measurements the run no longer claims. "
            "RECOVERY: re-run the same command."
        )
    return shard_paths, planned


# ---------------------------------------------------------------------------
# The array task body
# ---------------------------------------------------------------------------


@click.command("aggregation-shard")
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--task-index", type=int, required=True)
@click.option(
    "--epoch",
    default=None,
    help="Active SLURM lifecycle generation, or omitted for a local run.",
)
def run_aggregation_shard(
    output_dir: Path, task_index: int, epoch: str | None
) -> None:
    """Run one aggregation shard, dispatched by ``$SLURM_ARRAY_TASK_ID``.

    **This is a task body and never a submitter.** Indices ``0..K-1`` reach
    here; index K is the pre-existing dependent finalizer command, unchanged.
    Nothing in this module submits work to the scheduler, which is what makes
    *no standalone parallel job is submitted* true by construction rather than
    by discipline -- there is no submission site here to get wrong.
    """
    import json

    from ._cli_preload import preload_custom_operation_modules
    from ._cli_recompile_slurm_scripts import TASK_MEASUREMENTS

    preload_custom_operation_modules()
    manifest_path = aggregation_task_manifest_path(output_dir, epoch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks") or []
    task = tasks[task_index]
    if task.get("task_type") != TASK_MEASUREMENTS:
        raise RuntimeError(
            f"Task {task_index} is {task.get('task_type')!r}, not a "
            "measurement shard. Index K is the dependent finalizer and is "
            "dispatched to the checkpoint handler, not to this module."
        )

    shards = int(tasks[-1]["expected_non_finalizer_tasks"])
    shard_id = int(task["shard_id"])
    try:
        merged = write_measurement_shard(
            output_dir,
            scheduler_epoch=epoch,
            shard_id=shard_id,
            shards=shards,
        )
    except Exception:
        # The status file is the ONLY thing index K waits on. A shard that
        # died without writing one would hang the finalizer to its timeout and
        # then report a missing shard rather than the failure that caused it.
        write_shard_status(
            output_dir,
            scheduler_epoch=epoch,
            shard_id=shard_id,
            source_work_ids=[],
            status="failed",
        )
        raise
    write_shard_status(
        output_dir,
        scheduler_epoch=epoch,
        shard_id=shard_id,
        source_work_ids=merged,
    )


if __name__ == "__main__":  # pragma: no cover - module entry point
    run_aggregation_shard()


# ---------------------------------------------------------------------------
# Task 3 -- the local driver, same decomposition, different K
# ---------------------------------------------------------------------------


def local_shard_count(*, n_sources: int, njobs: int) -> int:
    """Return K for a local fan-out: one shard per worker, capped by the work.

    **Deliberately NOT :func:`shard_count`, and the plan's instruction to reuse
    it would have made local ``--njobs`` a no-op.** The two answer different
    questions:

    * :func:`shard_count` asks *"how many tasks fit a 900 s walltime budget?"*
      At S-2's measured 0.026 s/image that is **1** for every N below ~34,600 --
      including every N this project has ever run. On SLURM that is the right
      answer, because the array's purpose is the reserved ``TASK_FINALIZE``
      index and the partial-failure semantics, not parallelism.
    * Locally the question is *"how many cores may I use?"*, which is
      ``--njobs``. Sizing a local fan-out with :func:`shard_count` yields K = 1
      at every realistic N, so ``--njobs 8`` would run one shard on one core
      and the byte-identical test across ``njobs in {1, 2, 8}`` would compare
      three runs of the *same* single-shard path -- green, and proving nothing
      about the decomposition it claims to test.

    Delegates to :func:`~phenotypic._cli._cli_utils.resolve_local_worker_count`
    rather than restating the clamp: it already handles ``-1``, caps to the
    SLURM CPU allocation when one is present, and never exceeds the work item
    count. A second producer for "how many local workers" is exactly the defect
    ``_cli/CLAUDE.md`` names.

    Args:
        n_sources: Authorized measurement tables to aggregate.
        njobs: The ``--njobs`` value. ``-1`` means all allocated CPUs.

    Returns:
        K in ``[1, n_sources]``.
    """
    from ._cli_utils import resolve_local_worker_count

    return resolve_local_worker_count(njobs, n_sources)


def _run_one_local_shard(
    task: "tuple[Path, int, int]",
) -> "tuple[int, list[str]]":
    """Pool entry point: one shard, by index.

    Returns the shard id alongside its merged work ids so the driver can
    reassemble results in shard order without depending on completion order.

    (It took a single packed tuple because an earlier draft used a *process*
    pool and had to pickle its arguments. That constraint is gone -- the pool
    is threads -- and the shape is kept only because the id-with-result return
    is still what the driver wants.)
    """
    output_dir, shard_id, shards = task
    return shard_id, write_measurement_shard(
        output_dir,
        scheduler_epoch=None,
        shard_id=shard_id,
        shards=shards,
    )


def run_local_aggregation_fanout(
    output_dir: Path,
    *,
    dataset_names: Sequence[str],
    njobs: int,
    shard_timeout_seconds: float = 2 * TARGET_TASK_SECONDS,
) -> "tuple[list[Path], list[str]] | None":
    """Fan out aggregation over a local process pool.

    Spec §8: *"Local ``--njobs`` uses the same decomposition with a process
    pool."* Same shard worker, same shard files, same completeness gate, same
    merge order -- an array of SLURM tasks replaced by a pool of processes, and
    nothing else.

    **The clear happens FIRST, as this function's opening act**, which is the
    local counterpart of clearing at submission time on SLURM. It is the same
    logical point -- the moment the driver starts -- and it is what carries
    §7.5 locally, where the namespace cannot: ``_scheduler_epoch`` is ``None``
    for every local run, so consecutive invocations share one shard directory
    and a prior run's shards would otherwise be merged into this run's master.

    Args:
        output_dir: Run output root.
        dataset_names: Datasets this run finalizes.
        njobs: The ``--njobs`` value.
        shard_timeout_seconds: Backstop for a shard that never returns.
            Defaults to twice :data:`TARGET_TASK_SECONDS`, since a shard is
            *sized* to fit that budget and the doubling is headroom for a slow
            filesystem.

    Returns:
        ``(shard_paths, planned_work_ids)`` for :func:`finalize_run`, or
        ``None`` when there is nothing authorized to aggregate -- in which case
        the caller finalizes directly, exactly as before this phase.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    from ._cli_completion import authorized_measurement_sources

    authorized = authorized_measurement_sources(output_dir)
    if not authorized:
        return None

    shards = local_shard_count(n_sources=len(authorized), njobs=njobs)
    # `workers` is equal to `shards` today **by construction, not by
    # identity**, and they are named apart because they answer different
    # questions: `shards` is a property of the OUTPUT -- how many Parquet
    # files the finalizer must find and merge, carried as K in the manifest --
    # while `workers` is a property of the MACHINE. `max_workers=shards` would
    # read as an identity when it is a coincidence, and nothing would catch
    # them diverging.
    workers = shards
    begin_aggregation_fanout(
        output_dir,
        scheduler_epoch=None,
        shards=shards,
        dataset_names=dataset_names,
    )

    tasks = [(Path(output_dir), shard_id, shards) for shard_id in range(shards)]
    if workers == 1:
        # One worker means no pool, matching `_cli_overlay_rendering.py:166`.
        # The single-shard path still goes through the same worker rather than
        # a special case that could drift from it.
        results = [_run_one_local_shard(tasks[0])]
    else:
        # **Threads, matching the house pattern at
        # `_cli_overlay_rendering.py:180`**, whose rationale
        # (`phenotypicCLI.py:3156-3160`) carries here directly: the heavy work
        # releases the GIL -- polars reads and writes Parquet and is
        # internally multithreaded -- and per-item memory is large enough that
        # fanning out to processes risks exhausting RAM. S-3 measured 2.5 GB
        # peak for N=6,529 in one process; N processes each materialising a
        # slice is exactly the multiplication that comment warns about.
        #
        # An earlier draft used `ProcessPoolExecutor` and **deadlocked**: it
        # defaults to `fork` on Linux, the pytest parent had 82 threads, and
        # `fork()` copies a lock held by any of the other 81 in its held state
        # into a child that never releases it. `spawn` would have fixed that,
        # by paying an interpreter start and a pickle round-trip per shard to
        # avoid a hazard threads never create.
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {
                shard_id: pool.submit(_run_one_local_shard, task)
                for shard_id, task in zip(range(shards), tasks)
            }
            # A gate that can hang has a green and a silence that are
            # indistinguishable from outside: no summary line, so every check
            # grepping for `failed` or parsing `N passed` reads it as success.
            deadline = time.monotonic() + shard_timeout_seconds
            results = []
            for shard_id, future in sorted(futures.items()):
                try:
                    results.append(
                        future.result(
                            timeout=max(deadline - time.monotonic(), 0.0)
                        )
                    )
                except TimeoutError as exc:
                    raise RuntimeError(
                        f"Local aggregation shard {shard_id} of {shards} did "
                        f"not finish within {shard_timeout_seconds:.0f}s. "
                        "Nothing was published. RECOVERY: re-run the same "
                        "command; shards are per-invocation scratch and are "
                        "cleared when the next fan-out begins."
                    ) from exc
        finally:
            # `wait=False` is load-bearing. A stuck thread cannot be killed,
            # and the default `shutdown(wait=True)` -- which a `with` block
            # performs on exit -- would block on it forever, turning the
            # timeout above back into the hang it exists to prevent.
            pool.shutdown(wait=False, cancel_futures=True)

    for shard_id, merged in results:
        write_shard_status(
            output_dir,
            scheduler_epoch=None,
            shard_id=shard_id,
            source_work_ids=merged,
        )
    return resolve_finalizer_shard_inputs(output_dir, None)


# ---------------------------------------------------------------------------
# What the finalizer needs to hold the master in memory (S-3)
# ---------------------------------------------------------------------------

#: Peak RSS of the in-memory merge, **measured**, not projected.
#:
#: S-3, lane ``merge``: 2,577.6 MB at N = 6,529 on the Maresca tree, against a
#: 32 GB threshold -- a 12.7x margin, which is why the verdict is `IN-MEMORY`
#: and ``TASK_FINALIZE`` uses ``pl.concat`` rather than ``sink_parquet``.
#: Streaming measured *more* peak RSS here (421.5 MB vs 314.6 MB on the same
#: comparison), so the usual reason to adopt it is absent.
S3_PEAK_RSS_GB = 2.5

#: The N at which :data:`S3_PEAK_RSS_GB` was measured.
#:
#: Carried beside the value because the value alone is unusable: memory scales
#: with N, so "2.5 GB" without "at 6,529 images" cannot be extrapolated to any
#: other run, and a constant floor derived from it would be correct only at
#: this one N.
S3_MEASURED_AT_N = 6529


def finalizer_memory_advisory(
    *, n_images: int, configured_mem_gb: float
) -> str | None:
    """Return a warning when the finalizer's ``--mem`` looks too small, else ``None``.

    **Projected from one measurement, and the message says so.** S-3 measured
    :data:`S3_PEAK_RSS_GB` at :data:`S3_MEASURED_AT_N` on a single tree. Peak
    RSS scales with the row count, so the floor is scaled linearly by
    ``n_images`` and doubled for headroom. A *constant* floor -- the 8 GB the
    plan names -- is correct only at the N the spike happened to run: a user
    at N = 60,000 clears 8 GB and is OOM-killed anyway, and a user at N = 500
    is warned about nothing.

    This **warns and proceeds**. It never rewrites the user's ``--slurm``
    profile: silently changing what an explicit flag means is worse than a
    finalizer that runs out of memory and says why. And it is silent whenever
    the configured value clears the floor, because a warning that fires on
    correct configurations trains people to ignore it.

    Args:
        n_images: Images this run will aggregate.
        configured_mem_gb: The run's configured per-task memory, in GB.

    Returns:
        A multi-line warning, or ``None`` when the configuration clears the
        projected floor.
    """
    if n_images <= 0 or configured_mem_gb <= 0:
        return None
    projected = 2.0 * S3_PEAK_RSS_GB * (n_images / S3_MEASURED_AT_N)
    if configured_mem_gb >= projected:
        return None
    return (
        f"Finalizer memory may be too small: {configured_mem_gb:.1f} GB "
        f"configured, ~{projected:.1f} GB projected for {n_images} images.\n"
        f"  This is a PROJECTION from a single measurement, not a measured "
        f"requirement for this run: spike S-3 measured "
        f"{S3_PEAK_RSS_GB} GB peak RSS at {S3_MEASURED_AT_N} images on one "
        f"tree, scaled linearly by image count and doubled for headroom.\n"
        f"  A tree with wider tables or more objects per image will need "
        f"more. Raise `--slurm mem_gb=<n>` if the finalizer is OOM-killed."
    )
