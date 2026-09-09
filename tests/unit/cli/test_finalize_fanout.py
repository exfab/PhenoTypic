"""Shard sizing for aggregation fan-out (spec §8, Task 1).

**This is correctness machinery, not a performance optimisation, and the
measurement says so.** P0's S-2 spike measured ``seconds_per_image = 0.026``
cold on GPFS against a 6,529-store tree, which puts ``K = 1`` at the design
target of N = 6,000 and the crossover where a second shard first earns its
keep at **N ≈ 34,600** -- 5.8x the N this design was drawn around
(``spikes/RESULTS.md``). Nothing here makes a realistic run faster. What it
does is give the reserved ``TASK_FINALIZE`` entry an index that is always
inside ``MaxArraySize``, so the partial-failure and shard-completeness
machinery built on top of it has somewhere legal to run.

⛔ STANDING RULE, inherited from ``test_finalize_run.py``: every assertion of
a bound is preceded by an assertion that the input actually reached it. A
clamp test whose input never engages the clamp passes on code that has no
clamp at all.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

#: The cluster's real ``MaxArraySize``, from ``scontrol show config`` as
#: recorded in ``spikes/RESULTS.md``. ``MaxSubmitJobs`` is 5,000 there and
#: therefore never the binding term, so these tests state the constant rather
#: than carrying an inert ``min()``.
MAX_ARRAY_SIZE = 2500


def test_the_finalize_trigger_is_counted_against_the_array_bound() -> None:
    """Project CLAUDE.md: 'Count every trigger entry when sizing chunks against
    MaxArraySize.' A formula that sizes K to the bound and THEN appends the
    finalizer produces an array one index too long, which sbatch rejects at
    submission with a message that names neither the trigger nor the formula.

    **The input is chosen so the clamp actually fires.** The plan's draft used
    ``n_images=1_000_000, seconds_per_image=1.0``, whose unclamped target is
    1,112 -- well under the bound -- so ``assert k <= 2499`` passed without
    the clamp ever engaging, and would have passed identically against a
    ``shard_count`` that had no clamp in it. The unclamped target is asserted
    first here, so the equality below is a statement about the clamp.
    """
    from phenotypic._cli._cli_finalize_fanout import (
        TARGET_TASK_SECONDS,
        shard_count,
    )

    n_images = 3_000_000
    seconds_per_image = 1.0
    unclamped = math.ceil(n_images * seconds_per_image / TARGET_TASK_SECONDS)
    assert unclamped > MAX_ARRAY_SIZE, (
        "the fixture does not reach the clamp, so the assertion below would "
        f"hold against an unclamped formula (unclamped={unclamped})"
    )

    k = shard_count(
        n_images=n_images,
        seconds_per_image=seconds_per_image,
        max_array_size=MAX_ARRAY_SIZE,
    )

    assert k == MAX_ARRAY_SIZE - 1, "K + TASK_FINALIZE must fit inside MaxArraySize"


def test_max_array_size_caps_the_index_not_the_task_count() -> None:
    """User's global CLAUDE.md: MaxArraySize (2500 here) caps the INDEX. The
    highest legal index is 2499 -- ``--array=0-2499`` works, ``--array=1-2500``
    is rejected.

    So the interesting quantity is the array spec's **top index**, not K. The
    plan's draft asserted ``array_spec(k) == f"0-{k}"``, which is the
    implementation restated as its own expectation and holds for every K. What
    is asserted instead: the top index the spec names is legal, and the task
    count it implies is K shards plus exactly one finalizer.
    """
    from phenotypic._cli._cli_finalize_fanout import array_spec, shard_count

    k = shard_count(
        n_images=3_000_000,
        seconds_per_image=1.0,
        max_array_size=MAX_ARRAY_SIZE,
    )

    spec = array_spec(k)
    low, _, high = spec.partition("-")
    assert low == "0", f"array spec must be 0-based, got {spec!r}"
    top_index = int(high)

    assert top_index == k, (
        "the finalizer's reserved index is the top of the array, so the top "
        f"index must be K itself; spec={spec!r}, K={k}"
    )
    assert top_index <= MAX_ARRAY_SIZE - 1, (
        f"array spec {spec!r} names index {top_index}, which sbatch rejects "
        f"against MaxArraySize={MAX_ARRAY_SIZE}"
    )
    assert k + 1 <= MAX_ARRAY_SIZE, (
        f"K aggregation shards plus one TASK_FINALIZE entry is {k + 1} tasks, "
        f"which does not fit MaxArraySize={MAX_ARRAY_SIZE}"
    )


def test_shard_count_is_one_for_a_small_run() -> None:
    """The measured case, not a corner case. S-2 puts K = 1 at N = 6,000."""
    from phenotypic._cli._cli_finalize_fanout import shard_count

    assert (
        shard_count(
            n_images=5, seconds_per_image=0.1, max_array_size=MAX_ARRAY_SIZE
        )
        == 1
    )


def test_the_design_target_run_needs_one_shard() -> None:
    """S-2's verdict, instantiated: K = clamp(ceil(6000 x 0.026 / 900), 1, 2499).

    Pins the constant to the measurement rather than to a comment. If
    ``TARGET_TASK_SECONDS`` is ever retuned, this is what says the design
    target moved with it -- the spike measured 169.6 s for the whole 6,529-store
    tree in one shard, against a 900 s budget.
    """
    from phenotypic._cli._cli_finalize_fanout import (
        SECONDS_PER_IMAGE_S2,
        shard_count,
    )

    assert (
        shard_count(
            n_images=6000,
            seconds_per_image=SECONDS_PER_IMAGE_S2,
            max_array_size=MAX_ARRAY_SIZE,
        )
        == 1
    )


def test_shards_are_namespaced_by_scheduler_epoch(tmp_path: Path) -> None:
    """§7.5: aggregation shards are per-invocation scratch, so a prior run's
    shards can never be merged. Recompile already does this
    (``recompile/attempts/<attempt_id>/...``); the pattern generalises.

    The namespace is **not** what makes this correct -- ``_scheduler_epoch``
    returns ``None`` for every local run, so the fan-out empties the directory
    at driver start instead (user ruling, P2 close). The namespace stays
    because it costs nothing and keeps one path shape across both drivers.
    """
    from phenotypic.sdk_ import aggregation_shard_dir

    a = aggregation_shard_dir(tmp_path, "epoch-a")
    b = aggregation_shard_dir(tmp_path, "epoch-b")

    assert a != b
    assert a.parent == b.parent


def test_a_local_run_with_no_scheduler_epoch_still_gets_a_namespace(
    tmp_path: Path,
) -> None:
    """``_scheduler_epoch`` (``sdk_/_run_state.py:246``) returns ``None`` when
    there is no ``slurm_lifecycle.json``, which is every local run.

    The path must stay well-formed rather than acquiring a ``None`` segment,
    and it must not collide with a real epoch that happens to be named for the
    local case.
    """
    from phenotypic.sdk_ import aggregation_shard_dir

    local = aggregation_shard_dir(tmp_path, None)
    scheduled = aggregation_shard_dir(tmp_path, "epoch-a")

    assert "None" not in local.parts, (
        f"a null epoch leaked into the path as a literal: {local}"
    )
    assert local != scheduled
    assert local.parent == scheduled.parent


def test_the_aggregation_shard_dir_does_not_collide_with_recompiles(
    tmp_path: Path,
) -> None:
    """The leaf name is deliberately NOT ``measurement_shards``.

    ``DIR_RECOMPILE_SHARDS`` is already the string ``"measurement_shards"``
    (``sdk_/_io_constants.py:794``) and already names two live and *different*
    paths -- ``recompile_dir(progress) / DIR_RECOMPILE_SHARDS``, which
    ``_cli_finalize_run.py:255`` deletes, and ``attempt_dir /
    DIR_RECOMPILE_SHARDS``, which ``_cli_recompile_worker.py:381`` writes. A
    third use of that string would make an existing ambiguity harder to see,
    so this directory is ``aggregation_shards``.
    """
    from phenotypic.sdk_ import (
        DIR_RECOMPILE_SHARDS,
        aggregation_shard_dir,
        progress_dir,
        recompile_dir,
    )

    shard_dir = aggregation_shard_dir(tmp_path, "epoch-a")
    recompile_shards = (
        recompile_dir(progress_dir(tmp_path)) / DIR_RECOMPILE_SHARDS
    )

    assert DIR_RECOMPILE_SHARDS not in shard_dir.parts, (
        f"the aggregation shard dir reuses recompile's leaf name: {shard_dir}"
    )
    assert recompile_shards not in shard_dir.parents
    assert shard_dir.is_relative_to(progress_dir(tmp_path)), (
        "aggregation shards are machine state and belong under .phenotypic/"
    )


# ---------------------------------------------------------------------------
# The shard worker and the CAN-5 gate
# ---------------------------------------------------------------------------
#
# Real stores throughout, via the promoted conftest helpers. The standing rule
# from `test_finalize_run.py` binds here for the same reason it binds there:
# a shard worker that was only ever handed a literal DataFrame would pass on
# day one against code that never read an embedded table.


def _fanout(tmp_path, *, shards: int, epoch: str | None = "epoch-1"):
    """Start a fan-out and run every shard, returning the statuses."""
    from phenotypic._cli._cli_finalize_fanout import (
        begin_aggregation_fanout,
        write_measurement_shard,
        write_shard_status,
    )

    begin_aggregation_fanout(
        tmp_path, scheduler_epoch=epoch, shards=shards, dataset_names=["plate"]
    )
    merged = []
    for shard_id in range(shards):
        work_ids = write_measurement_shard(
            tmp_path, scheduler_epoch=epoch, shard_id=shard_id, shards=shards
        )
        write_shard_status(
            tmp_path,
            scheduler_epoch=epoch,
            shard_id=shard_id,
            source_work_ids=work_ids,
        )
        merged.append(work_ids)
    return merged


@pytest.mark.parametrize("shards", [1, 2, 3])
def test_the_shards_partition_the_authorized_sources_exactly(
    tmp_path: Path, shards: int
) -> None:
    """Every authorized image lands in exactly one shard, for every K.

    Both halves matter and they fail differently: a **dropped** image is
    missing measurements, a **duplicated** one is double-counted rows in the
    master. A partition test that only checked the union would pass on the
    second.
    """
    import polars as pl

    from phenotypic._cli._cli_completion import authorized_measurement_sources
    from phenotypic._cli._cli_finalize_fanout import collect_shard_paths
    from .conftest import _publish_successful_images

    _publish_successful_images(tmp_path, stems=["a", "b", "c"])
    authorized = authorized_measurement_sources(tmp_path)
    assert authorized and len(authorized) == 3, (
        f"fixture produced {authorized}; the partition claim below would be "
        "vacuous against an empty source set"
    )

    merged = _fanout(tmp_path, shards=shards)

    flat = [work_id for shard in merged for work_id in shard]
    assert sorted(flat) == sorted(set(flat)), (
        f"an image landed in more than one shard: {flat}"
    )
    assert set(flat) == {"work-a", "work-b", "work-c"}, (
        f"the shards did not cover every authorized image: {sorted(set(flat))}"
    )

    paths = collect_shard_paths(tmp_path, "epoch-1")
    assert len(paths) == shards, (
        "one Parquet per shard, including empty ones -- index K counts FILES "
        f"against the carried K; got {paths}"
    )
    rows = sum(pl.read_parquet(path).height for path in paths if path.stat().st_size)
    assert rows == 6, f"3 images x 2 objects should survive sharding, got {rows}"


def test_a_shard_partition_is_stable_across_repeated_runs(
    tmp_path: Path,
) -> None:
    """Assignment is deterministic, which is what makes the local master
    byte-identical across ``--njobs`` (Task 3).

    If the assignment could vary, two runs of identical data would produce
    different masters and ``source_set_digest`` would certify nothing.
    """
    from phenotypic._cli._cli_completion import authorized_measurement_sources
    from phenotypic._cli._cli_finalize_fanout import shard_sources
    from .conftest import _publish_successful_images

    _publish_successful_images(tmp_path, stems=["a", "b", "c"])
    authorized = authorized_measurement_sources(tmp_path)
    assert authorized, "no authorized sources; the comparison is vacuous"

    first = [
        sorted(shard_sources(authorized, shard_id=i, shards=2)) for i in range(2)
    ]
    second = [
        sorted(shard_sources(dict(reversed(list(authorized.items()))), shard_id=i, shards=2))
        for i in range(2)
    ]
    assert first == second, (
        "shard assignment depended on dict iteration order, so the merge "
        "order and therefore the master's bytes are not reproducible"
    )


def test_a_missing_shard_refuses_to_publish_rather_than_certifying_a_short_master(
    tmp_path: Path,
) -> None:
    """CAN-5, at the writer.

    The finalizer is ``afterany``, so it runs when a shard task dies, and
    ``publish_aggregate_snapshot`` used to derive its source set from the
    MARKERS rather than from what was merged. Without this check a master
    missing a shard receives a proof asserting the full success set.

    **K is carried, not counted**: the manifest says 3, and deleting a shard
    file must be detected by comparing files against that carried 3 -- not
    against ``len(glob(...))``, which is the number of files that happen to
    exist and would compare the list against itself.
    """
    from phenotypic._cli._cli_finalize_fanout import (
        resolve_finalizer_shard_inputs,
    )
    from phenotypic.sdk_ import aggregation_shard_dir
    from .conftest import _publish_successful_images

    _publish_successful_images(tmp_path, stems=["a", "b", "c"])
    _fanout(tmp_path, shards=3)

    shards = sorted(aggregation_shard_dir(tmp_path, "epoch-1").glob("shard_*.parquet"))
    assert len(shards) == 3, f"fixture did not produce three shards: {shards}"
    shards[1].unlink()

    with pytest.raises(RuntimeError, match="shard"):
        resolve_finalizer_shard_inputs(tmp_path, "epoch-1")


def test_the_refusal_tells_the_user_what_to_do_next(tmp_path: Path) -> None:
    """The finalizer is TERMINAL: by the time anyone reads this message the
    job has exited, so a traceback containing the word 'shard' and no cue is
    the whole of the user's information. Re-running the same command is the
    documented recovery and must be *in the message*.
    """
    from phenotypic._cli._cli_finalize_fanout import (
        resolve_finalizer_shard_inputs,
    )
    from phenotypic.sdk_ import aggregation_shard_dir
    from .conftest import _publish_successful_images

    _publish_successful_images(tmp_path, stems=["a", "b"])
    _fanout(tmp_path, shards=2)
    next(iter(sorted(aggregation_shard_dir(tmp_path, "epoch-1").glob("shard_*.parquet")))).unlink()

    with pytest.raises(RuntimeError) as excinfo:
        resolve_finalizer_shard_inputs(tmp_path, "epoch-1")
    assert "re-run" in str(excinfo.value).lower()


def test_no_manifest_means_no_fanout_rather_than_an_error(tmp_path: Path) -> None:
    """``None`` is the ordinary local path, not a failure.

    A run that never fanned out has no task manifest, and the finalizer must
    aggregate the embedded tables directly exactly as it did before this
    phase. Raising here would break every non-fan-out invocation.
    """
    from phenotypic._cli._cli_finalize_fanout import (
        resolve_finalizer_shard_inputs,
    )
    from .conftest import _publish_successful_images

    _publish_successful_images(tmp_path, stems=["a"])
    assert resolve_finalizer_shard_inputs(tmp_path, "epoch-1") is None


def test_fanout_start_empties_a_prior_invocations_shards(
    tmp_path: Path,
) -> None:
    """§7.5 and the P2-close ruling: a prior run's shards can never be merged.

    ``_scheduler_epoch`` returns ``None`` for every local run, so consecutive
    local invocations share ONE shard directory and the namespace cannot carry
    this guarantee. Clearing at fan-out start is what carries it -- and the
    epoch used here is ``None`` deliberately, because that is the case the
    namespace does not cover.
    """
    import polars as pl

    from phenotypic._cli._cli_finalize_fanout import (
        begin_aggregation_fanout,
    )
    from phenotypic.sdk_ import aggregation_shard_dir
    from .conftest import _publish_successful_images

    _publish_successful_images(tmp_path, stems=["a", "b"])
    stale_dir = aggregation_shard_dir(tmp_path, None)
    stale_dir.mkdir(parents=True, exist_ok=True)
    ghost = stale_dir / "shard_0099.parquet"
    pl.DataFrame({"Metadata_ImageName": ["GHOST.tif"]}).write_parquet(ghost)
    assert ghost.is_file(), "the fixture planted no stale shard to clear"

    begin_aggregation_fanout(
        tmp_path, scheduler_epoch=None, shards=1, dataset_names=["plate"]
    )

    assert not ghost.exists(), (
        "a prior local invocation's shard survived fan-out start and would be "
        "merged into this run's master"
    )


# ---------------------------------------------------------------------------
# Task 3 -- the local driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("njobs", [1, 2, 8])
def test_local_fanout_produces_a_byte_identical_master(
    tmp_path: Path, njobs: int
) -> None:
    """§8: 'Local --njobs uses the same decomposition with a process pool.'

    Same decomposition means same answer. If the merge order could change the
    master's bytes, two runs of the same data would disagree and the aggregate
    proof's ``source_set_digest`` would certify nothing.

    **The parametrization is only meaningful because local K is NOT
    ``shard_count``.** That function answers a walltime question and returns 1
    for every N below ~34,600, so reusing it here -- which is what the plan
    said to do -- would run all three cases down the identical single-shard
    path and compare a run against itself three times.
    """
    from phenotypic._cli._cli_output_manager import aggregate_measurements
    from phenotypic.sdk_ import master_measurements_parquet_path

    from .conftest import _publish_successful_images

    def _run(root: Path, jobs: int) -> bytes:
        root.mkdir(parents=True, exist_ok=True)
        _publish_successful_images(root, stems=["a", "b", "c"])
        aggregate_measurements(
            output_dir=root,
            dataset_names=["plate"],
            include_dataset_column=True,
            njobs=jobs,
        )
        return master_measurements_parquet_path(root).read_bytes()

    reference = _run(tmp_path / "ref", 1)
    assert reference, "the reference run wrote an empty master"
    assert _run(tmp_path / str(njobs), njobs) == reference


def test_local_k_tracks_njobs_and_not_the_walltime_budget(
    tmp_path: Path,
) -> None:
    """The distinction the plan's 'same shard_count' instruction erases.

    ``shard_count`` is a walltime question and returns 1 at every N this
    project runs; ``--njobs`` is a core-count question. If local K came from
    the former, ``--njobs 4`` would fan out over one shard and the test above
    would be green by construction.
    """
    from phenotypic._cli._cli_finalize_fanout import (
        SECONDS_PER_IMAGE_S2,
        local_shard_count,
        shard_count,
    )

    assert (
        shard_count(
            n_images=6000,
            seconds_per_image=SECONDS_PER_IMAGE_S2,
            max_array_size=2500,
        )
        == 1
    ), "the premise moved; local K may no longer need its own producer"

    assert local_shard_count(n_sources=6000, njobs=4) == 4
    assert local_shard_count(n_sources=3, njobs=8) == 3, (
        "K must never exceed the work, or empty shards are manufactured for "
        "no reason"
    )
    assert local_shard_count(n_sources=6000, njobs=1) == 1


def test_shard_paths_and_njobs_together_are_refused(tmp_path: Path) -> None:
    """They mean contradictory things: 'the shards exist' vs 'build them'.

    Silently preferring one would make the SLURM finalizer's carried shard set
    quietly replaceable by a fresh local fan-out, which is a second producer
    for the master's inputs.
    """
    from phenotypic._cli._cli_output_manager import aggregate_measurements

    from .conftest import _publish_successful_images

    _publish_successful_images(tmp_path, stems=["a"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        aggregate_measurements(
            output_dir=tmp_path,
            dataset_names=["plate"],
            shard_paths=[tmp_path / "shard_0000.parquet"],
            njobs=4,
        )


@pytest.mark.parametrize("shards", [1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize("n_sources", [1, 2, 3, 7, 10])
def test_merging_shards_in_shard_order_reproduces_sorted_order(
    shards: int, n_sources: int
) -> None:
    """The invariant the merge depends on, checked over the whole grid.

    The finalizer merges by sorted shard filename. So unless shard *i* owns a
    **contiguous** block of the sorted sources, concatenating shards in shard
    order is not sorted order, and the master's row order becomes a function
    of K -- two runs of identical data disagreeing on bytes.

    **This is a pure-function test on purpose.** The end-to-end
    byte-identical test caught the strided version only at ``njobs=2``: with
    three sources, ``njobs=8`` clamps to K=3, where the stride degenerates to
    one source per shard and *is* sorted. It passed at both ends of its
    parametrization and failed in the middle, so the fixture size decided
    whether the bug was visible. A grid over (n_sources, shards) does not
    depend on that luck.
    """
    from phenotypic._cli._cli_finalize_fanout import shard_sources

    sources = {Path(f"/out/{i:03d}.parquet"): "plate" for i in range(n_sources)}
    ordered = sorted(sources, key=str)

    slices = [
        list(shard_sources(sources, shard_id=i, shards=shards))
        for i in range(shards)
    ]
    merged = [path for shard in slices for path in shard]

    assert merged == ordered, (
        f"K={shards}, n={n_sources}: merging in shard order gave "
        f"{[p.name for p in merged]}, not sorted order"
    )
    assert sorted(merged) == ordered, "an image was dropped or duplicated"
    sizes = [len(shard) for shard in slices]
    assert max(sizes) - min(sizes) <= 1, (
        f"K={shards}, n={n_sources}: shard sizes {sizes} differ by more than "
        "one, so the split is not balanced"
    )
