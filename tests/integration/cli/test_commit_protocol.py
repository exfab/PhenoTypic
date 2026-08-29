"""Commit-protocol cases that need a **real** ``save2zarr``, not a fake store.

Phase 1's ``tests/unit/sdk_/test_ngff_promote.py`` already pins the promote
primitive against a ``_fake_store`` fixture -- uuid ``.part`` names, the
move-aside, the rollback, the retry budget, the sweep's age guard, and the
durable-write flushes. Everything there is in-process and synthetic.

Two cases cannot be covered that way, and only those two live here:

* **(b) concurrency.** Several *real* operating-system processes writing the
  same stem through the real writer. Duplicate execution is benign today -- a
  SLURM array can dispatch the same image more than once -- and it must stay
  benign.
* **(c) a stale ``.part``.** A killed worker's leftovers must be ignored, never
  merged into, by a writer that builds its own uuid-suffixed sibling.

Case (a) -- an interrupted store reading as absent -- is Phase 3's
``test_interrupted_store_classifies_stage1``. It cannot be demonstrated here:
every byte is written into the ``.part`` sibling and only ``promote_store``
ever creates the published path, so ``not final.exists()`` holds under *any*
write order, including the reversed one.

What "root ``zarr.json`` last" actually buys is **flush** ordering (design
§3.7), not reader visibility: without it the kernel may make the root durable
before the chunks it describes, leaving a store that passes
``valid_staged_store`` while reading ``fill_value``. That is a durability
property, which this suite does not exercise -- see
``test_durable_promote_flushes_every_file_and_every_directory`` in
``tests/unit/sdk_/test_ngff_promote.py`` for the part of it that is testable.

**What the concurrency test is NOT.** It is a property test, not the mutation
gate for ``promote_store``'s retry loop, and the plan's Phase-7 exit criterion
claiming otherwise was measured and found wrong. With the loop reduced to a
single ``exists -> move-aside -> replace`` pass, four writers over six rounds
went red in **2 of 8** measured runs -- and the error observed was ``ENOENT``
(a sibling's move-aside had already taken ``final`` out from under the loser),
not the ``ENOTEMPTY`` the criterion names. The narrowest window in the race is
one ``rename``, so no un-instrumented multi-process test can close it reliably.

The mutant is killed **deterministically** by
``tests/unit/sdk_/test_ngff_promote.py::test_a_concurrent_promoter_appearing_mid_retry_is_benign``,
which injects the interleaving instead of hoping for it. That test, not this
one, is the gate; this one is the end-to-end property the gate exists to
protect, and it must never be relied on to go red.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_.ngff_ import PART_SUFFIX, TRASH_SUFFIX, valid_staged_store

#: Set once per spawned worker by :func:`_init_worker`. A ``multiprocessing``
#: primitive survives ``spawn`` only when it is inherited at process creation,
#: which is what ``Pool(initializer=..., initargs=...)`` does.
_BARRIER: "mp.Barrier | None" = None  # type: ignore[name-defined]

#: Rounds of the race, and writers per round. Two writers is the minimum that
#: races at all; four over six rounds widens the sample without changing the
#: property, and a SLURM array can genuinely dispatch the same image more than
#: twice. See the module docstring for what this does and does not prove.
_RACE_ROUNDS = 6
_RACE_WRITERS = 4


def _init_worker(barrier) -> None:
    global _BARRIER
    _BARRIER = barrier


def _write_one(args) -> str:
    """Build an image, then write it to *final* in lockstep with the sibling.

    The barrier sits after the expensive construction and immediately before
    the public ``save2zarr`` call -- not inside the promote -- so nothing here
    depends on the writer's internal shape. A behaviour-preserving refactor of
    ``promote_store`` cannot make this test red.
    """
    final, marker = args
    image = Image(load_synth_yeast_plate())
    image._metadata.public["Metadata_Strain"] = marker
    if _BARRIER is not None:
        _BARRIER.wait(timeout=300)
    image.save2zarr(final)
    return marker


def _leftovers(directory: Path) -> list[str]:
    """Every ``.part`` / ``.trash`` sibling. ``glob`` sees dotfiles."""
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.name.endswith((PART_SUFFIX, TRASH_SUFFIX))
    )


def test_two_concurrent_writers_produce_one_coherent_winner(tmp_path: Path) -> None:
    """The property the uuid ``.part`` and the retrying promote exist to provide.

    Four real processes race on one stem, released together by a barrier and
    repeated over several rounds. Neither the barrier nor the round count
    changes the property -- they only widen a race window whose narrowest part
    is a single ``rename`` -- and the mutation proof below records what one
    round of two writers actually caught.

    No writer may raise: ``pool.map`` re-raises a child's exception, so an
    ``ENOTEMPTY`` anywhere fails this test. The survivor must be one coherent
    store, not an interleaving of four writers' chunks.

    The metadata marker is what makes "coherent" checkable: an interleaved
    store would still validate, so the assertion is that the winner's
    ``Metadata_Strain`` is one writer's value, read back through the real
    loader alongside its own pixels.
    """
    context = mp.get_context("spawn")
    for round_number in range(_RACE_ROUNDS):
        target = tmp_path / f"round{round_number}"
        target.mkdir()
        final = target / "p.ome.zarr"
        markers = [chr(ord("A") + i) for i in range(_RACE_WRITERS)]
        barrier = context.Barrier(_RACE_WRITERS)
        with context.Pool(
            _RACE_WRITERS, initializer=_init_worker, initargs=(barrier,)
        ) as pool:
            pool.map(_write_one, [(final, m) for m in markers], chunksize=1)

        assert valid_staged_store(final) is True
        winner = Image.load_zarr(final)._metadata.public["Metadata_Strain"]
        assert winner in set(markers)
        assert _leftovers(target) == []


def test_a_new_write_does_not_reuse_a_stale_part(tmp_path: Path) -> None:
    """A killed worker's leftovers must never be merged into.

    The stale directory carries a chunk at the exact key the new write will
    produce and fills it with garbage. A writer that reused the ``.part``
    instead of building its own uuid-suffixed sibling would either promote
    those bytes or fail to overwrite them; either way ``gray`` would not match
    the source image.
    """
    image = Image(load_synth_yeast_plate())
    final = tmp_path / "p.ome.zarr"
    stale = final.parent / f".p.ome.zarr.deadbeefdeadbeef{PART_SUFFIX}"
    (stale / "gray" / "0").mkdir(parents=True)
    (stale / "gray" / "0" / "0.0").write_bytes(b"garbage")

    image.save2zarr(final)

    assert valid_staged_store(final) is True
    assert (Image.load_layer_zarr(final, "gray") == image.gray[:]).all()
    assert stale.is_dir(), "the sweep, not the writer, is what clears an orphan"
