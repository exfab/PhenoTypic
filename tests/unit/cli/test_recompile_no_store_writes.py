"""``--mode recompile`` is aggregate + finalize, and writes no store byte.

The two propositions the 2026-09-11 ruling rests on, each driven through the
real local handler (``_handle_recompile``) rather than through the aggregation
it delegates to:

1. a recompile leaves ``results/`` byte- **and inode-** identical; and
2. a tree built with ``--metadata`` -- whose stores are *inverted*, a
   measurements table beside its own ``pht-metadata.parquet`` -- recompiles
   end to end.

Both fail before the ruling was implemented, and **the first one only on the
right tree** -- see :func:`_publish_a_tree_the_old_rewrite_rewrote`, which is
the whole reason (1) is evidence rather than decoration. (2) failed with
``RuntimeError: ... inverted``: the rewrite used the pre-inversion producer,
which would have re-joined the metadata into the measurement table and
stranded the metadata table, and
``_refuse_inverted_stores_before_any_write`` refused the whole run rather
than let that happen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from phenotypic.schema import IMAGE
from phenotypic.sdk_ import (
    DIR_RESULTS,
    master_measurements_parquet_path,
    measurements_parquet_path,
    zarr_store_path,
)

from .conftest import (
    DATASET,
    _install_overlays,
    _install_snapshot,
    _publish_successful_images,
)

#: Keyed on ``Metadata_Well``, which ``_measurements`` puts in every store's
#: baseline, so the snapshot really joins. Both tests use it, for opposite
#: reasons: installed BEFORE the stores it makes them inverted (test 2's
#: subject), installed AFTER it leaves them un-inverted with a snapshot the
#: old rewrite would have joined in (test 1's, and what makes test 1 able to
#: fail at all).
_SNAPSHOT = "Metadata_Well,Metadata_Strain\nA01,WT\nA02,WT\nB01,MUT\nB02,MUT\n"

_STEMS = ["a", "b"]

#: What the master's ``Metadata_ImageName`` actually holds. ``_publish_store``
#: publishes each image with ``relative_image_path=f"{stem}.tiff"``, and the
#: identity the master carries is that **filename**, not the store stem the
#: fixture is keyed on. Spelled out here because getting it wrong makes the
#: coverage control below fail while the property it guards is fine.
_IMAGE_NAMES = [f"{stem}.tiff" for stem in _STEMS]


def _results_fingerprint(output_dir: Path) -> dict[str, tuple[str, int]]:
    """Content **and** identity for every file under ``results/``.

    The content digest alone cannot see a re-promote that reproduces the same
    bytes, which is exactly the shape the removed rewrite had on a
    metadata-free tree: same payload, new store directory, renamed into place.
    ``st_ino`` is what distinguishes "not rewritten" from "rewritten to the
    same value".
    """
    root = output_dir / DIR_RESULTS
    fingerprint: dict[str, tuple[str, int]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        fingerprint[path.relative_to(root).as_posix()] = (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_ino,
        )
    return fingerprint


def _publish_a_tree_the_old_rewrite_rewrote(output_dir: Path) -> None:
    """Stores written WITHOUT the snapshot; the snapshot installed after.

    **The shape matters, and the obvious fixture is vacuous.** A first
    version of this test used a metadata-free tree and passed against the old
    code, which made the change's headline claim evidence-free. The reason is
    exact: with no snapshot, ``prepare_embedded_measurement_table`` returns
    ``join_status="not_requested"`` and the frame is the store's own baseline
    unchanged, so the prepared bytes equal the bytes already on disk -- and
    ``promote_recompile_table_transition`` short-circuits on precisely that
    (``if current == intended: ... return table``, `_cli_recompile_recovery.py`
    at ``92be762a``:636-638) and returns **without writing**. The old rewrite
    was a genuine no-op on that tree. Nothing was wrong with the assertion;
    the tree could not exercise it.

    So the tree needs a snapshot that really joins, while staying
    *un-inverted* -- an inverted store is refused before the first write, and
    that refusal is the subject of the other test, not this one. Writing the
    stores first and installing the snapshot afterwards is exactly that
    combination: ``prepare_image_tables`` splits the tables only when
    ``deliverables/metadata.csv`` exists as the image is written, so these
    stores carry a single joined-shape measurement table and no metadata
    table, and a later ``recompile`` would have re-joined the snapshot into
    every one of them.

    It is also the shape ``--mode migrate`` leaves behind, so this is not a
    contrivance for the test: it is the tree on which the deleted rewrite did
    the most work.
    """
    _publish_successful_images(output_dir, stems=_STEMS, snapshot=None)
    _install_snapshot(output_dir, _SNAPSHOT)
    # `_handle_recompile` opens with the overlay pass, which refuses a store
    # whose overlay is absent and whose marker binds none. Overlays live
    # under `deliverables/`, so they are outside the `results/` fingerprint.
    _install_overlays(output_dir, _STEMS)


def _assert_the_snapshot_would_have_been_joined(output_dir: Path) -> None:
    """Fail loudly if the premise that makes the pin non-vacuous is gone.

    The old rewrite only wrote bytes when the prepared table differed from
    the one on disk, and that required the snapshot to share at least one
    column with the store's recorded ``measurement_columns``. If it ever
    stops sharing one, this test goes quietly green against a rewrite that
    short-circuited -- the exact failure the fixture above exists to fix. So
    the premise is asserted rather than assumed.
    """
    from phenotypic.sdk_ import PhenotypicAttr, read_phenotypic_attributes

    header = _SNAPSHOT.splitlines()[0].split(",")
    for stem in _STEMS:
        baseline = read_phenotypic_attributes(
            zarr_store_path(output_dir, DATASET, stem)
        )[PhenotypicAttr.TABLES]["measurements"]["measurement_columns"]
        shared = sorted(set(header) & set(baseline))
        assert shared, (
            f"{stem}'s baseline {sorted(baseline)} shares no column with the "
            f"snapshot header {header}; the pre-inversion producer would have "
            "returned the baseline unchanged and the old rewrite would have "
            "written nothing, so this test could not have failed against it"
        )


def _recompile(output_dir: Path) -> None:
    """Drive the local recompile handler exactly as ``--mode recompile`` does."""
    from phenotypic.phenotypicCLI import _handle_recompile

    _handle_recompile(
        output_dir,
        None,
        True,
        0.3,
        1,
        no_qc=True,
    )


def test_recompile_writes_no_store_byte(tmp_path: Path) -> None:
    """Every file under ``results/`` keeps its bytes and its inode.

    The change's headline claim, and a **real pre-change failure** rather
    than a mutation pin: against ``92be762a`` this reddens on both halves of
    the fingerprint, because the old rewrite re-joined the snapshot into
    every table and promoted the result root-last.

    **Two controls, because "nothing changed" is the easiest assertion in
    the world to satisfy by accident.** The master must exist and cover both
    stores -- a recompile that aggregated nothing also writes no store byte
    -- and the snapshot must actually share a join key with the stores, or
    the old rewrite would have short-circuited and there would have been no
    pre-change failure to speak of. See
    :func:`_assert_the_snapshot_would_have_been_joined`.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _publish_a_tree_the_old_rewrite_rewrote(output_dir)
    _assert_the_snapshot_would_have_been_joined(output_dir)

    before = _results_fingerprint(output_dir)
    assert before, "no files inventoried under results/; the equality is vacuous"

    _recompile(output_dir)

    master_path = master_measurements_parquet_path(output_dir)
    assert master_path.is_file(), "recompile published no master"
    master = pl.read_parquet(master_path)
    assert master.height > 0
    assert sorted(set(master[str(IMAGE.IMAGE_NAME)].to_list())) == _IMAGE_NAMES, (
        "the master does not cover both stores, so this recompile did not do "
        "the work whose side effects the equality below denies"
    )

    assert _results_fingerprint(output_dir) == before, (
        "recompile rewrote per-image store state; it is supposed to be "
        "aggregate + finalize only"
    )


def test_a_metadata_tree_recompiles_end_to_end(tmp_path: Path) -> None:
    """A run built with ``--metadata`` recompiles, and its stores survive.

    This is the gate the superseded P7 Task 5 Step 1e named as missing: every
    recompile fixture in the tree was built without a snapshot, so nothing
    exercised the case the mode refused outright.
    """
    from phenotypic.sdk_ import PhenotypicAttr, read_phenotypic_attributes
    from phenotypic.sdk_.ngff_ import (
        MEASUREMENT_TABLE_GROUP,
        METADATA_TABLE_GROUP,
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _publish_successful_images(output_dir, stems=_STEMS, snapshot=_SNAPSHOT)
    _install_overlays(output_dir, _STEMS)

    # STANDING RULE: the stores must really be INVERTED, or this passes on a
    # tree the old refusal would have accepted and decides nothing.
    for stem in _STEMS:
        tables = read_phenotypic_attributes(
            zarr_store_path(output_dir, DATASET, stem)
        )[PhenotypicAttr.TABLES]
        assert MEASUREMENT_TABLE_GROUP in tables
        assert METADATA_TABLE_GROUP in tables, (
            f"{stem}'s store is not inverted; the snapshot did not reach the "
            "forward writer and the refusal this test retires never applied"
        )

    before = _results_fingerprint(output_dir)
    _recompile(output_dir)

    master = pl.read_parquet(master_measurements_parquet_path(output_dir))
    # The master is un-joined (P4): intrinsic identity plus measurements.
    assert "Metadata_Strain" not in master.columns
    assert (
        sorted(set(master[str(IMAGE.IMAGE_NAME)].to_list())) == _IMAGE_NAMES
    ), "the master does not cover both stores; the assertions above are vacuous"

    # The join happens once, at finalization, into the mirror.
    mirror = pl.read_parquet(measurements_parquet_path(output_dir))
    assert "Metadata_Strain" in mirror.columns
    assert set(mirror["Metadata_Strain"].drop_nulls().to_list()) == {
        "WT",
        "MUT",
    }

    # And the stores are untouched -- including the metadata tables the old
    # rewrite would have stranded.
    assert _results_fingerprint(output_dir) == before
    for stem in _STEMS:
        tables = read_phenotypic_attributes(
            zarr_store_path(output_dir, DATASET, stem)
        )[PhenotypicAttr.TABLES]
        assert METADATA_TABLE_GROUP in tables
