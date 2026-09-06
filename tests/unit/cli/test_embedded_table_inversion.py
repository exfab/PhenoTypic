"""Spec §7.1-7.2: the embedded table carries measurements, not user metadata.

Every assertion of a negative or of an equality here is preceded by an
assertion that the fixture produced the thing whose absence or equality is
being claimed. ``assert x not in frame`` is satisfied by an empty frame and
``assert a == b`` by two empty things, so a bare one passes while testing
nothing -- and green is the expected colour, which is what makes that
invisible.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _measurements_with_metadata() -> pd.DataFrame:
    """One image's baseline: intrinsic identity plus one measured column.

    ``Metadata_ImageFile`` and ``Metadata_Dataset`` are **intrinsic** -- the
    writer knows them from the image it is measuring, not from ``--metadata``
    -- so they stay on the measurements table. ``Metadata_Strain`` appears
    only in the CSV fixtures below, which is what makes its absence here a
    statement about the split rather than about the fixture.
    """
    return pd.DataFrame(
        {
            "Object_Label": [1, 2],
            "Shape_Area": [4.0, 9.0],
            "Metadata_ImageFile": ["plate.tif", "plate.tif"],
            "Metadata_Dataset": ["plate", "plate"],
        }
    )


def _metadata_csv(tmp_path: Path) -> Path:
    """A snapshot sharing exactly one column with the measurements."""
    path = tmp_path / "metadata.csv"
    path.write_text(
        "Metadata_ImageFile,Metadata_Strain\nplate.tif,WT\n",
        encoding="utf-8",
    )
    return path


def _unrelated_metadata_csv(tmp_path: Path) -> Path:
    """A snapshot sharing no column with the measurements."""
    path = tmp_path / "unrelated.csv"
    path.write_text("Metadata_Strain\nWT\n", encoding="utf-8")
    return path


def _metadata_csv_with_duplicate_keys(tmp_path: Path) -> Path:
    """Three rows, two of which repeat one key that the image carries."""
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "Object_Label,Metadata_Strain\n1,WT-a\n1,WT-b\n2,MUT\n",
        encoding="utf-8",
    )
    return path


def test_intrinsic_identity_stays_in_the_measurement_table(
    tmp_path: Path,
) -> None:
    """Spec §7.1: a concatenated row that cannot say which image it came from
    is unusable. Metadata_ImageFile, Metadata_Dataset and the object label
    stay."""
    from phenotypic._cli._embedded_measurement_tables import (
        prepare_image_tables,
    )

    prepared = prepare_image_tables(
        _measurements_with_metadata(), _metadata_csv(tmp_path)
    )

    # The fixture must have reached the JOINED arm: on `not_requested` the
    # measurements frame is returned untouched, so every assertion below
    # would hold without the split ever having been exercised.
    assert prepared.join_status == "joined"
    assert prepared.metadata is not None
    assert "Metadata_ImageFile" in prepared.measurements.columns
    assert "Metadata_Dataset" in prepared.measurements.columns
    assert "Object_Label" in prepared.measurements.columns


def test_user_metadata_leaves_the_measurement_table(tmp_path: Path) -> None:
    """§7.3's contract change. Metadata_Strain came from --metadata, not from
    the image, so it belongs in pht-metadata.parquet."""
    from phenotypic._cli._embedded_measurement_tables import (
        prepare_image_tables,
    )

    metadata_csv = _metadata_csv(tmp_path)
    # Establish that the CSV really declares the column whose absence is
    # asserted below; a typo in the fixture would otherwise make the negative
    # assertion pass for the wrong reason.
    assert "Metadata_Strain" in metadata_csv.read_text(encoding="utf-8")

    prepared = prepare_image_tables(_measurements_with_metadata(), metadata_csv)

    assert prepared.join_status == "joined"
    assert prepared.metadata is not None
    assert prepared.measurements.shape[0] > 0, (
        "fixture produced no measured rows; the assertion below is vacuous"
    )
    assert "Metadata_Strain" not in prepared.measurements.columns
    assert "Metadata_Strain" in prepared.metadata.columns


def test_the_measurement_table_equals_the_pre_join_baseline_exactly(
    tmp_path: Path,
) -> None:
    """The boundary already has a name: measurement_columns, computed from the
    baseline BEFORE joining. This asserts the new split IS that projection
    rather than a re-derivation of it."""
    from phenotypic._cli._embedded_measurement_tables import (
        prepare_image_tables,
    )

    baseline = _measurements_with_metadata()
    prepared = prepare_image_tables(baseline, _metadata_csv(tmp_path))

    # Both halves of the equality must be non-empty, and the join must have
    # happened -- `() == ()` is true of a frame with no columns at all, and
    # an unjoined run proves nothing about what the join would have added.
    assert prepared.join_status == "joined"
    assert prepared.measurement_columns, "the baseline declared no columns"
    assert tuple(prepared.measurements.columns) == prepared.measurement_columns
    assert prepared.measurement_columns == tuple(baseline.columns)


def test_no_metadata_table_when_the_join_was_not_requested(
    tmp_path: Path,
) -> None:
    """§7.2: absence is the honest signal."""
    from phenotypic._cli._embedded_measurement_tables import (
        prepare_image_tables,
    )

    # The same measurements DO produce a metadata table when a snapshot is
    # supplied, so `metadata is None` below is a property of the argument
    # rather than of a producer that never builds one.
    with_snapshot = prepare_image_tables(
        _measurements_with_metadata(), _metadata_csv(tmp_path)
    )
    assert with_snapshot.metadata is not None

    prepared = prepare_image_tables(_measurements_with_metadata(), None)

    assert prepared.metadata is None
    assert prepared.join_status == "not_requested"
    assert prepared.join_keys == ()
    assert prepared.metadata_snapshot_sha256 == ""


def test_no_metadata_table_when_no_columns_are_in_common(
    tmp_path: Path,
) -> None:
    """A snapshot that matched nothing still had a snapshot."""
    from phenotypic._cli._embedded_measurement_tables import (
        prepare_image_tables,
    )

    unrelated = _unrelated_metadata_csv(tmp_path)
    # The CSV must be a real, non-empty snapshot: an unreadable or empty one
    # would also produce `no_common_keys`, for a different reason.
    assert pd.read_csv(unrelated).shape[0] > 0

    prepared = prepare_image_tables(_measurements_with_metadata(), unrelated)

    assert prepared.metadata is None
    assert prepared.join_status == "no_common_keys"
    assert prepared.join_keys == ()
    # The digest is recorded even with nothing to join, because the store was
    # still built against that snapshot -- a later edit to it is exactly the
    # divergence the advisory exists to surface.
    assert len(prepared.metadata_snapshot_sha256) == 64


def test_duplicate_metadata_keys_preserve_fan_out(tmp_path: Path) -> None:
    """The behaviour the joined producer already warns about. Losing it
    silently changes row counts in the mirror."""
    from phenotypic._cli._embedded_measurement_tables import (
        prepare_image_tables,
    )

    duplicate = _metadata_csv_with_duplicate_keys(tmp_path)
    source = pd.read_csv(duplicate)
    # Establish the fan-out exists in the source: `len(...) == 3` says
    # nothing if the CSV never carried a repeated key.
    assert source.shape[0] == 3
    assert source["Object_Label"].duplicated().any()

    prepared = prepare_image_tables(_measurements_with_metadata(), duplicate)

    assert prepared.metadata is not None
    assert len(prepared.metadata) == 3
    assert prepared.metadata["Metadata_Strain"].tolist() == [
        "WT-a",
        "WT-b",
        "MUT",
    ]


# ---------------------------------------------------------------------------
# Task 2: both tables are written in the store's own .part, before the root
# ---------------------------------------------------------------------------

DATASET = "dataset-a"
STEM = "plate"

_JOINABLE_SNAPSHOT = "Object_Label,Metadata_Strain\n1,WT\n2,MUT\n"
_EDITED_SNAPSHOT = "Object_Label,Metadata_Strain\n1,WT-edited\n2,MUT\n"
_UNRELATED_SNAPSHOT = "Metadata_Strain\nWT\n"


def _image():
    """One 8x8 RGB image carrying two labelled objects."""
    import numpy as np

    from phenotypic import Image

    image = Image(np.zeros((8, 8, 3), dtype=np.uint8), name="plate")
    objmap = np.zeros((8, 8), dtype=np.uint16)
    objmap[1:3, 1:3] = 1
    objmap[1:3, 5:7] = 2
    image.objmap[:] = objmap
    return image


def _manager(output_dir: Path):
    from phenotypic._cli._cli_output_manager import OutputManager

    return OutputManager.from_config(
        output_dir,
        ext=".tiff",
        include_dataset_column=True,
        save_overlays=False,
    )


def _install_metadata_snapshot(output_dir: Path, text: str | None) -> Path:
    """Write (or remove) ``deliverables/metadata.csv`` -- the run's snapshot.

    Both producers read the snapshot from this one path, so installing it
    here is what makes a fixture a *run with metadata* rather than a call
    with an argument.
    """
    from phenotypic.sdk_ import metadata_csv_deliverable_path

    path = metadata_csv_deliverable_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _store_measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Object_Label": [1, 2],
            "Shape_Area": [4.0, 4.0],
            "Metadata_ImageName": ["plate.tiff", "plate.tiff"],
        }
    )


def _save_store(output_dir: Path, *, snapshot: str | None) -> Path:
    """Promote one real store through the forward writer."""
    _install_metadata_snapshot(output_dir, snapshot)
    store = _manager(output_dir).save_image_store(
        _image(),
        DATASET,
        STEM,
        work_id="work-1",
        measurements=_store_measurements(),
    )
    assert store is not None, "the forward writer failed to promote a store"
    return store


def _replace_store_tables(output_dir: Path, *, snapshot: str | None) -> Path:
    """Re-measure one existing store -- the ``--mode measure`` table path."""
    from phenotypic.sdk_ import zarr_store_path

    _install_metadata_snapshot(output_dir, snapshot)
    store = zarr_store_path(output_dir, DATASET, STEM)
    _manager(output_dir).replace_image_store_measurements(
        store, _store_measurements(), DATASET
    )
    return store


def _root(store: Path) -> dict:
    import json

    from phenotypic.sdk_ import STORE_ROOT_JSON

    return json.loads((store / STORE_ROOT_JSON).read_text(encoding="utf-8"))


def _phenotypic(store: Path) -> dict:
    from phenotypic.sdk_.ngff_ import PhenotypicAttr

    return _root(store)["attributes"][PhenotypicAttr.ROOT]


def _snapshot_sha256(store: Path) -> str | None:
    from phenotypic.sdk_.ngff_ import PhenotypicAttr

    block = _phenotypic(store).get(PhenotypicAttr.METADATA_TABLE)
    if block is None:
        return None
    return block.get(PhenotypicAttr.SNAPSHOT_SHA256)


def _sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _embedded_measurements(store: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

    return pq.read_table(store / MEASUREMENT_TABLE_RELATIVE_PATH).to_pandas()


def _metadata_payload(store: Path) -> Path:
    from phenotypic.sdk_ import METADATA_TABLE_RELATIVE_PATH

    return store / METADATA_TABLE_RELATIVE_PATH


def test_both_tables_land_in_the_same_part_before_the_root(
    tmp_path: Path,
) -> None:
    """D-A / INV-PROVEN. The root zarr.json is written last and is the
    record's content anchor, so anything written after it is a mutation of a
    proven artifact. Writing metadata in the same .part is what makes the
    backfill unnecessary.

    The ordering itself is structural -- both tables go into the unpromoted
    part and ``promote_store`` follows the root -- so what is checkable on the
    promoted store is that it carries both files, the new group document, and
    a root that declares them.
    """
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from phenotypic.sdk_.ngff_ import (
        METADATA_TABLE_GROUP,
        STORE_ROOT_JSON,
        TABLES_GROUP,
        PhenotypicAttr,
    )

    store = _save_store(tmp_path, snapshot=_JOINABLE_SNAPSHOT)

    assert (store / MEASUREMENT_TABLE_RELATIVE_PATH).is_file()
    assert _metadata_payload(store).is_file()
    assert METADATA_TABLE_GROUP in _phenotypic(store)[PhenotypicAttr.TABLES]
    # M2: a Zarr v3 hierarchy needs a group document for the new group too,
    # and the contract compares it by exact equality.
    assert (
        store / TABLES_GROUP / METADATA_TABLE_GROUP / STORE_ROOT_JSON
    ).is_file()


def test_the_store_records_the_metadata_snapshot_it_was_built_against(
    tmp_path: Path,
) -> None:
    """D-A: stores keep the metadata they were built with, and say which one.
    That is what lets resolve_run_state DERIVE the divergence advisory instead
    of tracking a backfill stage.

    The key is ``metadata_table``, NOT ``metadata`` -- P1 shipped the reader
    against that spelling because ``phenotypic.metadata`` is taken by the
    image-metadata sections.
    """
    from phenotypic.sdk_ import metadata_csv_deliverable_path

    store = _save_store(tmp_path, snapshot=_JOINABLE_SNAPSHOT)

    snapshot = metadata_csv_deliverable_path(tmp_path)
    assert snapshot.is_file(), "the fixture installed no metadata snapshot"
    assert _snapshot_sha256(store) == _sha256_of(snapshot)


def test_a_metadata_free_run_records_no_metadata_table_block(
    tmp_path: Path,
) -> None:
    """H2. ``resolve_run_state``'s divergence advisory fires when a store's
    recorded snapshot is neither None nor the run's current
    ``metadata_sha256``. On a run with no --metadata that value is None and
    the producer's digest is "", so writing the block unconditionally makes
    ``"" not in (None, None)`` true and reports EVERY store on EVERY
    metadata-free run as diverged. An advisory that is always on teaches
    people to ignore the one that will matter.
    """
    from phenotypic.sdk_.ngff_ import METADATA_TABLE_GROUP, PhenotypicAttr

    # Establish the block is one this producer CAN write, so its absence
    # below is a property of the metadata-free run rather than of a writer
    # that never emits it at all.
    with_metadata = _save_store(
        tmp_path / "with", snapshot=_JOINABLE_SNAPSHOT
    )
    assert _snapshot_sha256(with_metadata)

    store = _save_store(tmp_path / "without", snapshot=None)

    phenotypic = _phenotypic(store)
    assert PhenotypicAttr.METADATA_TABLE not in phenotypic
    assert METADATA_TABLE_GROUP not in phenotypic[PhenotypicAttr.TABLES]
    assert not _metadata_payload(store).exists()


def test_a_no_common_keys_run_still_records_its_snapshot(
    tmp_path: Path,
) -> None:
    """The other half of H2, and the reason the rule is not simply "omit
    whenever there is no metadata table".

    ``no_common_keys`` means a metadata.csv WAS supplied and matched nothing.
    There is no pht-metadata.parquet to write, but the store was still built
    against a specific snapshot, and a later edit to that snapshot is exactly
    the divergence the advisory exists to surface. So: omit the block only for
    ``not_requested``.
    """
    from phenotypic.sdk_ import metadata_csv_deliverable_path
    from phenotypic.sdk_.ngff_ import METADATA_TABLE_GROUP, PhenotypicAttr

    store = _save_store(tmp_path, snapshot=_UNRELATED_SNAPSHOT)

    snapshot = metadata_csv_deliverable_path(tmp_path)
    assert snapshot.is_file(), "the fixture installed no metadata snapshot"
    assert _snapshot_sha256(store) == _sha256_of(snapshot)
    # Present block, absent table: there is genuinely nothing to join, so the
    # descriptor and the payload must both be absent rather than empty.
    assert (
        METADATA_TABLE_GROUP not in _phenotypic(store)[PhenotypicAttr.TABLES]
    )
    assert not _metadata_payload(store).exists()


def test_measure_mode_refreshes_the_table_and_the_root_together(
    tmp_path: Path,
) -> None:
    """INV-PROVEN, second obligation -- and the reason the invariant is
    stated the way it is (CAN-3).

    The stronger claim ("nothing ever writes into a proven store") is FALSE
    and was false before this change. --mode measure re-measures from stores
    and replaces the embedded table; the in-place branch it used to take fired
    whenever the descriptor was unchanged, rewriting table.parquet directly in
    the promoted store with no .part and NO ROOT REWRITE. Two things broke,
    both silently: the record's store digest still matched, so the proof
    certified content that changed underneath it; and ``snapshot_sha256``
    lives in the root, so the divergence advisory read a value that branch
    never refreshed.
    """
    from phenotypic.sdk_ import STORE_ROOT_JSON, metadata_csv_deliverable_path

    store = _save_store(tmp_path, snapshot=_JOINABLE_SNAPSHOT)
    original_digest = _sha256_of(metadata_csv_deliverable_path(tmp_path))
    # Establish the starting state: the store records the ORIGINAL snapshot,
    # and the edit below really changes the digest. Without both, the final
    # equality could hold with nothing having been refreshed.
    assert _snapshot_sha256(store) == original_digest
    root_before = (store / STORE_ROOT_JSON).read_bytes()

    _replace_store_tables(tmp_path, snapshot=_EDITED_SNAPSHOT)

    edited_digest = _sha256_of(metadata_csv_deliverable_path(tmp_path))
    assert edited_digest != original_digest, "the fixture edit changed nothing"

    root_after = (store / STORE_ROOT_JSON).read_bytes()
    # This one passes for weak reasons and its message overclaims; kept as a
    # cheap smoke check with an honest message. THE SECOND ASSERTION IS THE
    # LOAD-BEARING ONE and must not be weakened or reordered away.
    assert root_after != root_before, "the root was not rewritten at all"

    assert _snapshot_sha256(store) == edited_digest, (
        "the embedded table was rewritten without refreshing the root's "
        "recorded snapshot, so the per-image proof still certifies the old "
        "digest and the divergence advisory reads a stale value -- "
        "INV-PROVEN's second obligation"
    )


@pytest.mark.parametrize("producer", ["forward", "measure"])
def test_the_metadata_table_is_written_not_a_joined_one(
    tmp_path: Path, producer: str
) -> None:
    """INV-PROVEN, second obligation, other half.

    There are TWO producers, not one (B4). ``save_image_store`` is the
    forward one, on the path every ``full`` run takes;
    ``replace_image_store_measurements`` is the ``--mode measure`` one. If
    either keeps feeding the joined producer it silently un-inverts every
    image it touches -- a joined table.parquet and no pht-metadata.parquet, on
    a tree whose other stores are inverted.
    """
    import pyarrow.parquet as pq

    if producer == "forward":
        store = _save_store(tmp_path, snapshot=_JOINABLE_SNAPSHOT)
    else:
        _save_store(tmp_path, snapshot=None)
        store = _replace_store_tables(tmp_path, snapshot=_JOINABLE_SNAPSHOT)

    # Establish that the snapshot really carries the column whose absence is
    # asserted next, and that the table has rows at all: `not in` on the
    # columns of an empty frame is satisfied by anything.
    assert "Metadata_Strain" in _JOINABLE_SNAPSHOT
    measurements = _embedded_measurements(store)
    assert measurements.shape[0] > 0, (
        "the store embedded no measured rows; the assertion below is vacuous"
    )
    assert "Shape_Area" in measurements.columns

    assert "Metadata_Strain" not in measurements.columns
    assert _metadata_payload(store).is_file()

    metadata = pq.read_table(_metadata_payload(store)).to_pandas()
    assert "Metadata_Strain" in metadata.columns
    assert sorted(metadata["Metadata_Strain"]) == ["MUT", "WT"]


def test_the_recompile_guard_refuses_an_inverted_store(
    tmp_path: Path,
) -> None:
    """Recompile still builds its payload with the PRE-inversion producer, so
    running it on an inverted store rejoins metadata into the measurement
    table and drops ``pht-metadata.parquet`` -- silently.

    P4 Task 1 expected ``_replace_and_republish_table``'s ``isinstance``
    check to fail closed here. It does not: the producer still returns the
    legacy type, so the check passes. This guard restores the loud failure.
    """
    from phenotypic._cli._cli_recompile_tables import _refuse_inverted_store
    from phenotypic.sdk_.ngff_ import METADATA_TABLE_GROUP, PhenotypicAttr

    store = _save_store(tmp_path, snapshot=_JOINABLE_SNAPSHOT)
    # The store must actually BE inverted, or the raise below would be
    # testing a guard that refuses everything.
    assert METADATA_TABLE_GROUP in _phenotypic(store)[PhenotypicAttr.TABLES]

    with pytest.raises(RuntimeError, match="inverted store"):
        _refuse_inverted_store(store)


def test_the_recompile_guard_still_accepts_a_pre_inversion_store(
    tmp_path: Path,
) -> None:
    """The other half: a guard that refused every store would pass the test
    above while breaking `--mode recompile` outright.

    "Must not raise" is the plan's named vacuity trap, so the store is first
    shown to be a real one the guard actually inspects -- it declares a
    measurement table and no metadata table.
    """
    from phenotypic._cli._cli_recompile_tables import _refuse_inverted_store
    from phenotypic.sdk_.ngff_ import (
        MEASUREMENT_TABLE_GROUP,
        METADATA_TABLE_GROUP,
        PhenotypicAttr,
    )

    store = _save_store(tmp_path, snapshot=None)
    tables = _phenotypic(store)[PhenotypicAttr.TABLES]
    assert MEASUREMENT_TABLE_GROUP in tables, "the guard would read nothing"
    assert METADATA_TABLE_GROUP not in tables

    _refuse_inverted_store(store)


def test_the_recompile_guard_is_wired_into_the_rewrite_path() -> None:
    """A guard that is defined and never called is worse than no guard: it
    reads as coverage. The two tests above exercise ``_refuse_inverted_store``
    directly, because driving the whole rewrite needs records and store
    locks -- so this one pins the call site instead.

    It proves the guard is *invoked*, not that the surrounding transaction is
    correct. Delete all three with the recompile repoint.
    """
    import inspect

    from phenotypic._cli import _cli_recompile_tables

    source = inspect.getsource(
        _cli_recompile_tables._replace_and_republish_table
    )
    assert "_refuse_inverted_store(store_path)" in source


def test_re_measuring_without_metadata_clears_the_stores_metadata_block(
    tmp_path: Path,
) -> None:
    """NEW-2. The removal half of H2's rule.

    Omitting the block on a metadata-free BUILD is not enough: a store that
    LOSES its metadata must lose the block too, or ``_store_metadata_snapshot``
    keeps returning a digest that no longer describes anything and the
    divergence advisory fires on every such store forever.
    """
    from phenotypic.sdk_.ngff_ import METADATA_TABLE_GROUP, PhenotypicAttr

    store = _save_store(tmp_path, snapshot=_JOINABLE_SNAPSHOT)
    # Establish there IS something to clear; otherwise every assertion below
    # holds on a store that never had a metadata table.
    assert _snapshot_sha256(store)
    assert _metadata_payload(store).is_file()

    _replace_store_tables(tmp_path, snapshot=None)

    phenotypic = _phenotypic(store)
    assert PhenotypicAttr.METADATA_TABLE not in phenotypic
    assert METADATA_TABLE_GROUP not in phenotypic[PhenotypicAttr.TABLES]
    assert not _metadata_payload(store).exists()
