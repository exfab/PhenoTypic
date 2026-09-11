"""``finalize_run`` -- the one aggregation + join + publish path (spec §7.4).

**Every fixture here builds REAL stores** through the forward writer
(``OutputManager.save_image_store`` -> ``prepare_image_tables`` ->
``promote_store``), publishes a real per-image record, and aggregates them.
Nothing hands ``finalize_run`` a literal DataFrame.

That is the single most important property of this file. The tests that
*appeared* to cover post-master finalization before P4 either fed
``finalize_post_master_outputs`` a hand-built frame
(``test_cli_output_manager.py``) or hand-constructed a
``PreparedEmbeddedMeasurementTable`` (``test_embedded_measurement_aggregation.py``),
so every one of them stayed on the **pre-inversion** path and would have passed
on day one against code that never aggregated an inverted store.

⛔ STANDING RULE, applied throughout: every assertion of a negative or of an
equality is preceded by an assertion that the fixture produced the thing whose
absence or equality is claimed. ``assert x not in frame`` is satisfied by an
empty frame; ``assert a == b`` by two empty things.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from phenotypic._cli._cli_completion import (
    authorized_measurement_sources,
    publish_image_success,
)
from phenotypic._cli._cli_finalize_run import finalize_run
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    STORE_ROOT_JSON,
    DATASET_AGGREGATED_PARQUET,
    DIR_MEASUREMENTS,
    DIR_RESULTS,
    analysis_full_parquet_path,
    chunk_parquet_path,
    master_measurements_parquet_path,
    measurements_parquet_path,
    metadata_csv_deliverable_path,
    order_measurement_columns,
    progress_dir,
    zarr_store_path,
)

from .conftest import (
    DATASET,
    _install_snapshot,
    _install_state,
    _master_bytes,
    _measurements,
    _publish_store,
    _publish_successful_images,
    _run_mode,
)


#: The join key. ``Metadata_Well`` is shared by the measurements and the
#: snapshot, so `prepare_metadata_join_keys` picks it and nothing else --
#: `Metadata_ImageName` is deliberately kept OUT of the CSV so the key set is
#: one column and every assertion below is about that one join.
_SNAPSHOT = (
    "Metadata_Well,Metadata_Strain\n"
    "A01,WT\nA02,WT\nB01,MUT\nB02,MUT\n"
)
_SNAPSHOT_WITH_PHANTOM = _SNAPSHOT + "Z99,GHOSTSTRAIN\n"

#: Keyed on ``Metadata_ImageName`` instead, for the tests that read the
#: master's metadata namespace rather than its join. ``Metadata_Well`` is
#: not a schema member, so a baseline carrying it puts an *unowned*
#: metadata header in the master -- realistic for a custom operation, but
#: not for the forward pipeline, whose master metadata is IMAGE-owned plus
#: ``Metadata_Dataset``. See
#: ``test_master_carries_user_metadata_reads_ownership_not_the_prefix``.
_SNAPSHOT_BY_IMAGE = (
    "Metadata_ImageName,Metadata_Strain\na.tiff,WT\nb.tiff,MUT\n"
)
_EDITED_SNAPSHOT = (
    "Metadata_Well,Metadata_Strain\n"
    "A01,WT-edited\nA02,WT\nB01,MUT\nB02,MUT\nC01,MUT\nC02,MUT\n"
)


# ---------------------------------------------------------------------------
# Fixture machinery -- real stores, real records, real state
# ---------------------------------------------------------------------------
#
# **The store-building half now lives in ``conftest.py``** and is imported
# above: ``_publish_successful_images`` and everything it stands on. P5's
# fan-out suite (``test_finalize_fanout.py``) needs the same real stores, and
# a second copy is how the standing rule at the top of this file quietly
# becomes false in one of the two suites. What stays here is what only this
# file asks about -- the snapshot path and the store's recorded snapshot
# digest.


def _snapshot_path(tmp_path: Path) -> Path:
    return metadata_csv_deliverable_path(tmp_path)


def _store_snapshot_sha256(store: Path) -> str | None:
    from phenotypic.sdk_.ngff_ import PhenotypicAttr

    root = json.loads((store / STORE_ROOT_JSON).read_text(encoding="utf-8"))
    block = root["attributes"][PhenotypicAttr.ROOT].get(
        PhenotypicAttr.METADATA_TABLE
    )
    return None if block is None else block.get(PhenotypicAttr.SNAPSHOT_SHA256)


def _master(tmp_path: Path) -> pl.DataFrame:
    return pl.read_parquet(master_measurements_parquet_path(tmp_path))


def _mirror(tmp_path: Path) -> pl.DataFrame:
    return pl.read_parquet(measurements_parquet_path(tmp_path))


def _measured(mirror: pl.DataFrame) -> pl.DataFrame:
    return mirror.filter(pl.col("QC_MetadataOnly").fill_null(False).not_())


def _phantoms(mirror: pl.DataFrame) -> pl.DataFrame:
    return mirror.filter(pl.col("QC_MetadataOnly").fill_null(False))


def _concat_of_embedded_tables(tmp_path: Path) -> pl.DataFrame:
    from phenotypic._cli._measurement_sources import (
        add_metadata_image_name_from_filename,
    )
    from phenotypic._cli._cli_parquet_agg import aggregate_parquet_files

    sources = authorized_measurement_sources(tmp_path)
    assert sources, "no authorized sources; the comparison frame is vacuous"
    frame = aggregate_parquet_files(
        file_paths=list(sources),
        path_to_dataset=sources,
        include_dataset_column=True,
        keep_filename=True,
    )
    assert frame is not None
    return add_metadata_image_name_from_filename(frame)


# -- poison ----------------------------------------------------------------


def _poison() -> pl.DataFrame:
    """A row that exists in NO embedded table.

    ``Metadata_ImageName`` is mandatory, not decorative:
    ``_aggregate_needs_image_name_recovery`` returns ``True`` for a frame with
    no ``IMAGE.IMAGE_NAME`` column, and ``discover_measurement_sources`` then
    SKIPS the aggregate for recovery whenever individual Parquets exist. A
    poison frame without it would never be chosen, and the legacy-arm test
    below would be green because the poison was never read.
    """
    return pl.DataFrame(
        {
            "Metadata_ImageName": ["GHOST.tif"],
            "Metadata_Well": ["G01"],
            "Shape_Area": [0.0],
            "Object_Label": [99],
        }
    )


def _plant_stale_chunk_parquet(tmp_path: Path, poison: pl.DataFrame) -> Path:
    path = chunk_parquet_path(progress_dir(tmp_path), 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    poison.write_parquet(path)
    return path


def _plant_stale_shard(tmp_path: Path, poison: pl.DataFrame) -> Path:
    from phenotypic.sdk_ import DIR_RECOMPILE_SHARDS, recompile_dir

    path = (
        recompile_dir(progress_dir(tmp_path))
        / DIR_RECOMPILE_SHARDS
        / "shard_0.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    poison.write_parquet(path)
    return path


def _dataset_aggregate_path(tmp_path: Path) -> Path:
    return (
        tmp_path
        / DIR_RESULTS
        / DATASET
        / DIR_MEASUREMENTS
        / DATASET_AGGREGATED_PARQUET
    )


def _plant_stale_dataset_aggregate(
    tmp_path: Path, poison: pl.DataFrame
) -> Path:
    path = _dataset_aggregate_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    poison.write_parquet(path)
    return path


def _plant_stale_analysis_full(tmp_path: Path, poison: pl.DataFrame) -> Path:
    path = analysis_full_parquet_path(progress_dir(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    poison.write_parquet(path)
    return path


def _plant_stale_master(tmp_path: Path, poison: pl.DataFrame) -> Path:
    path = master_measurements_parquet_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    poison.write_parquet(path)
    return path


def _publish_legacy_external_record(output_dir: Path, stem: str) -> Path:
    """Publish a record whose measurement authority is a LEGACY external Parquet.

    ``authorized_measurement_sources`` reads each record's
    ``artifacts["measurements"]`` path verbatim, so a record pointing at
    ``results/<ds>/measurements/<stem>.parquet`` puts a legacy-shaped source
    into the **authorized** mapping. That is the only way the authorized arm
    can hold a mixture, and it is the state
    ``refuse_mixed_measurement_authority`` exists for -- the reachability half
    that the direct-call test below cannot supply.

    Args:
        output_dir: Run output root.
        stem: Image stem for the legacy Parquet and its record.

    Returns:
        The legacy external Parquet the record certifies.
    """
    path = (
        output_dir
        / DIR_RESULTS
        / DATASET
        / DIR_MEASUREMENTS
        / f"{stem}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(_measurements(stem)).write_parquet(path)
    publish_image_success(
        output_dir,
        work_id=f"work-{stem}",
        dataset=DATASET,
        relative_image_path=f"{stem}.tiff",
        image_stem=stem,
        mode="full",
        attempt_id="attempt-1",
        lifecycle_epoch="epoch-1",
        artifacts={"measurements": path},
    )
    return path


def _build_legacy_external_parquet_tree(tmp_path: Path) -> None:
    """A genuine pre-record tree: external Parquets, NO progress payloads.

    Precondition 1 of the legacy-arm test. ``authorized_measurement_sources``
    returns ``None`` **only** when neither progress tree holds a single
    ``*/*.json`` (``_cli_completion.py`` ``_sources_without_state``), so the
    fixture must publish no record and no marker at all -- publishing two
    images, by either shape, gets a non-``None`` mapping back and the test
    would pin the authorized arm it was written to avoid.
    """
    meas_dir = tmp_path / DIR_RESULTS / DATASET / DIR_MEASUREMENTS
    meas_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        pl.from_pandas(_measurements(stem)).write_parquet(
            meas_dir / f"{stem}.parquet"
        )


# ---------------------------------------------------------------------------
# Task 4 -- the three entry points, one master
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full", "measure", "recompile"])
def test_every_mode_produces_a_byte_identical_master(
    tmp_path: Path, mode: str
) -> None:
    """§7.4: recompile becomes "call finalize_run again", not a parallel
    implementation that must be kept in sync. Three modes, one master.

    ``PARQUET_WRITE_OPTIONS`` is deterministic, so byte equality is the
    strongest available statement of "the same frame, written the same way" --
    and it is what catches a mode that quietly adds a column, reorders one, or
    joins something the others do not.
    """
    output = _run_mode(tmp_path / "run", mode)
    reference = _run_mode(tmp_path / "ref", "full")

    # STANDING RULE. This is the phase's HEADLINE CLAIM, and the bare equality
    # is satisfied by two FAILED runs -- `finalize_run` returns None and writes
    # nothing when no source could be read. Establish that both runs produced a
    # master with rows in it, then compare.
    for out in (output, reference):
        master = master_measurements_parquet_path(out)
        assert master.is_file(), (
            f"{out} produced no master; the comparison is vacuous"
        )
        assert pl.read_parquet(master).height > 0, f"{out}'s master is empty"

    assert _master_bytes(output) == _master_bytes(reference)


# ---------------------------------------------------------------------------
# Step 1 -- INV-INPUTS, the phase's gate
# ---------------------------------------------------------------------------


def test_finalize_run_ignores_every_stale_intermediate(tmp_path: Path) -> None:
    """INV-INPUTS / spec §7.5.

    Plant a stale chunk parquet, a stale shard, a stale
    ``_dataset_aggregated.parquet``, a stale ``analysis_full.parquet`` and a
    stale master, each carrying a row that exists in NO embedded table. The
    new master must equal a concat of the embedded tables exactly.

    Those files are outputs and intermediates of a PREVIOUS finalization, not
    inputs to this one. Under a rolling input, reusing any of them silently
    omits images that arrived since the cache was built, or retains rows for
    an image whose content changed and therefore has a new ``work_id``.
    """
    _publish_successful_images(tmp_path)
    poison = _poison()
    _plant_stale_chunk_parquet(tmp_path, poison)
    _plant_stale_shard(tmp_path, poison)
    _plant_stale_dataset_aggregate(tmp_path, poison)
    _plant_stale_analysis_full(tmp_path, poison)
    _plant_stale_master(tmp_path, poison)

    expected = _concat_of_embedded_tables(tmp_path)

    finalize_run(tmp_path, dataset_names=[DATASET])
    master = _master(tmp_path)

    # STANDING RULE. Both assertions below hold when both frames are EMPTY:
    # the `not in` trivially, and `.equals` because two empty frames are
    # equal. That is the CAN-22 outcome -- an empty master written with no
    # exception -- passing the test written to catch stale inputs.
    assert master.height > 0, (
        "finalize_run wrote an empty master; both asserts are vacuous"
    )
    assert expected.height > 0, (
        "fixture published no embedded tables to compare against"
    )

    assert "GHOST.tif" not in master["Metadata_ImageName"].to_list()
    assert master.equals(expected)


def test_the_legacy_arm_still_prefers_its_dataset_aggregate(
    tmp_path: Path,
) -> None:
    """B7, and the ⚠ RULED "KEEP the arm" decision, pinned as DELIBERATE.

    ⛔ **This test asserts the OPPOSITE of the plan's test body, and the plan
    contradicts itself here.** Task 3 Step 1 spells this test as
    ``assert "GHOST.tif" not in master``, which was written for the *drop the
    arm* option. The later user ruling chose *keep the arm and narrow §7.5 to
    the authorized path*, and says in terms: "The legacy arm remains free to
    prefer ``_dataset_aggregated.parquet``, which is what the second test
    above pins as **deliberate** rather than accidental." Both cannot be
    true: on a legacy tree ``discover_measurement_sources`` prefers the
    aggregate (`_measurement_sources.py:161-166`), so the poison IS the
    source. Pinning the ruling.

    The two preconditions are asserted rather than trusted, because each one
    silently makes this test green for the wrong reason.
    """
    from phenotypic._cli._measurement_sources import (
        _aggregate_needs_image_name_recovery,
    )

    _build_legacy_external_parquet_tree(tmp_path)
    aggregate = _plant_stale_dataset_aggregate(tmp_path, _poison())

    # Precondition 1: the legacy arm is actually reached.
    assert authorized_measurement_sources(tmp_path) is None, (
        "the fixture produced a progress payload; finalize_run takes the "
        "authorized arm and this test pins nothing"
    )
    # Precondition 2: the aggregate survives the recovery predicate. Without
    # `Metadata_ImageName` on the poison frame the predicate returns True,
    # individual Parquets are preferred, and the aggregate is never read.
    assert not _aggregate_needs_image_name_recovery(aggregate), (
        "the poisoned aggregate would be skipped for identity recovery; the "
        "preference this test is about is never exercised"
    )

    finalize_run(tmp_path, dataset_names=[DATASET])
    master = _master(tmp_path)

    assert master.height > 0, "the legacy arm wrote an empty master"
    assert "GHOST.tif" in master["Metadata_ImageName"].to_list(), (
        "the legacy arm stopped preferring _dataset_aggregated.parquet -- "
        "that preference is deliberate until P7 arms the schema gate"
    )


def test_finalize_run_invalidates_the_intermediates_on_success(
    tmp_path: Path,
) -> None:
    """§7.5: so a later invocation cannot mistake them for inputs."""
    _publish_successful_images(tmp_path)
    chunk = _plant_stale_chunk_parquet(tmp_path, _poison())
    shard = _plant_stale_shard(tmp_path, _poison())
    rolling = _plant_stale_analysis_full(tmp_path, _poison())

    # STANDING RULE. `not chunk.exists()` is trivially true if the fixture
    # never created it, which makes the invalidation this test pins
    # unobservable.
    assert chunk.is_file(), "fixture planted nothing; the negatives are vacuous"
    assert shard.is_file()
    assert rolling.is_file()

    finalize_run(tmp_path, dataset_names=[DATASET])

    assert not chunk.exists()
    assert not shard.exists()
    assert not rolling.exists()


def test_the_master_carries_no_user_metadata(tmp_path: Path) -> None:
    """§7.3's contract change, stated as a test.

    The one genuinely dangerous failure mode in §7 is code that filters the
    master on a user-metadata column: it returns EMPTY rather than erroring.
    NO schema stamp guards that (user ruling); the guard is the v1/v2
    discrimination in ``sdk_/_master_io.py``.
    """
    from phenotypic.sdk_ import master_carries_user_metadata

    # Keyed on Metadata_ImageName, with no Metadata_Well in the baseline: this
    # test asks what the master's metadata NAMESPACE contains, so the fixture
    # must not itself put an unowned metadata header there. See
    # `test_master_carries_user_metadata_reads_ownership_not_the_prefix`.
    _publish_successful_images(
        tmp_path, snapshot=_SNAPSHOT_BY_IMAGE, include_well=False
    )
    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )

    master = _master(tmp_path)
    mirror = _mirror(tmp_path)
    assert master.height > 0, "the master is empty; the negative is vacuous"
    assert mirror.height > 0

    assert "Metadata_Strain" not in master.columns
    assert "Metadata_Strain" in mirror.columns
    assert not master_carries_user_metadata(master)
    assert master_carries_user_metadata(mirror)


def test_curation_re_keying_still_works_against_the_intrinsic_master(
    tmp_path: Path,
) -> None:
    """§7.3 names this as needing an explicit test rather than assumption.

    Curation deliberately reads the CLEAN master so labels survive for
    curated-out objects, and keys on dataset / image / object-label -- all
    intrinsic, so it should be unaffected. Test it; do not assume it.
    """
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels
    from phenotypic.sdk_ import BundleLayout

    _publish_successful_images(tmp_path, snapshot=_SNAPSHOT)
    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )

    layout = BundleLayout.detect(tmp_path)
    master = _master(tmp_path)
    assert master.height > 0, "no master rows; there is nothing to re-key"

    labels = CurationLabels.load(layout, master)
    labels.mark("a.tiff", 1, "debris")
    assert labels.labels, "the mark did not land; the survival check is vacuous"

    # Re-load from disk: the re-key runs against the CLEAN master, which after
    # the inversion carries intrinsic identity only.
    reloaded = CurationLabels.load(layout, master)
    assert ("a.tiff", 1) in reloaded.labels, (
        "a curated label did not survive a reload against the un-joined "
        "master -- curation keys on dataset / image / object-label, all of "
        "which are intrinsic and must be unaffected by §7.3"
    )
    assert reloaded.filtered_df(master).height == master.height - 1, (
        "the curated-out object was not filtered out of the master"
    )


def test_master_measurements_csv_is_gone(tmp_path: Path) -> None:
    """D8: master is parquet-only.

    The un-joined master is no longer the file a human opens -- the mirror is.
    """
    _publish_successful_images(tmp_path)
    finalize_run(tmp_path, dataset_names=[DATASET])

    # STANDING RULE: a finalize that wrote nothing at all also leaves no CSV.
    assert master_measurements_parquet_path(tmp_path).is_file(), (
        "finalize_run wrote no master; the CSV's absence proves nothing"
    )
    assert not (
        tmp_path / "deliverables" / "master_measurements.csv"
    ).exists()


def test_finalize_run_writes_no_byte_into_a_proven_store(
    tmp_path: Path,
) -> None:
    """INV-PROVEN, first obligation: no NEW path writes into a promoted store.

    Publish a record, snapshot every mtime under the store, run finalize_run,
    and assert not one file moved. This is the test that would have caught
    the backfill D-A cut, if it had shipped.
    """
    stores = _publish_successful_images(tmp_path, snapshot=_SNAPSHOT)
    store = stores[0]
    before = {
        p: p.stat().st_mtime_ns for p in sorted(store.rglob("*")) if p.is_file()
    }

    # STANDING RULE. This is INV-PROVEN's ONLY gate, and `before == after` is
    # satisfied by `{} == {}` -- which is what a store path that does not
    # exist, or a publish that silently did nothing, produces.
    assert before, "fixture published no store files; before == after is vacuous"
    assert (store / STORE_ROOT_JSON) in before, (
        "the root zarr.json is not in the snapshot"
    )

    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )
    after = {
        p: p.stat().st_mtime_ns for p in sorted(store.rglob("*")) if p.is_file()
    }
    assert before == after, (
        "finalize_run mutated a store that carries a content proof"
    )


# ---------------------------------------------------------------------------
# Step 3 -- the promoted join_metadata path's observable behaviour
# ---------------------------------------------------------------------------


def test_the_mirrors_join_key_dtype_is_pinned(tmp_path: Path) -> None:
    """flow-r3 C1.

    ``join_metadata`` is the LEGACY branch and has not run on a forward tree
    since embedded tables landed. Promoting it changes the join keys' dtype --
    observable by the GUI and by any user script reading the mirror.
    """
    _publish_successful_images(
        tmp_path,
        snapshot="Metadata_Well,Metadata_Strain\n1,WT\n2,WT\n3,MUT\n4,MUT\n",
        well_dtype=pl.Int64,
    )
    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )

    master = _master(tmp_path)
    mirror = _mirror(tmp_path)
    assert master.schema["Metadata_Well"] == pl.Int64, (
        "the fixture did not produce an integer join key; the dtype change "
        "this test pins never happened"
    )
    assert mirror.schema["Metadata_Well"] == pl.String, (
        "join_metadata casts join keys to String; if this changed, the GUI's "
        "filters and every downstream script keyed on the old dtype changed "
        "with it"
    )


def test_a_heterogeneous_master_loses_no_measured_row(tmp_path: Path) -> None:
    """The ragged join. **This test was RED, by design, until the user ruled.**

    A key present in some stores and absent in others, concatenated
    ``diagonal_relaxed``, then joined. `Grid_RowNum` is in both the snapshot
    and image ``a``'s measurements, so column intersection makes it a join
    key; image ``b`` never measured it, so ``diagonal_relaxed`` fills null for
    b's rows, and a null key **anti-matches**. Measured before the fix:
    ``{'a.tiff'} == {'a.tiff', 'b.tiff'}`` -- 100% of b's rows gone from the
    mirror, and b's metadata rows appearing as phantoms instead.

    **Ruled (user, 2026-09-06): leave key selection alone, fix the ragged
    nulls.** `_join_ragged_key_groups` groups the frame by which key columns
    its rows actually carry and joins each group on the keys it has. Nothing
    is fabricated -- a concat cannot invent the value an image never measured
    -- and `Grid_RowNum` remains eligible, and is still used for image ``a``.
    """
    _install_snapshot(
        tmp_path,
        "Metadata_Well,Grid_RowNum,Metadata_Strain\n"
        "A01,0,WT\nA02,1,WT\nB01,0,MUT\nB02,1,MUT\n",
    )
    _publish_store(
        tmp_path, "a", _measurements("a", extra_columns=["Grid_RowNum"])
    )
    _publish_store(tmp_path, "b", _measurements("b"))  # ragged
    _install_state(tmp_path, ["a", "b"])

    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )

    master = _master(tmp_path)
    mirror = _mirror(tmp_path)

    # STANDING RULE, and it is the whole fixture here: if BOTH images carried
    # Grid_RowNum the frame would not be ragged, the ordinary single-join path
    # would run, and this test would pass while exercising nothing.
    assert master.height > 0, "no master rows; the set comparison is vacuous"
    assert "Grid_RowNum" in master.columns, (
        "the fixture produced no ragged column; there is no raggedness to test"
    )
    ragged_rows = master.filter(pl.col("Grid_RowNum").is_null())
    assert ragged_rows.height > 0, (
        "no row has a structurally-absent join key -- the concat was not "
        "ragged and the ragged path was never reached"
    )
    assert set(ragged_rows["Metadata_ImageName"]) == {"b.tiff"}, (
        "the ragged rows are not the image that lacked the column"
    )

    measured = _measured(mirror)
    assert set(measured["Metadata_ImageName"]) == set(
        master["Metadata_ImageName"]
    ), "a measured row was lost between the master and the mirror"

    # The other half of the same defect: b's metadata rows must be JOINED, not
    # turned into phantoms. Asserting only the image set would pass on a
    # mirror where b's rows survived with null metadata.
    b_rows = measured.filter(pl.col("Metadata_ImageName") == "b.tiff")
    assert b_rows.height > 0
    assert b_rows["Metadata_Strain"].null_count() == 0, (
        "the ragged image's rows survived but were not joined to metadata"
    )
    assert _phantoms(mirror).height == 0, (
        "a metadata row that matched a measured object was reported as a "
        "phantom -- phantoms must be computed once, globally, not per group"
    )


def test_the_ragged_path_is_reached_only_by_a_ragged_frame(
    tmp_path: Path,
) -> None:
    """The dispatch predicate, tested directly.

    ``_join_ragged_key_groups`` changes row order and recomputes phantoms, so
    it must not run for the overwhelmingly common non-ragged frame -- the
    guarantee that the ordinary path is untouched rests entirely on this
    predicate returning exactly one pattern there.
    """
    from phenotypic._cli._cli_output_manager import _structural_key_patterns

    uniform = pl.DataFrame(
        {"Metadata_Well": ["A01", "A02"], "Grid_RowNum": [0, 1]}
    )
    assert _structural_key_patterns(
        uniform, ["Metadata_Well", "Grid_RowNum"]
    ) == [(False, False)]

    ragged = pl.DataFrame(
        {"Metadata_Well": ["A01", "B01"], "Grid_RowNum": [0, None]}
    )
    assert len(
        _structural_key_patterns(ragged, ["Metadata_Well", "Grid_RowNum"])
    ) == 2

    # An empty frame has no rows to disagree, and `.unique()` on it returns no
    # patterns at all -- which would read as "ragged" and send an empty frame
    # down a path that expects groups.
    empty = uniform.clear()
    assert _structural_key_patterns(
        empty, ["Metadata_Well", "Grid_RowNum"]
    ) == [(False, False)]


def test_the_mirror_keeps_canonical_column_order_after_the_join(
    tmp_path: Path,
) -> None:
    """``join_metadata`` returns metadata-first; ``order_measurement_columns``
    restores the canonical shape. The call lives inside the function this
    phase rewrites (``_cli_output_manager.py``, just after the join)."""
    _publish_successful_images(tmp_path, snapshot=_SNAPSHOT)
    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )

    cols = _mirror(tmp_path).columns

    # STANDING RULE. `cols == order_measurement_columns(cols)` is a FIXPOINT
    # check, and [] is a fixpoint -- so an absent or empty mirror passes.
    assert cols, "the mirror has no columns; the fixpoint check is vacuous"
    assert "Metadata_Strain" in cols, (
        "the join did not happen; ordering proves nothing"
    )

    assert cols == order_measurement_columns(cols), (
        "the mirror is not canonically ordered -- the "
        "order_measurement_columns call in finalize_post_master_outputs was "
        "dropped in the rewrite"
    )


def test_the_master_inherits_its_column_order_from_the_embedded_tables(
    tmp_path: Path,
) -> None:
    """The master is written BEFORE the ordering call, so it inherits its
    order from its inputs. The inversion removes columns from those inputs;
    this asserts that does not disturb the rest.

    ⛔ **The plan's assertion here was `cols == order_measurement_columns(cols)`,
    and it is false against real code for a reason that predates P4.**
    ``OutputManager.save_image_store`` inserts ``Metadata_Dataset`` with
    ``baseline.insert(len(baseline.columns), ...)`` -- i.e. **appended last**,
    after ``ImagePipeline.measure`` has already applied
    ``order_measurement_columns``. Canonical order puts ``Metadata_Dataset``
    FIRST (``EXPERIMENT``-owned, front block), so no embedded table and
    therefore no master has ever been canonically ordered. Measured: the
    master comes out ``[Metadata_Well, Shape_Area, Metadata_ImageName,
    Object_Label, Metadata_Dataset]`` where canonical is
    ``[Metadata_Dataset, Metadata_Well, Shape_Area, Metadata_ImageName,
    Object_Label]``.

    The plan's own prose states the property it wanted -- *"it inherits its
    order from the embedded tables ... removing user metadata does not
    disturb the survivors' relative order. Assert that rather than assuming
    it"* -- so that is what is asserted. The canonical-order defect is
    reported separately; fixing it means changing where the forward writer
    inserts the column, which changes stored bytes and is not Task 3's.

    The MIRROR is canonically ordered, and that is pinned by
    ``test_the_mirror_keeps_canonical_column_order_after_the_join``.
    """
    from phenotypic.schema import EXPERIMENT, IMAGE

    stores = _publish_successful_images(tmp_path, snapshot=_SNAPSHOT)
    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )

    embedded = pl.read_parquet(stores[0] / MEASUREMENT_TABLE_RELATIVE_PATH)
    master = _master(tmp_path)
    assert embedded.columns, "the embedded table has no columns; vacuous"

    assert master.columns == embedded.columns, (
        "the master did not inherit its inputs' column order -- aggregation "
        "reordered the frame, so the master is no longer the exact "
        "concatenation of the authorized embedded tables"
    )
    assert "Metadata_Strain" not in embedded.columns, (
        "the embedded table was never inverted; there is no removal to check"
    )
    assert str(EXPERIMENT.DATASET) in master.columns
    assert str(IMAGE.IMAGE_NAME) in master.columns, (
        "intrinsic image identity left the master"
    )


def test_master_carries_user_metadata_reads_ownership_not_the_prefix(
    tmp_path: Path,
) -> None:
    """The v1/v2 discrimination, and the one thing it cannot see.

    A v2 master DOES carry ``Metadata_*`` columns -- ``Metadata_Dataset`` and
    the ``IMAGE``-owned provenance block -- so the plan's stated rule ("a v1
    master carries ``Metadata_*`` columns, a v2 does not") would classify
    every v2 master as v1. Ownership is the workable test, and
    ``Metadata_Strain`` is itself a schema member (``GENETIC.STRAIN``), so
    "is it in the schema" does not separate them either.

    **The known limit, pinned rather than left to be rediscovered:** column
    provenance is not recoverable from a column name. A master carrying a
    non-``IMAGE`` metadata header that a custom operation produced reads as
    v1 even though no CSV was ever joined into it. The forward pipeline does
    not emit such a column -- its master metadata is ``IMAGE``-owned plus
    ``Metadata_Dataset`` -- which is what keeps this a limit rather than a
    defect, and it is why the sibling test's fixture drops
    ``Metadata_Well``.
    """
    from phenotypic.sdk_ import (
        master_carries_user_metadata,
        user_metadata_headers,
    )

    intrinsic = pl.DataFrame(
        {
            "Metadata_Dataset": ["plate"],
            "Metadata_ImageName": ["a.tiff"],
            "Shape_Area": [4.0],
            "Object_Label": [1],
        }
    )
    assert not master_carries_user_metadata(intrinsic), (
        "a v2 master carrying only intrinsic identity was read as v1 -- the "
        "discrimination is testing the Metadata_ prefix, not ownership"
    )
    assert user_metadata_headers(intrinsic.columns) == ()

    joined = intrinsic.with_columns(pl.lit("WT").alias("Metadata_Strain"))
    assert master_carries_user_metadata(joined)
    assert user_metadata_headers(joined.columns) == ("Metadata_Strain",)

    # The limit. `Metadata_Well` is not a schema member at all, so it is an
    # unowned metadata header; nothing in the name says the pipeline put it
    # there rather than a metadata.csv join.
    unowned = intrinsic.with_columns(pl.lit("A01").alias("Metadata_Well"))
    assert master_carries_user_metadata(unowned), (
        "if this ever returns False the helper started guessing provenance "
        "from the column name, which it cannot do"
    )


# ---------------------------------------------------------------------------
# Step 3b -- CAN-1 and CAN-2
# ---------------------------------------------------------------------------


def test_an_authorized_metadata_run_does_not_lose_the_join(
    tmp_path: Path,
) -> None:
    """The STEP ZERO probe, ported with its preconditions intact.

    **This is the regression test for a defect that was live in the tree**
    between the inversion (Task 2) and this task, and it is the only thing
    here that aggregates a master from inverted stores *and* asserts the
    stores were inverted before drawing any conclusion from an absence.

    The defect, measured rather than reasoned: ``finalize_post_master_outputs``
    discriminated on ``metadata_join_keys is None``. After the inversion every
    store's **measurements** table correctly records ``not_requested`` / ``[]``
    / ``""`` -- the join is not a property of that file any more -- so the
    retired ``_consistent_embedded_join_keys`` returned the **empty tuple**.
    ``()`` is not ``None``, so finalization took the append-phantoms branch,
    hit ``if not join_keys: return measured``, and produced a master **and** a
    mirror with no metadata column and no phantom row, raising nothing. An
    authorized ``--metadata`` run silently discarded the entire join.

    The fix is not to special-case ``()``, and not to re-point the reader at
    the metadata table's Parquet KV: it is that **the finalizer needs no
    recorded join keys at all**. ``join_metadata`` derives its own common
    columns from the master and the snapshot, so the discriminator and both
    of its branches are gone (CAN-2). A per-store reader would have had to be
    reconciled across snapshot generations -- exactly the mixed state D-A
    deliberately manufactures and the retired guard aborted on.

    Every precondition below is asserted, because each one on its own makes
    the finding unfalsifiable: an empty authorized set, or a tree that never
    inverted, produces the same absences for a completely different reason.
    """
    stores = _publish_successful_images(
        tmp_path, snapshot=_SNAPSHOT_WITH_PHANTOM
    )

    # Precondition 1 -- there is authority, and it covers every image.
    sources = authorized_measurement_sources(tmp_path)
    assert sources is not None, "no marker authority; the authorized arm is unreached"
    assert len(sources) == len(stores), (
        f"{len(sources)} authorized source(s) for {len(stores)} published "
        "images; the aggregation below is not over this fixture"
    )

    # Precondition 2 -- the stores really are inverted. Without this the
    # assertions below pass on a pre-inversion tree, where the join was
    # already in the embedded table and finalization had nothing to do.
    embedded = pl.read_parquet(stores[0] / MEASUREMENT_TABLE_RELATIVE_PATH)
    assert "Metadata_Strain" not in embedded.columns, (
        "the embedded measurement table still carries user metadata; this "
        "tree was never inverted and proves nothing about finalization"
    )

    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )

    master = _master(tmp_path)
    mirror = _mirror(tmp_path)
    assert master.height > 0 and mirror.height > 0, (
        "finalization produced nothing; every assertion below is vacuous"
    )

    assert "Metadata_Strain" not in master.columns, (
        "the master is the un-joined archival set (§7.3)"
    )
    measured = _measured(mirror)
    assert measured.height > 0, "no measured rows survived into the mirror"
    assert "Metadata_Strain" in mirror.columns, (
        "the mirror carries no metadata column at all -- the finalizer took "
        "a branch that joined nothing, which is the defect this test exists "
        "to catch"
    )
    assert measured["Metadata_Strain"].null_count() == 0, (
        "the metadata column exists but every measured row is null -- the "
        "phantoms were appended and no measured row was joined"
    )
    assert _phantoms(mirror).height == 1, (
        "the metadata-only identity was dropped; the join kept measured rows "
        "but lost the half that reports a strain nobody detected"
    )


def test_the_mirror_carries_both_joined_rows_and_phantoms(
    tmp_path: Path,
) -> None:
    """CAN-1.

    Neither pre-P4 branch does both halves: one joins and drops every phantom,
    the other appends phantoms and joins nothing. Assert them in ONE frame,
    because each half passes a test that only looks at the other.
    """
    _publish_successful_images(tmp_path, snapshot=_SNAPSHOT_WITH_PHANTOM)
    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )

    mirror = _mirror(tmp_path)
    measured = _measured(mirror)
    phantoms = _phantoms(mirror)

    assert measured.height > 0 and phantoms.height == 1
    assert measured["Metadata_Strain"].null_count() == 0, (
        "measured rows were not joined"
    )
    assert "Z99" in phantoms["Metadata_Well"].to_list(), "phantoms were dropped"


def test_a_measured_row_absent_from_metadata_is_dropped_deliberately(
    tmp_path: Path,
) -> None:
    """The asymmetry is by design (user ruling, round 2), so PIN it rather
    than leaving it as an accident of which frame is on the left.

    ``metadata.csv`` describes the experiment. A measured object whose key
    appears in no metadata row is an object outside that description, and
    ``join_metadata``'s docstring states the intent: it keeps
    metadata-unmatched rows -- "a strain that failed to grow, or that
    detection missed, is exactly what the user needs to see" -- and drops
    measurement-unmatched ones.

    This test exists because an earlier draft proposed reversing the
    orientation. Without it, a future reader sees only "left join" and cannot
    tell which way round was intended.
    """
    _install_snapshot(tmp_path, _SNAPSHOT)
    _publish_store(tmp_path, "a", _measurements("a"))
    _publish_store(
        tmp_path, "b", _measurements("b", extra_objects=[(7, "ZZZ")])
    )
    _install_state(tmp_path, ["a", "b"])

    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )

    mirror = _mirror(tmp_path)

    # STANDING RULE. `orphan.height == 0` is satisfied by an EMPTY mirror, and
    # by a fixture that never added the orphan object. Establish both before
    # reading the zero as evidence of a deliberate drop.
    assert mirror.height > 0, "the mirror is empty; orphan.height == 0 is vacuous"
    master = _master(tmp_path)
    assert (
        master.filter(
            (pl.col("Metadata_ImageName") == "b.tiff")
            & (pl.col("Object_Label") == 7)
        ).height
        == 1
    ), "the fixture never created the orphan; there is nothing to drop"

    orphan = mirror.filter(
        (pl.col("Metadata_ImageName") == "b.tiff")
        & (pl.col("Object_Label") == 7)
    )
    assert orphan.height == 0, (
        "an object outside the described experiment reached the mirror; the "
        "join orientation was reversed"
    )


def test_the_master_keeps_the_object_the_mirror_drops(tmp_path: Path) -> None:
    """Where the dropped object DOES survive, and why that is the right split.

    §7.3: the master is the un-joined archival set -- intrinsic identity,
    every authorized measured row. The mirror is the post-applied,
    metadata-joined display frame.
    """
    _install_snapshot(tmp_path, _SNAPSHOT)
    _publish_store(tmp_path, "a", _measurements("a"))
    _publish_store(
        tmp_path, "b", _measurements("b", extra_objects=[(7, "ZZZ")])
    )
    _install_state(tmp_path, ["a", "b"])

    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )

    master = _master(tmp_path)
    kept = master.filter(
        (pl.col("Metadata_ImageName") == "b.tiff")
        & (pl.col("Object_Label") == 7)
    )
    assert kept.height == 1, (
        "the master must retain every authorized measured row"
    )


def test_metadata_added_after_the_stores_still_joins_every_measured_row(
    tmp_path: Path,
) -> None:
    """CAN-2, with DF-2's assertion verbatim -- and the hazard STEP ZERO
    measured.

    The stores are built with NO ``--metadata``, so every one of them records
    ``join_status="not_requested"`` and ``join_keys=[]``. The retired
    ``_consistent_embedded_join_keys`` returned ``()`` for that -- which is
    **not** ``None`` -- so the pre-P4 finalizer took the append-phantoms
    branch, hit ``if not join_keys: return measured``, and joined nothing at
    all while raising nothing.

    The ``measured.height > 0`` guard matters: without it the assertion is
    vacuously true on an all-phantom frame.
    """
    _publish_successful_images(tmp_path, snapshot=None)
    snapshot = _install_snapshot(tmp_path, _SNAPSHOT)

    finalize_run(
        tmp_path, dataset_names=[DATASET], metadata_csv=snapshot
    )

    measured = _measured(_mirror(tmp_path))
    assert measured.height > 0, "fixture produced no measured rows to check"
    assert "Metadata_Strain" in measured.columns
    assert measured["Metadata_Strain"].null_count() == 0, (
        "user metadata reached the mirror only as metadata-only phantoms; "
        "every measured row is null. The join keys were () rather than None, "
        "so finalize took the append-phantoms branch and joined nothing."
    )


def test_stores_with_mixed_metadata_snapshots_do_not_abort_finalization(
    tmp_path: Path,
) -> None:
    """CAN-2.

    D-A manufactures this state on the normal rolling-input path: stores keep
    the snapshot they were built against, so any run that gains images after
    a ``metadata.csv`` edit has two generations on disk. The retired guard
    raised on exactly that, in the finalizer, on the normal path -- while
    D-A's contract says divergence is an advisory, and an advisory is never a
    gate.
    """
    stores = _publish_successful_images(tmp_path, snapshot=_SNAPSHOT)
    _install_snapshot(tmp_path, _EDITED_SNAPSHOT)
    stores.append(_publish_store(tmp_path, "c", _measurements("c")))
    _install_state(tmp_path, ["a", "b", "c"])

    # STANDING RULE. "Must not raise" is satisfied by a fixture that produced
    # ONE snapshot generation instead of two -- in which case nothing was
    # mixed and the guard this test exists to retire was never reachable.
    digests = {_store_snapshot_sha256(store) for store in stores}
    assert len(digests) == 2, (
        f"fixture produced {len(digests)} snapshot generation(s), not 2; "
        "the mixed-digest state was never created and this test is vacuous"
    )

    finalize_run(
        tmp_path,
        dataset_names=[DATASET],
        metadata_csv=_snapshot_path(tmp_path),
    )
    # must not raise; divergence is an advisory, per D-A
    assert _master(tmp_path).height > 0, (
        "finalization survived the mixed state but produced nothing"
    )


# ---------------------------------------------------------------------------
# Step 6b -- the designated falsifier for the "no schema stamp" ruling
# ---------------------------------------------------------------------------

#: The readers P4 owns. The six GUI readers are P6's, by the same ruling that
#: created `master_carries_user_metadata` -- extending this collector is P6's
#: step, and the count below is what makes that extension visible rather than
#: silent.
_EXPECTED_READER_COUNT = 4


def _reader_outcomes(master_path: Path) -> dict[str, object]:
    """Run each P4-owned master reader and record WHAT IT PRODUCED.

    Outcomes, never column sets: the column sets are equal by construction,
    which is the whole premise being falsified. Exceptions are recorded as
    outcomes rather than swallowed -- a collector that returns ``{}`` on
    failure would make the equality below hold for two broken runs.
    """
    from phenotypic.sdk_ import (
        master_carries_user_metadata,
        user_metadata_headers,
    )

    outcomes: dict[str, object] = {}
    frame = pl.read_parquet(master_path)
    outcomes["carries_user_metadata"] = master_carries_user_metadata(frame)
    outcomes["user_metadata_headers"] = user_metadata_headers(frame.columns)
    outcomes["canonical_order"] = tuple(
        order_measurement_columns(frame.columns)
    )
    try:
        from phenotypic.gui.results_viewer._metadata import (
            normalize_viewer_frame,
        )

        normalized = normalize_viewer_frame(frame)
        outcomes["viewer_frame"] = (normalized.height, tuple(normalized.columns))
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        outcomes["viewer_frame"] = f"{type(exc).__name__}: {exc}"
    return outcomes


def _build_v1_tree(
    output_dir: Path,
    *,
    snapshot: str | None,
    measurements: dict[str, object] | None = None,
) -> Path | None:
    """Build a genuine PRE-INVERSION tree, without finalizing it.

    **The v1 arm cannot be a second call to the forward writer.** That is two
    runs of one function, and an equality between them holds by construction
    -- which is what the falsifier below used to do, and what made it unable
    to fail.

    It is built through the producer the inversion replaced instead.
    ``prepare_embedded_measurement_table`` still ships (``06809fbc`` retains
    it for the consumers that read and rewrite pre-inversion stores), and
    ``recompile_embedded_measurement_tables`` is the caller that drives it
    over a whole tree: it projects each store's recorded baseline, re-joins
    the snapshot into ``table.parquet``, and writes no
    ``pht-metadata.parquet``. That is the v1 store shape -- the shape
    ``--mode migrate`` leaves in every store it embeds a legacy Parquet into
    -- so the master aggregated from those stores is a v1 master.

    The stores are written WITHOUT a snapshot and the snapshot installed
    afterwards, because a store built with ``--metadata`` is inverted and
    ``_refuse_inverted_stores_before_any_write`` refuses the whole run. Absent
    the snapshot at write time the tree is un-inverted, which is exactly the
    pre-inversion shape this arm needs.

    Args:
        output_dir: Directory to build the run in.
        snapshot: The metadata CSV text to re-join, or ``None`` for the
            metadata-free arm.
        measurements: Per-stem baselines. ``None`` is the two-image,
            ``Metadata_Well``-free baseline the falsifier needs.

    Returns:
        The installed snapshot, or ``None`` for the metadata-free arm.
    """
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_tables,
    )

    frames = measurements or {
        stem: _measurements(stem, include_well=False) for stem in ("a", "b")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _install_snapshot(output_dir, None)
    for stem, frame in frames.items():
        _publish_store(output_dir, stem, frame)  # type: ignore[arg-type]
    _install_state(output_dir, list(frames))
    metadata_csv = (
        _install_snapshot(output_dir, snapshot) if snapshot is not None else None
    )
    rewritten = recompile_embedded_measurement_tables(output_dir, metadata_csv)
    assert rewritten == len(frames), (
        f"the pre-inversion producer rewrote {rewritten} of {len(frames)} "
        "stores; this arm is not a v1 tree and every comparison against it is "
        "about something else"
    )
    return metadata_csv


def _publish_v1_run(output_dir: Path, *, snapshot: str | None) -> None:
    """Build a genuine pre-inversion tree (:func:`_build_v1_tree`) and finalize it.

    **Only the metadata-free arm may be built this way.** ``finalize_run``
    projects every embedded table onto its descriptor (P7 Task 4). On a
    ``not_requested`` table that projection is the identity -- the table
    carries exactly its ``measurement_columns`` and no join fan-out -- so
    finalizing still writes the bytes a pre-projection build wrote for such a
    tree. A JOINED tree is different: finalizing it now yields a v2-shaped
    master, so a joined v1 master must be written as the pre-projection
    artifact instead -- see :func:`_write_pre_projection_v1_master`.
    """
    _build_v1_tree(output_dir, snapshot=snapshot)
    assert finalize_run(output_dir, dataset_names=[DATASET]) is not None


def _write_pre_projection_v1_master(output_dir: Path, *, snapshot: str) -> None:
    """Build a joined v1 tree and write the master a pre-projection build left.

    ``finalize_run`` projects each table onto its own descriptor (P7 Task 4),
    so **no current code path writes a joined v1 master** any more. Such
    masters still exist in the wild -- every finalization before this change,
    including every earlier ``--mode migrate``, wrote one -- and the
    falsifier's positive control needs a real one, so it is written as that
    artifact: the pre-projection aggregation of the joined embedded tables
    (:func:`_concat_of_embedded_tables`, i.e. the unchanged
    ``aggregate_parquet_files`` plus identity recovery, exactly what
    ``build_master_frame`` did before), written to the master path.
    """
    from phenotypic.sdk_ import PARQUET_WRITE_OPTIONS

    _build_v1_tree(output_dir, snapshot=snapshot)
    master = master_measurements_parquet_path(output_dir)
    master.parent.mkdir(parents=True, exist_ok=True)
    _concat_of_embedded_tables(output_dir).write_parquet(
        master, **PARQUET_WRITE_OPTIONS
    )


def _embedded_table(output_dir: Path, stem: str) -> pl.DataFrame:
    return pl.read_parquet(
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )


def _join_status(output_dir: Path, stem: str) -> str:
    """The ``phenotypic.join.status`` a store's table Parquet records."""
    import pyarrow.parquet as pq

    from phenotypic.sdk_.ngff_ import (
        EMBEDDED_MEASUREMENT_PARQUET_METADATA_KEYS as keys,
    )

    metadata = pq.read_schema(
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    ).metadata or {}
    return metadata[keys.JOIN_STATUS.encode()].decode()


def test_a_v1_metadata_free_master_is_indistinguishable_from_v2_and_that_is_harmless(
    tmp_path: Path,
) -> None:
    """The "no schema stamp" ruling's own falsifier (user, 2026-09-06).

    A v1 run with no ``metadata.csv`` also has no user-metadata columns, so
    column presence conflates it with v2. That is *expected* to be harmless --
    neither has anything to join -- but it is an inference, and this change
    has been punished for exactly that twice.

    If v1-no-metadata and v2 turn out BEHAVIOURALLY DISTINGUISHABLE -- some
    reader does something different, and something wrong, on one of them --
    the ruling flips: mint the stamp and register it as tracked state
    properly. **That raises the state-artifact count, which is a HARD STOP:
    stop and report, do not decide.**

    ⛔ **THE ARMS MUST COME FROM DIFFERENT PRODUCERS, or this test cannot
    fail.** An earlier version built both with ``_publish_successful_images``
    + ``finalize_run`` -- same code, same inputs -- so ``outcomes_v1 ==
    outcomes_v2`` was true by construction and could only break on
    nondeterminism. Inverting ``sdk_/_master_io.py``'s predicate changed both
    dicts identically and left it green. The v1 arm now runs the retained
    **pre-inversion** producer (see :func:`_publish_v1_run`), so the equality
    is a claim about two producers agreeing on the metadata-free path rather
    than about one function being deterministic.

    **``v1_joined`` is the positive control**, and it is what makes the
    equality mean something: a v1 run that WAS given a snapshot must come out
    distinguishable. Without it, a ``_reader_outcomes`` that discriminated
    nothing at all would satisfy the equality just as well.
    """
    from phenotypic.sdk_ import master_carries_user_metadata

    # v2: the real post-inversion path, no --metadata. `include_well=False`
    # matches `test_the_master_carries_no_user_metadata`'s fixture: with
    # `Metadata_Well` in the baseline BOTH arms would carry an unowned
    # metadata header and `master_carries_user_metadata` would be True for
    # every arm, collapsing the control below.
    v2 = tmp_path / "v2"
    v2.mkdir()
    _publish_successful_images(v2, snapshot=None, include_well=False)
    assert finalize_run(v2, dataset_names=[DATASET]) is not None

    v1 = tmp_path / "v1"
    _publish_v1_run(v1, snapshot=None)

    v1_joined = tmp_path / "v1-joined"
    # Written as the pre-projection artifact, not finalized: `finalize_run`
    # now projects a joined table, so finalizing this arm would yield a v2
    # master and collapse the positive control below.
    _write_pre_projection_v1_master(v1_joined, snapshot=_SNAPSHOT_BY_IMAGE)

    # STANDING RULE, the control half. Establish that the pre-inversion
    # producer really did join user metadata into the master when given a
    # snapshot, and that the v2 arm really did not -- otherwise "the two are
    # indistinguishable" is a statement about two empty differences.
    joined_master = _master(v1_joined)
    assert joined_master.height > 0, "the joined v1 arm produced no master"
    assert "Metadata_Strain" in joined_master.columns, (
        "the pre-inversion producer did not join the snapshot into the "
        "embedded tables; the v1_joined arm is not a joined v1 master"
    )
    assert master_carries_user_metadata(joined_master) is True
    assert master_carries_user_metadata(_master(v2)) is False

    outcomes_v1 = _reader_outcomes(master_measurements_parquet_path(v1))
    outcomes_v2 = _reader_outcomes(master_measurements_parquet_path(v2))
    outcomes_joined = _reader_outcomes(
        master_measurements_parquet_path(v1_joined)
    )

    # STANDING RULE, and the highest stakes in the file: this test is the
    # designated falsifier for a HARD-STOP ruling. An outcome collector that
    # swallowed exceptions would return {} for both, the equality would hold,
    # and a false green here silently confirms "no stamp needed" -- the exact
    # question the ruling said to settle by test rather than by reasoning.
    assert outcomes_v1, "no reader outcomes collected for v1; equality vacuous"
    assert set(outcomes_v1) == set(outcomes_v2), (
        "the two runs exercised different readers"
    )
    assert len(outcomes_v1) == _EXPECTED_READER_COUNT, (
        "a reader was added or dropped without updating this falsifier"
    )
    assert outcomes_joined != outcomes_v2, (
        "a JOINED v1 master read the same as a v2 master through every "
        "reader -- `_reader_outcomes` discriminates nothing, so the "
        "equality below proves nothing either"
    )

    # The two claims, in the order they can fail. Byte equality is the
    # stronger one and is a statement about the two PRODUCERS; the outcome
    # equality is the ruling's own wording.
    assert _master_bytes(v1) == _master_bytes(v2), (
        "the pre-inversion and post-inversion producers disagree on a "
        "metadata-free run -- v1 and v2 masters are not interchangeable and "
        "the no-stamp ruling has to be revisited"
    )
    assert outcomes_v1 == outcomes_v2


# ---------------------------------------------------------------------------
# P7 Task 4 -- a legacy joined embedded table is projected at read
# ---------------------------------------------------------------------------
#
# Migrate leaves every store's embedded table exactly as it found it -- joined,
# for a pre-inversion tree -- and `finalize_run` projects each table onto the
# store's own recorded `measurement_columns` before concatenating. That makes
# the master v2-shaped, so the single global join in
# `finalize_post_master_outputs` runs on intrinsic keys only.

#: The field regression's shape (a 6,657-image migration): keyed on image
#: identity, integer-valued user columns, and ``b.tiff`` ABSENT, so store
#: ``b``'s joined table carries every user column as null.
_SNAPSHOT_INT_USER_COLUMNS_WITHOUT_B = (
    "Metadata_ImageName,Metadata_pH,Metadata_Replicate\na.tiff,7,1\n"
)

#: Three metadata rows per image key. A legacy joined table therefore holds
#: every measured row three times (CAN-10(a)).
_SNAPSHOT_FANOUT = (
    "Metadata_ImageName,Metadata_Replicate\n"
    "a.tiff,1\na.tiff,2\na.tiff,3\nb.tiff,1\nb.tiff,2\nb.tiff,3\n"
)


def _strip_measurement_descriptor(output_dir: Path, stem: str) -> Path:
    """Remove one store's ``tables.measurements`` descriptor, then re-certify.

    Step 0b's shape: ``table.parquet`` on disk, no descriptor declaring it.
    The record is republished because a store is fingerprinted by its root
    ``zarr.json`` -- without that the image silently stops being authorized,
    and the skip under test is never reached.
    """
    from phenotypic.sdk_ import atomic_write_json
    from phenotypic.sdk_.ngff_ import MEASUREMENT_TABLE_GROUP, PhenotypicAttr

    store = zarr_store_path(output_dir, DATASET, stem)
    root_path = store / STORE_ROOT_JSON
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["attributes"][PhenotypicAttr.ROOT][PhenotypicAttr.TABLES].pop(
        MEASUREMENT_TABLE_GROUP
    )
    atomic_write_json(root_path, root)
    publish_image_success(
        output_dir,
        work_id=f"work-{stem}",
        dataset=DATASET,
        relative_image_path=f"{stem}.tiff",
        image_stem=stem,
        mode="full",
        attempt_id="attempt-1",
        lifecycle_epoch="epoch-1",
        artifacts={
            "measurements": store / MEASUREMENT_TABLE_RELATIVE_PATH,
            "store": store,
        },
    )
    return store


def _warnings_naming(
    caplog: pytest.LogCaptureFixture, needle: str | Path
) -> list[str]:
    """WARNING+ messages naming *needle*, by its given or resolved spelling."""
    import logging

    spellings = {str(needle)}
    if isinstance(needle, Path):
        spellings.add(str(needle.resolve()))
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and any(spelling in record.getMessage() for spelling in spellings)
    ]


def test_a_legacy_joined_master_carries_no_user_metadata(tmp_path: Path) -> None:
    """P7 Task 4: finalization projects each table onto its own descriptor.

    Classified by OWNERSHIP (``master_carries_user_metadata``), never by the
    ``Metadata_`` prefix: the projected master still carries
    ``Metadata_ImageName``, and must.
    """
    from phenotypic.sdk_ import master_carries_user_metadata

    snapshot = _build_v1_tree(tmp_path, snapshot=_SNAPSHOT_BY_IMAGE)

    # STANDING RULE, the positive control. Without a JOINED table in the store
    # there is no user metadata to project away, and the negative below holds
    # on an aggregator that projects nothing.
    assert _join_status(tmp_path, "a") == "joined"
    assert master_carries_user_metadata(_embedded_table(tmp_path, "a")) is True, (
        "the fixture's embedded table carries no user metadata; this is not a "
        "legacy joined store"
    )

    finalize_run(tmp_path, dataset_names=[DATASET], metadata_csv=snapshot)

    master = _master(tmp_path)
    assert master.height > 0, "the master is empty; the negative is vacuous"
    assert master_carries_user_metadata(master) is False, (
        "a joined embedded table reached the master unprojected -- the master "
        "is v1-shaped and the global join will run on user columns again"
    )
    assert "Metadata_ImageName" in master.columns, (
        "intrinsic identity was projected away along with the user metadata"
    )
    assert master_carries_user_metadata(_mirror(tmp_path)) is True, (
        "the join did not happen at finalization"
    )


def test_a_legacy_joined_tree_keeps_its_measured_rows_when_an_image_has_no_metadata(
    tmp_path: Path,
) -> None:
    """The field failure, at unit scale.

    A real 6,657-image ``--mode migrate`` produced a correct master and a
    mirror with **0 measured rows** -- every row ``QC_MetadataOnly=True`` --
    plus duplicate ``*_right`` columns. Mechanism: each legacy table is
    already joined; an image absent from ``metadata.csv`` carries the user
    columns as null, so the concat widened integer columns (``7`` -> ``7.0``);
    the global join then re-joined on those user columns, where ``"7.0"``
    never equals the CSV's ``"7"``; and the ragged path, grouping by null
    pattern, matched nothing and suffixed the colliding columns ``_right``.
    """
    from phenotypic.sdk_ import user_metadata_headers

    snapshot = _build_v1_tree(
        tmp_path, snapshot=_SNAPSHOT_INT_USER_COLUMNS_WITHOUT_B
    )

    # Preconditions -- the defect's two ingredients, asserted rather than
    # trusted. Without either, the mirror below is correct on unprojected code.
    table_a = _embedded_table(tmp_path, "a")
    table_b = _embedded_table(tmp_path, "b")
    user_a = user_metadata_headers(table_a.columns)
    user_b = user_metadata_headers(table_b.columns)
    assert user_a and user_b, "the stores carry no joined user metadata"
    assert all(table_a[c].null_count() == 0 for c in user_a), (
        "image a's user metadata is not populated"
    )
    assert all(table_b[c].null_count() == table_b.height for c in user_b), (
        "image b's user metadata is not all-null; the image-absent-from-"
        "metadata ingredient is missing"
    )
    # The third ingredient: the two tables DISAGREE on the user columns'
    # dtype (a's integers, b's all-null columns stored as float), which is what
    # renders 7 as "7.0" after the concat. Without it this test reduces to
    # "an image is absent from metadata" alone.
    assert set(user_a) == set(user_b)
    assert all(table_a.schema[c] != table_b.schema[c] for c in user_a), (
        "a's and b's user columns share a dtype; the concat-widening "
        "ingredient of the field failure is missing"
    )

    finalize_run(tmp_path, dataset_names=[DATASET], metadata_csv=snapshot)

    master = _master(tmp_path)
    mirror = _mirror(tmp_path)
    assert master.height > 0 and mirror.height > 0, (
        "finalization produced nothing; every assertion below is vacuous"
    )
    assert set(master["Metadata_ImageName"]) == {"a.tiff", "b.tiff"}, (
        "the master must keep every authorized measured row"
    )

    measured = _measured(mirror)
    assert measured.height > 0, (
        "every measured row was dropped from the mirror -- the field failure"
    )
    # b is outside the described experiment, so the mirror drops it by design
    # (test_a_measured_row_absent_from_metadata_is_dropped_deliberately).
    assert set(measured["Metadata_ImageName"]) == {"a.tiff"}
    assert measured.height == (
        master.filter(pl.col("Metadata_ImageName") == "a.tiff").height
    ), "some of image a's measured rows were lost from the mirror"
    assert measured["QC_MetadataOnly"].to_list() == [False] * measured.height

    joined_user = user_metadata_headers(measured.columns)
    assert joined_user, "the mirror carries no user metadata column"
    for column in joined_user:
        assert measured[column].null_count() == 0, (
            f"{column} is null on measured rows -- the join matched nothing"
        )
    suffixed = [column for column in mirror.columns if column.endswith("_right")]
    assert not suffixed, f"the join collided on user columns: {suffixed}"
    assert _phantoms(mirror).height == 0, (
        "a.tiff's metadata row was reported as a phantom although it matched"
    )


def test_a_legacy_fanout_store_and_a_fresh_store_produce_equal_masters(
    tmp_path: Path,
) -> None:
    """CAN-10(a). Projection fixes the column set, not the row count.

    A legacy table joined against *k* metadata rows per key holds each
    measured row *k* times; the global join then fans it out again, so the
    mirror would carry *k*² rows. Compare ROW COUNTS and full equality
    against a fresh tree, which the column-membership test above cannot see.
    """
    legacy = tmp_path / "legacy"
    _build_v1_tree(legacy, snapshot=_SNAPSHOT_FANOUT)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    _publish_successful_images(
        fresh, snapshot=_SNAPSHOT_FANOUT, include_well=False
    )

    # STANDING RULE. The equality below is satisfied by a legacy tree that
    # never fanned out, and by a "fresh" tree that was never inverted.
    fresh_rows = _embedded_table(fresh, "a").height
    assert fresh_rows > 0
    assert _join_status(legacy, "a") == "joined"
    assert _embedded_table(legacy, "a").height == 3 * fresh_rows, (
        "the legacy table does not carry the 3x fan-out this test is about"
    )
    assert "Metadata_Replicate" not in _embedded_table(fresh, "a").columns, (
        "the fresh arm is not inverted; it is a second legacy tree"
    )

    for root in (legacy, fresh):
        finalize_run(
            root, dataset_names=[DATASET], metadata_csv=_snapshot_path(root)
        )

    for read in (_master, _mirror):
        got, want = read(legacy), read(fresh)
        assert want.height > 0, f"{read.__name__}: the fresh arm is empty"
        assert got.height == want.height, (
            f"{read.__name__}: legacy {got.height} rows != fresh {want.height}"
        )
        assert got.equals(want), f"{read.__name__}: legacy != fresh"


def test_the_fanout_shards_project_legacy_tables_to_the_same_master(
    tmp_path: Path,
) -> None:
    """The P5 shards are a second read path into the master.

    A projection applied only where ``build_master_frame`` reads tables
    directly would leave ``--njobs N`` aggregating unprojected, fanned-out
    shards -- a different master for the same tree, certified by the same
    proof.
    """
    from phenotypic._cli._cli_output_manager import aggregate_measurements
    from phenotypic.sdk_ import aggregation_shard_dir, master_carries_user_metadata

    def _aggregate(root: Path, njobs: int) -> Path:
        _build_v1_tree(root, snapshot=_SNAPSHOT_FANOUT)
        aggregate_measurements(
            output_dir=root,
            dataset_names=[DATASET],
            include_dataset_column=True,
            metadata_csv=_snapshot_path(root),
            njobs=njobs,
        )
        return root

    direct = _aggregate(tmp_path / "direct", 1)
    fanned = _aggregate(tmp_path / "fanned", 2)

    # STANDING RULE: if the allocation clamped K to 1, both arms took the same
    # single-shard path and the byte equality compares a run against itself.
    shards = sorted(aggregation_shard_dir(fanned, None).glob("shard_*.parquet"))
    assert len(shards) == 2, f"the fan-out ran {len(shards)} shard(s), not 2"

    for root in (direct, fanned):
        master = _master(root)
        assert master.height == 4, (
            f"{root.name}: {master.height} master rows for 2 images x 2 objects"
        )
        assert master_carries_user_metadata(master) is False, (
            f"{root.name}: the master carries user metadata"
        )
    assert _master_bytes(direct) == _master_bytes(fanned)


@pytest.mark.parametrize("njobs", [1, 2])
def test_a_string_drifted_join_key_widens_the_master_and_keeps_the_mirrors_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    njobs: int,
) -> None:
    """CAN-10(b), as ruled R4 (2026-09-10): pinned and advised, not repaired.

    ``_restore_join_key_dtypes`` logs and keeps the string-safe type when it
    cannot restore a join key, so a legacy store can carry ``Metadata_Well``
    as ``String`` beside stores carrying ``Int64``. The plan predicted that
    widening at concat would null those rows' metadata after the global join.
    **Measured false**: ``join_metadata`` casts every key to ``String`` on
    both sides and the drifted values render identically. So the widening is
    advised -- a warning naming the column and its dtypes -- rather than
    repaired.

    **Both finalization paths, because that advisory is the whole of what R4
    kept.** ``njobs=1`` concatenates every store at once. ``njobs=2`` puts
    ``a`` and ``b`` in different shards, each internally uniform, so only the
    merge sees the disagreement -- and the forward CLI's default ``--njobs
    -1`` takes that path.

    If the metadata assertion ever fails, the harm the plan predicted became
    real and the R4 ruling must be revisited.
    """
    import logging

    from phenotypic._cli import _embedded_measurement_tables as producer
    from phenotypic._cli._cli_output_manager import aggregate_measurements
    from phenotypic.sdk_ import aggregation_shard_dir

    restore = producer._restore_join_key_dtypes

    def _restore_fails_for_b(frame, baseline, keys):
        # The producer's own failure branch: the string-safe type is kept.
        if "b.tiff" in set(frame["Metadata_ImageName"]):
            return frame.copy()
        return restore(frame, baseline, keys)

    monkeypatch.setattr(
        producer, "_restore_join_key_dtypes", _restore_fails_for_b
    )
    snapshot = _build_v1_tree(
        tmp_path,
        snapshot="Metadata_Well,Metadata_Strain\n1,WT\n2,WT\n3,MUT\n4,MUT\n",
        measurements={
            stem: _measurements(stem, well_dtype=pl.Int64) for stem in ("a", "b")
        },
    )
    monkeypatch.undo()

    # Preconditions: the drift exists, in one store and not the other.
    assert _embedded_table(tmp_path, "a").schema["Metadata_Well"] == pl.Int64
    assert _embedded_table(tmp_path, "b").schema["Metadata_Well"] == pl.String, (
        "the fixture did not drift b's join key; there is nothing to pin"
    )

    with caplog.at_level(logging.WARNING):
        aggregate_measurements(
            output_dir=tmp_path,
            dataset_names=[DATASET],
            include_dataset_column=True,
            metadata_csv=snapshot,
            njobs=njobs,
        )

    # STANDING RULE: at njobs=2 the drift must actually straddle a shard
    # boundary, or the merge advisory is never the one under test.
    shards = sorted(aggregation_shard_dir(tmp_path, None).glob("shard_*.parquet"))
    if njobs == 2:
        assert [set(pl.read_parquet(s)["Metadata_ImageName"]) for s in shards] == [
            {"a.tiff"},
            {"b.tiff"},
        ], f"the drifted stores did not land in separate shards: {shards}"
    else:
        assert not shards, "njobs=1 fanned out; both arms take the same path"

    master = _master(tmp_path)
    assert master.height > 0, "the master is empty; every pin below is vacuous"
    assert master.schema["Metadata_Well"] == pl.String, (
        "the concat no longer widens a drifted key -- if it now repairs it, "
        "update this pin and the R4 ruling it records"
    )
    assert any(
        "Int64" in message and "String" in message
        for message in _warnings_naming(caplog, "Metadata_Well")
    ), "the dtype widening happened silently"

    measured = _measured(_mirror(tmp_path))
    assert set(measured["Metadata_ImageName"]) == {"a.tiff", "b.tiff"}, (
        "a drifted store's rows left the mirror"
    )
    assert measured["Metadata_Strain"].null_count() == 0, (
        "the drifted key nulled its rows' metadata -- the harm CAN-10(b) "
        "predicted is real; revisit R4"
    )


@pytest.mark.parametrize("njobs", [1, 2])
def test_a_finalization_that_excludes_every_store_publishes_nothing(
    tmp_path: Path, njobs: int
) -> None:
    """MF-2 (Task 4 review): the direct path's outcome is the contract.

    When the projection excludes every authorized store, ``finalize_run``
    returns ``None`` and publishes nothing -- *"no measurement source could be
    read"*. Fan-out reached a different outcome: each shard wrote its empty
    sentinel, the merge concatenated them into a 0x0 master that overwrote the
    previous one, an empty mirror was seeded, and an aggregate proof
    certified 0 images while the live success count was 2. The outcome may
    not depend on ``njobs``.
    """
    from phenotypic._cli._cli_completion import valid_aggregate_snapshot
    from phenotypic._cli._cli_output_manager import aggregate_measurements
    from phenotypic.sdk_ import aggregation_shard_dir

    _publish_successful_images(tmp_path, include_well=False)
    for stem in ("a", "b"):
        _strip_measurement_descriptor(tmp_path, stem)
    master = master_measurements_parquet_path(tmp_path)
    master.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"Object_Label": [1, 2], "Shape_Area": [4.0, 4.0]}).write_parquet(
        master
    )
    planted = master.read_bytes()

    # Preconditions. Both stores are still AUTHORIZED, so the live success
    # count is 2 -- not the zero-success case where a proof is refused anyway
    # -- and neither a mirror nor a proof exists for "still none" to be about.
    sources = authorized_measurement_sources(tmp_path) or {}
    assert len(sources) == 2, "stripping a descriptor de-authorized a store"
    assert not measurements_parquet_path(tmp_path).exists()
    assert valid_aggregate_snapshot(tmp_path) is None

    result = aggregate_measurements(
        output_dir=tmp_path,
        dataset_names=[DATASET],
        include_dataset_column=True,
        njobs=njobs,
    )

    shards = sorted(aggregation_shard_dir(tmp_path, None).glob("shard_*.parquet"))
    assert len(shards) == (2 if njobs == 2 else 0), (
        f"njobs={njobs} ran {len(shards)} shard(s); the arms are not the two paths"
    )
    assert result is None, "a finalization that aggregated no store reported a master"
    assert master.read_bytes() == planted, "the previous master was overwritten"
    assert not measurements_parquet_path(tmp_path).exists(), (
        "a mirror was published for a master built from no store"
    )
    assert valid_aggregate_snapshot(tmp_path) is None, (
        "an aggregate proof was published for a master built from no store"
    )


def test_a_projection_that_raises_still_removes_its_scratch_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FU-2 (Task 4 review): the projection raises by design -- a newer store
    schema, a table missing a column its descriptor declares -- and
    ``build_master_frame`` has already staged every table to ``$SCRATCH`` by
    then. Removing the staging directory must not depend on it returning."""
    from phenotypic._cli import _cli_parquet_agg

    run = tmp_path / "run"
    run.mkdir()
    _publish_successful_images(run, include_well=False)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SCRATCH", str(scratch))

    staged: list[tuple[Path | None, bool]] = []

    def _raising_projection(table_path, *, read_path=None):
        staged.append((read_path, read_path is not None and read_path.is_file()))
        raise RuntimeError("projection failed")

    monkeypatch.setattr(
        _cli_parquet_agg, "project_embedded_measurement_table", _raising_projection
    )
    with pytest.raises(RuntimeError, match="projection failed"):
        finalize_run(run, dataset_names=[DATASET])

    # STANDING RULE: the staged copy existed while the projection ran, under
    # $SCRATCH, or its directory's absence below proves nothing.
    assert staged, "the projection was never reached"
    read_path, existed = staged[0]
    assert read_path is not None and existed, (
        "nothing was staged to $SCRATCH; there is no leak to test"
    )
    assert read_path.parent.parent == scratch
    assert not read_path.parent.exists(), (
        "the $SCRATCH staging directory leaked when the projection raised"
    )


def test_a_joined_store_whose_duplicate_labels_disagree_is_excluded_not_collapsed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """CAN-10(a)'s proof obligation: never collapse two distinct objects.

    Collapsing a joined store's rows on ``Object_Label`` is safe only when
    every row sharing a label is identical across the projected columns --
    then the duplicates are fan-out copies of one object. When they disagree
    the store is excluded with an advisory, and the aggregate proof must not
    certify the image the master omits.
    """
    import logging

    from phenotypic._cli._cli_completion import valid_aggregate_snapshot

    def _conflicting() -> dict[str, object]:
        # Label 1 twice, on different wells: two rows that are NOT copies.
        return {
            "a": _measurements("a", extra_objects=[(1, "A02")]),
            "b": _measurements("b"),
        }

    joined = tmp_path / "joined"
    snapshot = _build_v1_tree(
        joined, snapshot=_SNAPSHOT_BY_IMAGE, measurements=_conflicting()
    )
    unjoined = tmp_path / "unjoined"
    _build_v1_tree(unjoined, snapshot=None, measurements=_conflicting())

    # Preconditions: the conflict is in the joined store, and the control arm
    # carries the same rows under a status the rule does not apply to.
    assert _join_status(joined, "a") == "joined"
    conflict = _embedded_table(joined, "a").filter(pl.col("Object_Label") == 1)
    assert conflict.height == 2 and conflict["Metadata_Well"].n_unique() == 2, (
        "the fixture did not produce two disagreeing rows under one label"
    )
    assert _join_status(unjoined, "a") == "not_requested"

    with caplog.at_level(logging.WARNING):
        finalize_run(joined, dataset_names=[DATASET], metadata_csv=snapshot)

    master = _master(joined)
    assert master.height > 0, "the whole master is empty, not one store"
    assert set(master["Metadata_ImageName"]) == {"b.tiff"}, (
        "a store whose same-label rows disagree was aggregated -- collapsed "
        "(losing an object) or kept"
    )
    store_a = str(zarr_store_path(joined, DATASET, "a"))
    assert _warnings_naming(caplog, store_a), "the store was excluded silently"
    aggregate = valid_aggregate_snapshot(joined)
    assert aggregate is not None
    assert aggregate["source_image_count"] == 1, (
        "the aggregate proof certifies an image the master does not carry"
    )

    # Control: the rule is about JOIN fan-out, so a table the join never
    # touched keeps every row it has.
    finalize_run(unjoined, dataset_names=[DATASET])
    assert (
        _master(unjoined).filter(pl.col("Metadata_ImageName") == "a.tiff").height
        == 3
    ), "a not_requested store lost rows to a rule that only concerns joins"


def test_a_store_without_a_measurement_descriptor_is_skipped_with_an_advisory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Step 0b. ``read_embedded_measurement_descriptor`` documents an absent
    descriptor as a normal state, while ``embedded_measurement_columns``
    raises ``KeyError`` on it -- so projection has a reachable unhandled
    path. Skip the store, advise, and degrade the proof toward incomplete."""
    import logging

    from phenotypic._cli._cli_completion import valid_aggregate_snapshot
    from phenotypic.sdk_._measurement_tables import (
        read_embedded_measurement_descriptor,
    )

    _publish_successful_images(tmp_path, include_well=False)
    store = _strip_measurement_descriptor(tmp_path, "a")

    # Preconditions: the store is still AUTHORIZED (otherwise the skip is
    # never reached) and the descriptor really is gone.
    sources = authorized_measurement_sources(tmp_path) or {}
    table = (store / MEASUREMENT_TABLE_RELATIVE_PATH).resolve()
    assert table in {Path(path).resolve() for path in sources}, (
        "stripping the descriptor de-authorized the store; the skip is unreached"
    )
    with pytest.raises(KeyError):
        read_embedded_measurement_descriptor(store)

    with caplog.at_level(logging.WARNING):
        finalize_run(tmp_path, dataset_names=[DATASET])

    master = _master(tmp_path)
    assert master.height > 0, "the whole master is empty, not one store"
    assert set(master["Metadata_ImageName"]) == {"b.tiff"}, (
        "a store with no descriptor was aggregated without its column list"
    )
    assert _warnings_naming(caplog, str(store)), "the store was skipped silently"
    aggregate = valid_aggregate_snapshot(tmp_path)
    assert aggregate is not None
    assert aggregate["source_image_count"] == 1, (
        "the aggregate proof certifies an image the master does not carry"
    )


# ---------------------------------------------------------------------------
# H6 -- what survives the retirement of _consistent_embedded_join_keys
# ---------------------------------------------------------------------------


def test_mixed_embedded_and_legacy_authority_is_still_refused(
    tmp_path: Path,
) -> None:
    """H6: retiring the mixed-GENERATION guard must not retire the
    mixed-AUTHORITY one.

    ``_consistent_embedded_join_keys`` carried two independent refusals. D-A
    manufactures the state that trips the second; nothing about that says a
    tree holding both embedded tables and legacy external Parquets became
    safe to aggregate silently.
    """
    from phenotypic._cli._cli_finalize_run import (
        refuse_mixed_measurement_authority,
    )

    embedded = (
        zarr_store_path(tmp_path, DATASET, "a") / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    legacy = tmp_path / DIR_RESULTS / DATASET / DIR_MEASUREMENTS / "b.parquet"

    # Each half alone is fine -- assert that, or "raises on the mixture" is
    # indistinguishable from "raises on everything".
    refuse_mixed_measurement_authority([embedded])
    refuse_mixed_measurement_authority([legacy])

    with pytest.raises(
        ValueError, match="mixed embedded and legacy measurement authority"
    ):
        refuse_mixed_measurement_authority([embedded, legacy])


def test_the_authorized_arm_refuses_a_mixed_authority_tree(
    tmp_path: Path,
) -> None:
    """H6, reachability half 1 -- ``select_measurement_sources``.

    The test above proves the guard REFUSES; it does not prove anything ever
    hands it a mixture. ``_cli_finalize_run.py``'s call site runs on every
    authorized finalization, but every other fixture in this file builds a
    homogeneous authorized set, so deleting that line left the whole suite
    green.

    The mixture is manufactured the way production would reach it: a record
    whose ``artifacts["measurements"]`` names a legacy external Parquet
    instead of a store's embedded table. ``authorized_measurement_sources``
    reads that path verbatim, so the authorized mapping carries both shapes.
    """
    from phenotypic._cli._cli_completion import authorized_measurement_sources

    _publish_successful_images(tmp_path, stems=["a", "b"])

    # POSITIVE CONTROL, on this same tree and before the mixture exists: a
    # homogeneous authorized set finalizes. Without it, the raise below is
    # indistinguishable from a fixture that could never finalize at all.
    assert finalize_run(tmp_path, dataset_names=[DATASET]) is not None, (
        "the homogeneous tree did not finalize; the refusal below would be "
        "attributable to the fixture rather than to the mixture"
    )

    legacy = _publish_legacy_external_record(tmp_path, "c")
    _install_state(tmp_path, ["a", "b", "c"])

    # STANDING RULE. Establish the authorized set really is MIXED -- if the
    # legacy record failed to authorize, `pytest.raises` would be waiting for
    # an exception nothing could throw.
    sources = authorized_measurement_sources(tmp_path)
    assert sources is not None, "the tree fell back to the legacy arm"
    assert legacy.resolve() in sources, (
        "the legacy external Parquet did not reach the AUTHORIZED set; the "
        "mixture this test needs was never built"
    )
    suffix = MEASUREMENT_TABLE_RELATIVE_PATH.parts
    embedded = [
        path
        for path in sources
        if tuple(Path(path).parts[-len(suffix) :]) == suffix
    ]
    assert embedded, "no embedded tables in the authorized set"
    assert len(embedded) != len(sources), "the authorized set is not mixed"

    with pytest.raises(
        ValueError, match="mixed embedded and legacy measurement authority"
    ):
        finalize_run(tmp_path, dataset_names=[DATASET])


def test_the_recompile_finalizer_refuses_a_mixed_authority_task(
    tmp_path: Path,
) -> None:
    """H6, reachability half 2 -- ``_run_post_master_steps``.

    The recompile SLURM finalizer re-checks the mixture over the task's
    serialized ``measurement_sources`` rather than over a live scan, because
    by the time it runs the sources were chosen in a different job. Nothing
    drove that call site with a mixed list, so deleting the whole
    ``if measurement_sources is not None:`` block left the suite green.
    """
    from phenotypic.sdk_ import DIR_RECOMPILE_SHARDS
    from phenotypic._cli._cli_recompile_worker import _run_post_master_steps

    _publish_successful_images(tmp_path, stems=["a", "b"])

    attempt_dir = progress_dir(tmp_path) / "recompile" / "attempt-1"
    shard_dir = attempt_dir / DIR_RECOMPILE_SHARDS
    shard_dir.mkdir(parents=True, exist_ok=True)
    _concat_of_embedded_tables(tmp_path).write_parquet(
        shard_dir / "shard_0.parquet"
    )

    embedded = [
        str(
            zarr_store_path(tmp_path, DATASET, stem)
            / MEASUREMENT_TABLE_RELATIVE_PATH
        )
        for stem in ("a", "b")
    ]
    legacy = str(
        tmp_path / DIR_RESULTS / DATASET / DIR_MEASUREMENTS / "c.parquet"
    )

    def _task(sources: list[str]) -> dict[str, object]:
        return {
            "dataset_names": [DATASET],
            "include_dataset_column": True,
            "measurement_sources": sources,
        }

    # POSITIVE CONTROL: the identical call with a HOMOGENEOUS source list runs
    # all the way through `finalize_run` and publishes a master. So the raise
    # below is the mixture, not the task shape, the shard, or the tree.
    master = _run_post_master_steps(
        tmp_path, _task(embedded), attempt_dir=attempt_dir
    )
    assert master is not None and master.is_file(), (
        "the homogeneous task published no master; the refusal below is "
        "attributable to the fixture"
    )

    with pytest.raises(
        ValueError, match="mixed embedded and legacy measurement authority"
    ):
        _run_post_master_steps(
            tmp_path, _task([*embedded, legacy]), attempt_dir=attempt_dir
        )


def test_the_run_proof_copies_the_aggregates_source_set_digest(
    tmp_path: Path,
) -> None:
    """U-4, and the COPY-not-recompute half of it (NEW-7).

    The copy IS the binding: the run proof asserts "I was published against
    THAT aggregate", which rule 1 then checks against a live re-derivation.
    A recomputation would agree on an unchanged tree and diverge on a changed
    one -- letting a stale aggregate proof sit beside a fresh run proof with
    both passing independently.
    """
    from phenotypic._cli._cli_completion import (
        publish_run_completion_evidence,
        valid_aggregate_snapshot,
    )

    _publish_successful_images(tmp_path)
    finalize_run(tmp_path, dataset_names=[DATASET])

    aggregate = valid_aggregate_snapshot(tmp_path)
    assert aggregate is not None, "no aggregate proof; the copy is vacuous"
    assert aggregate.get("source_set_digest"), (
        "the aggregate proof carries no source_set_digest to copy"
    )
    assert "publication_id" not in aggregate, (
        "publication_id was not cut from the aggregate proof"
    )

    proof_path = publish_run_completion_evidence(
        tmp_path, execution_epoch="local"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))

    assert "publication_id" not in proof
    assert proof["source_set_digest"] == aggregate["source_set_digest"]
    assert proof["source_image_count"] == aggregate["source_image_count"]


def test_the_run_proof_binding_is_checked_end_to_end(tmp_path: Path) -> None:
    """The binding must be a live comparison, not a field nobody reads.

    ``valid_run_completion`` compares the run proof against the aggregate's
    values; corrupt one and the verdict must flip. Without this, replacing
    ``publication_id`` with ``source_set_digest`` could have produced another
    ``None == None`` tautology and nothing in the tree would fail.
    """
    from phenotypic._cli._cli_completion import (
        publish_run_completion_evidence,
        valid_run_completion,
    )
    from phenotypic.sdk_ import (
        aggregate_publication_marker_path,
        run_completion_marker_path,
    )

    _publish_successful_images(tmp_path)
    finalize_run(tmp_path, dataset_names=[DATASET])
    publish_run_completion_evidence(tmp_path, execution_epoch="local")

    assert valid_run_completion(tmp_path) is not None, (
        "the run does not validate at all; the corruption below proves nothing"
    )

    proof_path = run_completion_marker_path(tmp_path)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["source_set_digest"] = hashlib.sha256(b"different").hexdigest()
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    assert aggregate_publication_marker_path(tmp_path).is_file()
    assert valid_run_completion(tmp_path) is None, (
        "a run proof bound to a different source set still validated -- the "
        "aggregate<->run binding stopped being checked"
    )
