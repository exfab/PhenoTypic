"""#281 P3 Task 4, cluster C14d: the state readers and the GUI-facing ``sdk_`` readers.

Two kinds of reader move onto the v2 resolver here:

- the run-state reader (``sdk_/_run_state.py``), which judges a finished
  image by its record and projects facts out of the store root; and
- the three ``sdk_`` readers the results viewer calls
  (``read_embedded_measurement_descriptor``, ``embedded_measurement_columns``,
  ``read_embedded_measurement_column``), which hold no record and so read
  spec D6a's default: the highest-numbered run folder that holds
  ``measurements`` (``latest_measurement_run``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from phenotypic.sdk_ import latest_measurement_run, list_table_runs
from tests._tiny_run import one_store

#: A third pipeline digest, owned by neither ``pipeline_file_a`` nor ``_b``.
SHA_GRID = "c0ffee" + "0" * 58


def add_grid_only_run(store: Path) -> str:
    """Add a run entry with no ``measurements`` to *store*, through a real rewrite.

    The shape of a grid-only process folder (spec D6a's example): a run that
    is newer than every measuring run and declares no measurements. Written
    through ``write_run_entry`` inside the root-last ``_rewrite_store_tables``
    transaction, so the store stays one the writer could have produced.

    Returns:
        The new run's id.
    """
    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_._measurement_tables import (
        _rewrite_store_tables,
        write_run_entry,
    )
    from phenotypic.sdk_._table_runs import next_table_run_order, table_run_id

    block = ngff_.require_readable_store(store)
    order = next_table_run_order(block)
    run_id = table_run_id(order, SHA_GRID)

    def _plan(phenotypic: dict):
        def _populate(part: Path) -> None:
            entry = write_run_entry(
                part, run_id, order=order, pipeline_sha256=SHA_GRID
            )
            tables = phenotypic.setdefault(
                ngff_.PhenotypicAttr.TABLES,
                {"schema_version": ngff_.TABLES_SCHEMA_VERSION, "runs": {}, "upstream": {}},
            )
            tables["runs"][run_id] = entry

        return _populate

    _rewrite_store_tables(
        store, plan=_plan, durable=False, commit_guard=None, clear_table_run=run_id
    )
    return run_id


def test_add_grid_only_run_adds_a_newer_run_without_measurements(store_with_two_runs):
    """The helper's own contract, so the reader test below cannot pass vacuously."""
    run_id = add_grid_only_run(store_with_two_runs)
    runs = list_table_runs(store_with_two_runs)
    assert [run.order for run in runs] == [1, 2, 3]
    assert runs[-1].run_id == run_id and not runs[-1].has_measurements
    assert all(run.has_measurements for run in runs[:2])


def test_marker_rejects_a_corrupted_run_table(finished_tiny_run):
    from phenotypic.sdk_ import resolve_run_state

    assert resolve_run_state(finished_tiny_run, depth="deep").completion == "complete"
    latest_measurement_run(
        one_store(finished_tiny_run, stem="plate_1")
    ).measurements_path().write_bytes(
        b"corrupt"
    )
    assert resolve_run_state(finished_tiny_run, depth="deep").completion != "complete"


def test_gui_facing_readers_default_to_the_latest_folder_with_measurements(
    store_with_two_runs,
):
    from phenotypic.sdk_ import embedded_measurement_columns

    add_grid_only_run(store_with_two_runs)            # a folder with no `measurements`
    expected = list_table_runs(store_with_two_runs)[1]
    assert embedded_measurement_columns(store_with_two_runs) == tuple(
        pq.read_schema(expected.measurements_path()).names
    )


def test_the_descriptor_reader_returns_the_latest_measuring_run(store_with_two_runs):
    from phenotypic.sdk_ import read_embedded_measurement_descriptor

    add_grid_only_run(store_with_two_runs)
    expected = list_table_runs(store_with_two_runs)[1]
    descriptor = read_embedded_measurement_descriptor(store_with_two_runs)
    assert descriptor == dict(expected.descriptor["measurements"])
    assert descriptor["path"] == expected.measurements_path().relative_to(
        store_with_two_runs
    ).as_posix()


def test_the_column_reader_reads_the_latest_runs_payload(store_with_two_runs):
    """Runs 1 and 2 hold different ``Pipeline_Marker`` values, so the payload read is visible."""
    from phenotypic.sdk_ import read_embedded_measurement_column
    from phenotypic.schema import SHAPE

    first, second = list_table_runs(store_with_two_runs)
    column = str(SHAPE.CIRCULARITY)       # only pipeline B (run 2) measures shape
    assert column not in pq.read_schema(first.measurements_path()).names
    values = read_embedded_measurement_column(store_with_two_runs, column)
    table = pq.read_table(second.measurements_path(), columns=["Object_Label", column])
    assert values == {
        int(label): (None if value is None else float(value))
        for label, value in zip(
            table.column("Object_Label").to_pylist(),
            table.column(column).to_pylist(),
            strict=True,
        )
    }
    assert values, "run 2 measured no object, so the comparison is vacuous"


def test_a_store_whose_runs_hold_no_measurements_declares_no_table(tmp_path, measured_image):
    """A store with only a measurement-less run reads like one with no tables: ``KeyError``."""
    from phenotypic.sdk_ import (
        embedded_measurement_columns,
        read_embedded_measurement_column,
        read_embedded_measurement_descriptor,
    )

    store = measured_image.save2zarr(tmp_path / "bare.ome.zarr")
    add_grid_only_run(store)
    assert list_table_runs(store) and latest_measurement_run(store) is None
    for read in (
        read_embedded_measurement_descriptor,
        embedded_measurement_columns,
        lambda s: read_embedded_measurement_column(s, "Object_Label"),
    ):
        with pytest.raises(KeyError):
            read(store)


def test_staged_completeness_reads_the_journals_run_not_the_latest(store_with_two_runs, tmp_path):
    """Markers off: the staged guard asks for the table the image's own save wrote.

    The store's journal names pipeline B, so the guard reads run 2's table.
    Deleting run 1's table must not change the answer; deleting run 2's must.
    """
    import shutil

    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_._run_state import staged_image_is_complete

    target = zarr_store_path(tmp_path / "out", "ds", "plate")
    target.parent.mkdir(parents=True)
    shutil.copytree(store_with_two_runs, target)
    first, second = list_table_runs(target)

    def _complete() -> bool:
        return staged_image_is_complete(
            tmp_path / "out", "ds", "plate", markers_required=False, resume=True
        )

    assert _complete() is True
    first.measurements_path().unlink()
    assert _complete() is True
    second.measurements_path().unlink()
    assert _complete() is False


def test_staged_completeness_ignores_a_newer_run_the_journal_did_not_write(
    store_with_two_runs, tmp_path, pipeline_file_a
):
    """The journal names A (run 1) while B's run 2 is the latest: run 2 cannot stand in.

    A table-less save by A (a Stage-1 shape) carries both folders and makes A
    the journal's pipeline again. With A's own table gone the image is not
    complete, however complete the newer B folder is.
    """
    import shutil

    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_._run_state import staged_image_is_complete
    from tests._tiny_run import resave_with_pipeline

    target = zarr_store_path(tmp_path / "out", "ds", "plate")
    target.parent.mkdir(parents=True)
    shutil.copytree(store_with_two_runs, target)
    resave_with_pipeline(target, pipeline_file_a, tables=False)
    first, second = list_table_runs(target)
    assert latest_measurement_run(target) == second

    def _complete() -> bool:
        return staged_image_is_complete(
            tmp_path / "out", "ds", "plate", markers_required=False, resume=True
        )

    assert _complete() is True
    first.measurements_path().unlink()
    assert second.measurements_path().is_file()
    assert _complete() is False


def test_staged_completeness_is_false_for_a_corrupt_descriptor(tmp_path, store_with_two_runs):
    """A probe must degrade, never raise: a corrupt root routes the image back."""
    import shutil

    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_._run_state import staged_image_is_complete

    target = zarr_store_path(tmp_path / "out", "ds", "plate")
    target.parent.mkdir(parents=True)
    shutil.copytree(store_with_two_runs, target)
    root = target / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    runs = payload["attributes"]["phenotypic"]["tables"]["runs"]
    for entry in runs.values():
        entry["order"] = 1                         # two runs share an order
    root.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        staged_image_is_complete(
            tmp_path / "out", "ds", "plate", markers_required=False, resume=True
        )
        is False
    )


# ---------------------------------------------------------------------------
# The deep pass projects from the run the record certifies (spec D6a)
# ---------------------------------------------------------------------------

_DIGEST = "e" * 64
_STORE_RELATIVE = "results/ds/zarr/plate.ome.zarr"


def _published_two_run_tree(
    tmp_path: Path,
    store: Path,
    *,
    certify: int,
    snapshots: dict[int, str] | None = None,
    strip_measurements: int | None = None,
) -> Path:
    """Copy a two-run *store* into a run tree and publish a record certifying run *certify*.

    The root is edited **before** the record is published, so the record's
    store fingerprint matches it and the image verifies: *snapshots* writes a
    ``metadata_table`` block into the runs of those orders, and
    *strip_measurements* removes one run's ``measurements`` descriptor (the
    finalize projection's Step 0b shape, per run).
    """
    import shutil

    from phenotypic._cli._cli_completion import publish_image_success
    from phenotypic.sdk_ import atomic_write_json, zarr_store_path
    from phenotypic.sdk_.ngff_ import MEASUREMENT_TABLE_GROUP, STORE_ROOT_JSON, PhenotypicAttr

    out = tmp_path / "out"
    target = zarr_store_path(out, "ds", "plate")
    target.parent.mkdir(parents=True)
    shutil.copytree(store, target)
    runs = list_table_runs(target)
    certified = runs[certify - 1].measurements_path()
    root_json = target / STORE_ROOT_JSON
    payload = json.loads(root_json.read_text(encoding="utf-8"))
    entries = payload["attributes"][PhenotypicAttr.ROOT][PhenotypicAttr.TABLES]["runs"]
    for order, digest in (snapshots or {}).items():
        entries[runs[order - 1].run_id][PhenotypicAttr.METADATA_TABLE] = {
            PhenotypicAttr.SNAPSHOT_SHA256: digest
        }
    if strip_measurements is not None:
        entries[runs[strip_measurements - 1].run_id].pop(MEASUREMENT_TABLE_GROUP)
    atomic_write_json(root_json, payload)
    publish_image_success(
        out,
        work_id="work-plate",
        dataset="ds",
        relative_image_path="plate.tif",
        image_stem="plate",
        mode="full",
        attempt_id="attempt-1",
        lifecycle_epoch="local",
        artifacts={"store": target, "measurements": certified},
    )
    return out


def _measured_stage(out: Path) -> dict:
    from phenotypic.sdk_._image_record import STAGE_MEASURED
    from phenotypic.sdk_._run_state import _verify_image

    verified = _verify_image(
        out,
        out.resolve(),
        dataset="ds",
        image_name="plate.tif",
        work_id="work-plate",
        failures={},
    )
    assert verified.state.verdict == "verified", verified.state.reason
    return verified.state.stages[STAGE_MEASURED]


@pytest.mark.parametrize("certify", [1, 2])
def test_the_metadata_snapshot_is_the_certified_runs(tmp_path, store_with_two_runs, certify):
    """Only run 2 records a snapshot; it is seen exactly when run 2 is certified."""
    from phenotypic.sdk_._run_state import _SNAPSHOT_SHA256_ATTR

    out = _published_two_run_tree(
        tmp_path, store_with_two_runs, certify=certify, snapshots={2: _DIGEST}
    )
    stage = _measured_stage(out)
    assert stage.get(_SNAPSHOT_SHA256_ATTR) == (_DIGEST if certify == 2 else None)


@pytest.mark.parametrize(
    ("certify", "strip", "projectable"),
    [(1, 2, True), (2, 1, True), (1, 1, False), (2, 2, False)],
)
def test_projectability_is_judged_on_the_certified_run(
    tmp_path, store_with_two_runs, certify, strip, projectable
):
    """A run the record does not certify cannot make the image (un)projectable."""
    from phenotypic.sdk_._run_state import (
        _DECLARES_MEASUREMENTS,
        _PROJECTABLE_MEASUREMENTS,
    )

    out = _published_two_run_tree(
        tmp_path, store_with_two_runs, certify=certify, strip_measurements=strip
    )
    stage = _measured_stage(out)
    assert stage[_DECLARES_MEASUREMENTS] is True
    assert stage[_PROJECTABLE_MEASUREMENTS] is projectable


def test_a_corrupt_tables_descriptor_certifies_nothing():
    """The deep pass never raises on a store: a corrupt descriptor projects nothing."""
    from phenotypic.sdk_._run_state import _certified_tables_entry
    from tests._tiny_run import v2_block

    sha_a, sha_b = "a" * 64, "b" * 64
    block = v2_block(runs={
        "1-" + sha_a[:12]: {"order": 1, "pipeline_sha256": sha_a, "measurements": {}},
        "1-" + sha_b[:12]: {"order": 1, "pipeline_sha256": sha_b, "measurements": {}},
    })
    artifact = f"{_STORE_RELATIVE}/tables/1-{sha_a[:12]}/measurements/table.parquet"
    for measurements_artifact in (None, artifact):
        assert (
            _certified_tables_entry(
                block,
                store_relative=_STORE_RELATIVE,
                measurements_artifact=measurements_artifact,
            )
            is None
        )


def test_a_record_without_measurements_reads_the_latest_measuring_runs_snapshot():
    """No certified run: the snapshot comes from D6a's record-less default."""
    from phenotypic.sdk_._run_state import _certified_tables_entry
    from tests._tiny_run import v2_block

    sha_a, sha_b, sha_c = "a" * 64, "b" * 64, "c" * 64
    block = v2_block(runs={
        "1-" + sha_a[:12]: {"order": 1, "pipeline_sha256": sha_a, "measurements": {},
                            "metadata_table": {"snapshot_sha256": "1" * 64}},
        "2-" + sha_b[:12]: {"order": 2, "pipeline_sha256": sha_b, "measurements": {},
                            "metadata_table": {"snapshot_sha256": "2" * 64}},
        "3-" + sha_c[:12]: {"order": 3, "pipeline_sha256": sha_c},   # grid-only
    })
    entry = _certified_tables_entry(
        block, store_relative=_STORE_RELATIVE, measurements_artifact=None
    )
    assert entry is not None and entry["metadata_table"]["snapshot_sha256"] == "2" * 64
