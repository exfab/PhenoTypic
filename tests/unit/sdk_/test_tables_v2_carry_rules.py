"""#281 P3 review fixes (C16r): what a store rewrite keeps, replaces and warns about.

- **M1.** A writer that knows its input replaces the ``upstream`` block with
  what it read just now, so an entry the input no longer has does not linger;
  a run that shares an id or an order with an upstream entry is logged.
- **M7.** ``--restart`` keeps another pipeline's run folder byte for byte, and
  a same-pipeline ``replace_image_tables`` rewrites that run's folder as a
  total description of the write: given ``embeddings=``, it writes them;
  given none, the run keeps none (#281 P4 T2).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from phenotypic._cli._cli_failure_tracker import file_sha256
from phenotypic.sdk_ import list_table_runs
from phenotypic.sdk_._table_runs import run_upstream_collisions
from tests._tiny_run import (
    _close_open_application,
    _load_store,
    one_store,
    root_tables,
    run_cli,
    tables_for,
    tree_digest,
)

_TABLES_LOGGER = "phenotypic.sdk_._measurement_tables"


def _resave_input(store: Path, pipeline: Path, *, from_scratch: bool) -> None:
    """Re-save the input store as a finished run of *pipeline* would leave it.

    ``from_scratch`` deletes the store first, so its runs are renumbered and
    only *pipeline*'s survives; otherwise its existing runs are carried and
    *pipeline*'s is added. The application is closed, as a finished CLI run
    leaves it, so a forward run can read the store as its input.
    """
    from phenotypic._core._provenance import initialize_cli_provenance

    image = _load_store(store)
    if from_scratch:
        shutil.rmtree(store)
    _close_open_application(image)
    initialize_cli_provenance(image, pipeline, kind="full", input_filename="plate.tif")
    _close_open_application(image)
    image.save2zarr(store, measurement_table=tables_for(image))


def _forward_over_input(input_store: Path, out: Path, pipeline: Path, *extra: str):
    result = run_cli(input_store.parent.parent, out, pipeline=pipeline, extra=list(extra))
    assert result.exit_code == 0, result.output
    return one_store(out)


# ---------------------------------------------------------------------------
# M1: upstream is replaced, never merged; collisions are logged
# ---------------------------------------------------------------------------


def test_a_restart_drops_an_upstream_entry_the_input_no_longer_has(
    input_store_v2, tmp_path, pipeline_file_a, pipeline_file_b
):
    out = tmp_path / "out"
    result_store = _forward_over_input(input_store_v2, out, pipeline_file_a)
    (old_entry,) = root_tables(result_store)["upstream"]

    # The input is rebuilt by another pipeline: its one run is now 1-<B>.
    _resave_input(input_store_v2, pipeline_file_b, from_scratch=True)
    (new_entry,) = (run.run_id for run in list_table_runs(input_store_v2))
    assert new_entry != old_entry

    _forward_over_input(input_store_v2, out, pipeline_file_a, "--restart")
    assert set(root_tables(result_store)["upstream"]) == {new_entry}


def test_an_order_collision_with_the_input_is_logged(
    input_store_v2, tmp_path, pipeline_file_a, pipeline_file_b, caplog
):
    out = tmp_path / "out"
    result_store = _forward_over_input(input_store_v2, out, pipeline_file_a)
    own = f"2-{file_sha256(pipeline_file_a)[:12]}"
    assert [run.run_id for run in list_table_runs(result_store)] == [own]

    # The input grows a run numbered 2, the number the result's own run keeps.
    _resave_input(input_store_v2, pipeline_file_b, from_scratch=False)
    grown = [run.run_id for run in list_table_runs(input_store_v2)]
    assert grown[-1].startswith("2-")

    with caplog.at_level(logging.WARNING, logger=_TABLES_LOGGER):
        _forward_over_input(input_store_v2, out, pipeline_file_a, "--restart")
    tables = root_tables(result_store)
    assert set(tables["upstream"]) == set(grown)
    assert list(tables["runs"]) == [own]  # D6: the re-save keeps its number
    assert any(
        "shares order 2" in record.getMessage() and own in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_stage3_still_carries_the_upstream_stage1_recorded(tmp_path, measured_image):
    """A save given no ``upstream`` carries the existing block unchanged."""
    entry = {"source": "in.ome.zarr", "root_sha256": "0" * 64, "pipeline_sha256": None}
    _close_open_application(measured_image)
    store = measured_image.save2zarr(
        tmp_path / "p.ome.zarr", upstream={"1-unknown": entry}
    )
    measured_image.save2zarr(store, measurement_table=tables_for(measured_image))
    assert root_tables(store)["upstream"] == {"1-unknown": entry}


def test_run_upstream_collisions_names_ids_and_orders():
    runs = {"2-aaaaaaaaaaaa": {}, "3-bbbbbbbbbbbb": {}}
    upstream = {"2-cccccccccccc": {}, "3-bbbbbbbbbbbb": {}}
    assert run_upstream_collisions(runs, upstream) == [
        "run '3-bbbbbbbbbbbb' is also an upstream entry",
        "run '2-aaaaaaaaaaaa' shares order 2 with upstream entry '2-cccccccccccc'",
    ]
    assert run_upstream_collisions({"3-bbbbbbbbbbbb": {}}, {"1-x": {}}) == []


# ---------------------------------------------------------------------------
# M7: what --restart and replace_image_tables keep
# ---------------------------------------------------------------------------


def test_restart_keeps_another_pipelines_folder_byte_identical(
    finished_tiny_run, tiny_inputs, pipeline_file_b
):
    out = finished_tiny_run
    result = run_cli(None, out, mode="measure", pipeline=pipeline_file_b)
    assert result.exit_code == 0, result.output
    store = one_store(out, stem="plate_1")
    own, other = list_table_runs(store)
    assert other.pipeline_sha256 == file_sha256(pipeline_file_b)
    before = tree_digest(other.folder)

    result = run_cli(tiny_inputs, out, extra=["--restart"])
    assert result.exit_code == 0, result.output
    runs = list_table_runs(store)
    assert [run.run_id for run in runs] == [own.run_id, other.run_id]
    assert tree_digest(other.folder) == before
    assert root_tables(store)["runs"][other.run_id] == dict(other.descriptor)


def test_a_same_pipeline_replace_image_tables_without_embeddings_drops_that_runs_embeddings(
    store_with_crop_entry,
):
    """``embeddings=None`` means "this write has no embeddings" (#281 P4 T2).

    The measure rewrite owns its pipeline's whole run folder, and the folder
    it writes is a total description of the write (spec D6: no row of the
    previous write survives). So a rewrite given no embeddings leaves the run
    with none -- as a measurer dropped from the pipeline drops its columns.
    ``--mode measure`` passes its pipeline's ``embed()`` payloads, so a
    pipeline that embeds keeps (and rewrites) its embeddings; that case is
    ``test_replace_image_tables_writes_the_given_embeddings``.
    """
    from phenotypic.sdk_ import replace_image_tables

    store = store_with_crop_entry
    (run,) = list_table_runs(store)
    assert run.descriptor.get("embeddings")
    image = _load_store(store)
    replace_image_tables(
        store,
        tables_for(image),
        pipeline_sha256=run.pipeline_sha256,
        durable=False,
    )
    (after,) = list_table_runs(store)
    assert after.run_id == run.run_id
    assert not after.descriptor.get("embeddings")
    assert not (after.folder / "embeddings").exists()


def test_another_pipelines_replace_with_embeddings_leaves_the_first_folder_byte_identical(
    store_with_crop_entry, pipeline_file_b, toy_embedder
):
    """A rewrite by pipeline B adds B's folder, embeddings included, and carries A's whole."""
    from phenotypic import ImagePipeline
    from phenotypic.sdk_ import replace_image_tables

    store = store_with_crop_entry
    (first,) = list_table_runs(store)
    before = tree_digest(first.folder)
    entry_before = dict(first.descriptor)
    image = _load_store(store)
    written = replace_image_tables(
        store,
        tables_for(image),
        pipeline_sha256=file_sha256(pipeline_file_b),
        embeddings=ImagePipeline(meas={"emb": toy_embedder}).embed(image),
        durable=False,
    )
    runs = list_table_runs(store)
    assert [run.run_id for run in runs] == [first.run_id, written.run_id]
    assert tree_digest(first.folder) == before
    assert root_tables(store)["runs"][first.run_id] == entry_before
    assert set(runs[1].descriptor["embeddings"]) == {"toy-v1/crops"}
    assert runs[1].descriptor["embeddings"]["toy-v1/crops"]["stale"] is False
