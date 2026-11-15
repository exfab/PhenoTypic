"""The upstream reference: a result store records its input's table runs (#281 P3 Task 6, spec D2, D8)."""

from __future__ import annotations

import json
import shutil

import pytest

import phenotypic
from phenotypic import ImagePipeline
from phenotypic._cli._cli_failure_tracker import file_sha256
from phenotypic._cli._cli_process_only import process_only_output_path
from phenotypic.measure import MeasureSize
from phenotypic.sdk_ import (
    UpstreamChangedError,
    follow_upstream,
    latest_table_run,
    list_table_runs,
    upstream_entries_for_input,
)
from tests._fakes.fake_gpu_detector import FakeGpuDetector
from tests._tiny_run import (
    _close_open_application,
    one_store,
    root_tables,
    run_cli,
    tables_for,
    touch_root,
)

SHA_P = "3f9c2a1b7d04" + "1" * 52


def _only_upstream_id(store) -> str:
    (run_id,) = root_tables(store)["upstream"]
    return run_id


# --- the plan's two tests ---------------------------------------------------


def test_result_store_records_input_runs_and_numbers_after_them(
    input_store_v2, tmp_path, pipeline_file_b
):
    out = tmp_path / "out"
    result = run_cli(input_store_v2.parent.parent, out, pipeline=pipeline_file_b)
    assert result.exit_code == 0, result.output
    tables = root_tables(one_store(out))
    assert set(tables["upstream"]) == {latest_table_run(input_store_v2).run_id}  # "1-<A>"
    assert tables["upstream"][next(iter(tables["upstream"]))]["root_sha256"] == file_sha256(
        input_store_v2 / "zarr.json"
    )
    assert [r.run_id for r in list_table_runs(one_store(out))] == [
        f"2-{file_sha256(pipeline_file_b)[:12]}"
    ]


def test_follow_upstream_detects_a_reprocessed_input(input_store_v2, tmp_path, pipeline_file_b):
    out = tmp_path / "out"
    result = run_cli(input_store_v2.parent.parent, out, pipeline=pipeline_file_b)
    assert result.exit_code == 0, result.output
    touch_root(input_store_v2)  # simulate an AutoConvertRaw-GC reprocess
    with pytest.raises(UpstreamChangedError):
        follow_upstream(
            one_store(out),
            next(iter(root_tables(one_store(out))["upstream"])),
            input_root=input_store_v2.parent.parent,
        )


# --- the entry, field by field ----------------------------------------------


def test_an_entry_names_its_source_root_and_pipeline(input_store_v2, pipeline_file_a):
    run = latest_table_run(input_store_v2)
    entries = upstream_entries_for_input(input_store_v2, input_store_v2.parent.parent)
    assert entries == {
        run.run_id: {
            "source": "ds/plate.ome.zarr",
            "root_sha256": file_sha256(input_store_v2 / "zarr.json"),
            "pipeline_sha256": file_sha256(pipeline_file_a),
        }
    }


def test_a_direct_store_input_is_its_own_root(input_store_v2):
    entries = upstream_entries_for_input(input_store_v2, input_store_v2)
    (entry,) = entries.values()
    assert entry["source"] == "plate.ome.zarr"
    (run_id,) = entries
    # A direct store input is followed through the same --input it was read from.
    own = _store_with_upstream(input_store_v2.parent / "result.ome.zarr", entries)
    assert follow_upstream(own, run_id, input_root=input_store_v2).store == input_store_v2


# --- inputs with nothing to record ------------------------------------------


def test_a_plain_tiff_input_records_no_upstream(tmp_path, finished_tiny_run):
    from phenotypic.data import load_synth_yeast_plate

    tif = tmp_path / "plate.tif"
    load_synth_yeast_plate().rgb.imsave(filepath=tif)
    assert upstream_entries_for_input(tif, tmp_path) == {}
    # And a forward run over TIFFs records none.
    assert root_tables(one_store(finished_tiny_run, stem="plate_1"))["upstream"] == {}


def test_a_v1_input_store_records_no_upstream(v1_store, v1_tree):
    assert upstream_entries_for_input(v1_store, v1_tree) == {}


def test_a_third_party_store_records_no_upstream(tmp_path):
    store = tmp_path / "napari.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": {}}}),
        encoding="utf-8",
    )
    assert upstream_entries_for_input(store, tmp_path) == {}


def test_a_v2_input_with_no_runs_records_no_upstream(tmp_path, measured_image):
    _close_open_application(measured_image)
    store = measured_image.save2zarr(tmp_path / "in" / "ds" / "bare.ome.zarr")
    assert upstream_entries_for_input(store, tmp_path / "in") == {}


# --- the chain ---------------------------------------------------------------


def test_the_inputs_own_upstream_is_merged_transitively(tmp_path, measured_image):
    grandparent = {"source": "raw/plate.ome.zarr", "root_sha256": "0" * 64, "pipeline_sha256": SHA_P}
    _close_open_application(measured_image)
    store = measured_image.save2zarr(
        tmp_path / "in" / "ds" / "plate.ome.zarr",
        measurement_table=tables_for(measured_image),
        upstream={"1-" + SHA_P[:12]: grandparent},
    )
    own = latest_table_run(store)
    assert own.order == 2  # numbered after its own upstream entry (D2)
    entries = upstream_entries_for_input(store, tmp_path / "in")
    assert set(entries) == {"1-" + SHA_P[:12], own.run_id}
    assert entries["1-" + SHA_P[:12]] == grandparent  # carried verbatim
    assert entries[own.run_id]["source"] == "ds/plate.ome.zarr"


def test_a_chained_result_numbers_after_the_whole_chain(tmp_path, measured_image, pipeline_file_b):
    _close_open_application(measured_image)
    store = measured_image.save2zarr(
        tmp_path / "in" / "ds" / "plate.ome.zarr",
        measurement_table=tables_for(measured_image),
        upstream={"1-" + SHA_P[:12]: {"source": "x", "root_sha256": "0" * 64, "pipeline_sha256": SHA_P}},
    )
    out = tmp_path / "out"
    result = run_cli(store.parent.parent, out, pipeline=pipeline_file_b)
    assert result.exit_code == 0, result.output
    assert set(root_tables(one_store(out))["upstream"]) == {
        "1-" + SHA_P[:12],
        latest_table_run(store).run_id,
    }
    assert [r.order for r in list_table_runs(one_store(out))] == [3]


# --- following a reference ---------------------------------------------------


def test_follow_upstream_returns_the_inputs_run(input_store_v2, tmp_path, pipeline_file_b):
    out = tmp_path / "out"
    assert run_cli(input_store_v2.parent.parent, out, pipeline=pipeline_file_b).exit_code == 0
    run_id = _only_upstream_id(one_store(out))
    run = follow_upstream(one_store(out), run_id, input_root=input_store_v2.parent.parent)
    assert run.store == input_store_v2 and run.run_id == run_id
    assert run.measurements_path().is_file()


def test_follow_upstream_on_an_unreachable_input_raises(input_store_v2, tmp_path):
    entries = upstream_entries_for_input(input_store_v2, input_store_v2.parent.parent)
    own = _store_with_upstream(tmp_path / "result.ome.zarr", entries)
    shutil.rmtree(input_store_v2)
    with pytest.raises(FileNotFoundError):
        follow_upstream(own, next(iter(entries)), input_root=input_store_v2.parent.parent)


def test_upstream_changed_error_carries_both_digests(input_store_v2, tmp_path):
    entries = upstream_entries_for_input(input_store_v2, input_store_v2.parent.parent)
    own = _store_with_upstream(tmp_path / "result.ome.zarr", entries)
    (run_id,) = entries
    expected = entries[run_id]["root_sha256"]
    touch_root(input_store_v2)
    with pytest.raises(UpstreamChangedError) as caught:
        follow_upstream(own, run_id, input_root=input_store_v2.parent.parent)
    assert caught.value.run_id == run_id
    assert caught.value.expected == expected
    assert caught.value.actual == file_sha256(input_store_v2 / "zarr.json") != expected


def test_follow_upstream_on_an_unrecorded_run_raises_key_error(input_store_v2, tmp_path):
    own = _store_with_upstream(
        tmp_path / "result.ome.zarr",
        upstream_entries_for_input(input_store_v2, input_store_v2.parent.parent),
    )
    with pytest.raises(KeyError):
        follow_upstream(own, "9-" + "a" * 12, input_root=input_store_v2.parent.parent)


# --- the other writers -------------------------------------------------------


def test_a_process_store_records_its_inputs_runs(input_store_v2, tmp_path, pipeline_file_b):
    root = input_store_v2.parent.parent
    out = tmp_path / "out"
    result = run_cli(
        root,
        out,
        mode="process",
        pipeline=pipeline_file_b,
        extra=["--layer", "gray", "--process-format", "zarr"],
    )
    assert result.exit_code == 0, result.output
    store = process_only_output_path(out, input_store_v2, root, "gray", fmt="zarr")
    assert set(root_tables(store)["upstream"]) == {latest_table_run(input_store_v2).run_id}
    assert list_table_runs(store) == ()  # a process store measures nothing


def test_a_staged_store_records_its_inputs_runs(input_store_v2, tmp_path, monkeypatch):
    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False)
    pipeline = tmp_path / "staged.pht-pipe.json"
    pipeline.write_text(
        ImagePipeline(
            name="staged-upstream",
            ops=[FakeGpuDetector(threshold=0.3)],
            meas={"size": MeasureSize()},
        ).to_json()
        or "",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    result = run_cli(
        input_store_v2.parent.parent, out, pipeline=pipeline, extra=["--skip-validation"]
    )
    assert result.exit_code == 0, result.output
    store = one_store(out)
    assert set(root_tables(store)["upstream"]) == {latest_table_run(input_store_v2).run_id}
    assert [r.run_id for r in list_table_runs(store)] == [f"2-{file_sha256(pipeline)[:12]}"]


# --- helpers -----------------------------------------------------------------


def _store_with_upstream(path, entries):
    """A table-less store at *path* that records *entries* as its ``upstream``."""
    from phenotypic.data import load_synth_yeast_plate

    return phenotypic.Image(load_synth_yeast_plate()).save2zarr(path, upstream=entries)
