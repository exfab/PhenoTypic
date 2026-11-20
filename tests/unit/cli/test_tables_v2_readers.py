"""Readers and per-image records over the v2 table layout (#281 P3 Task 4, spec D6a).

The per-image record declares, as its ``measurements`` artifact, the
``measurements/table.parquet`` of the run folder the forward save wrote --
``tables/<n>-<sha12>/`` -- never the v1 flat ``tables/measurements/`` path.
``publish_image_success`` resolves every artifact ``strict=True``, so a record
naming the flat path fails every image after all of its work is done.

The record tests assert the image's record, not the run's exit code: the exit
code also needs finalization to read the run folders (C14a), which these tests
do not pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import phenotypic
from phenotypic._cli._cli_completion import valid_image_success
from phenotypic._cli._cli_failure_tracker import file_sha256
from phenotypic.sdk_ import list_table_runs
from phenotypic.sdk_._image_record import read_image_record
from tests._tiny_run import one_store, owned_table_run, run_cli


def _record_measurements(out: Path, dataset: str, stem: str) -> Path:
    """The ``measurements`` artifact of the image's VALID record."""
    record = read_image_record(out, dataset, stem)
    assert record is not None, f"no record for {dataset}/{stem}"
    assert valid_image_success(
        out, dataset=dataset, image_stem=stem, work_id=str(record["work_id"])
    ), f"the record for {dataset}/{stem} does not verify against disk"
    return out / record["artifacts"]["measurements"]["path"]


def test_a_local_full_run_certifies_the_run_folder_it_wrote(
    tiny_inputs, tmp_path, pipeline_file_a
):
    out = tmp_path / "out"
    # Explicit: ``run_cli``'s default pipeline file serializes a fresh random
    # ``name``, so its bytes (and sha) are not ``pipeline_file_a``'s.
    run_cli(tiny_inputs, out, pipeline=pipeline_file_a)

    for stem in ("plate_1", "plate_2"):
        store = one_store(out, stem=stem)
        run = owned_table_run(store, file_sha256(pipeline_file_a))
        assert run is not None and run.run_id == f"1-{file_sha256(pipeline_file_a)[:12]}"
        assert run.measurements_path().is_file()
        assert _record_measurements(out, "ds", stem) == run.measurements_path()


def test_a_rerun_certifies_its_own_folder_not_the_latest(
    tiny_inputs, tmp_path, pipeline_file_a, pipeline_file_b
):
    # D6a: the record names the folder THIS save owns. After A then B, a
    # re-run of A replaces folder 1 and carries B's folder 2, so the latest
    # folder with measurements is B's -- the wrong answer for A's record.
    # ``--restart`` because a plain re-run with changed pipeline bytes is
    # refused ("Pipeline contents changed"); a restart keeps the stores.
    out = tmp_path / "out"
    for index, pipeline in enumerate((pipeline_file_a, pipeline_file_b, pipeline_file_a)):
        run_cli(tiny_inputs, out, pipeline=pipeline, extra=["--restart"] if index else [])

    store = one_store(out, stem="plate_1")
    assert [run.order for run in list_table_runs(store)] == [1, 2]
    own = owned_table_run(store, file_sha256(pipeline_file_a))
    assert own is not None and own.order == 1
    assert _record_measurements(out, "ds", "plate_1") == own.measurements_path()


@pytest.fixture
def _fake_gpu_detector(monkeypatch):
    from tests._fakes.fake_gpu_detector import FakeGpuDetector

    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False)
    return FakeGpuDetector


def test_a_local_staged_run_certifies_the_run_folder_it_wrote(tmp_path, _fake_gpu_detector):
    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.measure import MeasureSize
    from phenotypic.phenotypicCLI import phenotypic_cli

    images = tmp_path / "images"
    images.mkdir()
    load_synth_yeast_plate().rgb.imsave(filepath=images / "img.tiff")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        ImagePipeline(
            ops=[_fake_gpu_detector(threshold=0.3)], meas=[MeasureSize()]
        ).to_json(),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline", str(pipeline), "--input", str(images), "--output", str(out),
            "--image-type", "Image", "--force-local", "--skip-validation", "--njobs", "1",
        ],
    )

    store = one_store(out, stem="img")
    run = owned_table_run(store, file_sha256(pipeline))
    assert run is not None and run.run_id == f"1-{file_sha256(pipeline)[:12]}"
    assert _record_measurements(out, images.name, "img") == run.measurements_path()


# ---------------------------------------------------------------------------
# C14a: the finalizer and aggregation readers (spec D6a, C4)
# ---------------------------------------------------------------------------


def write_marker_certifying(output_root: Path, run) -> Path:
    """Write a completion marker whose ``measurements`` artifact is *run*'s table."""
    import json

    relative = run.measurements_path().resolve().relative_to(output_root.resolve())
    marker = output_root / "marker.json"
    marker.write_text(
        json.dumps({"artifacts": {"measurements": {"path": relative.as_posix()}}}),
        encoding="utf-8",
    )
    return marker


def test_master_reads_the_folder_the_record_certifies(store_with_two_runs, tmp_path):
    from phenotypic._cli._cli_parquet_agg import project_embedded_measurement_table
    from phenotypic._cli._cli_recompile_recovery import certified_table_run

    first, latest = list_table_runs(store_with_two_runs)
    marker = write_marker_certifying(tmp_path, first)
    run = certified_table_run(tmp_path, marker)
    assert run is not None and run.run_id == first.run_id
    frame = project_embedded_measurement_table(run.measurements_path())
    assert frame is not None
    assert set(frame["Pipeline_Marker"]) == {"A"}
    # Projected onto the CERTIFIED run's own column list, not the latest
    # run's: B measures shape, so B's list names columns A's table lacks.
    own = list(first.descriptor["measurements"]["measurement_columns"])
    assert frame.columns == own
    assert own != list(latest.descriptor["measurements"]["measurement_columns"])


def test_a_marker_naming_no_declared_run_certifies_nothing(store_with_two_runs, tmp_path):
    from phenotypic._cli._cli_recompile_recovery import certified_table_run

    first = list_table_runs(store_with_two_runs)[0]
    marker = write_marker_certifying(tmp_path, first)
    marker.write_text(
        marker.read_text(encoding="utf-8").replace(first.run_id, "9-unknown"),
        encoding="utf-8",
    )
    assert certified_table_run(tmp_path, marker) is None


def test_v2_table_paths_are_embedded_authority(store_with_two_runs):
    from phenotypic._cli._cli_chunk_writer import _chunk_state_key
    from phenotypic._cli._cli_parquet_agg import is_embedded_measurement_table

    store = store_with_two_runs
    for run in list_table_runs(store):
        assert is_embedded_measurement_table(run.measurements_path())
    # The v1 flat shape stays recognised (migrate's legacy read, until C15).
    assert is_embedded_measurement_table(store / "tables/measurements/table.parquet")
    assert not is_embedded_measurement_table(
        store / "tables/1-unknown/embeddings/m/objects/table.parquet"
    )
    assert not is_embedded_measurement_table(
        store / "tables/0-abc/measurements/table.parquet"
    )
    relative = list_table_runs(store)[1].measurements_path().relative_to(store)
    placed = store.parent / "out/results/ds/zarr/plate_1.ome.zarr" / relative
    assert _chunk_state_key(placed) == "ds/plate_1.parquet"


def test_a_one_image_local_full_run_finalizes(tmp_path, pipeline_file_a):
    import json

    import polars as pl
    from PIL import Image as PILImage

    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import master_measurements_parquet_path, metrics_json_path

    inputs = tmp_path / "in"
    (inputs / "ds").mkdir(parents=True)
    PILImage.fromarray(load_synth_yeast_plate().rgb[:].astype("uint8")).save(
        inputs / "ds" / "plate_1.tif"
    )
    out = tmp_path / "out"
    result = run_cli(inputs, out, pipeline=pipeline_file_a)
    assert result.exit_code == 0, result.output

    run = owned_table_run(one_store(out), file_sha256(pipeline_file_a))
    assert run is not None
    master = pl.read_parquet(master_measurements_parquet_path(out))
    assert master.height == pl.read_parquet(run.measurements_path()).height > 0
    # Identity comes from the v2 run-folder path, not the file stem "table".
    assert set(master[str(IMAGE.IMAGE_NAME)]) == {"plate_1"}
    metrics = json.loads(metrics_json_path(out).read_text(encoding="utf-8"))
    assert metrics["pipeline_sha256"] == file_sha256(pipeline_file_a)


def test_metrics_pipeline_sha_is_null_when_certified_runs_differ(store_with_two_runs):
    from phenotypic._cli._cli_finalize_run import _aggregated_pipeline_digests
    from phenotypic.sdk_._run_metrics import shared_pipeline_sha256

    first, latest = list_table_runs(store_with_two_runs)
    same = {first.measurements_path(): "ds"}
    assert _aggregated_pipeline_digests(same) == [first.pipeline_sha256]
    assert shared_pipeline_sha256(_aggregated_pipeline_digests(same)) == first.pipeline_sha256
    mixed = {first.measurements_path(): "ds", latest.measurements_path(): "ds"}
    assert shared_pipeline_sha256(_aggregated_pipeline_digests(mixed)) is None


# ---------------------------------------------------------------------------
# C13 (Task 3, Review Focus 3): ``--mode measure`` over the v2 layout
# ---------------------------------------------------------------------------

#: The two stores :func:`finished_tiny_run` writes.
_TINY_STEMS = ("plate_1", "plate_2")


def _master_height(out: Path) -> int:
    import polars as pl

    from phenotypic.sdk_ import master_measurements_parquet_path

    return pl.read_parquet(master_measurements_parquet_path(out)).height


def test_measure_twice_with_the_same_pipeline_keeps_one_folder_and_row_count(finished_tiny_run):
    out = finished_tiny_run
    # The forward run's own pipeline file, so the measure digest is its digest.
    pipeline = out.parent / "tiny_run.pht-pipe.json"
    rows = _master_height(out)
    for _ in range(2):
        result = run_cli(None, out, mode="measure", pipeline=pipeline)
        assert result.exit_code == 0, result.output
    for stem in _TINY_STEMS:
        store = one_store(out, stem=stem)
        [run] = list_table_runs(store)
        assert run.pipeline_sha256 == file_sha256(pipeline)
        assert _record_measurements(out, "ds", stem) == run.measurements_path()
    assert _master_height(out) == rows


def test_measure_with_another_pipeline_adds_a_folder_and_recompile_keeps_it(
    finished_tiny_run, pipeline_file_b
):
    import json

    from phenotypic.sdk_ import metrics_json_path
    from tests._tiny_run import tree_digest

    out = finished_tiny_run
    firsts = {stem: list_table_runs(one_store(out, stem=stem))[0] for stem in _TINY_STEMS}
    before = {stem: tree_digest(run.folder) for stem, run in firsts.items()}
    result = run_cli(None, out, mode="measure", pipeline=pipeline_file_b)
    assert result.exit_code == 0, result.output
    for stem, first in firsts.items():
        store = one_store(out, stem=stem)
        runs = list_table_runs(store)
        assert [run.order for run in runs] == [1, 2]
        assert tree_digest(first.folder) == before[stem]
        # D6a: the record certifies the folder measure mode wrote, not the
        # one the store's journal (still the forward run's) names.
        own = owned_table_run(store, file_sha256(pipeline_file_b))
        assert own is not None and own.order == 2
        assert _record_measurements(out, "ds", stem) == own.measurements_path()
    metrics = json.loads(metrics_json_path(out).read_text(encoding="utf-8"))
    assert metrics["pipeline_sha256"] == file_sha256(pipeline_file_b)
    result = run_cli(None, out, mode="recompile")
    assert result.exit_code == 0, result.output
    metrics = json.loads(metrics_json_path(out).read_text(encoding="utf-8"))
    assert metrics["pipeline_sha256"] == file_sha256(pipeline_file_b)    # I17: B's folder, not A's


# ---------------------------------------------------------------------------
# C14c: recompile reads the run folder each record certifies (spec D6a)
# ---------------------------------------------------------------------------


# `certified_older_than_latest` lives in `tests/_tiny_run.py` (shared with the
# migrate re-run tests).


def _store_digests(out: Path) -> dict[str, str]:
    from tests._tiny_run import tree_digest

    return {stem: tree_digest(one_store(out, stem=stem)) for stem in _TINY_STEMS}


def test_recompile_reads_the_certified_folder_not_the_latest(certified_older_than_latest):
    import json

    import polars as pl

    from phenotypic.schema import SHAPE
    from phenotypic.sdk_ import master_measurements_parquet_path, metrics_json_path

    out = certified_older_than_latest
    before = _store_digests(out)
    result = run_cli(None, out, mode="recompile")
    assert result.exit_code == 0, result.output
    master = pl.read_parquet(master_measurements_parquet_path(out))
    # Only B measures shape, and B's folder is the latest but certified by none.
    assert str(SHAPE.CIRCULARITY) not in master.columns
    metrics = json.loads(metrics_json_path(out).read_text(encoding="utf-8"))
    assert metrics["pipeline_sha256"] == file_sha256(out.parent / "tiny_run.pht-pipe.json")
    # Recompile writes no store byte: both run folders survive byte for byte.
    assert _store_digests(out) == before


def test_local_recompile_repairs_an_overlay_bound_to_the_certified_folder(
    certified_older_than_latest,
):
    from phenotypic.sdk_ import dataset_overlays_dir

    out = certified_older_than_latest
    overlay = dataset_overlays_dir(out, "ds") / "plate_1.png"
    overlay.unlink()
    before = _store_digests(out)
    result = run_cli(None, out, mode="recompile")
    assert result.exit_code == 0, result.output
    assert overlay.is_file()
    store = one_store(out, stem="plate_1")
    assert _record_measurements(out, "ds", "plate_1") == list_table_runs(store)[0].measurements_path()
    assert _store_digests(out) == before


def test_slurm_recompile_worker_aggregates_and_repairs_the_certified_folder(
    certified_older_than_latest, monkeypatch
):
    import json

    import polars as pl

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        TASK_MEASUREMENTS,
        build_recompile_tasks,
        recompile_attempt_dir,
    )
    from phenotypic._cli._cli_recompile_worker import main
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.schema import SHAPE
    from phenotypic.sdk_ import (
        DIR_RECOMPILE_SHARDS,
        RECOMPILE_TASK_MANIFEST_JSON,
        dataset_overlays_dir,
        shard_parquet_filename,
    )

    out = certified_older_than_latest
    certified = {
        stem: list_table_runs(one_store(out, stem=stem))[0].measurements_path()
        for stem in _TINY_STEMS
    }
    overlay = dataset_overlays_dir(out, "ds") / "plate_1.png"
    overlay.unlink()
    before = _store_digests(out)
    generation = "c14c-recompile-generation"
    initialize_slurm_lifecycle(out, generation=generation, mode="recompile")

    tasks = build_recompile_tasks(
        out, ["ds"], include_dataset_column=True, overlay_alpha=0.3,
        shard_size=10, attempt_id=generation,
    )
    [task] = [item for item in tasks if item["task_type"] == TASK_MEASUREMENTS]
    assert sorted(Path(path) for path in task["files"]) == sorted(certified.values())
    [repair] = task["overlay_repairs"]
    assert Path(repair["table_path"]) == certified["plate_1"]

    manifest = recompile_attempt_dir(out, generation) / RECOMPILE_TASK_MANIFEST_JSON
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"tasks": [task]}), encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "987654")
    result = CliRunner().invoke(
        main,
        [
            "--output-dir", str(out), "--task-manifest", str(manifest),
            "--task-index", "0", "--slurm-generation", generation,
            "--attempt-id", generation,
        ],
    )
    assert result.exit_code == 0, result.output
    shard = pl.read_parquet(
        manifest.parent / DIR_RECOMPILE_SHARDS / shard_parquet_filename(int(task["shard_id"]))
    )
    assert shard.height > 0 and str(SHAPE.CIRCULARITY) not in shard.columns
    assert overlay.is_file()
    assert _record_measurements(out, "ds", "plate_1") == certified["plate_1"]
    assert _store_digests(out) == before
