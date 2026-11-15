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
from phenotypic.sdk_ import list_table_runs, table_run_for_pipeline
from phenotypic.sdk_._image_record import read_image_record
from tests._tiny_run import one_store, run_cli


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
        run = table_run_for_pipeline(store, file_sha256(pipeline_file_a))
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
    own = table_run_for_pipeline(store, file_sha256(pipeline_file_a))
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
    run = table_run_for_pipeline(store, file_sha256(pipeline))
    assert run is not None and run.run_id == f"1-{file_sha256(pipeline)[:12]}"
    assert _record_measurements(out, images.name, "img") == run.measurements_path()
