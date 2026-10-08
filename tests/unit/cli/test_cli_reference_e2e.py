"""SubtractBlank through the real CLI: process + full mode, continuation, staged GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner
from PIL import Image as PILImage
from pydantic import Field

import phenotypic
from phenotypic import Image, ImagePipeline
from phenotypic._cli import (
    _cli_execution_strategies,
    _cli_reference,
    _cli_staged_slurm,
    _cli_staged_strategy,
)
from phenotypic._cli._cli_failure_tracker import (
    read_terminal_failures,
    work_identity_for_image,
)
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_stage2_token import detector_slot, stage2_result_replayable
from phenotypic._cli._cli_staged_resume import staged_store_matches_work_id
from phenotypic._cli._cli_staged_orchestration import (
    initialize_orchestration,
    load_staged_manifest,
    write_staged_manifest,
)
from phenotypic._cli._cli_types import Dataset
from phenotypic.abc_ import ImageOperation
from phenotypic.abc_._ref_metadata import RefMetadata
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import SubtractBlank
from phenotypic.measure import MeasureSize
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import RefColumn, resolve_event_log_path, zarr_store_path
from phenotypic.sdk_._io_constants import (
    image_record_path,
    reference_manifest_path,
    reference_metadata_snapshot_path,
    terminal_failures_jsonl_path,
)
from phenotypic.sdk_.typing_ import OperationField
from tests._fakes.fake_gpu_detector import FakeGpuDetector
from tests.unit.cli._preflight_support import make_config

#: A lid scratch present in every frame of the plate, blank included. It is as
#: bright as a colony, so only a subtracted frame leaves the colony alone.
SCRATCH = 200
COLONY = 230


def _rgb(level: int, colony: tuple[slice, slice] | None = None) -> np.ndarray:
    image = np.full((32, 32, 3), level, dtype=np.uint8)
    image[0:4, 0:4] = SCRATCH
    if colony is not None:
        image[colony] = COLONY
    return image


@pytest.fixture
def run_inputs(tmp_path: Path):
    root = tmp_path / "images" / "plate1"
    root.mkdir(parents=True)
    tifffile.imwrite(root / "blank.tiff", _rgb(20))
    tifffile.imwrite(root / "blank2.tiff", _rgb(30))
    tifffile.imwrite(root / "t01.tiff", _rgb(20, (slice(10, 16), slice(10, 16))))
    tifffile.imwrite(root / "t02.tiff", _rgb(20, (slice(18, 26), slice(18, 26))))
    manifest = tmp_path / "frames.txt"
    manifest.write_text("plate1/t01.tiff\nplate1/t02.tiff\n", encoding="utf-8")
    table = tmp_path / "blank_map.csv"
    pd.DataFrame({"ImageName": ["t01", "t02"], "BlankImage": ["blank", "blank"]}).to_csv(
        table, index=False
    )
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        ImagePipeline(
            ops={"sb": SubtractBlank(), "det": OtsuDetector()}, meas={"size": MeasureSize()}
        ).to_json(),
        encoding="utf-8",
    )
    return tmp_path, root, manifest, table, pipeline


def _cli(*args: str):
    return CliRunner().invoke(phenotypic_cli, [*args, "--njobs", "1"], catch_exceptions=False)


def _process(base, manifest, table, pipeline, *extra):
    args = [
        "--pipeline", str(pipeline), "--input", str(base / "images"),
        "--output", str(base / "out"), "--image-manifest", str(manifest),
        "--mode", "process", "--layer", "detect_mat",
    ]
    if table is not None:
        args += ["--metadata", str(table)]
    return _cli(*args, *extra)


def _outputs(out: Path, pattern: str) -> list[Path]:
    """Exported files matching *pattern*, never the run's machine state."""
    return [p for p in out.rglob(pattern) if ".phenotypic" not in p.relative_to(out).parts]


def _output(base: Path, stem: str) -> Path:
    (found,) = _outputs(base / "out", f"{stem}.*")
    return found


def _expected_subtraction(frame: Path, blank: Path) -> np.ndarray:
    """What SubtractBlank must produce, from Image.imread alone (no hard-coded levels)."""
    target = Image.imread(frame).detect_mat[:]
    background = Image.imread(blank).detect_mat[:]
    return np.clip(target - background, 0.0, 1.0)


def test_process_mode_subtracts_the_blank(run_inputs):
    base, root, manifest, table, pipeline = run_inputs
    result = _process(base, manifest, table, pipeline)
    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(
        tifffile.imread(_output(base, "t01")),
        _expected_subtraction(root / "t01.tiff", root / "blank.tiff"),
        atol=1e-6,
    )
    assert (base / "out" / ".phenotypic" / "reference_metadata.csv").read_bytes() == (
        table.read_bytes()
    )


def test_process_mode_without_metadata_is_refused_before_any_write(run_inputs):
    base, _, manifest, _, pipeline = run_inputs
    result = _process(base, manifest, None, pipeline)
    assert result.exit_code != 0
    assert "PF-REF-NO-TABLE" in result.output
    # The refusal is in the read-only half: nothing under --output exists.
    assert not (base / "out").exists()


def _started(base: Path, image: str) -> int:
    """``started`` events for *image* (``timestamp|dataset|image|status|...``)."""
    log = resolve_event_log_path(base / "out")
    if not log.is_file():
        return 0
    lines = log.read_text(encoding="utf-8").splitlines()
    return sum(
        1 for line in lines if line.split("|")[2:4] == [image, "started"]
    )


def test_continuation_reruns_only_the_image_whose_blank_changed(run_inputs):
    base, _, manifest, table, pipeline = run_inputs
    assert _process(base, manifest, table, pipeline).exit_code == 0
    t01, t02 = _output(base, "t01"), _output(base, "t02")
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in (t01, t02)}
    started = {name: _started(base, name) for name in ("t01.tiff", "t02.tiff")}

    # Same command, no --metadata: falls back to the snapshot, reuses both.
    again = _process(base, manifest, None, pipeline)
    assert again.exit_code == 0, again.output
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in (t01, t02)} == before
    assert {name: _started(base, name) for name in started} == started

    edited = base / "blank_map_v2.csv"
    pd.DataFrame({"ImageName": ["t01", "t02"], "BlankImage": ["blank2", "blank"]}).to_csv(
        edited, index=False
    )
    third = _process(base, manifest, edited, pipeline)
    assert third.exit_code == 0, third.output
    assert _output(base, "t01").read_bytes() != before[t01.name][1]
    assert _output(base, "t02").stat().st_mtime_ns == before[t02.name][0]
    assert _started(base, "t01.tiff") == started["t01.tiff"] + 1
    assert _started(base, "t02.tiff") == started["t02.tiff"]


def _no_terminal_failures(out: Path) -> bool:
    path = terminal_failures_jsonl_path(out)
    return not path.exists() or not path.read_text(encoding="utf-8").strip()


def test_a_table_changed_after_planning_is_retried_without_retry_failures(
    run_inputs, monkeypatch
):
    """Review F2: a stale plan is not the image's fault, so it is never terminal."""
    base, _, manifest, table, pipeline = run_inputs
    out = base / "out"
    publish = _cli_reference.publish_reference_inputs

    def publish_then_edit_the_snapshot(config, datasets, output_dir):
        publish(config, datasets, output_dir)
        snapshot = reference_metadata_snapshot_path(output_dir)
        snapshot.write_text(
            snapshot.read_text(encoding="utf-8") + "t09,blank\n", encoding="utf-8"
        )

    monkeypatch.setattr(
        _cli_reference, "publish_reference_inputs", publish_then_edit_the_snapshot
    )
    first = _process(base, manifest, table, pipeline)
    assert first.exit_code != 0, first.output
    assert not _outputs(out, "t0*.tiff")
    assert _no_terminal_failures(out)

    monkeypatch.setattr(_cli_reference, "publish_reference_inputs", publish)
    second = _process(base, manifest, table, pipeline)
    assert second.exit_code == 0, second.output
    assert "recorded failure" not in second.output
    assert len(_outputs(out, "t0*.tiff")) == 2


# ---- a blank re-exported after planning (code review #1) --------------------


def _reexport_blank(root: Path) -> None:
    """Someone re-exports the plate's blank: same name, different pixels."""
    tifffile.imwrite(root / "blank.tiff", _rgb(35))


def _full(base, manifest, table, pipeline, *extra):
    args = [
        "--pipeline", str(pipeline), "--input", str(base / "images"),
        "--output", str(base / "out"), "--image-manifest", str(manifest),
        "--metadata", str(table), "--no-qc",
    ]
    return _cli(*args, *extra)


@pytest.mark.parametrize("mode", ["process", "full"])
def test_a_blank_replaced_after_planning_is_never_certified(
    run_inputs, monkeypatch, mode
):
    """The work-id names the blank's planned bytes, so the worker must use those.

    A blank re-exported between startup and the worker's load would otherwise be
    subtracted and certified under the old work-id, and continuation would then
    treat that output as current. The refusal is not the image's fault, so it
    is never terminal; the same command re-plans and completes it.
    """
    base, root, manifest, table, pipeline = run_inputs
    out = base / "out"
    run = _process if mode == "process" else _full
    publish = _cli_reference.publish_reference_inputs

    def publish_then_reexport(config, datasets, output_dir):
        publish(config, datasets, output_dir)
        _reexport_blank(root)

    monkeypatch.setattr(
        _cli_reference, "publish_reference_inputs", publish_then_reexport
    )
    first = run(base, manifest, table, pipeline)
    assert first.exit_code != 0, first.output
    for stem in ("t01", "t02"):
        assert not image_record_path(out, "plate1", stem).exists(), stem
    assert _no_terminal_failures(out)

    monkeypatch.setattr(_cli_reference, "publish_reference_inputs", publish)
    second = run(base, manifest, table, pipeline)
    assert second.exit_code == 0, second.output
    assert "recorded failure" not in second.output
    for stem in ("t01", "t02"):
        assert image_record_path(out, "plate1", stem).exists(), stem
    # The re-plan moved the work-id; the refused run's failed checkpoint is
    # obsolete, not a reason to fail the image (gate 29535357).
    assert _no_terminal_failures(out)


def test_the_staged_stage_1_refuses_a_blank_replaced_after_submission(
    run_inputs, monkeypatch
):
    from phenotypic._cli import _cli_staged_slurm_worker as worker

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    pipeline = _staged_pipeline(run_inputs, monkeypatch, "stage1")
    out = run_inputs[0] / "slurm_out"
    _, _, manifest = _slurm_setup(run_inputs, pipeline, out)
    epoch = _active_epoch(out)
    _reexport_blank(run_inputs[1])
    with pytest.raises(_cli_reference.ReferencePlanStaleError, match="blank"):
        worker.run_stage1_step(
            pipeline, out, "Image", manifest, 0, ".tiff", epoch=epoch
        )
    assert not image_record_path(out, "plate1", "t01").exists()
    assert _no_terminal_failures(out)

    # The next submission re-plans: a new digest, so a new work-id, over the
    # refused attempt's failed checkpoint. Stage 1 starts fresh under it.
    _, _, replanned = _slurm_setup(run_inputs, pipeline, out)
    assert replanned[0].work_id != manifest[0].work_id
    worker.run_stage1_step(
        pipeline, out, "Image", replanned, 0, ".tiff", epoch=epoch
    )
    assert staged_store_matches_work_id(
        zarr_store_path(out, "plate1", "t01"), replanned[0].work_id
    )
    assert _no_terminal_failures(out)


# ---- --metadata must be the CSV the run snapshots (code review #3) ----------


def _tree_bytes(root: Path) -> dict[str, bytes | None]:
    """Every path under *root*, with file bytes (``None`` for a directory)."""
    return {
        p.relative_to(root).as_posix(): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


@pytest.mark.parametrize(
    ("mode", "reads_references"),
    [("process", True), ("full", True), ("full", False)],
)
def test_a_parquet_metadata_is_refused_before_overwrite_deletes_anything(
    run_inputs, mode, reads_references
):
    base, _, manifest, table, pipeline = run_inputs
    out = base / "out"
    if not reads_references:
        pipeline = base / "plain_pipeline.json"
        pipeline.write_text(
            ImagePipeline(
                ops={"det": OtsuDetector()}, meas={"size": MeasureSize()}
            ).to_json(),
            encoding="utf-8",
        )
    run = _process if mode == "process" else _full
    assert run(base, manifest, table, pipeline).exit_code == 0
    parquet = base / "blank_map.parquet"
    pd.read_csv(table).to_parquet(parquet)
    before = _tree_bytes(out)

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline", str(pipeline), "--input", str(base / "images"),
            "--output", str(out), "--image-manifest", str(manifest),
            "--metadata", str(parquet), "--overwrite", "--njobs", "1",
            *(
                ["--mode", "process", "--layer", "detect_mat"]
                if mode == "process"
                else ["--no-qc"]
            ),
        ],
    )

    assert result.exit_code != 0, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        result.exception
    )
    assert "csv" in result.output.lower(), result.output
    assert _tree_bytes(out) == before


def _replan(out: Path, dataset: str, stem: str) -> None:
    """A concurrent invocation re-planned *stem*: its manifest digest changed."""
    path = reference_manifest_path(out)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["datasets"][dataset]["digests"][stem] = "replanned-by-another-invocation"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest), encoding="utf-8")
    tmp.replace(path)


def test_a_local_worker_never_certifies_an_image_replanned_mid_run(run_inputs, monkeypatch):
    """Review F1: the local process worker publishes under its pre-apply identity only."""
    base, _, manifest, table, pipeline = run_inputs
    out = base / "out"
    identity = _cli_execution_strategies.work_identity_for_image

    def identity_then_replan(config, dataset, image_path):
        computed = identity(config, dataset, image_path)
        if Path(image_path).name == "t01.tiff":
            _replan(out, dataset, "t01")
        return computed

    monkeypatch.setattr(
        _cli_execution_strategies, "work_identity_for_image", identity_then_replan
    )
    result = _process(base, manifest, table, pipeline)
    assert result.exit_code != 0, result.output
    assert not image_record_path(out, "plate1", "t01").exists()
    assert image_record_path(out, "plate1", "t02").exists()
    assert _no_terminal_failures(out)

    monkeypatch.setattr(_cli_execution_strategies, "work_identity_for_image", identity)
    again = _process(base, manifest, table, pipeline)
    assert again.exit_code == 0, again.output
    assert image_record_path(out, "plate1", "t01").exists()


def test_same_named_frames_resolve_per_dataset(tmp_path):
    """Both plates have a t01 and both blank files; the dataset decides which (spec §9).

    The rows name different blanks, so a lookup that drops the dataset key finds
    two answers for "t01" and fails, and a resolver that drops the dataset's
    directory reads the other plate's file (review F5).
    """
    images = tmp_path / "images"
    levels = {"plate1": {"blank": 20, "blank_b": 45}, "plate2": {"blank": 50, "blank_b": 60}}
    for plate, blanks in levels.items():
        root = images / plate
        root.mkdir(parents=True)
        for name, level in blanks.items():
            tifffile.imwrite(root / f"{name}.tiff", _rgb(level))
        frame = blanks["blank" if plate == "plate1" else "blank_b"]
        tifffile.imwrite(root / "t01.tiff", _rgb(frame, (slice(10, 16), slice(10, 16))))
    manifest = tmp_path / "frames.txt"
    manifest.write_text("plate1/t01.tiff\nplate2/t01.tiff\n", encoding="utf-8")
    table = tmp_path / "blank_map.csv"
    pd.DataFrame({
        "Dataset": ["plate1", "plate2"],
        "ImageName": ["t01", "t01"],
        "BlankImage": ["blank", "blank_b"],
    }).to_csv(table, index=False)
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(ImagePipeline(ops={"sb": SubtractBlank()}).to_json(), encoding="utf-8")
    result = _cli(
        "--pipeline", str(pipeline), "--input", str(images), "--output", str(tmp_path / "out"),
        "--image-manifest", str(manifest), "--metadata", str(table),
        "--mode", "process", "--layer", "detect_mat",
    )
    assert result.exit_code == 0, result.output
    for plate, blank in (("plate1", "blank"), ("plate2", "blank_b")):
        (out,) = _outputs(tmp_path / "out" / plate, "t01.*")
        np.testing.assert_allclose(
            tifffile.imread(out),
            _expected_subtraction(images / plate / "t01.tiff", images / plate / f"{blank}.tiff"),
            atol=1e-6,
        )


def test_full_mode_runs_and_joins_the_blank_column(run_inputs):
    base, _, manifest, table, pipeline = run_inputs
    result = _cli(
        "--pipeline", str(pipeline), "--input", str(base / "images"),
        "--output", str(base / "full"), "--image-manifest", str(manifest),
        "--metadata", str(table),
    )
    assert result.exit_code == 0, result.output
    measurements = pd.read_csv(base / "full" / "deliverables" / "measurements.csv")
    assert set(measurements["Metadata_ImageName"].astype(str)) == {"t01", "t02"}
    assert set(measurements["Metadata_BlankImage"].dropna()) == {"blank"}


def _with_the_reference_plate(base: Path) -> tuple[Path, Path]:
    """The input includes the plate's own blank frame, which names itself."""
    manifest = base / "with_blank.txt"
    manifest.write_text(
        "plate1/blank.tiff\nplate1/t01.tiff\nplate1/t02.tiff\n", encoding="utf-8"
    )
    table = base / "with_blank.csv"
    pd.DataFrame({
        "ImageName": ["blank", "t01", "t02"],
        "BlankImage": ["blank", "blank", "blank"],
    }).to_csv(table, index=False)
    return manifest, table


def test_a_full_run_may_include_the_reference_plate(run_inputs):
    """Every plate is treated alike: the reference plate subtracts itself to an
    empty image, so in full mode it fails measurement like any plate with no
    colonies -- an ordinary terminal failure, not a reference-metadata one. The
    other frames are subtracted and certified."""
    base, root, _, _, pipeline = run_inputs
    out = base / "out"
    manifest, table = _with_the_reference_plate(base)

    result = _full(base, manifest, table, pipeline)

    assert "PF-REF-SELF" not in result.output
    for stem in ("t01", "t02"):
        assert image_record_path(out, "plate1", stem).exists(), stem
    assert not image_record_path(out, "plate1", "blank").exists()
    (failure,) = read_terminal_failures(out)
    assert Path(failure.relative_image_path).name == "blank.tiff"
    detail = f"{failure.exception_type}: {failure.exception_message}\n{failure.traceback}"
    assert "No objects" in detail, detail
    assert "ReferenceLookupError" not in detail and "ReferenceImageError" not in detail
    np.testing.assert_allclose(
        Image.load_zarr(zarr_store_path(out, "plate1", "t01")).detect_mat[:],
        _expected_subtraction(root / "t01.tiff", root / "blank.tiff"),
        atol=1e-6,
    )


def test_a_process_run_exports_the_reference_plate_as_an_empty_image(run_inputs):
    base, root, _, _, pipeline = run_inputs
    manifest, table = _with_the_reference_plate(base)

    result = _process(base, manifest, table, pipeline)

    assert result.exit_code == 0, result.output
    plate = tifffile.imread(_output(base, "blank"))
    np.testing.assert_array_equal(plate, np.zeros_like(plate))
    np.testing.assert_allclose(
        tifffile.imread(_output(base, "t01")),
        _expected_subtraction(root / "t01.tiff", root / "blank.tiff"),
        atol=1e-6,
    )
    assert _no_terminal_failures(base / "out")


def _path_blank_inputs(run_inputs) -> tuple[Path, Path, Path]:
    """A blank outside the input folder, named by a path relative to ``run/``."""
    base = run_inputs[0]
    run = base / "run"
    blank = run / "blanks" / "b0.tiff"
    blank.parent.mkdir(parents=True)
    tifffile.imwrite(blank, _rgb(20))
    table = base / "path_blanks.csv"
    pd.DataFrame({
        "ImageName": ["t01", "t02"], "BlankImage": ["blanks/b0.tiff"] * 2,
    }).to_csv(table, index=False)
    return run, blank, table


def test_a_relative_file_path_blank_is_planned_once_and_used_from_any_cwd(
    run_inputs, monkeypatch
):
    """Startup resolves it against the submitter's cwd; the workers then run
    from another cwd holding a different file at the same relative path, and
    must still subtract the planned one."""
    base, root, manifest, _, pipeline = run_inputs
    run, blank, table = _path_blank_inputs(run_inputs)
    decoy = base / "elsewhere" / "blanks" / "b0.tiff"
    decoy.parent.mkdir(parents=True)
    tifffile.imwrite(decoy, _rgb(90))
    publish = _cli_reference.publish_reference_inputs

    def publish_then_move(config, datasets, output_dir):
        publish(config, datasets, output_dir)
        monkeypatch.chdir(base / "elsewhere")

    monkeypatch.chdir(run)
    monkeypatch.setattr(_cli_reference, "publish_reference_inputs", publish_then_move)
    result = _process(base, manifest, table, pipeline)
    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(
        tifffile.imread(_output(base, "t01")),
        _expected_subtraction(root / "t01.tiff", blank),
        atol=1e-6,
    )


def test_file_path_blanks_own_file_and_same_stem_are_ordinary_blanks(run_inputs):
    """t01 names its own file by path: it subtracts itself to an empty image.
    t02 names ``blanks/t02.tiff``, a different file with the same stem: an
    ordinary blank, subtracted as usual."""
    base, root, manifest, _, pipeline = run_inputs
    same_stem = base / "blanks" / "t02.tiff"
    same_stem.parent.mkdir()
    tifffile.imwrite(same_stem, _rgb(20))
    table = base / "own_paths.csv"
    pd.DataFrame({
        "ImageName": ["t01", "t02"],
        "BlankImage": [str(root / "t01.tiff"), str(same_stem)],
    }).to_csv(table, index=False)

    result = _process(base, manifest, table, pipeline)

    assert result.exit_code == 0, result.output
    own = tifffile.imread(_output(base, "t01"))
    np.testing.assert_array_equal(own, np.zeros_like(own))
    np.testing.assert_allclose(
        tifffile.imread(_output(base, "t02")),
        _expected_subtraction(root / "t02.tiff", same_stem),
        atol=1e-6,
    )
    assert _no_terminal_failures(base / "out")


def test_a_file_path_blank_replaced_after_planning_is_never_certified(
    run_inputs, monkeypatch
):
    """t01's blank is a file path, rewritten after planning: refused as stale.
    t02's blank is the bare name ``blank``, untouched: certified. That control
    is what proves the path resolved (a refused path would fail t01 alone too,
    but the refusal would not name a changed file)."""
    base, _, manifest, _, pipeline = run_inputs
    out = base / "out"
    run, blank, _ = _path_blank_inputs(run_inputs)
    table = base / "mixed_blanks.csv"
    pd.DataFrame({
        "ImageName": ["t01", "t02"], "BlankImage": ["blanks/b0.tiff", "blank"],
    }).to_csv(table, index=False)
    publish = _cli_reference.publish_reference_inputs

    def publish_then_reexport(config, datasets, output_dir):
        publish(config, datasets, output_dir)
        tifffile.imwrite(blank, _rgb(35))

    monkeypatch.chdir(run)
    monkeypatch.setattr(_cli_reference, "publish_reference_inputs", publish_then_reexport)
    first = _process(base, manifest, table, pipeline)
    assert first.exit_code != 0, first.output
    assert not image_record_path(out, "plate1", "t01").exists()
    assert image_record_path(out, "plate1", "t02").exists()
    assert _no_terminal_failures(out)
    events = resolve_event_log_path(out).read_text(encoding="utf-8")
    assert "changed since the run planned its references" in events
    assert str(blank.resolve()) in events


# ---- staged GPU: SubtractBlank inside the detector's own branch (review B1) --


class ReadsBlankName(ImageOperation, RefMetadata):
    """Test-only: reads this image's blank name from the active context."""

    blank_column: RefColumn = "Metadata_BlankImage"

    def _operate(self, image):
        self._ref_values(image)
        return image


class MeasureSizeReadingReferences(MeasureSize):
    """Test-only measurer whose private operation reads reference metadata."""

    probe: OperationField = Field(default_factory=ReadsBlankName)  # type: ignore[valid-type]

    def _operate(self, image):
        self.probe._ref_values(image)
        return super()._operate(image)


def _placements() -> dict[str, dict]:
    # The detector reads detect_mat, so the label count says whether the
    # detector saw the subtracted frame: unsubtracted, the lid scratch is a
    # second object as bright as the colony.
    gpu = FakeGpuDetector(input_layer="detect_mat")
    return {
        # Inside the detector's branch: SubtractBlank runs in the Stage-2 prefix.
        "branch": {"branch": ImagePipeline(ops={"sb": SubtractBlank(), "gpu": gpu})},
        # Top level, before the detector: it runs in Stage 1 (review F6).
        "stage1": {"sb": SubtractBlank(), "gpu": gpu},
    }


def _staged_pipeline(run_inputs, monkeypatch, placement: str) -> Path:
    """Write the staged pipeline for *placement*, its classes resolvable by name."""
    # Register FakeGpuDetector the permanent way first, as conftest's
    # write_stageable_pipeline does. PHENOTYPIC_PRELOAD_MODULES below makes the
    # CLI import this module, and that import is cached: had the attribute been
    # created by monkeypatch alone, its undo would delete it while the module
    # stayed imported, and every later test relying on the registration would
    # fail with UnknownOperationClassError.
    import tests._fakes.register_fake_gpu  # noqa: F401  (import side effect)

    for cls in (ReadsBlankName, MeasureSizeReadingReferences):
        monkeypatch.setattr(phenotypic, cls.__name__, cls, raising=False)
    monkeypatch.setenv("PHENOTYPIC_PRELOAD_MODULES", "tests._fakes.register_fake_gpu")
    base = run_inputs[0]
    pipeline = base / "staged_pipeline.json"
    # The measurer reads reference metadata too, so Stage 3's measure step
    # needs the context as much as its apply does (review F6).
    pipeline.write_text(
        ImagePipeline(
            ops=_placements()[placement],
            meas={"size": MeasureSizeReadingReferences()},
        ).to_json(),
        encoding="utf-8",
    )
    return pipeline


def _staged_args(run_inputs, pipeline: Path) -> list[str]:
    base, _, manifest, table, _ = run_inputs
    return [
        "--pipeline", str(pipeline), "--input", str(base / "images"),
        "--image-manifest", str(manifest), "--metadata", str(table),
    ]


@pytest.fixture(params=["branch", "stage1"])
def staged(run_inputs, monkeypatch, request):
    pipeline = _staged_pipeline(run_inputs, monkeypatch, request.param)
    base, root = run_inputs[:2]
    return base, root, _staged_args(run_inputs, pipeline)


def _labels(path: Path) -> int:
    return int(np.count_nonzero(np.unique(np.asarray(PILImage.open(path)))))


def test_unsubtracted_frames_would_show_the_scratch(run_inputs):
    """Control for the staged tests: without the blank the detector finds two objects."""
    _, root, _, _, _ = run_inputs
    frame = Image.imread(root / "t01.tiff")
    FakeGpuDetector(input_layer="detect_mat").apply(frame, inplace=True)
    assert frame.num_objects == 2


def test_staged_objmap_export_runs_the_blank_in_stage_2(staged):
    base, _, common = staged
    out = base / "staged_objmap"
    result = _cli(*common, "--output", str(out), "--mode", "process", "--layer", "objmap")
    assert result.exit_code == 0, result.output
    for stem in ("t01", "t02"):
        (exported,) = _outputs(out, f"{stem}.png")
        assert _labels(exported) == 1, stem


def test_staged_full_run_with_subtract_blank_in_the_detector_branch(staged):
    base, _, common = staged
    out = base / "staged_full"
    result = _cli(*common, "--output", str(out))
    assert result.exit_code == 0, result.output
    measurements = pd.read_csv(out / "deliverables" / "measurements.csv")
    per_image = measurements.groupby("Metadata_ImageName").size().to_dict()
    assert {str(k): v for k, v in per_image.items()} == {"t01": 1, "t02": 1}


# ---- staged GPU: one identity per image across all three stages (F1) ------
#
# A later invocation re-plans an image between the run's identity and a
# stage's apply. The stage must refuse with ReferencePlanStaleError -- never
# apply under one plan and certify under another -- and the refusal is never a
# terminal failure, so the next run retries the image.


#: stage -> (placement whose stage reads reference metadata, the core it calls).
#: Stage 2 applies reference operations only through the branch prefix.
_STAGE_CORES = {
    "stage1": ("stage1", "stage1_preprocess_core"),
    "stage2": ("branch", "stage2_detect_core"),
    "stage3": ("stage1", "stage3_merge_measure_core"),
}


@pytest.mark.parametrize("stage", sorted(_STAGE_CORES))
def test_a_local_staged_stage_refuses_an_image_replanned_mid_run(
    run_inputs, monkeypatch, stage
):
    placement, core_name = _STAGE_CORES[stage]
    common = _staged_args(
        run_inputs, _staged_pipeline(run_inputs, monkeypatch, placement)
    )
    out = run_inputs[0] / "staged_pin"
    real = getattr(_cli_staged_strategy, core_name)
    stale: list[str] = []

    def replan_then_run(*args, **kwargs):
        stem = args[3]  # every staged core takes the image stem fourth
        if stem == "t01":
            _replan(out, "plate1", "t01")
        try:
            return real(*args, **kwargs)
        except _cli_reference.ReferencePlanStaleError:
            stale.append(stem)
            raise

    monkeypatch.setattr(_cli_staged_strategy, core_name, replan_then_run)
    result = _cli(*common, "--output", str(out))
    assert stale == ["t01"], result.output
    assert not image_record_path(out, "plate1", "t01").exists()
    assert image_record_path(out, "plate1", "t02").exists()
    assert _no_terminal_failures(out)

    # The same command again re-plans from the table and completes the image.
    monkeypatch.setattr(_cli_staged_strategy, core_name, real)
    again = _cli(*common, "--output", str(out))
    assert again.exit_code == 0, again.output
    assert image_record_path(out, "plate1", "t01").exists()
    assert _no_terminal_failures(out)


def test_the_staged_objmap_export_refuses_an_image_replanned_after_stage_1(
    run_inputs, monkeypatch
):
    """The export publishes under the identity Stages 1-2 ran under, not a fresh one."""
    common = _staged_args(
        run_inputs, _staged_pipeline(run_inputs, monkeypatch, "branch")
    )
    out = run_inputs[0] / "staged_objmap_pin"
    strategy = _cli_staged_strategy.StagedGpuStrategy
    real = strategy._export_objmap_layer

    def replan_then_export(self, *args, **kwargs):
        _replan(out, "plate1", "t01")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(strategy, "_export_objmap_layer", replan_then_export)
    _cli(*common, "--output", str(out), "--mode", "process", "--layer", "objmap")
    assert not image_record_path(out, "plate1", "t01").exists()
    assert image_record_path(out, "plate1", "t02").exists()
    assert _no_terminal_failures(out)


def _slurm_setup(run_inputs, pipeline: Path, out: Path):
    """Startup + the staged submitter's manifest, as ``StagedSlurmStrategy`` builds it."""
    base, root, _, table, _ = run_inputs
    config = make_config(
        pipeline_json=pipeline, input_path=base / "images", output_dir=out,
        image_type="Image", metadata_csv=table,
    )
    dataset = Dataset(
        name="plate1", images=[root / "t01.tiff", root / "t02.tiff"],
        input_dir=root, output_dir=out / "plate1",
    )
    _cli_reference.publish_reference_inputs(config, [dataset], out)
    manifest = [
        _cli_staged_slurm.staged_manifest_entry(config, "plate1", image)
        for image in dataset.images
    ]
    return config, dataset, manifest


def _active_epoch(out: Path) -> str:
    """A live staged epoch, as the submitter initialises one.

    Without it the SLURM recorder (``_record_terminal_scientific_failure``)
    returns early on ``epoch is None``, and "no terminal failure was recorded"
    would hold whatever the stages raised (review T2).
    """
    epoch = "epoch-reference-pin"
    initialize_orchestration(
        out, epoch=epoch, mode="fresh",
        controller_config_path=out / "staged_controller.json",
    )
    return epoch


def test_the_staged_slurm_manifest_carries_each_images_reference_digest(run_inputs):
    """The submitter's one identity per image holds the digest its work-id names."""
    base = run_inputs[0]
    out = base / "out"
    config, dataset, manifest = _slurm_setup(run_inputs, run_inputs[4], out)
    for image, entry in zip(dataset.images, manifest):
        identity = work_identity_for_image(config, "plate1", image)
        assert entry.work_id == identity.work_id
        assert entry.reference_digest == identity.reference_digest
        assert entry.reference_digest == _cli_reference.reference_digest_for(
            out, "plate1", image.stem
        )
        assert entry.reference_digest is not None
    # The digest survives the manifest file the workers read.
    path = write_staged_manifest(base / "staged_manifest.json", manifest)
    assert load_staged_manifest(path) == manifest


@pytest.mark.parametrize("version", [2, 3])
def test_a_staged_manifest_written_before_the_digest_field_still_loads(
    tmp_path, version
):
    """An in-flight run's manifest has no ``reference_digest`` key.

    It loads with ``None``, which pins "no reference plan" -- true of every
    run planned before reference metadata existed (the field's docstring).
    """
    path = tmp_path / "staged_manifest.json"
    path.write_text(
        json.dumps({
            "version": version,
            "images": [{
                "dataset": "plate1", "image_name": "t01.tiff", "stem": "t01",
                "input_path": "/in/plate1/t01.tiff", "work_id": "w",
                "relative_image_path": "plate1/t01.tiff", "attempt_id": "a",
            }],
        }),
        encoding="utf-8",
    )
    (entry,) = load_staged_manifest(path)
    assert entry.work_id == "w"
    assert entry.reference_digest is None


def test_an_old_shape_entry_is_refused_under_a_reference_plan(
    run_inputs, monkeypatch
):
    """A missing digest never reads as "unpinned": Stage 1 refuses, non-terminally."""
    from dataclasses import replace

    from phenotypic._cli import _cli_staged_slurm_worker as worker

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    pipeline = _staged_pipeline(run_inputs, monkeypatch, "stage1")
    out = run_inputs[0] / "slurm_out"
    _, _, manifest = _slurm_setup(run_inputs, pipeline, out)
    old_shape = [replace(entry, reference_digest=None) for entry in manifest]
    epoch = _active_epoch(out)
    with pytest.raises(_cli_reference.ReferencePlanStaleError, match="t01"):
        worker.run_stage1_step(
            pipeline, out, "Image", old_shape, 0, ".tiff", epoch=epoch
        )
    assert _no_terminal_failures(out)


@pytest.mark.parametrize("replan_before", [1, 2, 3])
def test_the_staged_slurm_workers_refuse_an_image_replanned_after_submission(
    run_inputs, monkeypatch, replan_before
):
    """Each SLURM stage pins the digest its manifest entry was computed from.

    t02 is never re-planned: it is the control, and completes every stage.
    """
    from phenotypic._cli import _cli_staged_slurm_worker as worker

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # Stage 2 applies reference operations only through the branch prefix.
    placement = "branch" if replan_before == 2 else "stage1"
    pipeline = _staged_pipeline(run_inputs, monkeypatch, placement)
    out = run_inputs[0] / "slurm_out"
    _, _, manifest = _slurm_setup(run_inputs, pipeline, out)
    epoch = _active_epoch(out)
    slot = detector_slot(
        split_pipeline_at_gpu(ImagePipeline.from_json(pipeline)).gpu_path
    )

    def stage1(index: int) -> None:
        worker.run_stage1_step(
            pipeline, out, "Image", manifest, index, ".tiff", epoch=epoch
        )

    def stage3(index: int) -> None:
        worker.run_stage3_step(
            pipeline, out, "Image", manifest, index, ".tiff", epoch=epoch
        )

    if replan_before == 1:
        _replan(out, "plate1", "t01")
        with pytest.raises(_cli_reference.ReferencePlanStaleError, match="t01"):
            stage1(0)
    else:
        stage1(0)
    stage1(1)

    if replan_before == 2:
        _replan(out, "plate1", "t01")
    # One shard over both images: a refusal must not stop the control.
    worker.run_stage2_shard(pipeline, out, "Image", manifest, 0, 1, epoch=epoch)
    assert stage2_result_replayable(out, "plate1", "t01", slot) is (replan_before == 3)
    assert stage2_result_replayable(out, "plate1", "t02", slot)

    if replan_before == 3:
        _replan(out, "plate1", "t01")
        with pytest.raises(_cli_reference.ReferencePlanStaleError, match="t01"):
            stage3(0)
    stage3(1)

    assert not image_record_path(out, "plate1", "t01").exists()
    assert image_record_path(out, "plate1", "t02").exists()
    assert _no_terminal_failures(out)


def test_stage3_measure_refuses_a_replan_between_apply_and_measure(
    run_inputs, monkeypatch
):
    """Review M1: Stage 3's measure pins on its own, not through its apply.

    Each context checks the pin once, on entry. A re-plan that lands after the
    replay apply's check would otherwise be measured under the new plan and
    published under the old work-id -- F1 again.
    """
    from phenotypic._cli import _cli_staged_slurm_worker as worker

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    pipeline = _staged_pipeline(run_inputs, monkeypatch, "stage1")
    out = run_inputs[0] / "slurm_out"
    _, _, manifest = _slurm_setup(run_inputs, pipeline, out)
    epoch = _active_epoch(out)
    worker.run_stage1_step(
        pipeline, out, "Image", manifest, 0, ".tiff", epoch=epoch
    )
    worker.run_stage2_shard(pipeline, out, "Image", manifest, 0, 1, epoch=epoch)

    # Installed only around Stage 3: the replay pipeline is a plain
    # ImagePipeline (`substitute_at_path` copies `post_pipeline`).
    real_apply = ImagePipeline.apply
    applied: list[str] = []

    def apply_then_replan(self, image, *args, **kwargs):
        result = real_apply(self, image, *args, **kwargs)
        applied.append(image.name)
        _replan(out, "plate1", "t01")  # after the apply context's check
        return result

    monkeypatch.setattr(ImagePipeline, "apply", apply_then_replan)
    with pytest.raises(_cli_reference.ReferencePlanStaleError, match="t01"):
        worker.run_stage3_step(
            pipeline, out, "Image", manifest, 0, ".tiff", epoch=epoch
        )
    # The replay apply passed its own check, so the refusal is the measure's.
    assert applied, "the replay apply never ran; the apply context refused"
    assert not image_record_path(out, "plate1", "t01").exists()
    assert _no_terminal_failures(out)
