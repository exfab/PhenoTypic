"""SubtractBlank through the real CLI: process + full mode, continuation, staged GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner
from PIL import Image as PILImage

import phenotypic
from phenotypic import Image, ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import SubtractBlank
from phenotypic.measure import MeasureSize
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import resolve_event_log_path
from tests._fakes.fake_gpu_detector import FakeGpuDetector

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
    assert not (base / "out").exists() or not _outputs(base / "out", "t01.*")


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


def test_same_named_blanks_resolve_per_dataset(tmp_path):
    """Two datasets each hold their own "blank"; each frame uses its own (spec §9)."""
    images = tmp_path / "images"
    for plate, level in (("plate1", 20), ("plate2", 60)):
        root = images / plate
        root.mkdir(parents=True)
        tifffile.imwrite(root / "blank.tiff", _rgb(level))
        tifffile.imwrite(root / "t01.tiff", _rgb(level, (slice(10, 16), slice(10, 16))))
    manifest = tmp_path / "frames.txt"
    manifest.write_text("plate1/t01.tiff\nplate2/t01.tiff\n", encoding="utf-8")
    table = tmp_path / "blank_map.csv"
    pd.DataFrame({
        "Dataset": ["plate1", "plate2"],
        "ImageName": ["t01", "t01"],
        "BlankImage": ["blank", "blank"],
    }).to_csv(table, index=False)
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(ImagePipeline(ops={"sb": SubtractBlank()}).to_json(), encoding="utf-8")
    result = _cli(
        "--pipeline", str(pipeline), "--input", str(images), "--output", str(tmp_path / "out"),
        "--image-manifest", str(manifest), "--metadata", str(table),
        "--mode", "process", "--layer", "detect_mat",
    )
    assert result.exit_code == 0, result.output
    for plate in ("plate1", "plate2"):
        (out,) = _outputs(tmp_path / "out" / plate, "t01.*")
        np.testing.assert_allclose(
            tifffile.imread(out),
            _expected_subtraction(images / plate / "t01.tiff", images / plate / "blank.tiff"),
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


# ---- staged GPU: SubtractBlank inside the detector's own branch (review B1) --


@pytest.fixture
def staged(run_inputs, monkeypatch):
    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False)
    monkeypatch.setenv("PHENOTYPIC_PRELOAD_MODULES", "tests._fakes.register_fake_gpu")
    base, root, manifest, table, _ = run_inputs
    # The detector reads detect_mat, so the label count says whether Stage 2
    # saw the subtracted frame: unsubtracted, the lid scratch is a second
    # object as bright as the colony.
    branch = ImagePipeline(
        ops={"sb": SubtractBlank(), "gpu": FakeGpuDetector(input_layer="detect_mat")}
    )
    pipeline = base / "staged_pipeline.json"
    pipeline.write_text(
        ImagePipeline(ops={"branch": branch}, meas={"size": MeasureSize()}).to_json(),
        encoding="utf-8",
    )
    common = [
        "--pipeline", str(pipeline), "--input", str(base / "images"),
        "--image-manifest", str(manifest), "--metadata", str(table),
    ]
    return base, root, common


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
