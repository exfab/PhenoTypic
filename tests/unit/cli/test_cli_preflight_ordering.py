"""Nothing under ``--output`` changes before validation and the dry-run exit.

Spec: ``docs/superpowers/specs/2026-09-24-cli-preflight/design.md`` §1
(findings F1, F2, F3, F27). Before this change ``phenotypic_cli`` cleared
``.phenotypic/`` for ``--restart`` and ran ``shutil.rmtree`` for
``--overwrite`` *above* pipeline validation and the ``--dry-run`` exit, so:

- a corrupt pipeline with ``--overwrite`` deleted the previous run and only
  then reported "Pipeline loading failed";
- ``--overwrite --dry-run`` deleted the output and exited 0;
- ``--restart --dry-run`` cleared machine state and bumped the restart epoch;
- ``--overwrite`` deleted any run input stored under ``--output``.

Every case asserts the exit status, the message and the on-disk state,
because each assertion alone passes on broken code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.phenotypicCLI import phenotypic_cli

PREVIOUS = "previous-run.txt"


def _plate_image() -> np.ndarray:
    """A small RGB plate: dark agar with two bright colonies."""
    image = np.full((32, 32, 3), 20, dtype=np.uint8)
    image[6:12, 6:12] = 220
    image[20:27, 18:25] = 200
    return image


def _write_tree(root: Path) -> Path:
    image = root / "plate1" / "img001.tiff"
    image.parent.mkdir(parents=True)
    tifffile.imwrite(image, _plate_image())
    return root


@pytest.fixture
def image_tree(tmp_path: Path) -> Path:
    """One dataset holding one decodable TIFF (review R12: never placeholder bytes)."""
    return _write_tree(tmp_path / "images")


@pytest.fixture
def valid_pipeline(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.json"
    pipeline = ImagePipeline(
        ops={"det": OtsuDetector()}, meas={"size": MeasureSize()}
    )
    # write_text, not to_json(path): the latter appends ``.pht-pipe``.
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


@pytest.fixture
def previous_run(tmp_path: Path) -> Path:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / PREVIOUS).write_text("keep me", encoding="utf-8")
    return output_dir


def _invoke(*args: str):
    return CliRunner().invoke(phenotypic_cli, [*args])


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_overwrite_with_a_corrupt_pipeline_keeps_the_previous_run(
    image_tree: Path, previous_run: Path, tmp_path: Path
) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ not json", encoding="utf-8")

    result = _invoke(
        "--pipeline", str(corrupt),
        "--input", str(image_tree),
        "--output", str(previous_run),
        "--overwrite",
    )

    assert result.exit_code != 0, result.output
    assert "Pipeline loading failed" in result.output, result.output
    assert (previous_run / PREVIOUS).read_text(encoding="utf-8") == "keep me"


def test_overwrite_dry_run_previews_and_deletes_nothing(
    valid_pipeline: Path, image_tree: Path, previous_run: Path
) -> None:
    result = _invoke(
        "--pipeline", str(valid_pipeline),
        "--input", str(image_tree),
        "--output", str(previous_run),
        "--overwrite",
        "--dry-run",
    )

    assert result.exit_code == 0, result.output
    assert "would delete" in result.output, result.output
    assert sorted(p.name for p in previous_run.iterdir()) == [PREVIOUS]
    assert (previous_run / PREVIOUS).read_text(encoding="utf-8") == "keep me"


def test_restart_dry_run_leaves_machine_state_untouched(
    valid_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    common = (
        "--pipeline", str(valid_pipeline),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--image-type", "Image",
        "--njobs", "1",
        "--no-qc",
    )
    first = _invoke(*common)
    assert first.exit_code == 0, first.output
    machine_state = output_dir / ".phenotypic"
    assert machine_state.is_dir(), first.output
    before = _tree_digest(machine_state)
    assert before, "the real run left no machine state to protect"

    result = _invoke(*common, "--restart", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "would clear" in result.output, result.output
    assert _tree_digest(machine_state) == before


@pytest.mark.parametrize(
    "option", ["--input", "--pipeline", "--metadata", "--image-manifest"]
)
def test_overwrite_refuses_each_run_input_inside_the_output(
    option: str,
    valid_pipeline: Path,
    image_tree: Path,
    previous_run: Path,
) -> None:
    """F3 and F27: the delete would remove a file the run is about to read."""
    paths = {
        "--input": image_tree,
        "--pipeline": valid_pipeline,
        "--metadata": None,
        "--image-manifest": None,
    }
    inside = previous_run / "deliverables"
    inside.mkdir()
    if option == "--input":
        target = _write_tree(inside / "images")
    elif option == "--pipeline":
        target = inside / "pipeline.json"
        target.write_bytes(valid_pipeline.read_bytes())
    elif option == "--metadata":
        target = inside / "metadata.csv"
        target.write_text("ImageName,Strain\nimg001,WT\n", encoding="utf-8")
    else:
        target = inside / "approved.images"
        target.write_text("plate1/img001.tiff\n", encoding="utf-8")
    paths[option] = target

    args = [
        "--pipeline", str(paths["--pipeline"]),
        "--input", str(paths["--input"]),
        "--output", str(previous_run),
        "--overwrite",
    ]
    if paths["--metadata"] is not None:
        args += ["--metadata", str(paths["--metadata"])]
    if paths["--image-manifest"] is not None:
        args += ["--image-manifest", str(paths["--image-manifest"])]

    result = _invoke(*args)

    assert result.exit_code != 0, result.output
    assert option in result.output, result.output
    assert target.exists()
    assert (previous_run / PREVIOUS).read_text(encoding="utf-8") == "keep me"


def test_restart_refuses_a_run_input_inside_machine_state(
    valid_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    hidden = output_dir / ".phenotypic" / "pipeline.json"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(valid_pipeline.read_bytes())

    result = _invoke(
        "--pipeline", str(hidden),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--restart",
    )

    assert result.exit_code != 0, result.output
    assert "--pipeline" in result.output, result.output
    assert hidden.exists()


def test_overlap_refusal_survives_skip_validation(
    valid_pipeline: Path, previous_run: Path
) -> None:
    """The overlap refusal protects the run's own inputs, so it is not skippable."""
    inputs = _write_tree(previous_run / "images")

    result = _invoke(
        "--pipeline", str(valid_pipeline),
        "--input", str(inputs),
        "--output", str(previous_run),
        "--overwrite",
        "--skip-validation",
    )

    assert result.exit_code != 0, result.output
    assert "--input" in result.output, result.output
    assert (inputs / "plate1" / "img001.tiff").exists()
