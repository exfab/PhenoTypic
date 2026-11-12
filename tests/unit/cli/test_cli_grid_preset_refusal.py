"""``--nrows``/``--ncols`` that contradict the pipeline's grid preset are refused.

In ``--mode full`` the flags shape each ``GridImage``, while a pipeline that
presets ``nrows``/``ncols`` makes ``measure()`` add a ``CenteredAutoGridFinder``
with the preset shape. Two different grids disagree on the colonies' ``Grid_*``
positions; ``measure()`` used to drop those colonies silently and now raises
``MeasurementConflictError`` -- so without a refusal, every image would fail one
at a time. The refusal is one usage error, before anything under ``--output``
changes, and regardless of ``--skip-validation``.

Each refusal case asserts the exit status, the message, the absence of a
traceback, and that ``--output`` was never created, because each alone passes
on broken code. The last case is the other half of the contract: a grid
conflict the startup check does not cover (an explicit grid finder in ``meas``)
fails per image, and the terminal-failure record carries the error's message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tifffile
from click.testing import CliRunner

from phenotypic import ImagePipeline
from phenotypic._cli._cli_failure_tracker import read_terminal_failures
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.grid import CenteredAutoGridFinder
from phenotypic.measure import MeasureSize
from phenotypic.phenotypicCLI import phenotypic_cli


@pytest.fixture(scope="module")
def plate_rgb():
    """The synthetic 8 x 12 plate's pixels: 96 colonies, one per well."""
    return load_synth_yeast_plate().rgb[:]


@pytest.fixture
def image_tree(tmp_path: Path, plate_rgb) -> Path:
    root = tmp_path / "images"
    image = root / "plate1" / "img001.tiff"
    image.parent.mkdir(parents=True)
    tifffile.imwrite(image, plate_rgb)
    return root


def _pipeline_file(tmp_path: Path, pipeline: ImagePipeline) -> Path:
    path = tmp_path / "pipeline.json"
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


@pytest.fixture
def preset_pipeline(tmp_path: Path) -> Path:
    """A pipeline presetting the plate's own 8 x 12 grid."""
    return _pipeline_file(
        tmp_path,
        ImagePipeline(
            ops={"det": OtsuDetector()},
            meas={"size": MeasureSize()},
            nrows=8,
            ncols=12,
        ),
    )


def _invoke(*args: str):
    return CliRunner().invoke(phenotypic_cli, [*args, "--njobs", "1"])


def _assert_refused(result, output_dir: Path, image_grid: str) -> None:
    assert result.exit_code != 0, result.output
    assert f"lay the images out as {image_grid}," in result.output, result.output
    assert "presets nrows=8, ncols=12" in result.output, result.output
    assert "--nrows 8 --ncols 12" in result.output, result.output
    assert "Traceback" not in result.output, result.output
    assert not output_dir.exists(), sorted(output_dir.rglob("*"))


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param([], id="full"),
        pytest.param(["--dry-run"], id="dry-run"),
        pytest.param(["--skip-validation"], id="skip-validation"),
        pytest.param(["--overwrite"], id="overwrite"),
    ],
)
def test_flags_unlike_the_preset_are_refused_before_any_write(
    extra: list[str], preset_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"

    result = _invoke(
        *extra,
        "--pipeline", str(preset_pipeline),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--nrows", "4",
        "--ncols", "6",
    )

    _assert_refused(result, output_dir, "4 rows x 6 columns")


def test_one_flag_unlike_the_preset_is_refused(
    preset_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    """``--nrows`` alone still reshapes the image: 4 x 12 against 8 x 12."""
    output_dir = tmp_path / "out"

    result = _invoke(
        "--pipeline", str(preset_pipeline),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--nrows", "4",
    )

    _assert_refused(result, output_dir, "4 rows x 12 columns")


def test_flags_like_the_preset_run_and_measure_every_colony(
    preset_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"

    result = _invoke(
        "--pipeline", str(preset_pipeline),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--nrows", "8",
        "--ncols", "12",
    )

    assert result.exit_code == 0, result.output
    assert "presets nrows" not in result.output
    assert read_terminal_failures(output_dir) == []


def test_process_mode_is_not_refused(
    preset_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    """``--mode process`` applies ops only and never measures."""
    output_dir = tmp_path / "out"

    result = _invoke(
        "--mode", "process", "--layer", "objmap",
        "--pipeline", str(preset_pipeline),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--nrows", "4",
        "--ncols", "6",
    )

    assert result.exit_code == 0, result.output
    assert "presets nrows" not in result.output


def test_an_explicit_grid_finder_conflict_is_recorded_per_image(
    image_tree: Path, tmp_path: Path
) -> None:
    """Not a preset, so not refused at startup; the image fails with the reason."""
    pipeline = _pipeline_file(
        tmp_path,
        ImagePipeline(
            ops={"det": OtsuDetector()},
            meas={
                "plate_grid": CenteredAutoGridFinder(nrows=4, ncols=6),
                "size": MeasureSize(),
            },
        ),
    )
    output_dir = tmp_path / "out"

    result = _invoke(
        "--pipeline", str(pipeline),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--nrows", "8",
        "--ncols", "12",
    )

    assert "presets nrows" not in result.output, result.output
    (failure,) = read_terminal_failures(output_dir)
    detail = (
        f"{failure.exception_type}: {failure.exception_message}\n"
        f"{failure.traceback}"
    )
    assert "MeasurementConflictError" in detail, detail
    assert "Grid_RowNum" in failure.exception_message, detail
    assert "'plate_grid' (CenteredAutoGridFinder)" in failure.exception_message, detail


def test_an_explicit_grid_finder_beside_a_preset_is_not_refused(
    image_tree: Path, tmp_path: Path
) -> None:
    """``measure()`` adds no finder for the preset when ``meas`` has one.

    So the preset's 8 x 12 lays out nothing, and flags matching the explicit
    finder's 4 x 6 run cleanly. Refusing here would refuse a valid run.
    """
    pipeline = _pipeline_file(
        tmp_path,
        ImagePipeline(
            ops={"det": OtsuDetector()},
            meas={
                "plate_grid": CenteredAutoGridFinder(nrows=4, ncols=6),
                "size": MeasureSize(),
            },
            nrows=8,
            ncols=12,
        ),
    )
    output_dir = tmp_path / "out"

    result = _invoke(
        "--pipeline", str(pipeline),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--nrows", "4",
        "--ncols", "6",
    )

    assert result.exit_code == 0, result.output
    assert "presets nrows" not in result.output
    assert read_terminal_failures(output_dir) == []


@pytest.mark.parametrize(
    ("preset_nrows", "preset_ncols"),
    [
        pytest.param(None, None, id="no-preset"),
        pytest.param(8, None, id="nrows-only"),
        pytest.param(None, 12, id="ncols-only"),
    ],
)
def test_a_partial_or_absent_preset_is_not_refused(
    image_tree: Path,
    tmp_path: Path,
    preset_nrows: int | None,
    preset_ncols: int | None,
) -> None:
    """``measure()`` adds a finder only for a preset of both; the flags rule."""
    pipeline = _pipeline_file(
        tmp_path,
        ImagePipeline(
            ops={"det": OtsuDetector()},
            meas={"size": MeasureSize()},
            nrows=preset_nrows,
            ncols=preset_ncols,
        ),
    )
    output_dir = tmp_path / "out"

    result = _invoke(
        "--pipeline", str(pipeline),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--nrows", "4",
        "--ncols", "6",
    )

    assert result.exit_code == 0, result.output
    assert "presets nrows" not in result.output
    assert read_terminal_failures(output_dir) == []
